"""Compare PlainUNet, MultiResUNet (regular), and MultiResUNet (harmonic) architectures.

Example:
    python scripts/compare_with_harmonic.py --clip_id synthetic_01 --gap_ms 400 --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.audio.loader import load_audio
from src.audio.stft import (
    channels_to_stft,
    compute_stft,
    crop_to_shape,
    frame_mask,
    inverse_stft,
    pad_to_multiple,
    stft_to_channels,
    tf_mask,
)
from src.evaluation.nmse import missing_region_nmse, nmse
from src.models.multires_unet import MultiResUNet
from src.models.unet import PlainUNet
from src.training.deep_prior import optimize_deep_prior
from src.training.noise import make_input_noise
from src.utils.visualization import plot_comparison_curves, plot_spectrogram


def _config() -> dict:
    with (REPO / "configs" / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _args(cfg: dict) -> argparse.Namespace:
    optimization = cfg["optimization"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--clip_id", required=True)
    parser.add_argument("--gap_ms", type=float, required=True)
    parser.add_argument("--seed_idx", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=optimization["epochs"])
    parser.add_argument("--lr", type=float, default=optimization["learning_rate"])
    parser.add_argument("--out_dir", default="outputs/comparison_harmonic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["all", "plainunet", "multiresunet", "multiresunet_harmonic"],
        default="all",
        help="Which model(s) to evaluate (default: all)",
    )
    return parser.parse_args()


def _run_model(
    model_name: str,
    model_cls: type,
    model_kwargs: dict,
    noise: torch.Tensor,
    observed_channels: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    clean_channels: torch.Tensor,
    clean_audio: torch.Tensor,
    sample_mask: torch.Tensor,
    original_shape: tuple[int, int],
    args: argparse.Namespace,
    stft_cfg: dict,
    audio_cfg: dict,
    out_dir: Path,
) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = model_cls(**model_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    started = time.perf_counter()
    result = optimize_deep_prior(
        model,
        noise,
        observed_channels.to(device),
        tf_mask_tensor.to(device),
        optimizer,
        args.epochs,
        perturbation_variance=_config()["optimization"]["perturbation_variance"],
        log_every=args.log_every,
        reference=clean_channels.to(device),
        generator=generator,
    )
    elapsed = time.perf_counter() - started

    output_channels = crop_to_shape(result["final_output"].cpu(), original_shape)
    reconstruction = inverse_stft(
        channels_to_stft(output_channels),
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        length=len(clean_audio),
    )

    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "spectrograms").mkdir(parents=True, exist_ok=True)

    clip_tag = f"{args.clip_id}_{int(args.gap_ms)}ms"
    wav_path = out_dir / "audio" / f"{clip_tag}_{model_name}_reconstructed.wav"
    sf.write(str(wav_path), reconstruction.numpy(), audio_cfg["sample_rate"])

    frame = frame_mask(sample_mask, **stft_cfg)
    png_path = out_dir / "spectrograms" / f"{clip_tag}_{model_name}_reconstruction.png"
    plot_spectrogram(
        channels_to_stft(output_channels),
        audio_cfg["sample_rate"],
        stft_cfg["hop_length"],
        stft_cfg["n_fft"],
        title=f"{model_name} reconstruction",
        out_path=png_path,
        lost_frames=frame.numpy(),
    )

    summary = {
        "model": model_name,
        "final_train_loss": float(result["loss_history"][-1]),
        "final_val_loss": float(result["val_loss_history"][-1][1]) if result["val_loss_history"] else None,
        "nmse_tot": float(nmse(clean_audio, reconstruction)),
        "nmse_miss": float(missing_region_nmse(clean_audio, reconstruction, sample_mask)),
        "parameter_count": model.count_parameters(),
        "wall_clock_seconds": float(elapsed),
        "loss_history": [float(v) for v in result["loss_history"]],
        "val_loss_history": [(int(e), float(v)) for e, v in result["val_loss_history"]],
    }

    summary_path = out_dir / f"{clip_tag}_{model_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    cfg = _config()
    args = _args(cfg)
    device = torch.device(args.device)

    manifest_path = (REPO / args.manifest).resolve()
    rows = pd.read_csv(manifest_path)
    selected = rows[
        (rows["clip_id"] == args.clip_id)
        & np.isclose(rows["gap_ms"].astype(float), args.gap_ms)
        & (rows["seed_idx"] == args.seed_idx)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one manifest row, found {len(selected)}")
    row = selected.iloc[0]

    audio_cfg, stft_cfg = cfg["audio"], cfg["stft"]
    resolve = lambda value: (REPO / str(value)).resolve()

    clean = load_audio(resolve(row["clean_path"]), audio_cfg["sample_rate"], audio_cfg["mono"])
    corrupted = load_audio(resolve(row["corrupted_path"]), audio_cfg["sample_rate"], audio_cfg["mono"])
    sample_mask = torch.from_numpy(np.load(resolve(row["mask_path"])).astype(np.float32))

    clean_stft = compute_stft(clean, **stft_cfg)
    corrupted_stft = compute_stft(corrupted, **stft_cfg)
    clean_channels = stft_to_channels(clean_stft)
    observed_channels = stft_to_channels(corrupted_stft)
    frame = frame_mask(sample_mask, **stft_cfg)
    tf = tf_mask(frame, clean_channels.shape[-2])

    observed_channels, original_shape = pad_to_multiple(observed_channels)
    clean_channels, _ = pad_to_multiple(clean_channels)
    tf, _ = pad_to_multiple(tf.unsqueeze(0))
    tf = tf.squeeze(0)

    # Generate noise ONCE to guarantee identical input to all models
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = make_input_noise(
        tuple(observed_channels.shape),
        variance=cfg["optimization"]["noise_variance"],
        generator=generator,
        device=device,
    )

    out_dir = (REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = []
    if args.model in ["all", "plainunet"]:
        models_to_run.append(("plainunet", PlainUNet, {
            "in_channels": cfg["model"]["input_channels"],
            "out_channels": cfg["model"]["output_channels"],
            "negative_slope": cfg["model"]["negative_slope"],
        }))
    if args.model in ["all", "multiresunet"]:
        models_to_run.append(("multiresunet", MultiResUNet, {
            "in_channels": cfg["model"]["input_channels"],
            "out_channels": cfg["model"]["output_channels"],
            "negative_slope": cfg["model"]["negative_slope"],
            "use_harmonic": False,
        }))
    if args.model in ["all", "multiresunet_harmonic"]:
        models_to_run.append(("multiresunet_harmonic", MultiResUNet, {
            "in_channels": cfg["model"]["input_channels"],
            "out_channels": cfg["model"]["output_channels"],
            "negative_slope": cfg["model"]["negative_slope"],
            "use_harmonic": True,
            "harmonic_anchors": cfg["model"].get("harmonic_anchors"),
        }))

    results = {}
    for name, cls, kwargs in models_to_run:
        summary = _run_model(
            name,
            cls,
            kwargs,
            noise.clone(),
            observed_channels,
            tf,
            clean_channels,
            clean,
            sample_mask,
            original_shape,
            args,
            stft_cfg,
            audio_cfg,
            out_dir,
        )
        results[name] = summary

    clip_tag = f"{args.clip_id}_{int(args.gap_ms)}ms"

    if args.model == "all":
        comparison_png = out_dir / f"{clip_tag}_comparison.png"
        models_dict = {
            "PlainUNet": {
                "train_loss": results["plainunet"]["loss_history"],
                "val_loss": results["plainunet"]["val_loss_history"],
            },
            "MultiResUNet": {
                "train_loss": results["multiresunet"]["loss_history"],
                "val_loss": results["multiresunet"]["val_loss_history"],
            },
            "MultiResUNet (Harmonic)": {
                "train_loss": results["multiresunet_harmonic"]["loss_history"],
                "val_loss": results["multiresunet_harmonic"]["val_loss_history"],
            },
        }
        plot_comparison_curves(models_dict=models_dict, out_path=comparison_png)

        comparison_table = {
            "clip_id": args.clip_id,
            "gap_ms": args.gap_ms,
            "epochs": args.epochs,
            "plainunet": {
                "parameter_count": results["plainunet"]["parameter_count"],
                "final_train_loss": results["plainunet"]["final_train_loss"],
                "final_val_loss": results["plainunet"]["final_val_loss"],
                "nmse_tot": results["plainunet"]["nmse_tot"],
                "nmse_miss": results["plainunet"]["nmse_miss"],
                "wall_clock_seconds": results["plainunet"]["wall_clock_seconds"],
            },
            "multiresunet": {
                "parameter_count": results["multiresunet"]["parameter_count"],
                "final_train_loss": results["multiresunet"]["final_train_loss"],
                "final_val_loss": results["multiresunet"]["final_val_loss"],
                "nmse_tot": results["multiresunet"]["nmse_tot"],
                "nmse_miss": results["multiresunet"]["nmse_miss"],
                "wall_clock_seconds": results["multiresunet"]["wall_clock_seconds"],
            },
            "multiresunet_harmonic": {
                "parameter_count": results["multiresunet_harmonic"]["parameter_count"],
                "final_train_loss": results["multiresunet_harmonic"]["final_train_loss"],
                "final_val_loss": results["multiresunet_harmonic"]["final_val_loss"],
                "nmse_tot": results["multiresunet_harmonic"]["nmse_tot"],
                "nmse_miss": results["multiresunet_harmonic"]["nmse_miss"],
                "wall_clock_seconds": results["multiresunet_harmonic"]["wall_clock_seconds"],
            },
        }
        table_path = out_dir / f"{clip_tag}_comparison_summary.json"
        table_path.write_text(json.dumps(comparison_table, indent=2), encoding="utf-8")
        print("--- Comparison Summary (Harmonic) ---")
        print(json.dumps(comparison_table, indent=2))


if __name__ == "__main__":
    main()
