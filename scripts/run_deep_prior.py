"""Run one CPU-friendly plain U-Net deep-prior reconstruction.

Example:
    python scripts/run_deep_prior.py --clip_id synthetic_01 --gap_ms 400 --seed_idx 0

The default 5000 epochs is intentionally a research setting; on CPU, a
five-second clip can take several minutes depending on hardware. Use
``--epochs 1000`` for a quicker smoke run.
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
from src.models.unet import PlainUNet
from src.training.deep_prior import optimize_deep_prior
from src.training.noise import make_input_noise
from src.utils.visualization import plot_loss_curves, plot_spectrogram


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
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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

    torch.manual_seed(args.seed)
    model = PlainUNet(
        in_channels=cfg["model"]["input_channels"],
        out_channels=cfg["model"]["output_channels"],
        negative_slope=cfg["model"]["negative_slope"],
    ).to(device)
    print(f"Trainable parameters: {model.count_parameters()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = make_input_noise(
        tuple(observed_channels.shape),
        variance=cfg["optimization"]["noise_variance"],
        generator=generator,
        device=device,
    )

    started = time.perf_counter()
    result = optimize_deep_prior(
        model,
        noise,
        observed_channels.to(device),
        tf.to(device),
        optimizer,
        args.epochs,
        perturbation_variance=cfg["optimization"]["perturbation_variance"],
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
        length=len(clean),
    )

    out_dir = (REPO / args.out_dir).resolve()
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "spectrograms").mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "audio" / f"{args.clip_id}_{int(args.gap_ms)}ms_reconstructed.wav"), reconstruction.numpy(), audio_cfg["sample_rate"])
    plot_loss_curves(
        result["loss_history"],
        result["val_loss_history"],
        out_dir / "spectrograms" / f"{args.clip_id}_{int(args.gap_ms)}ms_loss.png",
    )
    plot_spectrogram(
        channels_to_stft(output_channels),
        audio_cfg["sample_rate"],
        stft_cfg["hop_length"],
        stft_cfg["n_fft"],
        title="Deep-prior reconstruction",
        out_path=out_dir / "spectrograms" / f"{args.clip_id}_{int(args.gap_ms)}ms_reconstruction.png",
        lost_frames=frame.numpy(),
    )
    summary = {
        "final_train_loss": result["loss_history"][-1],
        "final_val_loss": result["val_loss_history"][-1][1] if result["val_loss_history"] else None,
        "nmse_tot": nmse(clean, reconstruction),
        "nmse_miss": missing_region_nmse(clean, reconstruction, sample_mask),
        "parameter_count": model.count_parameters(),
        "wall_clock_seconds": elapsed,
    }
    summary_path = out_dir / f"{args.clip_id}_{int(args.gap_ms)}ms_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
