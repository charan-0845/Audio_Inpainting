"""Resumable benchmark runner.

    python scripts/run_benchmark.py --variant unet --tier quick
    python scripts/run_benchmark.py --variant unet --tier quick --dummy --limit 3   # CPU plumbing test

Appends one row per (clip, gap level, seed) to experiments/<variant>/results.csv
and saves the reconstruction to outputs/audio/<variant>/. Rows already present
in the CSV are skipped, so an interrupted GPU session can simply be restarted.
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio import benchmark as bm  # noqa: E402
from src.audio.stft import (  # noqa: E402
    channels_to_stft,
    compute_stft,
    crop_to_shape,
    frame_mask,
    inverse_stft,
    pad_to_multiple,
    stft_to_channels,
    tf_mask,
)
from src.models.multires_unet import MultiResUNet  # noqa: E402
from src.models.unet import PlainUNet  # noqa: E402
from src.training.deep_prior import optimize_deep_prior  # noqa: E402
from src.training.noise import make_input_noise  # noqa: E402

FIELDS = ["variant", "tier", "clip", "type", "level_ms", "seed", "epochs",
          "nmse_tot_db", "nmse_miss_db", "pesq_delta", "runtime_s",
          "git", "note"]


# --------------------------------------------------------------------------
# >>> WIRE THIS TO YOUR EXISTING PIPELINE <<<
# --------------------------------------------------------------------------
def run_one(variant, corrupted, mask, epochs, seed, dummy=False):
    """Reconstruct one corrupted clip.

    Must return a float32 waveform of length bm.N_SAMPLES.
    Suggested body (adapt names to your code):
        1. x_tilde = compute_stft(corrupted)  -> 2-channel (real, imag) tensor
        2. S = time-frequency mask from `mask` (frame is lost if it overlaps a gap)
        3. torch.manual_seed(seed); build the model for `variant`
           ('unet' | 'multires' | 'multires_harmonic')
        4. noise input Z ~ U(.), variance 0.1; per-iteration perturbation 0.03
        5. run your deep-prior loop for `epochs` (Adam, lr 0.01, masked loss)
        6. inverse STFT of the final output -> waveform
    """
    if dummy:                       # plumbing test only: "reconstruction" = input
        return corrupted.copy()

    torch.manual_seed(seed)
    if variant == "unet":
        model = PlainUNet()
    elif variant == "multires":
        model = MultiResUNet()
    elif variant == "multires_harmonic":
        model = MultiResUNet(use_harmonic=True)
    else:
        raise ValueError(
            f"Unknown variant {variant!r}; expected "
            "'unet', 'multires', or 'multires_harmonic'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    corrupted_tensor = torch.as_tensor(
        np.asarray(corrupted, dtype=np.float32), device=device
    )
    sample_mask = torch.as_tensor(
        np.asarray(mask, dtype=np.float32), device=device
    )
    observed = stft_to_channels(
        compute_stft(corrupted_tensor, n_fft=1024, hop_length=120, win_length=600)
    )
    frame = frame_mask(
        sample_mask, n_fft=1024, hop_length=120, win_length=600
    )
    tf = tf_mask(frame, observed.shape[-2])
    observed, original_shape = pad_to_multiple(observed)
    tf, _ = pad_to_multiple(tf.unsqueeze(0))
    tf = tf.squeeze(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = make_input_noise(
        tuple(observed.shape),
        variance=0.1,
        generator=generator,
        device=device,
    )
    result = optimize_deep_prior(
        model,
        noise,
        observed,
        tf,
        optimizer,
        epochs,
        perturbation_variance=0.03,
        generator=generator,
    )

    output_channels = crop_to_shape(result["final_output"], original_shape)
    with torch.no_grad():
        reconstruction = inverse_stft(
            channels_to_stft(output_channels),
            n_fft=1024,
            hop_length=120,
            win_length=600,
            length=bm.N_SAMPLES,
        )
    return reconstruction.detach().cpu().numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------
def nmse_db(ref, est, sel=None):
    """10*log10(||ref-est||^2 / ||ref||^2), optionally on selected samples."""
    if sel is not None:
        ref, est = ref[sel], est[sel]
    return float(10 * np.log10(((ref - est) ** 2).sum() / (ref ** 2).sum() + 1e-12))


def pesq_delta(ref, corrupted, recon):
    """PESQ(ref, recon) - PESQ(ref, corrupted); speech only (needs `pip install pesq`)."""
    try:
        from pesq import pesq
    except ImportError:
        return ""
    return float(pesq(bm.SR, ref, recon, "wb") - pesq(bm.SR, ref, corrupted, "wb"))


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def done_keys(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["clip"], int(r["level_ms"]), int(r["seed"])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="e.g. unet, multires, multires_harmonic")
    ap.add_argument("--tier", default="quick", choices=["quick", "full"])
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--manifest", default="benchmark/manifest.json")
    ap.add_argument("--note", default="", help="e.g. 'mse-only loss'")
    ap.add_argument("--limit", type=int, default=None, help="run at most N new cases")
    ap.add_argument("--only", choices=["piano", "music", "speech"], default=None,
                    help="run only clips of this type")
    ap.add_argument("--worker", default=None,
                    help="worker letter; writes results_<worker>.csv instead of results.csv")
    ap.add_argument("--dummy", action="store_true", help="skip training (plumbing test)")
    args = ap.parse_args()

    manifest = bm.load_manifest(args.manifest)
    fname = f"results_{args.worker}.csv" if args.worker else "results.csv"
    csv_path = Path("experiments") / args.variant / fname
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path("outputs/audio") / args.variant
    audio_dir.mkdir(parents=True, exist_ok=True)
    finished = done_keys(csv_path)
    new_file = not csv_path.exists()
    commit, ran = git_hash(), 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for clip, level in bm.iter_cases(manifest, args.tier):
            if args.only and clip["type"] != args.only:
                continue
            for seed in args.seeds:
                if (clip["id"], level, seed) in finished:
                    continue
                if args.limit is not None and ran >= args.limit:
                    return
                clean, mask, corrupted = bm.load_case(clip, level)
                t0 = time.time()
                recon = np.asarray(run_one(args.variant, corrupted, mask,
                                           args.epochs, seed, args.dummy), dtype=np.float32)
                runtime = time.time() - t0
                missing = mask == 0
                row = {
                    "variant": args.variant, "tier": args.tier, "clip": clip["id"],
                    "type": clip["type"], "level_ms": level, "seed": seed,
                    "epochs": args.epochs,
                    "nmse_tot_db": round(nmse_db(clean, recon), 3),
                    "nmse_miss_db": round(nmse_db(clean, recon, missing), 3),
                    "pesq_delta": (round(pesq_delta(clean, corrupted, recon), 4)
                                   if clip["type"] == "speech" and not args.dummy else ""),
                    "runtime_s": round(runtime, 1), "git": commit,
                    "note": args.note + (" DUMMY" if args.dummy else ""),
                }
                sf.write(str(audio_dir / f"{clip['id']}_g{level}_s{seed}.wav"),
                         recon, bm.SR, subtype="PCM_16")
                writer.writerow(row)
                f.flush()                # never lose a finished row
                ran += 1
                print(f"{clip['id']} g{level} s{seed}: "
                      f"NMSE_tot={row['nmse_tot_db']} dB, NMSE_miss={row['nmse_miss_db']} dB, "
                      f"{runtime:.0f}s")


if __name__ == "__main__":
    main()
