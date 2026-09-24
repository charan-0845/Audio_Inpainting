"""Visualization script for audio inpainting pre-processing.

Generates spectrograms, waveform plots, mask visualizations, and audio files
for the DPAI reproduction project.

Usage
-----
    python scripts/visualize_stft.py [--audio PATH] [--cumulative_gap_ms N]
                                     [--seed N] [--out_dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path when called as a script.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import soundfile as sf
import torch
import yaml

from src.audio.corruption import apply_mask, create_mask
from src.audio.loader import load_audio, make_synthetic_audio
from src.audio.stft import (
    channels_to_stft,
    compute_stft,
    frame_mask,
    inverse_stft,
    stft_to_channels,
    tf_mask,
)
from src.utils.visualization import (
    plot_mask,
    plot_overview,
    plot_real_imag_channels,
    plot_roundtrip_error,
    plot_spectrogram,
    plot_waveform_with_gaps,
)


def _load_config() -> dict:
    cfg_path = _REPO / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _parse_args(cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPAI visualization script")
    p.add_argument("--audio", type=str, default=None,
                   help="Path to an audio file. Falls back to synthetic audio if omitted.")
    p.add_argument("--cumulative_gap_ms", type=float,
                   default=cfg["corruption"]["cumulative_gap_ms"],
                   help="Total gap duration in ms (default from config.yaml).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for corruption and synthetic audio.")
    p.add_argument("--out_dir", type=str, default="outputs/spectrograms",
                   help="Directory to save PNGs.")
    return p.parse_args()


def main() -> None:
    cfg = _load_config()
    args = _parse_args(cfg)

    # --- config values ---
    sample_rate: int = cfg["audio"]["sample_rate"]
    duration_s: float = cfg["audio"]["duration_seconds"]
    n_fft: int = cfg["stft"]["n_fft"]
    hop_length: int = cfg["stft"]["hop_length"]
    win_length: int = cfg["stft"]["win_length"]
    min_gap_ms: float = cfg["corruption"]["min_gap_ms"]
    max_gap_ms: float = cfg["corruption"]["max_gap_ms"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = _REPO / "outputs" / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load / generate audio
    # -----------------------------------------------------------------------
    if args.audio is not None:
        audio_clean = load_audio(
            args.audio,
            sample_rate=sample_rate,
            mono=True,
            duration_seconds=duration_s,
            peak_normalise=True,
        )
        print(f"Loaded audio from: {args.audio}")
    else:
        audio_clean = make_synthetic_audio(
            duration_seconds=duration_s,
            sample_rate=sample_rate,
            seed=args.seed,
        )
        print("Using synthetic audio (no --audio path provided).")

    N = audio_clean.shape[0]

    # -----------------------------------------------------------------------
    # 2. Create corruption mask
    # -----------------------------------------------------------------------
    mask_np = create_mask(
        num_samples=N,
        sample_rate=sample_rate,
        min_gap_ms=min_gap_ms,
        max_gap_ms=max_gap_ms,
        cumulative_gap_ms=args.cumulative_gap_ms,
        seed=args.seed,
    )
    mask_torch = torch.from_numpy(mask_np)
    audio_corrupted = audio_clean * mask_torch

    # -----------------------------------------------------------------------
    # 3. Compute STFTs
    # -----------------------------------------------------------------------
    X_clean = compute_stft(audio_clean, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length)
    X_corrupted = compute_stft(audio_corrupted, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length)

    s = frame_mask(mask_torch, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    n_bins = n_fft // 2 + 1
    S = tf_mask(s, n_bins)

    # -----------------------------------------------------------------------
    # 4. STFT round-trip
    # -----------------------------------------------------------------------
    audio_reconstructed = inverse_stft(
        X_clean, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, length=N,
    )
    roundtrip_err = (audio_clean - audio_reconstructed).abs()

    # -----------------------------------------------------------------------
    # 5. Channels
    # -----------------------------------------------------------------------
    channels_corr = stft_to_channels(X_corrupted)  # (2, M, L)

    # -----------------------------------------------------------------------
    # 6. Save audio
    # -----------------------------------------------------------------------
    sf.write(
        str(audio_out_dir / "clean.wav"),
        audio_clean.numpy(),
        samplerate=sample_rate,
    )
    sf.write(
        str(audio_out_dir / "corrupted.wav"),
        audio_corrupted.numpy(),
        samplerate=sample_rate,
    )

    # -----------------------------------------------------------------------
    # 7. Generate plots
    # -----------------------------------------------------------------------

    # 01_waveforms.png
    plot_waveform_with_gaps(
        audio_clean, audio_corrupted, mask_np, sample_rate,
        title="Clean vs Corrupted Waveform",
        out_path=out_dir / "01_waveforms.png",
    )

    # 02_mask_time.png + 02_mask_tf.png
    plot_mask(
        mask_np, S,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        out_path_time=out_dir / "02_mask_time.png",
        out_path_tf=out_dir / "02_mask_tf.png",
    )

    # 03_spec_clean.png
    plot_spectrogram(
        X_clean, sample_rate, hop_length, n_fft,
        title="Clean Spectrogram (dB)",
        out_path=out_dir / "03_spec_clean.png",
    )

    # 04_spec_corrupted.png
    plot_spectrogram(
        X_corrupted, sample_rate, hop_length, n_fft,
        title="Corrupted Spectrogram (dB)",
        out_path=out_dir / "04_spec_corrupted.png",
        lost_frames=s.numpy(),
    )

    # 05_real_imag_channels.png
    plot_real_imag_channels(
        X_corrupted, sample_rate, hop_length, n_fft,
        out_path=out_dir / "05_real_imag_channels.png",
    )

    # 06_overview.png
    plot_overview(
        X_clean, X_corrupted, S,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        out_path=out_dir / "06_overview.png",
    )

    # 07_roundtrip_error.png
    plot_roundtrip_error(
        audio_clean, audio_reconstructed, sample_rate,
        out_path=out_dir / "07_roundtrip_error.png",
    )

    # -----------------------------------------------------------------------
    # 8. Print summary
    # -----------------------------------------------------------------------
    ms_per_sample = 1000.0 / sample_rate

    # gap stats
    from src.utils.visualization import _sample_times  # local helper
    gap_segments = []
    in_gap = False
    g_start = 0
    for i, v in enumerate(mask_np):
        if v == 0.0 and not in_gap:
            g_start = i
            in_gap = True
        elif v != 0.0 and in_gap:
            gap_segments.append((g_start, i - g_start))
            in_gap = False
    if in_gap:
        gap_segments.append((g_start, N - g_start))

    n_gaps = len(gap_segments)
    gap_lengths_ms = [length * ms_per_sample for (_, length) in gap_segments]
    n_lost = int((s == 0.0).sum().item())
    n_frames = s.shape[0]

    print()
    print("=" * 60)
    print("  VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"  Audio duration      : {duration_s:.1f} s  ({N} samples @ {sample_rate} Hz)")
    print(f"  Cumulative gap      : {args.cumulative_gap_ms:.0f} ms  "
          f"({int(mask_np == 0.0).sum() if False else int((mask_np == 0.0).sum())} samples)")
    print(f"  Number of gaps      : {n_gaps}")
    print(f"  Gap lengths (ms)    : {[f'{g:.1f}' for g in gap_lengths_ms]}")
    print(f"  Spectrogram shape   : {X_clean.shape[0]} x {X_clean.shape[1]}  (bins x frames)")
    print(f"  Frames total        : {n_frames}")
    print(f"  Lost frames         : {n_lost}  ({100*n_lost/n_frames:.1f}%)")
    print(f"  Round-trip max err  : {roundtrip_err.max().item():.2e}")
    print(f"  Round-trip mean err : {roundtrip_err.mean().item():.2e}")
    print()
    print("  Output files:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"    {p}")
    print(f"    {audio_out_dir / 'clean.wav'}")
    print(f"    {audio_out_dir / 'corrupted.wav'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
