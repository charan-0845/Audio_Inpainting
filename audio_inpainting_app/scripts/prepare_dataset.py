"""Prepare dataset script for audio inpainting reproduction.

Generates clean audio clips, corruption masks, corrupted audio clips, and a
manifest.csv mapping clip parameters.

Usage:
    python scripts/prepare_dataset.py [--raw_dir DIR] [--clean_dir DIR]
                                      [--corrupted_dir DIR] [--masks_dir DIR]
                                      [--manifest PATH] [--gap_ms_list LIST]
                                      [--seeds_per_gap N] [--base_seed N]
                                      [--overwrite] [--synthetic]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

# Ensure repo root is on sys.path when called as a script
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.audio.corruption import apply_mask, create_mask
from src.audio.dataset import PreparedSample, list_raw_clips
from src.audio.loader import load_audio, make_synthetic_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare_dataset")


def _load_config() -> dict:
    cfg_path = _REPO / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def derive_seed(base_seed: int, clip_id: str, gap_ms: float, seed_idx: int) -> int:
    """Derive a deterministic integer seed from parameters using SHA-256."""
    key = f"{base_seed}_{clip_id}_{gap_ms:.1f}_{seed_idx}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def count_gaps(mask: np.ndarray) -> int:
    """Count contiguous zero regions in a binary mask."""
    gaps = 0
    in_gap = False
    for v in mask:
        if v == 0.0 and not in_gap:
            gaps += 1
            in_gap = True
        elif v != 0.0 and in_gap:
            in_gap = False
    return gaps


def get_dir_size_bytes(dir_path: Path) -> int:
    """Calculate total size of files in a directory in bytes."""
    if not dir_path.exists():
        return 0
    return sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())


def parse_args(cfg: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare audio inpainting dataset.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw audio clips.")
    parser.add_argument("--clean_dir", type=str, default="data/clean", help="Output directory for clean clips.")
    parser.add_argument("--corrupted_dir", type=str, default="data/corrupted", help="Output directory for corrupted clips.")
    parser.add_argument("--masks_dir", type=str, default="data/masks", help="Output directory for mask npy files.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest CSV.")
    parser.add_argument(
        "--gap_ms_list",
        type=str,
        default="200,400,600,800,1000",
        help="Comma-separated gap durations in milliseconds.",
    )
    parser.add_argument("--seeds_per_gap", type=int, default=1, help="Number of corruption seeds per gap setting.")
    parser.add_argument("--base_seed", type=int, default=0, help="Base random seed for hash derivation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset files and manifest.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic clips if raw_dir is empty.")
    return parser.parse_args()


def main() -> None:
    cfg = _load_config()
    args = parse_args(cfg)

    sample_rate: int = cfg["audio"]["sample_rate"]
    duration_s: float = cfg["audio"]["duration_seconds"]
    mono: bool = cfg["audio"]["mono"]
    min_gap_ms: float = cfg["corruption"]["min_gap_ms"]
    max_gap_ms: float = cfg["corruption"]["max_gap_ms"]

    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    corrupted_dir = Path(args.corrupted_dir)
    masks_dir = Path(args.masks_dir)
    manifest_path = Path(args.manifest)

    clean_dir.mkdir(parents=True, exist_ok=True)
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    gap_ms_values = [float(g.strip()) for g in args.gap_ms_list.split(",") if g.strip()]

    raw_clips = list_raw_clips(raw_dir)
    is_synthetic_run = False

    if len(raw_clips) == 0:
        logger.info(f"No audio clips found in {raw_dir}.")
        logger.info("Generating 3 synthetic audio clips in data/raw to exercise pipeline...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 4):
            synth_path = raw_dir / f"synthetic_{i:02d}.wav"
            synth_audio = make_synthetic_audio(duration_seconds=duration_s, sample_rate=sample_rate, seed=i * 10)
            sf.write(str(synth_path), synth_audio.numpy(), sample_rate)
            logger.info(f"Created synthetic clip: {synth_path}")
        raw_clips = list_raw_clips(raw_dir)
        is_synthetic_run = True

    skipped_clips: list[str] = []
    rows_written = 0
    rows_skipped = 0
    manifest_rows: list[dict] = []

    # If not overwriting and manifest exists, read existing rows
    existing_manifest_dict: dict[tuple[str, str, str], dict] = {}
    if manifest_path.exists() and not args.overwrite:
        try:
            df_existing = pd.read_csv(manifest_path)
            for _, r in df_existing.iterrows():
                key = (str(r["clip_id"]), str(float(r["gap_ms"])), str(int(r["seed_idx"])))
                existing_manifest_dict[key] = r.to_dict()
        except Exception as e:
            logger.warning(f"Could not parse existing manifest: {e}. Rebuilding manifest.")

    for clip_path in raw_clips:
        clip_id = clip_path.stem

        # Check raw duration without padding to skip short clips
        try:
            raw_audio = load_audio(
                clip_path,
                sample_rate=sample_rate,
                mono=mono,
                duration_seconds=None,
                peak_normalise=False,
            )
            raw_duration = len(raw_audio) / sample_rate
            if raw_duration < duration_s:
                logger.warning(
                    f"Skipping clip '{clip_path.name}': duration ({raw_duration:.2f}s) "
                    f"is shorter than target duration_seconds ({duration_s:.1f}s)."
                )
                skipped_clips.append(clip_path.name)
                continue
        except Exception as err:
            logger.warning(f"Failed to load clip '{clip_path.name}': {err}. Skipping.")
            skipped_clips.append(clip_path.name)
            continue

        # Load clean audio normalized and trimmed
        audio_clean = load_audio(
            clip_path,
            sample_rate=sample_rate,
            mono=mono,
            duration_seconds=duration_s,
            peak_normalise=True,
        )
        N = audio_clean.shape[0]

        clean_out_path = clean_dir / f"{clip_id}.wav"
        if args.overwrite or not clean_out_path.exists():
            sf.write(str(clean_out_path), audio_clean.numpy(), sample_rate)

        for gap_ms in gap_ms_values:
            gap_str = f"{int(gap_ms)}" if gap_ms.is_integer() else f"{gap_ms:.1f}"
            for seed_idx in range(args.seeds_per_gap):
                actual_seed = derive_seed(args.base_seed, clip_id, gap_ms, seed_idx)

                corrupted_out_path = corrupted_dir / f"{clip_id}_{gap_str}ms_s{seed_idx}.wav"
                mask_out_path = masks_dir / f"{clip_id}_{gap_str}ms_s{seed_idx}.npy"

                files_exist = clean_out_path.exists() and corrupted_out_path.exists() and mask_out_path.exists()

                if files_exist and not args.overwrite:
                    rows_skipped += 1
                    mask_np = np.load(mask_out_path)
                else:
                    mask_np = create_mask(
                        num_samples=N,
                        sample_rate=sample_rate,
                        min_gap_ms=min_gap_ms,
                        max_gap_ms=max_gap_ms,
                        cumulative_gap_ms=gap_ms,
                        seed=actual_seed,
                    )
                    audio_corrupted = apply_mask(audio_clean.numpy(), mask_np)

                    sf.write(str(corrupted_out_path), audio_corrupted, sample_rate)
                    np.save(mask_out_path, mask_np.astype(np.float32))
                    rows_written += 1

                n_gaps = count_gaps(mask_np)
                cum_gap_samples = int((mask_np == 0.0).sum())
                cum_gap_ms_actual = float(cum_gap_samples * 1000.0 / sample_rate)

                row_dict = {
                    "clip_id": clip_id,
                    "source_path": str(clip_path),
                    "sample_rate": sample_rate,
                    "duration_seconds": duration_s,
                    "gap_ms": gap_ms,
                    "seed_idx": seed_idx,
                    "actual_seed": actual_seed,
                    "clean_path": str(clean_out_path),
                    "corrupted_path": str(corrupted_out_path),
                    "mask_path": str(mask_out_path),
                    "num_gaps": n_gaps,
                    "cumulative_gap_samples": cum_gap_samples,
                    "cumulative_gap_ms_actual": cum_gap_ms_actual,
                }

                manifest_rows.append(row_dict)
                key = (str(clip_id), str(float(gap_ms)), str(int(seed_idx)))
                existing_manifest_dict[key] = row_dict

    # Write manifest CSV
    fieldnames = [
        "clip_id",
        "source_path",
        "sample_rate",
        "duration_seconds",
        "gap_ms",
        "seed_idx",
        "actual_seed",
        "clean_path",
        "corrupted_path",
        "mask_path",
        "num_gaps",
        "cumulative_gap_samples",
        "cumulative_gap_ms_actual",
    ]

    if args.overwrite:
        final_rows = manifest_rows
    else:
        final_rows = list(existing_manifest_dict.values())

    # Sort final rows for deterministic output
    final_rows.sort(key=lambda r: (r["clip_id"], float(r["gap_ms"]), int(r["seed_idx"])))

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    clean_size_mb = get_dir_size_bytes(clean_dir) / (1024 * 1024)
    corr_size_mb = get_dir_size_bytes(corrupted_dir) / (1024 * 1024)
    mask_size_mb = get_dir_size_bytes(masks_dir) / (1024 * 1024)
    total_size_mb = clean_size_mb + corr_size_mb + mask_size_mb

    print()
    print("=" * 65)
    print("  DATASET PREPARATION SUMMARY" + (" (SYNTHETIC RUN)" if is_synthetic_run else ""))
    print("=" * 65)
    print(f"  Clips found               : {len(raw_clips)}")
    print(f"  Clips skipped (too short) : {len(skipped_clips)}")
    if skipped_clips:
        for sc in skipped_clips:
            print(f"    - {sc}")
    print(f"  Rows written (new files)  : {rows_written}")
    print(f"  Rows skipped (existing)   : {rows_skipped}")
    print(f"  Total manifest rows       : {len(final_rows)}")
    print(f"  Manifest written to       : {manifest_path}")
    print(f"  Disk usage (clean)        : {clean_size_mb:.2f} MB")
    print(f"  Disk usage (corrupted)    : {corr_size_mb:.2f} MB")
    print(f"  Disk usage (masks)        : {mask_size_mb:.2f} MB")
    print(f"  Total disk usage          : {total_size_mb:.2f} MB")
    print("=" * 65)
    print()

    # Fail loudly if zero clips found or zero total rows generated
    if len(raw_clips) == 0 or len(final_rows) == 0:
        logger.error("No audio clips found or zero manifest rows generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
