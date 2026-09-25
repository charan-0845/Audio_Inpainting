"""Tests for dataset preparation script and utilities."""

from __future__ import annotations

import os
from pathlib import Path
import time
import pytest
import soundfile as sf
import pandas as pd
import numpy as np

from src.audio.loader import make_synthetic_audio
from src.audio.dataset import list_raw_clips
from scripts.prepare_dataset import derive_seed, main as prepare_main


@pytest.fixture
def temp_dataset_dir(tmp_path: Path):
    """Fixture creating temporary directory structure and 2 synthetic clips."""
    raw_dir = tmp_path / "raw"
    clean_dir = tmp_path / "clean"
    corrupted_dir = tmp_path / "corrupted"
    masks_dir = tmp_path / "masks"
    manifest_path = tmp_path / "manifest.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Write 2 valid synthetic clips (5.0 seconds @ 16000 Hz = 80000 samples)
    clip1 = make_synthetic_audio(duration_seconds=5.0, sample_rate=16000, seed=1)
    clip2 = make_synthetic_audio(duration_seconds=5.0, sample_rate=16000, seed=2)

    sf.write(str(raw_dir / "clip1.wav"), clip1.numpy(), 16000)
    sf.write(str(raw_dir / "clip2.wav"), clip2.numpy(), 16000)

    return {
        "raw_dir": raw_dir,
        "clean_dir": clean_dir,
        "corrupted_dir": corrupted_dir,
        "masks_dir": masks_dir,
        "manifest_path": manifest_path,
        "tmp_path": tmp_path,
    }


def test_prepare_dataset_manifest_row_count(temp_dataset_dir, monkeypatch):
    """1. prepare_dataset produces expected manifest rows for N clips x M gap settings x K seeds."""
    raw_dir = temp_dataset_dir["raw_dir"]
    clean_dir = temp_dataset_dir["clean_dir"]
    corrupted_dir = temp_dataset_dir["corrupted_dir"]
    masks_dir = temp_dataset_dir["masks_dir"]
    manifest_path = temp_dataset_dir["manifest_path"]

    gap_ms_list = "200,400"  # 2 gap settings
    seeds_per_gap = 2        # 2 seeds per gap

    test_args = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", gap_ms_list,
        "--seeds_per_gap", str(seeds_per_gap),
        "--overwrite",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    prepare_main()

    assert manifest_path.exists()
    df = pd.read_csv(manifest_path)
    # Expected: 2 clips x 2 gap settings x 2 seeds = 8 rows
    assert len(df) == 8


def test_prepare_dataset_no_overwrite_preserves_mtime(temp_dataset_dir, monkeypatch):
    """2. Re-running without --overwrite adds zero new rows and leaves file mtimes unchanged."""
    raw_dir = temp_dataset_dir["raw_dir"]
    clean_dir = temp_dataset_dir["clean_dir"]
    corrupted_dir = temp_dataset_dir["corrupted_dir"]
    masks_dir = temp_dataset_dir["masks_dir"]
    manifest_path = temp_dataset_dir["manifest_path"]

    test_args = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", "200",
        "--seeds_per_gap", "1",
        "--overwrite",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    prepare_main()

    df1 = pd.read_csv(manifest_path)
    clean_file = Path(df1.iloc[0]["clean_path"])
    mtime1 = clean_file.stat().st_mtime

    time.sleep(0.05)

    # Re-run without overwrite
    test_args_no_ov = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", "200",
        "--seeds_per_gap", "1",
    ]
    monkeypatch.setattr("sys.argv", test_args_no_ov)
    prepare_main()

    df2 = pd.read_csv(manifest_path)
    assert len(df2) == len(df1)
    mtime2 = clean_file.stat().st_mtime
    assert mtime1 == mtime2


def test_prepare_dataset_overwrite_updates_files(temp_dataset_dir, monkeypatch):
    """3. Re-running with --overwrite regenerates files."""
    raw_dir = temp_dataset_dir["raw_dir"]
    clean_dir = temp_dataset_dir["clean_dir"]
    corrupted_dir = temp_dataset_dir["corrupted_dir"]
    masks_dir = temp_dataset_dir["masks_dir"]
    manifest_path = temp_dataset_dir["manifest_path"]

    test_args = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", "200",
        "--seeds_per_gap", "1",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    prepare_main()

    df1 = pd.read_csv(manifest_path)
    corrupted_file = Path(df1.iloc[0]["corrupted_path"])
    mtime1 = corrupted_file.stat().st_mtime

    time.sleep(0.05)

    test_args_overwrite = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", "200",
        "--seeds_per_gap", "1",
        "--overwrite",
    ]
    monkeypatch.setattr("sys.argv", test_args_overwrite)
    prepare_main()

    mtime2 = corrupted_file.stat().st_mtime
    assert mtime2 >= mtime1


def test_short_clip_skipped(temp_dataset_dir, monkeypatch):
    """4. Clip shorter than duration_seconds is skipped, logged, not in manifest, run exits 0."""
    raw_dir = temp_dataset_dir["raw_dir"]
    clean_dir = temp_dataset_dir["clean_dir"]
    corrupted_dir = temp_dataset_dir["corrupted_dir"]
    masks_dir = temp_dataset_dir["masks_dir"]
    manifest_path = temp_dataset_dir["manifest_path"]

    # Write a short clip (2.0 seconds @ 16000 Hz, while config expects 5.0s)
    short_clip = make_synthetic_audio(duration_seconds=2.0, sample_rate=16000, seed=99)
    sf.write(str(raw_dir / "short_clip.wav"), short_clip.numpy(), 16000)

    test_args = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", "200",
        "--seeds_per_gap", "1",
        "--overwrite",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    prepare_main()

    df = pd.read_csv(manifest_path)
    # short_clip should NOT be in clip_id column
    clip_ids = df["clip_id"].tolist()
    assert "short_clip" not in clip_ids
    assert "clip1" in clip_ids
    assert "clip2" in clip_ids


def test_seed_determinism():
    """5. Same (clip_id, gap_ms, seed_idx) always yields same derived seed across runs."""
    s1 = derive_seed(base_seed=0, clip_id="clip1", gap_ms=200.0, seed_idx=0)
    s2 = derive_seed(base_seed=0, clip_id="clip1", gap_ms=200.0, seed_idx=0)
    assert s1 == s2

    s_diff = derive_seed(base_seed=0, clip_id="clip1", gap_ms=400.0, seed_idx=0)
    assert s1 != s_diff


def test_manifest_gap_ms_actual_within_bounds(temp_dataset_dir, monkeypatch):
    """6. Manifest's cumulative_gap_ms_actual is equal to or close to requested gap_ms."""
    raw_dir = temp_dataset_dir["raw_dir"]
    clean_dir = temp_dataset_dir["clean_dir"]
    corrupted_dir = temp_dataset_dir["corrupted_dir"]
    masks_dir = temp_dataset_dir["masks_dir"]
    manifest_path = temp_dataset_dir["manifest_path"]

    test_args = [
        "prepare_dataset.py",
        "--raw_dir", str(raw_dir),
        "--clean_dir", str(clean_dir),
        "--corrupted_dir", str(corrupted_dir),
        "--masks_dir", str(masks_dir),
        "--manifest", str(manifest_path),
        "--gap_ms_list", "200,400",
        "--seeds_per_gap", "1",
        "--overwrite",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    prepare_main()

    df = pd.read_csv(manifest_path)
    for _, row in df.iterrows():
        requested_ms = float(row["gap_ms"])
        actual_ms = float(row["cumulative_gap_ms_actual"])
        # Difference should be minimal (within rounding sample precision of 1 sample = 0.0625ms)
        assert abs(requested_ms - actual_ms) < 2.0
