"""Tests for the DPAIInpainter pipeline (src/pipeline/inpainter.py).

All tests run on CPU in seconds because epochs=10 and the synthetic
clip is 1 second long.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent


def _smoke_cfg() -> Dict[str, Any]:
    """Load the default config then apply smoke overrides for fast tests."""
    base = yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))
    base["optimization"]["epochs"] = 10
    base["model"]["channel_schedule"] = [8, 8, 8, 8, 8]
    base["model"]["res_path_lengths"] = [4, 3, 2, 1]
    base["model"]["use_harmonic"] = False
    base["pipeline"] = base.get("pipeline", {})
    base["pipeline"]["log_every"] = 5
    base["pipeline"]["preserve_observed"] = False
    base["pipeline"]["save_checkpoint"] = False
    return base


def _make_1s_clip(sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Return a 1-second synthetic waveform."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(sr)
    peak = np.abs(sig).max()
    return (sig * 0.9 / peak).astype(np.float32)


def _make_mask(n_samples: int, gap_ms: float = 200, sr: int = 16000) -> np.ndarray:
    from src.audio.corruption import create_mask
    return create_mask(n_samples, sr, cumulative_gap_ms=gap_ms, seed=42)


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def test_inpaint_returns_correct_length_and_finite():
    """inpaint() on 1 s clip with 10 epochs returns waveform of right length, finite values."""
    from src.pipeline.inpainter import DPAIInpainter

    sr = 16000
    wav = _make_1s_clip(sr)
    mask = _make_mask(len(wav))
    corrupted = wav * mask

    cfg = _smoke_cfg()
    inpainter = DPAIInpainter(cfg, device="cpu")
    result = inpainter.inpaint(corrupted, mask, epochs=10, seed=0)

    assert result.waveform.shape == (len(corrupted),), (
        f"Expected ({len(corrupted)},), got {result.waveform.shape}"
    )
    assert np.isfinite(result.waveform).all(), "Waveform contains inf/nan"
    assert result.n_params > 0
    assert result.epochs == 10


def test_loss_decreases_over_10_epochs():
    """Loss history should decrease at least slightly over 10 epochs."""
    from src.pipeline.inpainter import DPAIInpainter

    sr = 16000
    wav = _make_1s_clip(sr)
    mask = _make_mask(len(wav))
    corrupted = wav * mask

    cfg = _smoke_cfg()
    inpainter = DPAIInpainter(cfg, device="cpu")
    result = inpainter.inpaint(corrupted, mask, epochs=10, seed=0)

    assert result.loss_history[0] > result.loss_history[-1], (
        "Loss did not decrease: "
        f"first={result.loss_history[0]:.4e}, last={result.loss_history[-1]:.4e}"
    )


def test_same_seed_is_deterministic():
    """Calling inpaint twice with the same seed produces identical loss_history."""
    from src.pipeline.inpainter import DPAIInpainter

    sr = 16000
    wav = _make_1s_clip(sr)
    mask = _make_mask(len(wav))
    corrupted = wav * mask
    cfg = _smoke_cfg()

    inpainter = DPAIInpainter(cfg, device="cpu")
    r1 = inpainter.inpaint(corrupted, mask, epochs=10, seed=42)
    r2 = inpainter.inpaint(corrupted, mask, epochs=10, seed=42)

    assert r1.loss_history == r2.loss_history, (
        "Same seed produced different loss histories"
    )


def test_reference_does_not_affect_training_loss():
    """Providing a reference must not change the training loss (no leakage)."""
    from src.pipeline.inpainter import DPAIInpainter

    sr = 16000
    wav = _make_1s_clip(sr)
    mask = _make_mask(len(wav))
    corrupted = wav * mask
    clean = _make_1s_clip(sr, seed=99)  # different from wav
    cfg = _smoke_cfg()

    inpainter = DPAIInpainter(cfg, device="cpu")
    r_no_ref = inpainter.inpaint(corrupted, mask, reference=None, epochs=10, seed=0)
    r_with_ref = inpainter.inpaint(corrupted, mask, reference=clean, epochs=10, seed=0)

    assert r_no_ref.loss_history == r_with_ref.loss_history, (
        "Training loss changed when reference was provided (possible leakage!)"
    )


def test_preserve_observed_pastes_back_exactly():
    """With preserve_observed=True, mask==1 samples must equal the input exactly."""
    from src.pipeline.inpainter import DPAIInpainter

    sr = 16000
    wav = _make_1s_clip(sr)
    mask = _make_mask(len(wav))
    corrupted = wav * mask
    cfg = _smoke_cfg()
    cfg["pipeline"]["preserve_observed"] = True

    inpainter = DPAIInpainter(cfg, device="cpu")
    result = inpainter.inpaint(corrupted, mask, epochs=5, seed=0)

    observed_idx = mask > 0.5
    np.testing.assert_allclose(
        result.waveform[observed_idx],
        corrupted[observed_idx],
        rtol=1e-5,
        err_msg="preserve_observed=True: observed samples differ from input",
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_inpaint_creates_expected_artifacts():
    """app.py inpaint --gap_ms 200 --epochs 5 creates all expected output files."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, str(REPO / "app.py"),
            "inpaint",
            "--gap_ms", "200",
            "--epochs", "5",
            "--config", str(REPO / "configs" / "smoke.yaml"),
            "--device", "cpu",
            "--out_dir", tmp,
            "--seed", "0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode == 0, (
            f"app.py inpaint failed with returncode {proc.returncode}.\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

        # Find the run directory (timestamped subdir)
        run_dirs = list(Path(tmp).iterdir())
        assert len(run_dirs) == 1, f"Expected 1 run dir, found: {run_dirs}"
        run_dir = run_dirs[0]

        expected_files = {
            "config_snapshot.yaml",
            "meta.json",
            "reconstruction.wav",
            "corrupted.wav",
            "metrics.json",
            "loss_curve.png",
            "spec_corrupted.png",
            "spec_reconstructed.png",
        }
        produced = {p.name for p in run_dir.iterdir()}
        missing = expected_files - produced
        assert not missing, f"Missing artifacts: {missing}\nProduced: {produced}"
