"""Tests for FastAPI web server, job queue, and validation."""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from src.audio.loader import make_synthetic_audio
from src.pipeline.inpainter import DPAIInpainter
from src.web.server import (
    PRESETS,
    WHITELIST_FILES,
    JobStatus,
    create_app,
    parse_and_validate_gaps,
)


def _smoke_cfg() -> Dict[str, Any]:
    """Base smoke configuration for fast CPU testing."""
    return {
        "audio": {
            "sample_rate": 16000,
            "duration_seconds": 5,
            "mono": True,
        },
        "corruption": {
            "min_gap_ms": 40,
            "max_gap_ms": 80,
            "cumulative_gap_ms": 200,
        },
        "stft": {
            "n_fft": 1024,
            "hop_length": 120,
            "win_length": 600,
        },
        "model": {
            "name": "multiresunet",
            "input_channels": 2,
            "output_channels": 2,
            "channel_schedule": [8, 8, 8, 8, 8],
            "res_path_lengths": [4, 3, 2, 1],
            "norm": "batch",
            "alpha": 1.6,
            "negative_slope": 0.01,
            "use_harmonic": False,
        },
        "optimization": {
            "epochs": 5,
            "learning_rate": 0.01,
            "noise_variance": 0.1,
            "perturbation_variance": 0.03,
        },
        "loss": {
            "use_mss": False,
            "alpha_mse": 1.0,
            "alpha_mss": 0.1,
        },
        "pipeline": {
            "preserve_observed": False,
            "save_checkpoint": False,
            "log_every": 5,
            "variant": "smoke",
        },
        "run": {
            "out_dir": "outputs/runs",
        },
    }


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    """Create a valid in-memory mono WAV file."""
    wav = make_synthetic_audio(duration_seconds=duration_s, sample_rate=sr, seed=42).numpy()
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gap JSON -> mask tests
# ---------------------------------------------------------------------------

def test_parse_and_validate_gaps_exact_counts_and_bounds():
    sr = 16000
    dur_ms = 1000.0
    num_samples = 16000

    # 100 ms gap from 100 ms to 200 ms: exactly 1600 missing samples
    gaps_json = "[[100, 200]]"
    mask, warnings = parse_and_validate_gaps(gaps_json, dur_ms, sr, num_samples)

    assert mask.shape == (num_samples,)
    missing = (mask == 0.0).sum()
    assert missing == 1600
    assert np.all(mask[:1600] == 1.0)
    assert np.all(mask[1600:3200] == 0.0)
    assert np.all(mask[3200:] == 1.0)
    assert len(warnings) == 0

    # Two non-contiguous gaps: 50ms and 50ms
    gaps_list = [[100, 150], [400, 450]]
    mask2, warnings2 = parse_and_validate_gaps(gaps_list, dur_ms, sr, num_samples)
    assert (mask2 == 0.0).sum() == 800 + 800
    assert len(warnings2) == 0


def test_parse_and_validate_gaps_rejection_cases():
    sr = 16000
    dur_ms = 1000.0
    num_samples = 16000

    # Negative start
    with pytest.raises(ValueError, match="cannot be negative"):
        parse_and_validate_gaps("[[-10, 100]]", dur_ms, sr, num_samples)

    # Start >= End
    with pytest.raises(ValueError, match="strictly greater"):
        parse_and_validate_gaps("[[200, 100]]", dur_ms, sr, num_samples)
    with pytest.raises(ValueError, match="strictly greater"):
        parse_and_validate_gaps("[[100, 100]]", dur_ms, sr, num_samples)

    # End exceeds audio duration
    with pytest.raises(ValueError, match="exceeds total audio duration"):
        parse_and_validate_gaps("[[100, 1200]]", dur_ms, sr, num_samples)

    # Total missing exceeds 40% (450 ms missing out of 1000 ms)
    with pytest.raises(ValueError, match="exceeds the 40% limit"):
        parse_and_validate_gaps("[[100, 550]]", dur_ms, sr, num_samples)

    # Empty gaps
    with pytest.raises(ValueError, match="non-empty list"):
        parse_and_validate_gaps("[]", dur_ms, sr, num_samples)

    # Malformed JSON
    with pytest.raises(ValueError, match="Invalid gaps JSON"):
        parse_and_validate_gaps("not a json", dur_ms, sr, num_samples)


def test_parse_and_validate_gaps_warning_for_gap_over_100ms():
    sr = 16000
    dur_ms = 1000.0
    num_samples = 16000

    # 150 ms gap is < 40%, but > 100 ms
    mask, warnings = parse_and_validate_gaps("[[100, 250]]", dur_ms, sr, num_samples)
    assert len(warnings) == 1
    assert "exceeds 100 ms" in warnings[0]


# ---------------------------------------------------------------------------
# FastAPI Job Lifecycle Test
# ---------------------------------------------------------------------------

def test_job_lifecycle_with_smoke_settings():
    """Verify complete job flow produces expected artifacts matching test_pipeline.py."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _smoke_cfg()
        app = create_app(base_cfg=cfg, device="cpu", out_dir=tmp_dir, max_seconds=10.0)
        client = TestClient(app)

        # 1. Check health
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        # 2. Check presets
        presets_resp = client.get("/api/presets")
        assert presets_resp.status_code == 200
        assert "fast" in presets_resp.json()["presets"]

        # 3. Submit job
        wav_bytes = _make_wav_bytes(duration_s=1.0)
        submit_resp = client.post(
            "/api/jobs",
            data={
                "preset": "fast",
                "epochs": 5,
                "seed": 0,
                "mode": "gaps",
                "gaps_json": "[[100, 200]]",
                "preserve_observed": "true",
            },
            files={"audio": ("sample.wav", wav_bytes, "audio/wav")},
        )
        assert submit_resp.status_code == 201
        job_info = submit_resp.json()
        job_id = job_info["id"]

        # 4. Poll until completed
        completed = False
        for _ in range(60):
            time.sleep(0.5)
            status_resp = client.get(f"/api/jobs/{job_id}")
            assert status_resp.status_code == 200
            data = status_resp.json()
            if data["status"] == "completed":
                completed = True
                break
            elif data["status"] in ("failed", "cancelled"):
                pytest.fail(f"Job terminated prematurely with status: {data['status']}, error: {data.get('error')}")

        assert completed, "Job did not complete within timeout"

        # 5. Check run dir artifacts match test_pipeline.py expectations
        run_dirs = list(Path(tmp_dir).iterdir())
        assert len(run_dirs) >= 1
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
        assert not missing, f"Missing files in run directory: {missing}"

        # 6. Test downloading whitelisted files
        wav_resp = client.get(f"/api/jobs/{job_id}/files/reconstruction.wav")
        assert wav_resp.status_code == 200
        assert len(wav_resp.content) > 0

        metrics_resp = client.get(f"/api/jobs/{job_id}/files/metrics.json")
        assert metrics_resp.status_code == 200


# ---------------------------------------------------------------------------
# Path Traversal and Whitelist Security Tests
# ---------------------------------------------------------------------------

def test_whitelist_blocks_path_traversal():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _smoke_cfg()
        app = create_app(base_cfg=cfg, device="cpu", out_dir=tmp_dir)
        client = TestClient(app)

        wav_bytes = _make_wav_bytes(duration_s=1.0)
        submit_resp = client.post(
            "/api/jobs",
            data={"preset": "fast", "epochs": 2, "seed": 0, "mode": "synthetic", "gap_ms": 100},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        job_id = submit_resp.json()["id"]

        # Wait for completion
        for _ in range(40):
            time.sleep(0.3)
            if client.get(f"/api/jobs/{job_id}").json()["status"] == "completed":
                break

        # Non-whitelisted file
        bad_file_resp = client.get(f"/api/jobs/{job_id}/files/secret.txt")
        assert bad_file_resp.status_code == 400

        # Path traversal attempt
        traversal_resp = client.get(f"/api/jobs/{job_id}/files/../../app.py")
        assert traversal_resp.status_code in (400, 403, 404)


# ---------------------------------------------------------------------------
# Job Cancellation and Queue Order Tests
# ---------------------------------------------------------------------------

def test_cancel_works_and_second_job_queues_behind_first():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _smoke_cfg()
        app = create_app(base_cfg=cfg, device="cpu", out_dir=tmp_dir)
        client = TestClient(app)

        wav_bytes = _make_wav_bytes(duration_s=1.0)

        # Job 1: 500 epochs (long running)
        resp1 = client.post(
            "/api/jobs",
            data={"preset": "fast", "epochs": 500, "seed": 1, "mode": "synthetic", "gap_ms": 100},
            files={"audio": ("job1.wav", wav_bytes, "audio/wav")},
        )
        job1_id = resp1.json()["id"]

        # Job 2: 2 epochs
        resp2 = client.post(
            "/api/jobs",
            data={"preset": "fast", "epochs": 2, "seed": 2, "mode": "synthetic", "gap_ms": 100},
            files={"audio": ("job2.wav", wav_bytes, "audio/wav")},
        )
        job2_id = resp2.json()["id"]

        # Job 2 should initially be queued behind job 1
        st2 = client.get(f"/api/jobs/{job2_id}").json()
        assert st2["status"] in ("queued", "running")

        # Cancel job 1
        cancel_resp = client.post(f"/api/jobs/{job1_id}/cancel")
        assert cancel_resp.status_code == 200

        # Verify job 1 becomes cancelled
        for _ in range(40):
            time.sleep(0.2)
            if client.get(f"/api/jobs/{job1_id}").json()["status"] == "cancelled":
                break
        assert client.get(f"/api/jobs/{job1_id}").json()["status"] == "cancelled"

        # Verify job 2 runs to completion
        completed2 = False
        for _ in range(40):
            time.sleep(0.3)
            if client.get(f"/api/jobs/{job2_id}").json()["status"] == "completed":
                completed2 = True
                break
        assert completed2, "Job 2 failed to run after Job 1 was cancelled"


# ---------------------------------------------------------------------------
# Determinism Test with progress_callback
# ---------------------------------------------------------------------------

def test_progress_callback_does_not_change_loss_history():
    from src.audio.corruption import create_mask

    sr = 16000
    wav = make_synthetic_audio(duration_seconds=1.0, sample_rate=sr, seed=42).numpy()
    mask = create_mask(len(wav), sr, cumulative_gap_ms=100.0, seed=0)
    corrupted = wav * mask
    cfg = _smoke_cfg()

    inpainter = DPAIInpainter(cfg, device="cpu")

    # Run without callbacks
    r1 = inpainter.inpaint(corrupted, mask, epochs=5, seed=123)

    # Run with progress callback
    recorded = []
    def my_cb(ep, total, loss, out):
        recorded.append(loss)

    r2 = inpainter.inpaint(corrupted, mask, epochs=5, seed=123, callbacks=[my_cb])

    assert r1.loss_history == r2.loss_history
    assert recorded == r2.loss_history


# ---------------------------------------------------------------------------
# Demo-only mode test
# ---------------------------------------------------------------------------

def test_demo_only_mode_disables_submitting_jobs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _smoke_cfg()
        app = create_app(base_cfg=cfg, device="cpu", out_dir=tmp_dir, demo_only=True)
        client = TestClient(app)

        health = client.get("/api/health").json()
        assert health["demo_only"] is True

        wav_bytes = _make_wav_bytes(duration_s=1.0)
        resp = client.post(
            "/api/jobs",
            data={"preset": "fast", "epochs": 5, "mode": "synthetic"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 403
        assert "demo_only" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Static files test
# ---------------------------------------------------------------------------

def test_static_files_served():
    cfg = _smoke_cfg()
    app = create_app(base_cfg=cfg, device="cpu")
    client = TestClient(app)

    # Root serves index.html
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "DPAI Audio Inpainter" in root_resp.text

    # app.js and style.css
    js_resp = client.get("/app.js")
    assert js_resp.status_code == 200

    css_resp = client.get("/style.css")
    assert css_resp.status_code == 200

    # examples/index.json
    ex_resp = client.get("/examples/index.json")
    assert ex_resp.status_code == 200
    ex_data = ex_resp.json()
    assert "examples" in ex_data
    assert len(ex_data["examples"]) >= 1


