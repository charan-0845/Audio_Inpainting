"""Cluster tests that need no Redis, no GPU, no torch."""

import threading
import numpy as np
import pytest
import yaml
from pathlib import Path

from src.cluster.celery_app import dec, enc
from src.cluster import dispatch
from src.cluster.dispatch import ClusterCancelled, run_windows


def test_enc_dec_roundtrip():
    a = np.random.randn(16000).astype(np.float32)
    np.testing.assert_array_equal(dec(enc(a)), a)


class _H:
    def __init__(self, res): self._res = res
    def ready(self): return True
    def failed(self): return False
    def get(self): return dict(self._res)
    def revoke(self): pass
    def forget(self): pass


class FakeExecutor(dispatch.ClusterExecutor):
    def __init__(self):  # no redis
        self.submitted = []
        self.cancel_key = None
    def watch(self, *a, **k): return lambda: None
    def submit(self, job_id, k, cfg, c, m, ref, epochs, seed):
        self.submitted.append((k, seed))
        return _H({"wave": enc(c + k), "loss": [1.0], "n_params": 3, "runtime_s": 0.1})


def test_run_windows_order_and_seeds():
    ex = FakeExecutor()
    segs = [(np.zeros(8, np.float32), np.ones(8, np.float32), None) for _ in range(3)]
    out = run_windows(ex, "j", {}, segs, 5, 10, threading.Event(), lambda e: None)
    assert [s for _, s in ex.submitted] == [10, 11, 12]          # seed + k
    assert [float(o["wave"][0]) for o in out] == [0.0, 1.0, 2.0]  # order preserved


def test_cluster_cancel_exception_is_distinct():
    assert dispatch.ClusterCancelled is not None
    with pytest.raises(dispatch.ClusterCancelled):
        raise dispatch.ClusterCancelled("cancelled")


def test_worker_failure_propagates():
    class FailedHandle:
        def ready(self): return True
        def failed(self): return True
        def get(self): raise RuntimeError("worker blew up")
        def revoke(self): pass
        def forget(self): pass

    ex = dispatch.ClusterExecutor.__new__(dispatch.ClusterExecutor)
    ex.r = type("R", (), {"set": lambda *a, **k: None})()
    with pytest.raises(RuntimeError, match="worker blew up"):
        ex.collect("j", [FailedHandle()], threading.Event())


def test_inpaint_long_window_cap_rejection():
    from src.pipeline.long_inpaint import inpaint_long
    sr = 16000
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    sig = np.ones(20 * sr, dtype=np.float32)
    mask = np.ones_like(sig)
    mask[8 * sr:8 * sr + 1600] = 0.0
    with pytest.raises(ValueError, match="exceeds limit"):
        inpaint_long(
            sig * mask,
            mask,
            cfg,
            epochs=1,
            seed=0,
            executor=object(),
            options={"window_s": 5.0, "lpc_max_gap_ms": 40.0, "max_windows": 0},
        )


def test_fake_executor_inpaint_long_matches_sequential_seeded():
    from src.pipeline.inpainter import DPAIInpainter
    from src.pipeline.long_inpaint import inpaint_long

    sr = 16000
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    cfg["optimization"]["epochs"] = 5
    cfg["model"]["channel_schedule"] = [8, 8, 8, 8, 8]
    cfg["model"]["res_path_lengths"] = [4, 3, 2, 1]
    cfg["model"]["use_harmonic"] = False
    cfg["pipeline"]["preserve_observed"] = True

    t = np.linspace(0, 3, 3 * sr, endpoint=False, dtype=np.float32)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mask = np.ones_like(signal)
    mask[32000:33600] = 0.0
    corrupted = signal * mask

    class LocalFakeExecutor(FakeExecutor):
        def submit(self, job_id, k, cfg, c, m, ref, epochs, seed):
            result = DPAIInpainter(cfg, device="cpu").inpaint(
                corrupted=c,
                sample_mask=m,
                reference=ref,
                epochs=epochs,
                seed=seed,
            )
            return _H({
                "wave": enc(result.waveform),
                "loss": result.loss_history,
                "n_params": result.n_params,
                "runtime_s": result.runtime_s,
            })

    local = inpaint_long(
        corrupted, mask, cfg, epochs=5, seed=7,
        device="cpu",
    )
    clustered = inpaint_long(
        corrupted, mask, cfg, epochs=5, seed=7,
        device="cpu",
        executor=LocalFakeExecutor(),
        options={"window_s": 5.0, "lpc_max_gap_ms": 40.0, "max_windows": 24},
    )
    np.testing.assert_array_equal(clustered.waveform, local.waveform)
