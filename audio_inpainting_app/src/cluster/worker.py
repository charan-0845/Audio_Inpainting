"""Compute-only Celery worker: runs DPAIInpainter.inpaint for ONE window per task.

Start one process per GPU:

    DPAI_REDIS=redis://<gateway>:6379/0 DPAI_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
      celery -A src.cluster.worker worker --concurrency=1 -n w1@%h

The worker needs the repo code + a Python env, but NO shared filesystem:
the task carries cfg + audio segment and returns the waveform.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import redis

from src.cluster.celery_app import REDIS_URL, TASK_NAME, app, dec, enc


class Cancelled(Exception):
    """Raised inside the epoch callback when the gateway set cancel:<job_id>."""


_redis: Optional[redis.Redis] = None


def _r() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(REDIS_URL)
    return _redis


@app.task(name=TASK_NAME)
def run_window(
    job_id: str,
    k: int,
    cfg: Dict[str, Any],
    corrupted: str,
    mask: str,
    reference: Optional[str],
    epochs: int,
    seed: int,
) -> Dict[str, Any]:
    """Inpaint one window. Returns {wave(b64), loss, n_params, runtime_s}."""
    import torch  # heavy imports only on workers
    from src.pipeline.inpainter import DPAIInpainter
    from src.utils.preview import render_preview

    # One worker process should use all cores by default.  DPAI_TORCH_THREADS
    # lets operators deliberately split cores across multiple processes.
    torch.set_num_threads(int(os.environ.get("DPAI_TORCH_THREADS", str(os.cpu_count() or 1))))

    r = _r()
    cancel_key = f"cancel:{job_id}"
    if r.exists(cancel_key):
        raise Cancelled()

    device = os.environ.get("DPAI_DEVICE", "cpu")
    c, m = dec(corrupted), dec(mask)
    ref = dec(reference) if reference else None

    inp = DPAIInpainter(cfg, device=device)
    sr = int(cfg["audio"]["sample_rate"])
    stft_p = inp._stft_params()

    # Same padded-shape derivation the web server used for previews.
    obs, _ = inp._compute_observed_channels(c)
    _, tf = inp._build_tf_mask(m, obs.shape[-2])
    _, _, shape = inp._pad(obs, tf)

    chan = f"progress:{job_id}"
    pv_every = max(1, min(50, epochs // 10))
    state = {"t": 0.0}

    def cb(epoch: int, total: int, loss: float, out: "torch.Tensor") -> None:
        last = epoch + 1 == total
        now = time.time()
        # Throttle Redis traffic + cancel polling to ~2 Hz (always send first/last).
        if not (epoch == 0 or last or now - state["t"] >= 0.5):
            return
        state["t"] = now
        if r.exists(cancel_key):
            raise Cancelled()
        ev: Dict[str, Any] = {"window": k, "epoch": epoch + 1, "total": total, "loss": loss}
        if epoch == 0 or (epoch + 1) % pv_every == 0 or last:
            ev["preview"] = render_preview(out, shape, stft_p, sr, epoch + 1, total)
        r.publish(chan, json.dumps(ev))

    res = inp.inpaint(c, m, reference=ref, epochs=epochs, seed=seed, callbacks=[cb])
    return {
        "wave": enc(res.waveform),
        "loss": [float(x) for x in res.loss_history],
        "n_params": int(res.n_params),
        "runtime_s": float(res.runtime_s),
    }
