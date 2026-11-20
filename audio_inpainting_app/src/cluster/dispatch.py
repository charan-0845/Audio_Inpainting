"""Gateway-side dispatch: submit windows to workers, relay progress, collect results.

Imports no torch at module level (gateway stays light); `remote_inpaint` imports
what it needs lazily.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import redis
from celery.result import AsyncResult

from src.cluster.celery_app import (
    PRIORITY_STEPS, REDIS_URL, TASK_NAME, app, dec, enc,
)

Segment = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]  # (corrupted, mask, reference)


class ClusterCancelled(Exception):
    """Job cancelled by the user (map this to JobCancelledException in server.py)."""


class ClusterExecutor:
    """Thin wrapper over Celery + Redis used by the gateway."""

    def __init__(self, url: str = REDIS_URL) -> None:
        self.r = redis.Redis.from_url(url)

    # -- submit ---------------------------------------------------------
    def submit(self, job_id: str, k: int, cfg: Dict[str, Any], corrupted: np.ndarray,
               mask: np.ndarray, reference: Optional[np.ndarray], epochs: int,
               seed: int) -> AsyncResult:
        """Queue one window. Lower priority number = served first (window 0 of
        every job beats late windows of long jobs, so long files can't starve short ones)."""
        return app.send_task(
            TASK_NAME,
            args=[job_id, k, cfg, enc(corrupted), enc(mask),
                  enc(reference) if reference is not None else None, int(epochs), int(seed)],
            priority=min(k, PRIORITY_STEPS[-1]),
        )

    # -- progress -------------------------------------------------------
    def watch(self, job_id: str, n_windows: int, epochs: int, t_start: float,
              on_event: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        """Relay worker progress to `on_event` (e.g. job.publish_event).

        Subscribes BEFORE returning, so call it before submit(). Returns stop().
        """
        ps = self.r.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(f"progress:{job_id}")
        done = {k: 0 for k in range(n_windows)}
        stop = threading.Event()

        def loop() -> None:
            while not stop.is_set():
                try:
                    msg = ps.get_message(timeout=0.5)
                except Exception:
                    break
                if not msg:
                    continue
                e = json.loads(msg["data"])
                done[e["window"]] = max(done[e["window"]], e["epoch"])
                scaled, total = sum(done.values()), epochs * n_windows
                el = time.time() - t_start
                eta = round(el / scaled * (total - scaled), 1) if scaled else None
                ev = {"type": "progress", "epoch": scaled, "epochs": total,
                      "loss": e["loss"], "eta_s": eta, "preview": e.get("preview"),
                      "window_index": e["window"], "n_windows": n_windows}
                if e.get("preview"):
                    ev["preview_window"] = e["window"]
                on_event(ev)
            try:
                ps.close()
            except Exception:
                pass

        threading.Thread(target=loop, daemon=True).start()
        return stop.set

    # -- collect --------------------------------------------------------
    def collect(self, job_id: str, handles: Sequence[AsyncResult],
                cancel_event: threading.Event) -> List[Dict[str, Any]]:
        """Block until all windows finish. Raises ClusterCancelled on cancel,
        or the worker's exception if a window fails (remaining windows revoked)."""
        try:
            while not all(h.ready() for h in handles):
                if cancel_event.is_set():
                    self.r.set(f"cancel:{job_id}", 1, ex=3600)  # stops running windows
                    for h in handles:
                        h.revoke()                              # drops queued windows
                    raise ClusterCancelled()
                if any(h.failed() for h in handles):
                    break
                time.sleep(0.5)
            out = []
            for h in handles:
                res = h.get()           # re-raises a failed window's exception
                res["wave"] = dec(res["wave"])
                out.append(res)
            return out
        except ClusterCancelled:
            raise
        except Exception:
            self.r.set(f"cancel:{job_id}", 1, ex=3600)
            for h in handles:
                h.revoke()
            raise
        finally:
            for h in handles:
                try:
                    h.forget()
                except Exception:
                    pass

    # -- monitoring -----------------------------------------------------
    def queue_depth(self) -> int:
        """Approximate number of window tasks waiting (all priority sub-queues)."""
        sep = "\x06\x16"
        names = ["celery"] + [f"celery{sep}{p}" for p in PRIORITY_STEPS if p]
        try:
            return int(sum(self.r.llen(n) for n in names))
        except Exception:
            return -1


def run_windows(executor: ClusterExecutor, job_id: str, cfg: Dict[str, Any],
                segments: Sequence[Segment], epochs: int, seed: int,
                cancel_event: threading.Event,
                on_event: Callable[[Dict[str, Any]], None]) -> List[Dict[str, Any]]:
    """Fan all segments out to workers in parallel and return results in order.

    Window k uses seed+k. Use this from long-audio code: build one Segment per window.
    """
    n = len(segments)
    stop = executor.watch(job_id, n, epochs, time.time(), on_event)
    try:
        handles = [
            executor.submit(job_id, k, cfg, c, m, ref, epochs, seed + k)
            for k, (c, m, ref) in enumerate(segments)
        ]
        return executor.collect(job_id, handles, cancel_event)
    finally:
        stop()


def remote_inpaint(job: Any, executor: ClusterExecutor, sr: int, epochs: int):
    """Drop-in replacement for the short-clip `inpainter.inpaint(...)` call in server.py.

    Returns an InpaintResult (same type save_run expects).
    """
    import torch
    from src.audio.stft import compute_stft
    from src.pipeline.inpainter import InpaintResult

    corrupted = (job.wav * job.mask).astype(np.float32)
    mask = job.mask.astype(np.float32)
    res = run_windows(executor, job.id, job.cfg, [(corrupted, mask, job.clean)],
                      epochs, job.seed, job.cancel_event, job.publish_event)[0]

    wave = res["wave"].astype(np.float32)
    s = job.cfg.get("stft", {})
    spec = compute_stft(torch.from_numpy(wave), n_fft=int(s.get("n_fft", 1024)),
                        hop_length=int(s.get("hop_length", 120)),
                        win_length=int(s.get("win_length", 600))).numpy()
    return InpaintResult(
        waveform=wave, loss_history=res["loss"], val_loss_history=[],
        final_spectrogram=spec, runtime_s=res["runtime_s"],
        n_params=res["n_params"], seed=job.seed, epochs=epochs,
    )
