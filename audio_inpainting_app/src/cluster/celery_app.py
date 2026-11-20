"""Shared Celery app + array (de)serialisation.

Deliberately imports NO torch, so the gateway can import this cheaply.

Environment variables
---------------------
DPAI_REDIS          broker URL, default redis://127.0.0.1:6379/0
DPAI_REDIS_BACKEND  result backend URL, default = same host, db 1
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from celery import Celery

REDIS_URL = os.environ.get("DPAI_REDIS", "redis://127.0.0.1:6379/0")
BACKEND_URL = os.environ.get("DPAI_REDIS_BACKEND", REDIS_URL.rsplit("/", 1)[0] + "/1")
TASK_NAME = "dpai.run_window"
PRIORITY_STEPS = list(range(10))

app = Celery("dpai", broker=REDIS_URL, backend=BACKEND_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # A node holds exactly one task; everything else waits in Redis (= the queue).
    worker_prefetch_multiplier=1,
    # Re-queue the window if a node dies mid-run.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": 6 * 3600,   # exceeds worst-case CPU paper-window time; measure/raise if needed
        "priority_steps": PRIORITY_STEPS,
        "queue_order_strategy": "priority",
    },
)


def enc(a: np.ndarray) -> str:
    """ndarray -> base64 string (JSON-safe, no pickle)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(a, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def dec(s: str) -> np.ndarray:
    """base64 string -> float32 ndarray."""
    return np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False)
