# DPAI on a cluster: gateway + compute-only workers

```
Browser -> Gateway node (FastAPI, Redis, save_run, SSE, gallery, LPC)
              |  Celery tasks via Redis (priority queue)
   +----------+----------+
   v          v          v
 worker1    worker2 ...  workerN     (only DPAIInpainter.inpaint, 1 window per task)
```
Workers return only a waveform (~320 KB per 5 s window) and loss list. No shared filesystem needed.
When every worker is busy, window tasks wait in Redis (`worker_prefetch_multiplier=1`).

## 1. Install

Copy `src/cluster/`, `src/utils/preview.py`, `tests/test_cluster.py`, `scripts/*.sh` into the repo.
On every node: `pip install -r requirements.txt -r requirements-cluster.txt` (same torch build per hardware).

## 2. Start

Gateway node (do NOT run a worker here). CPU-only quick start:
```bash
scripts/start_redis.sh PASS &
python app.py serve --host 0.0.0.0 --cluster redis://:PASS@127.0.0.1:6379/0 \
       --worker_device cpu --coordinators 8 --max_seconds 10 --max_seconds_long 120
```
Each CPU worker node:
```bash
DPAI_REDIS=redis://:PASS@<gateway-ip>:6379/0 scripts/run_worker.sh <gateway-ip>
```
Use `redis://:PASS@...` in scripts and clients. The gateway script defaults to loopback;
for worker access, configure Redis on a protected network/firewall and use requirepass.
Never expose port 6379 unauthenticated to the public internet.
Firewall port 6379 to the worker nodes only (Redis has no auth here; add `requirepass` + `rediss://` for anything exposed).

## 3. Edits to `app.py`

In `_build_parser`, serve subparser:
```python
p_serve.add_argument("--cluster", default=None, help="Redis URL; enables distributed workers.")
p_serve.add_argument("--worker_device", default=None, help="Device the WORKERS use (for presets), e.g. cuda.")
p_serve.add_argument("--coordinators", type=int, default=4, help="Concurrent jobs the gateway coordinates.")
```
In `_cmd_serve`:
```python
executor = None
if args.cluster:
    from src.cluster.dispatch import ClusterExecutor
    executor = ClusterExecutor(args.cluster)
    device = args.worker_device or "cuda"      # presets must reflect the WORKERS, not the gateway CPU
app = create_app(base_cfg=cfg, device=device, out_dir=out_dir,
                 max_seconds=float(args.max_seconds), demo_only=bool(args.demo_only),
                 executor=executor, n_coordinators=int(args.coordinators))
```

## 4. Edits to `src/web/server.py`

**Imports**
```python
import threading
from src.cluster.dispatch import ClusterCancelled
from src.utils.preview import render_preview as _render_preview   # delete the local _render_preview
_PLOT_LOCK = threading.Lock()   # pyplot is not thread-safe; several coordinators call save_run
```

**`JobManager`**: N coordinator threads, set of running jobs
```python
def __init__(self, executor=None, n_coordinators: int = 1) -> None:
    self.registry = {}
    self._queue = queue.Queue()
    self.executor = executor
    self._running = set()
    self._running_lock = threading.Lock()
    self._stop_event = threading.Event()
    n = max(1, n_coordinators) if executor else 1      # local GPU/CPU: stay at 1
    self._threads = [threading.Thread(target=self._run_worker, daemon=True) for _ in range(n)]
    for t in self._threads:
        t.start()

@property
def current_job(self):                                   # keeps /api/health working
    with self._running_lock:
        return next(iter(self._running), None)
```
In `_run_worker`: replace `self.current_job = job` with
`with self._running_lock: self._running.add(job)` and in `finally` replace `self.current_job = None` with
`with self._running_lock: self._running.discard(job)`.
Both `registry` writes stay as they are (dict assignment is atomic enough here).

**Short-clip branch** (the final `else:` that builds `DPAIInpainter` and calls `inpainter.inpaint`)
```python
elif self.executor is not None:
    from src.cluster.dispatch import remote_inpaint
    result = remote_inpaint(job, self.executor, sr, total_epochs)
else:
    ... existing local code unchanged ...
```

**Cancellation mapping**
```python
except (JobCancelledException, ClusterCancelled):
```

**Serialise plotting**
```python
with _PLOT_LOCK:
    run_dir = save_run(...)
```

**`create_app`**: add parameters `executor=None, n_coordinators: int = 4`, and `job_manager = JobManager(executor, n_coordinators)`.

**`/api/health`**: add
```python
"cluster": app.state.job_manager.executor is not None,
"windows_queued": app.state.job_manager.executor.queue_depth() if app.state.job_manager.executor else None,
"running_jobs": len(app.state.job_manager._running),
```

## 5. Long audio: still to wire (needs `src/pipeline/long_inpaint.py`)

`inpaint_long` currently processes windows one after another on the gateway. To use many nodes it must
(1) plan windows and LPC/DPAI gap routing exactly as now, (2) build one `(corrupted_window, mask_window, ref_window)`
segment per DPAI window, (3) call
`run_windows(executor, job_id, cfg, segments, epochs, seed, cancel_event, on_event)`, (4) crossfade the returned waves
exactly as now, (5) run LPC gaps locally on the gateway. Add an `executor=None` argument that falls back to the current
sequential code, so existing tests keep passing. `run_windows` is already written for this.

## 6. Behaviour and limits

* Fairness: window k has priority min(k, 9); every job's first windows run before any job's late ones.
* Node crash: `acks_late` + `reject_on_worker_lost` re-queue the window.
* Cancel: sets `cancel:<job_id>` in Redis; running windows stop within ~0.5 s, queued ones are revoked.
* Gateway is a single point of failure: in-flight jobs are lost if it restarts.
* Seeds: window k uses `seed + k`, so a single-window cluster run matches a local run with the same seed.
* CPU workers only make sense for the `fast` preset.
* Cap windows per job (e.g. 24) in `submit_job` so one request can't flood the queue.

## 7. CPU deployment and smoke checklist

1. Start Redis with a password and firewall port 6379 to worker/gateway nodes only.
2. Start the gateway with `--worker_device cpu`.
3. Start one worker process on a CPU node.
4. Submit a 1 s short clip and verify it completes and reconstruction matches a local seed run.
5. Submit a 12 s clip with DPAI gaps and verify window events and `long_audio.json`.
6. Cancel a running/queued job and verify `cancelled` status.
7. Kill a worker while it is running a window and verify Celery re-queues it to another worker.

Run one worker process with all cores, or benchmark N processes with `DPAI_TORCH_THREADS=cores/N`.
Celery on Windows: use `--pool=solo`.

Known limits:
- gateway is a single point of failure
- jobs and running state are in memory
- Redis must be authenticated and firewall-protected
- CPU optimization speed depends strongly on core count and preset
- queued-window scaling and wall time were not measured in this environment

### Measured scaling table

| Windows | Workers | Wall time | Status |
|---|---|---|---|
| — | — | — | NOT MEASURED |
