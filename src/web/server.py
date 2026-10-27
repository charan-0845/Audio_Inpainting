"""FastAPI web demo server and single-worker job queue for DPAI."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import _auto_device, _build_cfg, _deep_merge, _load_config
from src.audio.corruption import create_mask, extract_mask_from_audio
from src.audio.loader import load_audio, make_synthetic_audio
from src.audio.stft import channels_to_stft, crop_to_shape
from src.pipeline.inpainter import DPAIInpainter
from src.pipeline.io import make_run_id, save_run
from src.utils.visualization import plot_spectrogram

logger = logging.getLogger("dpai.web")

# Whitelist of allowed artefact files that may be downloaded from /files/{name}
WHITELIST_FILES = {
    "reconstruction.wav",
    "corrupted.wav",
    "clean.wav",
    "metrics.json",
    "meta.json",
    "loss_curve.png",
    "spec_corrupted.png",
    "spec_reconstructed.png",
    "spec_clean.png",
    "spectrogram.png",
    "config_snapshot.yaml",
    "checkpoint.pt",
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "optimization": {"epochs": 300},
        "model": {
            "base_filters": 4,
            "channel_schedule": [4, 4, 4, 4, 4],
            "use_harmonic": False,
        },
    },
    "balanced": {
        "optimization": {"epochs": 1500},
        "model": {
            "channel_schedule": [68, 68, 68, 68, 68],
            "use_harmonic": True,
        },
    },
    "paper": {
        "optimization": {"epochs": 5000},
        "model": {
            "channel_schedule": [68, 68, 68, 68, 68],
            "use_harmonic": True,
        },
    },
}


def get_presets_for_device(device: str = "cpu") -> Dict[str, Dict[str, Any]]:
    """Return device-adapted presets to prevent memory exhaustion and timeout on CPU."""
    if device == "cpu":
        return {
            "fast": {
                "optimization": {"epochs": 100},
                "model": {
                    "base_filters": 4,
                    "channel_schedule": [4, 4, 4, 4, 4],
                    "use_harmonic": False,
                },
            },
            "balanced": {
                "optimization": {"epochs": 300},
                "model": {
                    "base_filters": 16,
                    "channel_schedule": [16, 16, 16, 16, 16],
                    "use_harmonic": False,
                },
            },
            "paper": {
                "optimization": {"epochs": 1000},
                "model": {
                    "base_filters": 32,
                    "channel_schedule": [32, 32, 32, 32, 32],
                    "use_harmonic": True,
                },
            },
        }
    return PRESETS


class JobCancelledException(Exception):
    """Raised when inpainting is cancelled by the user."""
    pass


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def parse_and_validate_gaps(
    gaps_json: str | list,
    duration_ms: float,
    sample_rate: int,
    num_samples: int,
) -> Tuple[np.ndarray, List[str]]:
    """Parse gap list [[start_ms, end_ms], ...], validate constraints, and return mask + warnings.

    Validation rules:
      - gaps in-bounds: 0 <= start < end <= duration_ms
      - total missing < 40%
      - warn if any gap > 100 ms
    """
    if isinstance(gaps_json, str):
        try:
            gaps = json.loads(gaps_json)
        except Exception as e:
            raise ValueError(f"Invalid gaps JSON: {e}")
    else:
        gaps = gaps_json

    if not isinstance(gaps, list) or len(gaps) == 0:
        raise ValueError("gaps_json must be a non-empty list of [start_ms, end_ms] pairs.")

    warnings: List[str] = []
    mask = np.ones(num_samples, dtype=np.float32)

    for idx, gap in enumerate(gaps):
        if not (isinstance(gap, (list, tuple)) and len(gap) == 2):
            raise ValueError(f"Gap item at index {idx} must be a 2-element list [start_ms, end_ms].")
        start_ms, end_ms = float(gap[0]), float(gap[1])
        if start_ms < 0:
            raise ValueError(f"Gap start_ms ({start_ms}) cannot be negative.")
        if end_ms <= start_ms:
            raise ValueError(f"Gap end_ms ({end_ms}) must be strictly greater than start_ms ({start_ms}).")
        if end_ms > duration_ms + 1e-3:
            raise ValueError(f"Gap end_ms ({end_ms:.1f} ms) exceeds total audio duration ({duration_ms:.1f} ms).")

        gap_dur = end_ms - start_ms
        if gap_dur > 100.0:
            warnings.append(
                f"Gap {start_ms:.0f}-{end_ms:.0f} ms ({gap_dur:.0f} ms) exceeds 100 ms. "
                "The paper's method is designed for gaps <= 100 ms; quality may degrade."
            )

        start_samp = max(0, min(num_samples, int(round(start_ms * sample_rate / 1000.0))))
        end_samp = max(0, min(num_samples, int(round(end_ms * sample_rate / 1000.0))))
        mask[start_samp:end_samp] = 0.0

    missing_samples = int((mask == 0.0).sum())
    missing_ratio = missing_samples / num_samples
    if missing_ratio >= 0.40:
        raise ValueError(
            f"Total missing audio ({missing_ratio * 100:.1f}%) exceeds the 40% limit."
        )
    if missing_ratio == 0:
        raise ValueError("No missing samples found in the provided gaps.")

    return mask, warnings


def _render_preview(
    output_padded: torch.Tensor,
    original_shape: Tuple[int, int],
    stft_cfg: Dict[str, int],
    sample_rate: int,
    epoch: int,
    total_epochs: int,
) -> str:
    """Render in-progress spectrogram to base64 PNG data URL."""
    try:
        t = output_padded.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        out_ch = crop_to_shape(t, original_shape)
        stft_out = channels_to_stft(out_ch)
        fig = plot_spectrogram(
            stft_out,
            sample_rate=sample_rate,
            hop_length=int(stft_cfg["hop_length"]),
            n_fft=int(stft_cfg["n_fft"]),
            title=f"Inpainting Progress – Epoch {epoch}/{total_epochs}",
        )
        fig.set_size_inches(6, 2.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
    except Exception as e:
        logger.warning("Failed to render preview: %s", e)
        return ""


@dataclass
class Job:
    """In-memory representation of an inpainting job."""
    id: str
    clip_id: str
    gap_ms: float
    seed: int
    wav: np.ndarray
    mask: np.ndarray
    cfg: Dict[str, Any]
    device: str
    out_dir: str
    clean: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    run_dir: Optional[Path] = None
    metrics: Optional[Dict[str, Any]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _listeners: List[asyncio.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latest_event: Optional[Dict[str, Any]] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None

    def cancel(self) -> None:
        """Cancel this job (whether queued or running)."""
        self.cancel_event.set()
        if self.status == JobStatus.QUEUED:
            self.status = JobStatus.CANCELLED
            self.publish_event({"type": "cancelled", "status": "cancelled"})

    def add_listener(self, q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        """Register an asyncio queue for SSE notifications."""
        with self._lock:
            self._loop = loop
            self._listeners.append(q)
            if self._latest_event:
                try:
                    q.put_nowait(self._latest_event)
                except Exception:
                    pass

    def remove_listener(self, q: asyncio.Queue) -> None:
        """Unregister an asyncio queue."""
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def publish_event(self, event: Dict[str, Any]) -> None:
        """Publish an event to all registered SSE listeners."""
        self._latest_event = event
        with self._lock:
            for q in list(self._listeners):
                try:
                    if self._loop and self._loop.is_running():
                        self._loop.call_soon_threadsafe(q.put_nowait, event)
                    else:
                        q.put_nowait(event)
                except Exception:
                    pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert job info to JSON-serializable dictionary."""
        produced_files: List[str] = []
        n_params = None
        if self.run_dir and self.run_dir.is_dir():
            produced_files = [
                p.name for p in sorted(self.run_dir.iterdir()) if p.name in WHITELIST_FILES
            ]
            meta_file = self.run_dir / "meta.json"
            if meta_file.is_file():
                try:
                    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                    n_params = meta_data.get("n_params")
                except Exception:
                    pass

        return {
            "id": self.id,
            "status": self.status.value,
            "clip_id": self.clip_id,
            "gap_ms": self.gap_ms,
            "seed": self.seed,
            "progress": self.progress,
            "warnings": self.warnings,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics": self.metrics,
            "n_params": n_params,
            "files": produced_files,
        }


class JobManager:
    """Single-worker FIFO queue and in-memory job registry."""

    def __init__(self) -> None:
        self.registry: Dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self.current_job: Optional[Job] = None
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def submit(self, job: Job) -> None:
        """Add job to registry and queue."""
        self.registry[job.id] = job
        self._queue.put(job)

    def get(self, job_id: str) -> Optional[Job]:
        """Look up job by job_id or run_dir name."""
        if job_id in self.registry:
            return self.registry[job_id]
        for job in self.registry.values():
            if job.run_dir and job.run_dir.name == job_id:
                return job
        return None

    def queue_length(self) -> int:
        """Return number of pending jobs in queue."""
        return self._queue.qsize()

    def shutdown(self) -> None:
        """Signal worker thread to stop."""
        self._stop_event.set()

    def _run_worker(self) -> None:
        """Single-worker background loop."""
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if job.cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                job.publish_event({"type": "cancelled", "status": "cancelled"})
                self._queue.task_done()
                continue

            self.current_job = job
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            total_epochs = int(job.cfg["optimization"]["epochs"])
            job.progress = {
                "epoch": 0,
                "epochs": total_epochs,
                "loss": None,
                "eta_s": None,
            }
            job.publish_event({
                "type": "status",
                "status": "running",
                "epoch": 0,
                "epochs": total_epochs,
            })

            try:
                inpainter = DPAIInpainter(job.cfg, device=job.device)
                sr = int(job.cfg["audio"]["sample_rate"])
                stft_params = inpainter._stft_params()
                corrupted = job.wav * job.mask

                # Derive padding & shape for preview rendering
                observed_ch, _ = inpainter._compute_observed_channels(corrupted)
                num_bins = observed_ch.shape[-2]
                _, tfmask = inpainter._build_tf_mask(job.mask, num_bins)
                _, _, original_shape = inpainter._pad(observed_ch, tfmask)

                def progress_cb(
                    epoch: int, ep_total: int, loss_val: float, output: torch.Tensor
                ) -> None:
                    if job.cancel_event.is_set():
                        raise JobCancelledException("Job was cancelled by client.")

                    elapsed = time.time() - (job.started_at or time.time())
                    time_per_ep = elapsed / (epoch + 1)
                    rem = max(0, ep_total - (epoch + 1))
                    eta_s = max(0.0, round(rem * time_per_ep, 1))

                    job.progress = {
                        "epoch": epoch + 1,
                        "epochs": ep_total,
                        "loss": round(loss_val, 6),
                        "eta_s": eta_s,
                    }

                    preview_interval = max(1, min(50, ep_total // 10))
                    preview_b64: Optional[str] = None
                    if epoch == 0 or (epoch + 1) % preview_interval == 0 or (epoch + 1) == ep_total:
                        preview_b64 = _render_preview(
                            output, original_shape, stft_params, sr, epoch + 1, ep_total
                        )

                    job.publish_event({
                        "type": "progress",
                        "epoch": epoch + 1,
                        "epochs": ep_total,
                        "loss": loss_val,
                        "eta_s": eta_s,
                        "preview": preview_b64,
                    })

                result = inpainter.inpaint(
                    corrupted=corrupted,
                    sample_mask=job.mask,
                    reference=job.clean,
                    epochs=total_epochs,
                    seed=job.seed,
                    callbacks=[progress_cb],
                )

                run_dir = save_run(
                    result=result,
                    corrupted=corrupted,
                    sample_mask=job.mask,
                    cfg=job.cfg,
                    clip_id=job.clip_id,
                    gap_ms=job.gap_ms,
                    device=job.device,
                    out_dir=job.out_dir,
                    clean=job.clean,
                    is_speech=False,
                    sample_rate=sr,
                )
                job.run_dir = run_dir
                if run_dir.name != job.id:
                    self.registry[run_dir.name] = job

                metrics_file = run_dir / "metrics.json"
                if metrics_file.is_file():
                    try:
                        job.metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                n_params = getattr(result, "n_params", None)
                meta_file = run_dir / "meta.json"
                if meta_file.is_file():
                    try:
                        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                        n_params = meta_data.get("n_params", n_params)
                    except Exception:
                        pass

                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.publish_event({
                    "type": "completed",
                    "status": "completed",
                    "run_id": run_dir.name,
                    "metrics": job.metrics,
                    "n_params": n_params,
                    "files": [
                        p.name for p in sorted(run_dir.iterdir()) if p.name in WHITELIST_FILES
                    ],
                })

            except JobCancelledException:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                job.publish_event({"type": "cancelled", "status": "cancelled"})
            except Exception as exc:
                logger.exception("Job %s failed: %s", job.id, exc)
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.completed_at = time.time()
                job.publish_event({
                    "type": "failed",
                    "status": "failed",
                    "error": str(exc),
                })
            finally:
                self.current_job = None
                self._queue.task_done()


def create_app(
    base_cfg: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    out_dir: str = "outputs/runs",
    max_seconds: float = 10.0,
    demo_only: bool = False,
) -> FastAPI:
    """Construct and configure the FastAPI web application.

    Args:
        base_cfg: Default base configuration dictionary.
        device: Device string (e.g. 'cpu' or 'cuda').
        out_dir: Directory where runs are saved.
        max_seconds: Maximum allowed audio clip duration in seconds.
        demo_only: If True, disallow submitting new jobs and only serve gallery.

    Returns:
        Configured FastAPI instance.
    """
    if base_cfg is None:
        base_cfg = _load_config("configs/config.yaml")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    app = FastAPI(title="DPAI Audio Inpainting Web Demo", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    job_manager = JobManager()

    # Store server state on app.state
    app.state.base_cfg = base_cfg
    app.state.device = device
    app.state.out_dir = out_dir
    app.state.max_seconds = max_seconds
    app.state.demo_only = demo_only
    app.state.job_manager = job_manager

    # -----------------------------------------------------------------------
    # API Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "device": app.state.device,
            "queue_length": job_manager.queue_length(),
            "running_job": job_manager.current_job.id if job_manager.current_job else None,
            "demo_only": app.state.demo_only,
            "max_seconds": app.state.max_seconds,
        }

    @app.get("/api/presets")
    async def get_presets() -> Dict[str, Any]:
        """Return available configuration presets."""
        presets = get_presets_for_device(app.state.device)
        is_cpu = app.state.device == "cpu"
        return {
            "presets": {
                "fast": {
                    "epochs": presets["fast"]["optimization"]["epochs"],
                    "base_filters": presets["fast"]["model"].get("base_filters", 4),
                    "harmonic": presets["fast"]["model"].get("use_harmonic", False),
                    "description": "Fast preview (100 epochs, 4 filters, ~30-60s on CPU)" if is_cpu else "Fast preview (300 epochs, base_filters 4, harmonic off)",
                },
                "balanced": {
                    "epochs": presets["balanced"]["optimization"]["epochs"],
                    "base_filters": presets["balanced"]["model"].get("base_filters", 16 if is_cpu else 68),
                    "harmonic": presets["balanced"]["model"].get("use_harmonic", False if is_cpu else True),
                    "description": "Balanced quality & speed (300 epochs, 16 filters)" if is_cpu else "Balanced quality & speed (1500 epochs, full MultiResUNet)",
                },
                "paper": {
                    "epochs": presets["paper"]["optimization"]["epochs"],
                    "base_filters": presets["paper"]["model"].get("base_filters", 32 if is_cpu else 68),
                    "harmonic": presets["paper"]["model"].get("use_harmonic", True),
                    "description": "High quality MultiResUNet (1000 epochs, harmonic on)" if is_cpu else "Paper reproduction (5000 epochs, full MultiResUNet)",
                },
            },
            "default": "fast" if is_cpu else "balanced",
            "device": app.state.device,
        }

    @app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
    async def submit_job(
        audio: Optional[UploadFile] = File(None),
        gaps_json: Optional[str] = Form(None),
        mode: str = Form("gaps"),
        gap_ms: float = Form(200.0),
        preset: str = Form("balanced"),
        epochs: Optional[int] = Form(None),
        seed: int = Form(0),
        preserve_observed: bool = Form(True),
        reference: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        """Submit a new inpainting job."""
        if app.state.demo_only:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Server is running in --demo_only mode. Submitting new jobs is disabled.",
            )

        sr = int(app.state.base_cfg["audio"]["sample_rate"])
        max_duration = float(app.state.max_seconds)
        max_bytes = 20 * 1024 * 1024  # 20 MB

        # 1. Load or synthesise primary audio
        if audio is not None and audio.filename:
            content = await audio.read()
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Uploaded audio ({len(content)/(1024*1024):.1f} MB) exceeds 20 MB limit.",
                )
            # Write to a secure temporary file with anonymous name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = Path(tmp_file.name)
            try:
                wav = load_audio(tmp_path, sample_rate=sr).numpy().astype(np.float32)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to decode audio file: {e}",
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            clip_id = "".join(
                c for c in Path(audio.filename).stem if c.isalnum() or c in ("-", "_")
            ) or "audio"
        else:
            if mode == "synthetic":
                wav = make_synthetic_audio(
                    duration_seconds=min(max_duration, 5.0),
                    sample_rate=sr,
                    seed=int(seed),
                ).numpy().astype(np.float32)
                clip_id = "synthetic"
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Audio file must be uploaded unless mode='synthetic'.",
                )

        num_samples = len(wav)
        duration_s = num_samples / sr
        if duration_s > max_duration + 1e-3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio duration ({duration_s:.2f}s) exceeds maximum allowed limit ({max_duration:.1f}s).",
            )

        # 2. Optional clean reference file
        clean_wav: Optional[np.ndarray] = None
        if reference is not None and reference.filename:
            ref_content = await reference.read()
            if len(ref_content) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Reference file ({len(ref_content)/(1024*1024):.1f} MB) exceeds 20 MB limit.",
                )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(ref_content)
                tmp_ref_path = Path(tmp_file.name)
            try:
                clean_tensor = load_audio(tmp_ref_path, sample_rate=sr)
                clean_wav = clean_tensor.numpy().astype(np.float32)
                if len(clean_wav) > num_samples:
                    clean_wav = clean_wav[:num_samples]
                elif len(clean_wav) < num_samples:
                    clean_wav = np.pad(clean_wav, (0, num_samples - len(clean_wav)))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to decode reference audio: {e}",
                )
            finally:
                if tmp_ref_path.exists():
                    tmp_ref_path.unlink()

        # 3. Derive mask and validate gaps
        warnings: List[str] = []
        actual_gap_ms: float = 0.0

        if mode == "detect":
            mask = extract_mask_from_audio(wav).astype(np.float32)
            missing = (mask == 0.0).sum()
            if missing == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Auto-detect found no silent/zeroed regions in this audio.",
                )
            missing_ratio = missing / num_samples
            if missing_ratio >= 0.40:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Auto-detected missing regions ({missing_ratio*100:.1f}%) exceed the 40% limit.",
                )
            actual_gap_ms = float(round(missing * 1000.0 / sr))
        elif mode == "synthetic":
            actual_gap_ms = float(gap_ms)
            try:
                mask = create_mask(
                    num_samples=num_samples,
                    sample_rate=sr,
                    cumulative_gap_ms=actual_gap_ms,
                    seed=int(seed),
                ).astype(np.float32)
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        else:
            # Mode = user gaps
            if not gaps_json:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="gaps_json is required when mode is not 'detect' or 'synthetic'.",
                )
            duration_ms = duration_s * 1000.0
            try:
                mask, warnings = parse_and_validate_gaps(
                    gaps_json, duration_ms, sr, num_samples
                )
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
            actual_gap_ms = float(round((mask == 0.0).sum() * 1000.0 / sr))

        # 4. Resolve configuration
        cfg = dict(app.state.base_cfg)
        device_presets = get_presets_for_device(app.state.device)
        if preset in device_presets:
            cfg = _deep_merge(cfg, device_presets[preset])
        elif preset in PRESETS:
            cfg = _deep_merge(cfg, PRESETS[preset])
        if epochs is not None and epochs > 0:
            cfg["optimization"]["epochs"] = int(epochs)
        cfg["pipeline"] = cfg.get("pipeline", {})
        cfg["pipeline"]["preserve_observed"] = bool(preserve_observed)

        # 5. Build Job and enqueue
        run_id = make_run_id(clip_id, actual_gap_ms, int(seed))
        job = Job(
            id=run_id,
            clip_id=clip_id,
            gap_ms=actual_gap_ms,
            seed=int(seed),
            wav=wav,
            mask=mask,
            cfg=cfg,
            device=app.state.device,
            out_dir=app.state.out_dir,
            clean=clean_wav,
            warnings=warnings,
        )
        job_manager.submit(job)

        return {
            "id": job.id,
            "status": job.status.value,
            "clip_id": job.clip_id,
            "gap_ms": job.gap_ms,
            "warnings": job.warnings,
            "queue_position": job_manager.queue_length(),
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> Dict[str, Any]:
        """Get job status and results."""
        job = job_manager.get(job_id)
        if job:
            return job.to_dict()

        # Disk fallback if job was completed earlier or server restarted
        potential_dir = Path(app.state.out_dir) / job_id
        if potential_dir.is_dir():
            meta_file = potential_dir / "meta.json"
            metrics_file = potential_dir / "metrics.json"
            meta = {}
            metrics = {}
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if metrics_file.is_file():
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            produced_files = [
                p.name for p in sorted(potential_dir.iterdir()) if p.name in WHITELIST_FILES
            ]

            return {
                "id": potential_dir.name,
                "status": "completed",
                "clip_id": meta.get("clip_id", ""),
                "gap_ms": meta.get("gap_ms", 0),
                "seed": meta.get("seed", 0),
                "progress": {"epoch": meta.get("epochs", 0), "epochs": meta.get("epochs", 0)},
                "warnings": [],
                "error": None,
                "created_at": None,
                "started_at": 0,
                "completed_at": meta.get("runtime_s", 0),
                "metrics": metrics,
                "n_params": meta.get("n_params"),
                "files": produced_files,
            }

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> Dict[str, Any]:
        """Cancel an in-progress or queued job."""
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        job.cancel()
        return {"id": job.id, "status": job.status.value}

    @app.get("/api/jobs/{job_id}/events")
    async def get_job_events(job_id: str) -> StreamingResponse:
        """Stream progress events via Server-Sent Events (SSE)."""
        job = job_manager.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )

        loop = asyncio.get_running_loop()

        async def sse_generator():
            q: asyncio.Queue = asyncio.Queue()
            job.add_listener(q, loop)
            try:
                # Emit current initial status
                init_event = {
                    "type": "status",
                    "status": job.status.value,
                    "progress": job.progress,
                    "warnings": job.warnings,
                }
                yield f"data: {json.dumps(init_event)}\n\n"

                # If already finished, emit terminal event immediately
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    term_event = {
                        "type": job.status.value,
                        "status": job.status.value,
                        "run_id": job.run_dir.name if job.run_dir else job.id,
                        "metrics": job.metrics,
                        "error": job.error,
                    }
                    yield f"data: {json.dumps(term_event)}\n\n"
                    return

                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=1.0)
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("type") in ("completed", "failed", "cancelled"):
                            break
                    except asyncio.TimeoutError:
                        # Check status in case worker finished without a queue item
                        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                            term_event = {
                                "type": job.status.value,
                                "status": job.status.value,
                                "run_id": job.run_dir.name if job.run_dir else job.id,
                                "metrics": job.metrics,
                                "error": job.error,
                            }
                            yield f"data: {json.dumps(term_event)}\n\n"
                            break
                        # Heartbeat comment to keep HTTP connection alive
                        yield ": keepalive\n\n"
            finally:
                job.remove_listener(q)

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs/{job_id}/files/{filename:path}")
    async def get_job_file(
        job_id: str,
        filename: str,
        download: bool = False,
    ) -> FileResponse:
        """Serve whitelisted artifact files from the run directory."""
        # Whitelist verification
        if filename not in WHITELIST_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{filename}' is not in the whitelist of permitted downloads.",
            )

        run_dir: Optional[Path] = None
        job = job_manager.get(job_id)
        if job and job.run_dir and job.run_dir.is_dir():
            run_dir = job.run_dir
        else:
            # Fallback to checking disk in case job_id matches a run directory name directly
            potential_dir = Path(app.state.out_dir) / job_id
            if potential_dir.is_dir():
                run_dir = potential_dir

        if not run_dir or not run_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run directory for job '{job_id}' is not yet available.",
            )

        # Path traversal guard
        target_path = (run_dir / filename).resolve()
        run_dir_resolved = run_dir.resolve()
        if not target_path.is_relative_to(run_dir_resolved):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: path traversal attempt detected.",
            )

        if not target_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' does not exist for this run.",
            )

        media_type = "application/octet-stream"
        if filename.endswith(".wav"):
            media_type = "audio/wav"
        elif filename.endswith(".png"):
            media_type = "image/png"
        elif filename.endswith(".json"):
            media_type = "application/json"
        elif filename.endswith(".yaml"):
            media_type = "text/yaml"

        disp_type = "attachment" if download else "inline"
        return FileResponse(
            target_path,
            media_type=media_type,
            filename=filename,
            content_disposition_type=disp_type,
        )

    # -----------------------------------------------------------------------
    # Static files mount
    # -----------------------------------------------------------------------
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
