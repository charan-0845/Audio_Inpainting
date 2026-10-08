"""Run-directory management and artifact saving for DPAI inpainting.

Every call to :func:`save_run` creates a self-contained directory under
``<out_dir>/<run_id>/`` where::

    run_id = <clip>_<level_ms>ms_s<seed>_<YYYYmmdd-HHMMSS>

The directory contains:

* ``config_snapshot.yaml`` – fully resolved config + CLI overrides.
* ``meta.json``            – git hash, torch version, device, seed, etc.
* ``reconstruction.wav``   – reconstructed waveform (16-bit PCM).
* ``corrupted.wav``        – corrupted input.
* ``clean.wav``            – clean reference (if provided).
* ``metrics.json``         – nmse_tot_db, nmse_miss_db, pesq_delta.
* ``loss_curve.png``       – training / diagnostic loss curves.
* ``spectrogram.png``      – triptych: corrupted / reconstructed / clean.
* ``checkpoint.pt``        – model state dict (optional; cfg flag).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import yaml

from src.pipeline.inpainter import InpaintResult
from src.pipeline.metrics import compute_metrics
from src.utils.visualization import plot_loss_curves, plot_spectrogram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return the short HEAD commit hash, or '' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def make_run_id(clip_id: str, gap_ms: float, seed: int) -> str:
    """Build a human-readable, time-stamped run identifier.

    Args:
        clip_id: Audio clip identifier (filename stem).
        gap_ms: Cumulative gap in milliseconds.
        seed: Optimisation seed.

    Returns:
        String of the form ``<clip>_<level_ms>ms_s<seed>_<YYYYmmdd-HHMMSS>``.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{clip_id}_{int(gap_ms)}ms_s{seed}_{ts}"


def save_run(
    result: InpaintResult,
    corrupted: np.ndarray,
    sample_mask: np.ndarray,
    cfg: Dict[str, Any],
    clip_id: str,
    gap_ms: float,
    device: str | torch.device,
    out_dir: str | Path = "outputs/runs",
    clean: Optional[np.ndarray] = None,
    is_speech: bool = False,
    sample_rate: int = 16000,
    model_state: Optional[Dict] = None,
    cli_overrides: Optional[Dict] = None,
) -> Path:
    """Persist all artefacts for one inpainting run.

    Creates the run directory and writes every output file.  Returns the
    run directory :class:`~pathlib.Path` so the caller can print it.

    Args:
        result: :class:`~src.pipeline.inpainter.InpaintResult` from
            :meth:`~src.pipeline.inpainter.DPAIInpainter.inpaint`.
        corrupted: Corrupted waveform ``(N,)`` float32.
        sample_mask: Sample-level binary mask ``(N,)``.
        cfg: Fully resolved config dict (written verbatim to YAML).
        clip_id: Clip identifier used in the run-directory name.
        gap_ms: Cumulative gap duration in ms (used in name).
        device: Device used for this run (logged in meta.json).
        out_dir: Parent directory for all run directories.
        clean: Optional clean reference waveform ``(N,)``.
        is_speech: Whether to attempt PESQ computation.
        sample_rate: Sample rate in Hz for WAV writing and PESQ.
        model_state: Optional model state dict (written as checkpoint.pt).
        cli_overrides: Dict of CLI-supplied overrides merged into the
            config snapshot (for audit trail).

    Returns:
        :class:`~pathlib.Path` to the run directory.
    """
    run_id = make_run_id(clip_id, gap_ms, result.seed)
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stft_cfg = cfg.get("stft", {})

    # --- config snapshot ---
    snapshot = dict(cfg)
    if cli_overrides:
        snapshot["_cli_overrides"] = cli_overrides
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.dump(snapshot, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    # --- meta.json ---
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "clip_id": clip_id,
        "gap_ms": gap_ms,
        "seed": result.seed,
        "epochs": result.epochs,
        "n_params": result.n_params,
        "runtime_s": round(result.runtime_s, 2),
        "device": str(device),
        "torch_version": torch.__version__,
        "git_hash": _git_hash(),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # --- audio files ---
    sf.write(str(run_dir / "reconstruction.wav"), result.waveform, sample_rate,
             subtype="PCM_16")
    sf.write(str(run_dir / "corrupted.wav"), corrupted.astype(np.float32),
             sample_rate, subtype="PCM_16")
    if clean is not None:
        sf.write(str(run_dir / "clean.wav"), clean.astype(np.float32),
                 sample_rate, subtype="PCM_16")

    # --- metrics ---
    if clean is not None:
        metrics = compute_metrics(
            clean=clean,
            recon=result.waveform,
            corrupted=corrupted,
            mask=sample_mask,
            is_speech=is_speech,
            sample_rate=sample_rate,
        )
    else:
        metrics = {"nmse_tot_db": None, "nmse_miss_db": None, "pesq_delta": None}
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # --- loss curve ---
    plot_loss_curves(
        result.loss_history,
        result.val_loss_history or None,
        out_path=run_dir / "loss_curve.png",
    )
    plt_close_all()

    # --- spectrogram triptych (corrupted / reconstructed / clean) ---
    _save_spectrogram_triptych(
        result=result,
        corrupted=corrupted,
        sample_mask=sample_mask,
        clean=clean,
        stft_cfg=stft_cfg,
        sample_rate=sample_rate,
        run_dir=run_dir,
    )

    # --- optional checkpoint ---
    save_ckpt = bool(cfg.get("pipeline", {}).get("save_checkpoint", False))
    if save_ckpt and model_state is not None:
        torch.save(model_state, str(run_dir / "checkpoint.pt"))

    return run_dir


def _save_spectrogram_triptych(
    result: InpaintResult,
    corrupted: np.ndarray,
    sample_mask: np.ndarray,
    clean: Optional[np.ndarray],
    stft_cfg: Dict,
    sample_rate: int,
    run_dir: Path,
) -> None:
    """Save spectrogram PNGs for the corrupted, reconstructed, and (optionally) clean signals.

    Args:
        result: Inpainting result (contains ``final_spectrogram``).
        corrupted: Corrupted waveform ``(N,)``.
        sample_mask: Sample-level mask ``(N,)`` for lost-frame overlay.
        clean: Optional clean reference ``(N,)``.
        stft_cfg: STFT parameters dict.
        sample_rate: Sample rate in Hz.
        run_dir: Directory where PNGs are written.
    """
    from src.audio.stft import compute_stft, frame_mask

    n_fft = int(stft_cfg.get("n_fft", 1024))
    hop = int(stft_cfg.get("hop_length", 120))
    win = int(stft_cfg.get("win_length", 600))

    sm = torch.from_numpy(np.asarray(sample_mask, dtype=np.float32))
    fmask = frame_mask(sm, n_fft=n_fft, hop_length=hop, win_length=win)
    lost = fmask.numpy()

    # corrupted spectrogram
    corr_wav = torch.from_numpy(np.asarray(corrupted, dtype=np.float32))
    corr_stft = compute_stft(corr_wav, n_fft=n_fft, hop_length=hop, win_length=win)
    plot_spectrogram(
        corr_stft, sample_rate=sample_rate, hop_length=hop, n_fft=n_fft,
        title="Corrupted spectrogram",
        out_path=run_dir / "spec_corrupted.png",
        lost_frames=lost,
    )
    plt_close_all()

    # reconstructed spectrogram
    plot_spectrogram(
        result.final_spectrogram, sample_rate=sample_rate, hop_length=hop, n_fft=n_fft,
        title="Reconstructed spectrogram",
        out_path=run_dir / "spec_reconstructed.png",
        lost_frames=lost,
    )
    plt_close_all()

    # clean spectrogram (if available)
    if clean is not None:
        clean_wav = torch.from_numpy(np.asarray(clean, dtype=np.float32))
        clean_stft = compute_stft(clean_wav, n_fft=n_fft, hop_length=hop, win_length=win)
        plot_spectrogram(
            clean_stft, sample_rate=sample_rate, hop_length=hop, n_fft=n_fft,
            title="Clean spectrogram",
            out_path=run_dir / "spec_clean.png",
        )
        plt_close_all()


def plt_close_all() -> None:
    """Close all matplotlib figures to avoid memory leaks in long batch runs."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
