"""Spectrogram and reconstruction visualization utilities.

All functions use the Agg backend and write to disk; they never call
``plt.show()``.  Axes are labelled in seconds and Hz; colorbars are in dB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless – must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CMAP = "magma"
_DPI = 150


def _db(X_abs: np.ndarray, ref: float = 1e-8) -> np.ndarray:
    """Convert amplitude spectrogram to dB, clipped at *ref*."""
    return 20.0 * np.log10(np.maximum(X_abs, ref))


def _time_axis(n_frames: int, hop_length: int, sample_rate: int) -> np.ndarray:
    """Return frame centre times in seconds."""
    return np.arange(n_frames) * hop_length / sample_rate


def _freq_axis(n_bins: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Return frequency bin centres in Hz."""
    return np.linspace(0, sample_rate / 2, n_bins)


def _sample_times(n_samples: int, sample_rate: int) -> np.ndarray:
    return np.arange(n_samples) / sample_rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_spectrogram(
    X: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    title: str = "Spectrogram",
    out_path: Optional[str | Path] = None,
    lost_frames: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Plot a magnitude spectrogram in dB and optionally save it.

    Args:
        X: Complex STFT tensor of shape ``(M, L)`` or real magnitude array.
        sample_rate: Sample rate in Hz (for axis labels).
        hop_length: Hop length in samples (for time axis).
        n_fft: FFT size (for frequency axis).
        title: Figure title.
        out_path: If given, save the figure at this path (PNG).
        lost_frames: Boolean/float array of length ``L``; frames where this is
            0 are overlaid with a semi-transparent red column.
        vmin: Lower dB clip for the colour scale.
        vmax: Upper dB clip for the colour scale.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    if torch.is_tensor(X):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if np.iscomplexobj(X_np):
        X_db = _db(np.abs(X_np))
    else:
        X_db = _db(X_np)

    M, L = X_db.shape
    times = _time_axis(L, hop_length, sample_rate)
    freqs = _freq_axis(M, n_fft, sample_rate)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        X_db,
        origin="lower",
        aspect="auto",
        cmap=_CMAP,
        extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(img, ax=ax, format="%.0f dB")
    cbar.set_label("dB", rotation=270, labelpad=12)

    if lost_frames is not None:
        lost = np.asarray(lost_frames, dtype=float)
        dt = (times[-1] - times[0]) / max(L - 1, 1) if L > 1 else 1.0 / sample_rate
        for l_idx, val in enumerate(lost):
            if val < 0.5:
                t_centre = times[l_idx]
                ax.axvspan(
                    t_centre - dt / 2,
                    t_centre + dt / 2,
                    alpha=0.35,
                    color="cyan",
                    linewidth=0,
                )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=_DPI)

    return fig


def plot_loss_curves(
    train_loss: Sequence[float],
    val_loss: Optional[Sequence[Tuple[int, float]]] = None,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot deep-prior training and optional missing-region diagnostic loss."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(train_loss)), train_loss, label="Observed-region training loss")
    if val_loss:
        epochs, values = zip(*val_loss)
        ax.plot(epochs, values, "o-", label="Missing-region diagnostic loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Deep-prior optimization")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=_DPI)
    return fig


def plot_waveform_with_gaps(
    audio_clean: torch.Tensor,
    audio_corrupted: torch.Tensor,
    mask: np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot clean and corrupted waveforms with gap regions shaded in red.

    Args:
        audio_clean: Clean waveform tensor ``(N,)``.
        audio_corrupted: Corrupted waveform tensor ``(N,)``.
        mask: Sample-level binary mask ``(N,)`` (1=observed, 0=missing).
        sample_rate: Sample rate in Hz.
        title: Figure title.
        out_path: If given, save the figure at this path.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    clean = audio_clean.detach().cpu().numpy()
    corrupted = audio_corrupted.detach().cpu().numpy()
    mask_np = np.asarray(mask, dtype=float)
    t = _sample_times(len(clean), sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ax, signal, label in zip(axes, [clean, corrupted], ["Clean", "Corrupted"]):
        ax.plot(t, signal, lw=0.6, color="#4ecdc4")
        # shade gap regions
        in_gap = False
        gap_start = 0
        for i, m in enumerate(mask_np):
            if m < 0.5 and not in_gap:
                gap_start = i
                in_gap = True
            elif m >= 0.5 and in_gap:
                ax.axvspan(t[gap_start], t[i - 1], color="red", alpha=0.35, linewidth=0)
                in_gap = False
        if in_gap:
            ax.axvspan(t[gap_start], t[-1], color="red", alpha=0.35, linewidth=0)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{label} waveform")
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=_DPI)

    return fig


def plot_mask(
    mask: np.ndarray,
    tf_mask_matrix: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    out_path_time: Optional[str | Path] = None,
    out_path_tf: Optional[str | Path] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Plot the sample-level mask and the TF mask S side by side (as separate files).

    Args:
        mask: Sample-level mask ``(N,)`` (1=observed, 0=missing).
        tf_mask_matrix: TF mask ``(M, L)`` float tensor.
        sample_rate: Sample rate in Hz.
        hop_length: Hop size in samples.
        n_fft: FFT size.
        out_path_time: Save path for the time-domain mask plot.
        out_path_tf: Save path for the TF mask plot.

    Returns:
        Tuple of two figures ``(fig_time, fig_tf)``.
    """
    mask_np = np.asarray(mask, dtype=float)
    N = len(mask_np)
    t = _sample_times(N, sample_rate)

    # --- sample-level mask ---
    fig_time, ax = plt.subplots(figsize=(12, 2))
    ax.fill_between(t, mask_np, step="post", color="#4ecdc4", alpha=0.8, label="Observed")
    ax.fill_between(t, 1 - mask_np, step="post", color="red", alpha=0.6, label="Missing")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mask value")
    ax.set_title("Sample-level mask (1=observed, 0=missing)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(t[0], t[-1])
    fig_time.tight_layout()
    if out_path_time is not None:
        Path(out_path_time).parent.mkdir(parents=True, exist_ok=True)
        fig_time.savefig(str(out_path_time), dpi=_DPI)

    # --- TF mask ---
    S_np = tf_mask_matrix.detach().cpu().numpy()
    M, L = S_np.shape
    times = _time_axis(L, hop_length, sample_rate)
    freqs = _freq_axis(M, n_fft, sample_rate)

    fig_tf, ax2 = plt.subplots(figsize=(10, 4))
    img = ax2.imshow(
        S_np,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        extent=[times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000],
    )
    fig_tf.colorbar(img, ax=ax2, label="Mask value")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (kHz)")
    ax2.set_title("TF mask S (green=present, red=lost)")
    fig_tf.tight_layout()
    if out_path_tf is not None:
        Path(out_path_tf).parent.mkdir(parents=True, exist_ok=True)
        fig_tf.savefig(str(out_path_tf), dpi=_DPI)

    return fig_time, fig_tf


def plot_real_imag_channels(
    X_corrupted: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot real and imaginary STFT channels side by side.

    Uses a symmetric colour scale so zero is centred.

    Args:
        X_corrupted: Complex STFT of the corrupted signal ``(M, L)``.
        sample_rate: Sample rate in Hz.
        hop_length: Hop size in samples.
        n_fft: FFT size.
        out_path: If given, save the figure at this path.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    if torch.is_tensor(X_corrupted):
        X_np = X_corrupted.detach().cpu().numpy()
    else:
        X_np = np.asarray(X_corrupted)

    real = X_np.real
    imag = X_np.imag

    M, L = real.shape
    times = _time_axis(L, hop_length, sample_rate)
    freqs = _freq_axis(M, n_fft, sample_rate)
    extent = [times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000]

    vmax = max(np.abs(real).max(), np.abs(imag).max())
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, data, label in zip(axes, [real, imag], ["Real", "Imaginary"]):
        img = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(img, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"STFT – {label} channel")
    axes[0].set_ylabel("Frequency (kHz)")

    fig.suptitle("Network input channels (corrupted STFT)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=_DPI)

    return fig


def plot_overview(
    X_clean: torch.Tensor,
    X_corrupted: torch.Tensor,
    tf_mask_matrix: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Three-row overview figure with shared time axis.

    Rows: clean spectrogram / corrupted spectrogram / TF mask.

    Args:
        X_clean: Complex STFT of clean signal ``(M, L)``.
        X_corrupted: Complex STFT of corrupted signal ``(M, L)``.
        tf_mask_matrix: TF mask ``(M, L)``.
        sample_rate: Sample rate in Hz.
        hop_length: Hop size in samples.
        n_fft: FFT size.
        out_path: If given, save the figure at this path.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    X_clean_np = _to_np(X_clean)
    X_corr_np = _to_np(X_corrupted)
    S_np = _to_np(tf_mask_matrix)

    M, L = X_clean_np.shape
    times = _time_axis(L, hop_length, sample_rate)
    freqs = _freq_axis(M, n_fft, sample_rate)
    extent = [times[0], times[-1], freqs[0] / 1000, freqs[-1] / 1000]

    db_clean = _db(np.abs(X_clean_np))
    db_corr = _db(np.abs(X_corr_np))
    vmin = min(db_clean.min(), db_corr.min())
    vmax = max(db_clean.max(), db_corr.max())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Row 0: clean
    img0 = axes[0].imshow(db_clean, origin="lower", aspect="auto",
                           cmap=_CMAP, extent=extent, vmin=vmin, vmax=vmax)
    fig.colorbar(img0, ax=axes[0], format="%.0f dB").set_label("dB", rotation=270, labelpad=12)
    axes[0].set_ylabel("Freq (kHz)")
    axes[0].set_title("Clean spectrogram (dB)")

    # Row 1: corrupted
    img1 = axes[1].imshow(db_corr, origin="lower", aspect="auto",
                           cmap=_CMAP, extent=extent, vmin=vmin, vmax=vmax)
    fig.colorbar(img1, ax=axes[1], format="%.0f dB").set_label("dB", rotation=270, labelpad=12)
    axes[1].set_ylabel("Freq (kHz)")
    axes[1].set_title("Corrupted spectrogram (dB)")

    # Row 2: TF mask
    img2 = axes[2].imshow(S_np, origin="lower", aspect="auto",
                           cmap="RdYlGn", extent=extent, vmin=0, vmax=1)
    fig.colorbar(img2, ax=axes[2]).set_label("Mask", rotation=270, labelpad=12)
    axes[2].set_ylabel("Freq (kHz)")
    axes[2].set_title("TF mask S")
    axes[2].set_xlabel("Time (s)")

    fig.suptitle("Audio Inpainting Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=_DPI)

    return fig


def plot_roundtrip_error(
    audio_clean: torch.Tensor,
    audio_reconstructed: torch.Tensor,
    sample_rate: int,
    out_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the absolute per-sample STFT round-trip reconstruction error.

    Args:
        audio_clean: Original waveform ``(N,)``.
        audio_reconstructed: Reconstructed waveform ``(N,)``.
        sample_rate: Sample rate in Hz.
        out_path: If given, save the figure at this path.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    clean = audio_clean.detach().cpu().numpy()
    recon = audio_reconstructed.detach().cpu().numpy()
    # align lengths
    min_len = min(len(clean), len(recon))
    error = np.abs(clean[:min_len] - recon[:min_len])
    t = _sample_times(min_len, sample_rate)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, error, lw=0.6, color="#ff6b6b")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|error|")
    ax.set_title(f"STFT round-trip error  (max={error.max():.2e}, mean={error.mean():.2e})")
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=_DPI)

    return fig
