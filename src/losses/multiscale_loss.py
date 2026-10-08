"""Multi-scale spectrogram loss (MSS) for audio inpainting.

Implements the multi-scale spectrogram loss described in Table II of:
  Miotello et al., "Deep Prior-Based Audio Inpainting Using
  Multi-Resolution Harmonic CNNs", IEEE/ACM TASLP 2024.

Three STFT scales at 16 kHz (Table II):
  Scale 0 (fine):   n_fft=256,  win_length=240,  hop_length=60
  Scale 1 (medium): n_fft=1024, win_length=600,  hop_length=120
  Scale 2 (coarse): n_fft=2048, win_length=1200, hop_length=240

For each scale the loss combines four spectral terms (eq. 14):
  1. Spectral convergence:  ||S*(|X|-|X_hat|)||_F / ||S*|X|||_F
  2. Log-STFT magnitude:    mean|S*(log|X|-log|X_hat|)|
  3. Linear STFT magnitude: mean|S*(|X|-|X_hat|)|
  4. Phase consistency:     mean|S*(1-cos(ang(X)-ang(X_hat)))|

The final value is the average over scales.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.audio.stft import frame_mask


# ---------------------------------------------------------------------------
# Default STFT scales  (Table II, 16 kHz)
# ---------------------------------------------------------------------------

DEFAULT_SCALES: List[Tuple[int, int, int]] = [
    (512, 240, 50),    # fine
    (1024, 600, 120),  # medium
    (2048, 1200, 240), # coarse
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_stft_mss(
    waveform: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:
    """Compute centred STFT of shape (M, L) (complex64).

    Args:
        waveform: 1-D real waveform ``(N,)``.
        n_fft: FFT size.
        win_length: Analysis window length.
        hop_length: Hop size.

    Returns:
        Complex STFT tensor of shape ``(M, L)``.
    """
    window = torch.hann_window(win_length, device=waveform.device)
    pad = n_fft // 2
    padded = F.pad(waveform.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
    return torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )  # (M, L)


def _single_scale_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    n_fft: int,
    win_length: int,
    hop_length: int,
    eps: float = 1e-8,
    sample_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the four-term spectral loss at a single STFT scale.

    Args:
        pred_wav: Predicted waveform ``(N,)`` – must support autograd.
        target_wav: Target waveform ``(N,)`` – detached reference.
        n_fft: FFT size.
        win_length: Window length.
        hop_length: Hop size.
        eps: Floor for log and denominators.
        sample_mask: 1-D sample-level mask ``(N,)`` (1=observed, 0=missing).
            Accurately converted to frame mask using this scale's params.

    Returns:
        Scalar loss tensor.
    """
    X_hat = _compute_stft_mss(pred_wav, n_fft, win_length, hop_length)
    X_tgt = _compute_stft_mss(target_wav, n_fft, win_length, hop_length)

    X_hat_mag = X_hat.abs().clamp(min=eps)
    X_tgt_mag = X_tgt.abs().clamp(min=eps)

    M, L = X_hat_mag.shape

    if sample_mask is not None:
        fm = frame_mask(sample_mask, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        S = fm.unsqueeze(0).expand(M, -1)
    else:
        S = torch.ones(M, L, device=pred_wav.device, dtype=pred_wav.dtype)

    num_obs = S.sum().clamp(min=1.0)

    diff_mag = (X_tgt_mag - X_hat_mag) * S
    ref_mag_masked = X_tgt_mag * S

    # 1. Spectral convergence
    sc = diff_mag.norm() / (ref_mag_masked.norm() + eps)

    # 2. Log-STFT magnitude
    log_mag = ((torch.log(X_tgt_mag) - torch.log(X_hat_mag)) * S).abs().sum() / num_obs

    # 3. Linear STFT magnitude
    lin_mag = diff_mag.abs().sum() / num_obs

    # 4. Phase consistency
    phase_term = ((1.0 - torch.cos(torch.angle(X_tgt) - torch.angle(X_hat))) * S).sum() / num_obs

    return sc + log_mag + lin_mag + phase_term


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multiscale_spectrogram_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    scales: Optional[List[Tuple[int, int, int]]] = None,
    eps: float = 1e-8,
    sample_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute multi-scale spectrogram reconstruction loss (MSS).

    Each scale contributes four spectral terms (spectral convergence,
    log-STFT magnitude, linear STFT magnitude, phase consistency).
    The final value is the mean across all scales.  Fully differentiable;
    no in-place operations are used.

    Args:
        pred_wav: Predicted waveform of shape ``(N,)``.  Differentiable.
        target_wav: Target (clean) waveform of shape ``(N,)``.  Should be
            detached to prevent gradient leakage.
        scales: List of ``(n_fft, win_length, hop_length)`` tuples.
            Defaults to :data:`DEFAULT_SCALES` (Table II, 16 kHz).
        eps: Small floor for log / denominator stability.
        sample_mask: Optional 1-D sample mask ``(N,)``.
            Accurately converted to per-scale frame masks.

    Returns:
        Scalar tensor – mean multi-scale spectrogram loss.
    """
    if scales is None:
        scales = DEFAULT_SCALES

    total = pred_wav.new_zeros(())
    for n_fft, win_length, hop_length in scales:
        total = total + _single_scale_loss(
            pred_wav, target_wav,
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            eps=eps, sample_mask=sample_mask,
        )
    return total / len(scales)
