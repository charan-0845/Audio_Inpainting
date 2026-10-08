"""Total loss builder combining masked MSE and multi-scale spectrogram loss.

Use :func:`build_loss` to get a callable ``(pred, target, mask) -> scalar``
that combines the two terms according to ``configs/config.yaml``:

    total = alpha_mse * masked_mse(pred, target, mask)
          + alpha_mss * multiscale_spectrogram_loss(pred_wav, target_wav)

The MSS term is disabled by default (``loss.use_mss: false`` in config) so
that the MSE-only behaviour is bit-for-bit identical to the original
``optimize_deep_prior`` call.  Enable it only after the MSS implementation
has been validated against the held-out test set.

Design notes
------------
* When ``use_mss`` is False the returned callable is exactly ``masked_mse``
  (no extra STFT computations).
* When ``use_mss`` is True the callable also computes the inverse STFT of
  the masked prediction to obtain ``pred_wav``, then calls
  :func:`multiscale_spectrogram_loss`.  The inverse STFT parameters are
  taken from the closure-captured STFT config so the caller does not need
  to pass them.
* The reference waveform is passed into the closure detached, and is used
  only inside the MSS term (never in the MSE loss computation).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from src.losses.masked_mse import masked_mse
from src.losses.multiscale_loss import multiscale_spectrogram_loss
from src.audio.stft import channels_to_stft, inverse_stft


# Type alias for the loss callable
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def build_loss(
    cfg: Dict,
    stft_cfg: Optional[Dict] = None,
    clean_wav: Optional[torch.Tensor] = None,
    frame_mask: Optional[torch.Tensor] = None,
    signal_length: Optional[int] = None,
) -> LossFn:
    """Build a combined MSE + MSS loss callable from the config dict.

    The returned function has signature::

        loss_fn(pred, target, mask) -> scalar

    where *pred* and *target* are real 2-channel STFT tensors of shape
    ``(2, M, L)`` and *mask* is a TF mask ``(M, L)`` with 1=observed.

    When ``loss.use_mss`` is False (the default) the function is simply
    ``masked_mse`` – no STFT inversion, no extra computation, and
    behaviour is bit-for-bit identical to the original deep-prior loop.

    When ``use_mss`` is True the function additionally:
      1. Applies the mask to the prediction channels (zero out missing frames).
      2. Inverts the masked STFT to a waveform.
      3. Calls :func:`multiscale_spectrogram_loss` against the detached
         *clean_wav* (passed via closure – never affects MSE gradient).
      4. Returns ``alpha_mse * MSE + alpha_mss * MSS``.

    Args:
        cfg: Full config dict (must contain ``loss`` sub-dict).
        stft_cfg: STFT parameters dict with keys ``n_fft``, ``hop_length``,
            ``win_length``.  Required when ``use_mss`` is True.
        clean_wav: Detached clean waveform ``(N,)``.  Required when
            ``use_mss`` is True (used only as MSS reference).
        frame_mask: Frame mask ``(L,)`` passed to
            :func:`multiscale_spectrogram_loss` so missing frames are
            excluded from MSS.  Optional even when ``use_mss`` is True.
        signal_length: Original waveform length ``N`` passed to
            :func:`~src.audio.stft.inverse_stft`.  Required when
            ``use_mss`` is True.

    Returns:
        A callable ``(pred, target, mask) -> scalar``.

    Raises:
        ValueError: If ``use_mss`` is True but required arguments are missing.
    """
    loss_cfg = cfg.get("loss", {})
    use_mss: bool = bool(loss_cfg.get("use_mss", False))
    alpha_mse: float = float(loss_cfg.get("alpha_mse", 1.0))
    alpha_mss: float = float(loss_cfg.get("alpha_mss", 0.1))

    if not use_mss:
        # Bit-for-bit identical to the original optimise loop.
        return masked_mse

    # --- MSS is enabled: validate required arguments ---
    if stft_cfg is None:
        raise ValueError("stft_cfg is required when loss.use_mss is True")
    if clean_wav is None:
        raise ValueError("clean_wav is required when loss.use_mss is True")
    if signal_length is None:
        raise ValueError("signal_length is required when loss.use_mss is True")

    _n_fft = int(stft_cfg["n_fft"])
    _hop = int(stft_cfg["hop_length"])
    _win = int(stft_cfg["win_length"])
    _clean_wav = clean_wav.detach()
    _signal_length = int(signal_length)
    _frame_mask = frame_mask.detach() if frame_mask is not None else None

    def _combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mse_val = masked_mse(pred, target, mask)

        # Invert the *masked* prediction (zero out lost TF bins, then iSTFT)
        masked_pred = pred * mask
        pred_stft = channels_to_stft(masked_pred)
        pred_wav = inverse_stft(
            pred_stft,
            n_fft=_n_fft,
            hop_length=_hop,
            win_length=_win,
            length=_signal_length,
        )

        mss_val = multiscale_spectrogram_loss(
            pred_wav,
            _clean_wav.to(pred_wav.device),
            frame_mask=_frame_mask,
        )
        return alpha_mse * mse_val + alpha_mss * mss_val

    return _combined_loss
