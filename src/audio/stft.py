"""STFT and inverse-STFT utilities, plus masking and padding helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_stft(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 120,
    win_length: int = 600,
) -> torch.Tensor:
    """Compute the Short-Time Fourier Transform of a 1-D waveform.

    Uses a Hann window and ``center=True`` (the waveform is reflected-padded
    by ``n_fft // 2`` on each side before analysis), matching librosa defaults.

    Args:
        audio: Real waveform of shape ``(N,)``, dtype ``torch.float32``.
        n_fft: FFT size.  Number of frequency bins is ``n_fft // 2 + 1``.
        hop_length: Hop size in samples.
        win_length: Analysis window length in samples (``<= n_fft``).

    Returns:
        Complex STFT tensor of shape ``(M, L)`` where
        ``M = n_fft // 2 + 1`` and ``L`` is the number of frames.
        dtype is ``torch.complex64``.
    """
    window = torch.hann_window(win_length, device=audio.device)

    # center=True: reflect-pad by n_fft // 2 on each side
    pad = n_fft // 2
    audio_padded = F.pad(audio.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)

    X = torch.stft(
        audio_padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,          # padding already applied manually
        return_complex=True,
    )  # (M, L)
    return X


def inverse_stft(
    X: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 120,
    win_length: int = 600,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Compute the inverse STFT (Griffin-Lim overlap-add).

    This is the exact inverse of :func:`compute_stft` when applied to an
    unmodified STFT (i.e. round-trip error is below floating-point precision).

    Args:
        X: Complex STFT of shape ``(M, L)``, dtype ``torch.complex64``.
        n_fft: FFT size used during analysis.
        hop_length: Hop size in samples used during analysis.
        win_length: Window length used during analysis.
        length: If given, the output is trimmed or zero-padded to exactly
            *length* samples.  Pass the original waveform length to undo
            center padding precisely.

    Returns:
        Reconstructed waveform of shape ``(N,)``, dtype ``torch.float32``.
    """
    window = torch.hann_window(win_length, device=X.device)

    y = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,           # istft handles center un-padding internally
        length=length,
        return_complex=False,
    )  # (N,)
    return y


def stft_to_channels(X: torch.Tensor) -> torch.Tensor:
    """Split a complex STFT into real and imaginary channel planes.

    Args:
        X: Complex tensor of shape ``(M, L)``.

    Returns:
        Real tensor of shape ``(2, M, L)`` where channel 0 is the real part
        and channel 1 is the imaginary part.  This is the two-channel
        representation fed to the network.
    """
    return torch.stack([X.real, X.imag], dim=0)  # (2, M, L)


def channels_to_stft(t: torch.Tensor) -> torch.Tensor:
    """Merge real/imaginary channel planes back into a complex STFT.

    This is the exact inverse of :func:`stft_to_channels`.

    Args:
        t: Real tensor of shape ``(2, M, L)`` where channel 0 is real and
           channel 1 is imaginary.

    Returns:
        Complex tensor of shape ``(M, L)``, dtype ``torch.complex64``.
    """
    return torch.complex(t[0], t[1])  # (M, L)


def frame_mask(
    sample_mask: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 120,
    win_length: int = 600,
) -> torch.Tensor:
    """Compute the per-frame binary mask vector **s** from the paper.

    A frame is **lost** (0) if its analysis window overlaps *any* missing
    sample (mask == 0).  A frame is **present** (1) only when all samples
    under its window are observed.

    With ``center=True`` frame *l* is centred at sample ``l * hop_length``,
    so its window support is::

        [l * hop_length - win_length // 2,  l * hop_length + win_length // 2)

    Samples outside the signal boundary are treated as observed (padded with
    1s), consistent with :func:`compute_stft`.

    The vectorised implementation uses a prefix-sum on the *missing* indicator
    so that the number of missing samples inside any window can be retrieved
    in O(1) per frame.

    Args:
        sample_mask: Float tensor of shape ``(N,)`` with 1 = observed,
            0 = missing.
        n_fft: FFT size (used only to compute the number of output frames *L*,
            not the window support width).
        hop_length: Hop size in samples.
        win_length: Analysis window length in samples.  **Only** this width is
            used for the overlap check (not *n_fft*).

    Returns:
        Float tensor of shape ``(L,)`` with values in ``{0.0, 1.0}`` where
        ``L`` is the number of STFT frames produced by :func:`compute_stft`.
    """
    N = sample_mask.shape[0]

    # Number of frames produced by compute_stft (center=True)
    pad = n_fft // 2
    padded_len = N + 2 * pad
    L = (padded_len - n_fft) // hop_length + 1

    half_win = win_length // 2  # window extends this many samples to each side

    # Build the missing indicator padded with zeros (= observed) outside signal
    # Pad: left = half_win (all observed), right = half_win (all observed)
    missing = 1.0 - sample_mask  # 1 where sample is missing, 0 where observed
    # We need the prefix sum over the region [-half_win, N + half_win)
    # which spans indices relative to the original signal.
    # Pad missing with zeros on both sides so we can index freely.
    padded_missing = F.pad(missing, (half_win, half_win), value=0.0)
    # padded_missing[i] corresponds to original sample index (i - half_win)

    # Prefix sum: prefix[i] = sum of padded_missing[0..i-1]
    prefix = F.pad(padded_missing.cumsum(dim=0), (1, 0), value=0.0)
    # prefix shape: (len(padded_missing) + 1,)

    # For frame l, window centre in original coordinates = l * hop_length
    # Window support in *padded_missing* coordinates:
    #   left  = (l * hop_length) - half_win + half_win = l * hop_length
    #   right = (l * hop_length) + half_win + half_win = l * hop_length + win_length
    # But we must clamp to valid indices.
    frame_indices = torch.arange(L, device=sample_mask.device, dtype=torch.long)
    starts = frame_indices * hop_length  # in padded_missing coords
    ends = starts + win_length           # exclusive

    # Clamp to valid prefix indices
    max_idx = prefix.shape[0] - 1
    starts_c = starts.clamp(0, max_idx)
    ends_c = ends.clamp(0, max_idx)

    missing_counts = prefix[ends_c] - prefix[starts_c]  # (L,)

    s = (missing_counts == 0).to(torch.float32)  # 1 if no missing, 0 otherwise
    return s


def tf_mask(s: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Broadcast the frame mask vector to a full TF mask matrix **S**.

    This implements the paper's ``S = j * s`` notation, where the per-frame
    scalar mask is tiled over all frequency bins.

    Args:
        s: Frame mask vector of shape ``(L,)``.
        num_bins: Number of frequency bins ``M = n_fft // 2 + 1``.

    Returns:
        Float tensor of shape ``(M, L)`` where every row equals *s*.
    """
    return s.unsqueeze(0).expand(num_bins, -1)  # (M, L)


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int = 32,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad the last two dimensions of *x* to the next multiple of *multiple*.

    The last two dimensions (height and width, i.e. frequency bins and time
    frames) are padded using **reflect** mode where possible (requires
    ``pad < input_size``), falling back to **zero** (constant) padding for
    very small inputs.  The original spatial shape is returned so the caller
    can crop back with :func:`crop_to_shape`.

    Args:
        x: Tensor of shape ``(..., M, L)``.
        multiple: Padding target.  Defaults to 32 (5-level stride-2 encoder).

    Returns:
        Tuple ``(x_padded, original_shape)`` where:

        * ``x_padded`` – tensor of shape ``(..., M', L')`` with
          ``M' % multiple == 0`` and ``L' % multiple == 0``.
        * ``original_shape`` – ``(M, L)`` tuple of ints needed by
          :func:`crop_to_shape`.
    """
    M, L = x.shape[-2], x.shape[-1]

    def _next_multiple(n: int, m: int) -> int:
        return n if n % m == 0 else n + (m - n % m)

    M_pad = _next_multiple(M, multiple)
    L_pad = _next_multiple(L, multiple)

    pad_M = M_pad - M
    pad_L = L_pad - L

    # F.pad pads from last dim backwards: (left_L, right_L, top_M, bottom_M)
    # Reflect mode requires pad < input_size; fall back to zero for tiny inputs.
    can_reflect = (pad_L < L) and (pad_M < M)
    mode = "reflect" if can_reflect else "constant"
    x_padded = F.pad(x, (0, pad_L, 0, pad_M), mode=mode)

    return x_padded, (M, L)


def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Crop the last two dimensions of *x* back to *shape*.

    Intended as the inverse of :func:`pad_to_multiple`.

    Args:
        x: Tensor of shape ``(..., M', L')``.
        shape: ``(M, L)`` target spatial shape.

    Returns:
        Tensor of shape ``(..., M, L)``.
    """
    M, L = shape
    return x[..., :M, :L]
