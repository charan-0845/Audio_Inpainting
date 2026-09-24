"""Tests for STFT utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.audio.loader import make_synthetic_audio
from src.audio.stft import (
    channels_to_stft,
    compute_stft,
    crop_to_shape,
    frame_mask,
    inverse_stft,
    pad_to_multiple,
    stft_to_channels,
    tf_mask,
)


# ---------------------------------------------------------------------------
# Constants (from config.yaml defaults)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION = 5.0
N_FFT = 1024
HOP_LENGTH = 120
WIN_LENGTH = 600
N_BINS = N_FFT // 2 + 1  # 513


# ---------------------------------------------------------------------------
# Brute-force reference for frame_mask
# ---------------------------------------------------------------------------

def _frame_mask_brute(
    sample_mask: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Reference implementation: iterate over every frame explicitly."""
    N = sample_mask.shape[0]
    pad = n_fft // 2
    padded_len = N + 2 * pad
    L = (padded_len - n_fft) // hop_length + 1
    half_win = win_length // 2

    s = []
    for l in range(L):
        centre = l * hop_length  # centre in padded_missing coords
        lo = centre             # = l*hop - half_win + half_win
        hi = centre + win_length  # exclusive

        # map back to original signal indices
        # padded_missing[i] <-> original sample index (i - half_win)
        orig_lo = lo - half_win  # = l * hop_length - half_win
        orig_hi = hi - half_win  # = l * hop_length + half_win

        # check any original sample in [orig_lo, orig_hi) that is valid
        is_lost = False
        for n in range(orig_lo, orig_hi):
            if 0 <= n < N:
                if sample_mask[n].item() == 0.0:
                    is_lost = True
                    break
        s.append(0.0 if is_lost else 1.0)

    return torch.tensor(s, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Test 1: STFT round-trip
# ---------------------------------------------------------------------------

def test_stft_roundtrip_error():
    """inverse_stft(compute_stft(x)) should reproduce x with max error < 1e-4."""
    audio = make_synthetic_audio(duration_seconds=DURATION, sample_rate=SAMPLE_RATE, seed=0)
    N = audio.shape[0]

    X = compute_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    y = inverse_stft(X, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, length=N)

    err = (audio - y).abs().max().item()
    assert err < 1e-4, f"Round-trip max error {err:.2e} exceeds 1e-4"


def test_stft_shape():
    """STFT output shape must be (513, ~667) for 5 s at default settings."""
    audio = make_synthetic_audio(duration_seconds=DURATION, sample_rate=SAMPLE_RATE, seed=0)
    X = compute_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    assert X.shape[0] == N_BINS, f"Expected {N_BINS} bins, got {X.shape[0]}"
    # Allow some tolerance around 667 frames
    assert 650 <= X.shape[1] <= 690, f"Unexpected number of frames: {X.shape[1]}"


# ---------------------------------------------------------------------------
# Test 2: stft_to_channels / channels_to_stft round-trip
# ---------------------------------------------------------------------------

def test_channels_roundtrip():
    """stft_to_channels -> channels_to_stft must be an exact inverse."""
    audio = make_synthetic_audio(duration_seconds=DURATION, sample_rate=SAMPLE_RATE, seed=1)
    X = compute_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    channels = stft_to_channels(X)
    X_reconstructed = channels_to_stft(channels)

    # Real and imaginary parts should be exactly equal
    assert torch.allclose(X.real, X_reconstructed.real, atol=0.0), "Real part mismatch"
    assert torch.allclose(X.imag, X_reconstructed.imag, atol=0.0), "Imag part mismatch"


def test_channels_shape():
    """stft_to_channels output shape must be (2, M, L)."""
    audio = make_synthetic_audio(duration_seconds=DURATION, sample_rate=SAMPLE_RATE, seed=2)
    X = compute_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    channels = stft_to_channels(X)

    assert channels.shape[0] == 2
    assert channels.shape[1] == X.shape[0]
    assert channels.shape[2] == X.shape[1]


# ---------------------------------------------------------------------------
# Test 3: frame_mask
# ---------------------------------------------------------------------------

def test_frame_mask_vs_brute_force():
    """frame_mask must agree with the brute-force reference on 20 random masks."""
    rng = np.random.default_rng(42)
    N = round(DURATION * SAMPLE_RATE)

    for _ in range(20):
        # Random mask: draw positions of missing blocks
        mask_np = np.ones(N, dtype=np.float32)
        n_gaps = rng.integers(1, 10)
        positions = rng.integers(0, N, size=n_gaps)
        lengths = rng.integers(1, 200, size=n_gaps)
        for pos, length in zip(positions, lengths):
            end = min(int(pos) + int(length), N)
            mask_np[int(pos):end] = 0.0
        mask = torch.from_numpy(mask_np)

        s_fast = frame_mask(mask, N_FFT, HOP_LENGTH, WIN_LENGTH)
        s_ref = _frame_mask_brute(mask, N_FFT, HOP_LENGTH, WIN_LENGTH)

        assert torch.allclose(s_fast, s_ref), (
            f"frame_mask disagrees with brute force. "
            f"Max diff: {(s_fast - s_ref).abs().max().item()}"
        )


def test_frame_mask_all_ones():
    """All-ones mask must produce all-ones frame mask."""
    N = round(DURATION * SAMPLE_RATE)
    mask = torch.ones(N, dtype=torch.float32)
    s = frame_mask(mask, N_FFT, HOP_LENGTH, WIN_LENGTH)
    assert s.min().item() == 1.0, "All-ones mask should give all-ones frame mask"
    assert s.max().item() == 1.0


def test_frame_mask_single_missing_sample():
    """A single missing sample should mark exactly the frames whose window covers it."""
    N = 4000
    gap_pos = 2000  # missing sample index
    mask = torch.ones(N, dtype=torch.float32)
    mask[gap_pos] = 0.0

    s_fast = frame_mask(mask, N_FFT, HOP_LENGTH, WIN_LENGTH)
    s_ref = _frame_mask_brute(mask, N_FFT, HOP_LENGTH, WIN_LENGTH)

    assert torch.allclose(s_fast, s_ref), (
        "Single-sample gap: fast and brute-force disagree"
    )
    # At least one frame must be lost
    assert s_fast.min().item() == 0.0, "Expected at least one lost frame"


# ---------------------------------------------------------------------------
# Test 4: tf_mask
# ---------------------------------------------------------------------------

def test_tf_mask_shape():
    """tf_mask must broadcast s over frequency to shape (M, L)."""
    L = 100
    s = torch.ones(L)
    S = tf_mask(s, N_BINS)
    assert S.shape == (N_BINS, L)


def test_tf_mask_values():
    """tf_mask values must equal s in every frequency row."""
    L = 50
    s = torch.rand(L)
    S = tf_mask(s, N_BINS)
    for row in range(N_BINS):
        assert torch.allclose(S[row], s)


# ---------------------------------------------------------------------------
# Test 5: pad_to_multiple / crop_to_shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (513, 667),
    (512, 640),
    (100, 100),
    (1, 1),
    (31, 63),
])
def test_pad_crop_roundtrip(shape):
    """pad_to_multiple followed by crop_to_shape must return the original tensor."""
    M, L = shape
    x = torch.randn(1, 1, M, L)
    x_padded, orig_shape = pad_to_multiple(x, multiple=32)

    assert orig_shape == (M, L)
    assert x_padded.shape[-2] % 32 == 0, "Height not divisible by 32"
    assert x_padded.shape[-1] % 32 == 0, "Width not divisible by 32"

    x_cropped = crop_to_shape(x_padded, orig_shape)
    assert x_cropped.shape == x.shape
    assert torch.allclose(x_cropped, x), "Crop did not recover original tensor exactly"
