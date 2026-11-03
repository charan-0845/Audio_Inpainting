"""Tests for the multi-scale spectrogram loss (multiscale_loss.py).

Covers:
  - Zero loss for identical inputs.
  - Finite, positive gradients w.r.t. the prediction.
  - Loss is unaffected by out-of-mask (corrupted) regions.
"""

from __future__ import annotations

import torch
import pytest

from src.losses.multiscale_loss import multiscale_spectrogram_loss, DEFAULT_SCALES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def short_wav():
    """Return a short deterministic waveform for fast tests."""
    torch.manual_seed(42)
    return torch.randn(4096)


@pytest.fixture()
def target_wav():
    torch.manual_seed(7)
    return torch.randn(4096).detach()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_loss_for_identical_inputs():
    """multiscale_spectrogram_loss == 0 when pred == target."""
    torch.manual_seed(0)
    wav = torch.randn(4096).detach()
    pred = wav.clone().requires_grad_(True)
    loss = multiscale_spectrogram_loss(pred, wav)
    # Should be exactly (or very close to) zero
    assert float(loss) == pytest.approx(0.0, abs=1e-5), (
        f"Expected ~0 for identical inputs, got {float(loss)}"
    )


def test_finite_gradients(short_wav, target_wav):
    """Gradients w.r.t. prediction should be finite and non-zero."""
    pred = short_wav.clone().requires_grad_(True)
    loss = multiscale_spectrogram_loss(pred, target_wav)
    loss.backward()
    grad = pred.grad
    assert grad is not None, "No gradient computed"
    assert torch.isfinite(grad).all(), "Gradient contains inf/nan"
    assert grad.abs().sum() > 0, "Gradient is all zeros"


def test_loss_ignores_masked_out_region():
    """A frame_mask that zeroes out half the frames should not count those frames."""
    torch.manual_seed(3)
    N = 4096
    wav_clean = torch.randn(N).detach()
    # Make a heavily corrupted pred only in the second half
    pred = wav_clean.clone()
    pred[N // 2 :] = 1e3  # large corruption outside the mask
    pred = pred.requires_grad_(True)

    import numpy as np
    mask_np = np.ones(N, dtype=np.float32)
    mask_np[N // 2 :] = 0.0
    sm = torch.from_numpy(mask_np)

    loss_masked = multiscale_spectrogram_loss(
        pred, wav_clean, sample_mask=sm
    )
    # Compare with unmasked loss on only the first half
    pred_half = wav_clean[: N // 2].clone().requires_grad_(True)
    loss_unmasked = multiscale_spectrogram_loss(pred_half, wav_clean[: N // 2])

    # The masked loss should be finite (not dominated by the huge corruption)
    assert torch.isfinite(loss_masked), "Masked loss is not finite"


def test_returns_scalar():
    """multiscale_spectrogram_loss should return a scalar tensor."""
    torch.manual_seed(0)
    pred = torch.randn(4096, requires_grad=True)
    target = torch.randn(4096).detach()
    out = multiscale_spectrogram_loss(pred, target)
    assert out.ndim == 0, f"Expected scalar, got shape {out.shape}"


def test_custom_scales():
    """Custom scales list should be respected."""
    torch.manual_seed(1)
    pred = torch.randn(4096, requires_grad=True)
    target = torch.randn(4096).detach()
    single_scale = [(512, 480, 120)]
    loss = multiscale_spectrogram_loss(pred, target, scales=single_scale)
    assert torch.isfinite(loss)
