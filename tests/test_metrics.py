"""Tests for masked_mse loss and NMSE metrics."""

from __future__ import annotations

import logging
import numpy as np
import pytest
import torch

from src.losses.masked_mse import masked_mse
from src.evaluation.nmse import nmse, missing_region_nmse


# ---------------------------------------------------------------------------
# masked_mse tests
# ---------------------------------------------------------------------------

def test_masked_mse_identical_inputs():
    """1. masked_mse(x, x, mask) == 0 for any mask."""
    x = torch.randn(2, 513, 100)
    mask = torch.randint(0, 2, (100,)).float()

    loss = masked_mse(x, x, mask)
    assert float(loss.item()) == pytest.approx(0.0)


def test_masked_mse_ignores_outside_mask():
    """2. masked_mse ignores differences outside the mask (perturb target only outside mask)."""
    target = torch.zeros(2, 513, 100)
    prediction = torch.zeros(2, 513, 100)

    # Mask: 1 for first 50 frames, 0 for last 50 frames
    mask = torch.zeros(100)
    mask[:50] = 1.0

    # Perturb prediction ONLY in unmasked (mask==0) region
    prediction[:, :, 50:] = 100.0

    loss = masked_mse(prediction, target, mask)
    # Since masked region (first 50 frames) is identical, loss should be 0.0
    assert float(loss.item()) == pytest.approx(0.0)

    # If we perturb inside masked region, loss should be non-zero
    prediction[:, :, :50] = 1.0
    loss_nonzero = masked_mse(prediction, target, mask)
    assert float(loss_nonzero.item()) == pytest.approx(1.0)


def test_masked_mse_all_zero_mask(caplog):
    """3. masked_mse on an all-zero mask returns 0.0, not NaN, and warns."""
    x = torch.randn(2, 513, 100)
    target = torch.randn(2, 513, 100)
    zero_mask = torch.zeros(100)

    with caplog.at_level(logging.WARNING):
        loss = masked_mse(x, target, zero_mask)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert float(loss.item()) == 0.0
    assert "Mask has no observed elements" in caplog.text


def test_masked_mse_complex_stft():
    """Test masked_mse works on complex STFT tensors (M, L)."""
    X_clean = torch.complex(torch.ones(513, 50), torch.ones(513, 50))
    X_pred = torch.complex(torch.zeros(513, 50), torch.zeros(513, 50))
    mask = torch.ones(50)

    loss = masked_mse(X_pred, X_clean, mask)
    # |(1+1j) - 0|^2 = 1^2 + 1^2 = 2.0
    assert float(loss.item()) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# NMSE and missing_region_nmse tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_torch", [True, False])
def test_nmse_values(use_torch):
    """4. nmse(x, x) == 0. nmse(x, 0) == 1.0."""
    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    zero_data = np.zeros_like(x_data)

    if use_torch:
        x = torch.from_numpy(x_data)
        zero = torch.from_numpy(zero_data)
    else:
        x = x_data
        zero = zero_data

    assert nmse(x, x) == pytest.approx(0.0)
    assert nmse(x, zero) == pytest.approx(1.0)


@pytest.mark.parametrize("use_torch", [True, False])
def test_missing_region_nmse_isolated(use_torch):
    """5. missing_region_nmse computed only on corrupted region (mask == 0)."""
    # 100 samples total
    ref_np = np.ones(100, dtype=np.float32) * 2.0
    rec_np = np.ones(100, dtype=np.float32) * 2.0  # Perfect prediction by default

    # Mask: 1 = observed (samples 0..49), 0 = missing (samples 50..99)
    mask_np = np.ones(100, dtype=np.float32)
    mask_np[50:] = 0.0

    # Perturb reconstruction ONLY in observed region (samples 0..49)
    rec_np[:50] = 999.0

    if use_torch:
        ref = torch.from_numpy(ref_np)
        rec = torch.from_numpy(rec_np)
        mask = torch.from_numpy(mask_np)
    else:
        ref = ref_np
        rec = rec_np
        mask = mask_np

    # missing_region_nmse should still be 0.0 because missing region (50..99) is perfect!
    val_perfect_missing = missing_region_nmse(ref, rec, mask)
    assert val_perfect_missing == pytest.approx(0.0)

    # Now perturb reconstruction in missing region (samples 50..99) to 0.0
    if use_torch:
        rec[50:] = 0.0
    else:
        rec[50:] = 0.0

    val_zero_missing = missing_region_nmse(ref, rec, mask)
    # Since rec in missing region is 0, NMSE_missing should be 1.0
    assert val_zero_missing == pytest.approx(1.0)


def test_nmse_zero_energy_reference(caplog):
    """Zero energy reference returns NaN with a warning."""
    ref = np.zeros(100, dtype=np.float32)
    rec = np.ones(100, dtype=np.float32)
    mask = np.zeros(100, dtype=np.float32)

    with caplog.at_level(logging.WARNING):
        val = nmse(ref, rec)
        val_missing = missing_region_nmse(ref, rec, mask)

    assert np.isnan(val)
    assert np.isnan(val_missing)
