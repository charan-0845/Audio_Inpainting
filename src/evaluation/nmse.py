"""Normalized Mean Squared Error (NMSE) evaluation metrics."""

from __future__ import annotations

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nmse(
    reference: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
) -> float:
    """Compute per-clip Normalized Mean Squared Error (NMSE) in time domain.

    Formula:
        NMSE(x, x_hat) = ||x - x_hat||_2^2 / ||x||_2^2
                       = sum((x - x_hat)^2) / sum(x^2)

    For perfect reconstruction (x == x_hat), NMSE is 0.0.
    For zero reconstruction (x_hat == 0), NMSE is 1.0.

    Args:
        reference: Time-domain clean reference waveform (N,).
        reconstruction: Time-domain reconstructed waveform (N,).

    Returns:
        Float value of NMSE, or float('nan') if reference energy is zero.
    """
    ref = _to_numpy(reference).astype(np.float64)
    rec = _to_numpy(reconstruction).astype(np.float64)

    ref_energy = np.sum(ref**2)
    if ref_energy == 0.0:
        logger.warning("Reference signal has zero energy; returning NaN for NMSE.")
        return float("nan")

    err_energy = np.sum((ref - rec) ** 2)
    val = err_energy / ref_energy
    return float(val)


def missing_region_nmse(
    reference: np.ndarray | torch.Tensor,
    reconstruction: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
) -> float:
    """Compute NMSE evaluated restricted strictly to originally missing samples.

    Note on mask convention:
    Following Task 1's create_mask convention, mask == 1.0 denotes OBSERVED
    samples, and mask == 0.0 denotes MISSING (corrupted) samples.
    This function evaluates reconstruction quality ONLY on the missing region
    where mask == 0.0 (the opposite of masked_mse loss which evaluates on 1.0).

    Formula:
        NMSE_missing = sum_{i: mask[i]==0} (x_i - x_hat_i)^2 / sum_{i: mask[i]==0} x_i^2

    Args:
        reference: Time-domain clean reference waveform (N,).
        reconstruction: Time-domain reconstructed waveform (N,).
        mask: Sample-level binary mask (N,) where 1=observed, 0=missing.

    Returns:
        Float value of missing region NMSE, or float('nan') if missing region
        reference energy is zero.
    """
    ref = _to_numpy(reference).astype(np.float64)
    rec = _to_numpy(reconstruction).astype(np.float64)
    m = _to_numpy(mask).astype(np.float64)

    missing_idx = (m < 0.5)
    if not np.any(missing_idx):
        logger.warning("Mask has no missing regions (all observed); returning NaN for missing_region_nmse.")
        return float("nan")

    ref_missing = ref[missing_idx]
    rec_missing = rec[missing_idx]

    ref_energy = np.sum(ref_missing**2)
    if ref_energy == 0.0:
        logger.warning("Reference signal in missing region has zero energy; returning NaN for missing_region_nmse.")
        return float("nan")

    err_energy = np.sum((ref_missing - rec_missing) ** 2)
    val = err_energy / ref_energy
    return float(val)
