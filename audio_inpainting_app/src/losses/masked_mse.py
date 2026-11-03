"""Masked reconstruction loss functions for audio inpainting."""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute Mean Squared Error only on observed / valid regions (mask == 1).

    This implements the paper's data fidelity term E(X) (equation 5/7/10), which
    evaluates reconstruction error solely over observed TF frames/samples:
    E(X) = || S * (X - X_hat) ||_F^2 / || S ||_F^2.
    Division is performed by the number of observed elements (where mask == 1),
    not the total element count.

    Args:
        prediction: Predicted STFT tensor of shape (..., 2, M, L) real channels
            or (..., M, L) complex STFT.
        target: Target STFT tensor of shape matching prediction.
        mask: Frame mask vector (..., L) or TF mask matrix (..., M, L) with
            1.0 = observed/keep, 0.0 = missing. Broadcastable to prediction shape.

    Returns:
        Scalar PyTorch float tensor containing the masked MSE loss.
    """
    if torch.is_complex(prediction) or torch.is_complex(target):
        diff = prediction - target
        sq_err = diff.real.square() + diff.imag.square()
    else:
        sq_err = (prediction - target).square()

    # Expand mask to match sq_err dimensions for exact element count calculation
    mask_expanded = mask.expand_as(sq_err)
    num_observed = mask_expanded.sum()

    if num_observed == 0:
        logger.warning("Mask has no observed elements (all zeros); returning 0.0 loss.")
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

    masked_sq_err = sq_err * mask_expanded
    loss = masked_sq_err.sum() / num_observed
    return loss
