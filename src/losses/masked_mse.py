"""Masked reconstruction losses."""


def masked_mse(prediction, target, mask):
    """Compute MSE only on observed/valid regions."""
    raise NotImplementedError
