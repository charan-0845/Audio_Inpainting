"""Deterministic noise construction for deep-prior optimization."""

from __future__ import annotations

import torch


def make_input_noise(
    shape: tuple[int, ...],
    variance: float = 0.1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create fixed uniform noise with the requested variance.

    ``variance`` is interpreted as the variance of the returned distribution,
    so the uniform interval is ``[-sqrt(3v), sqrt(3v)]``.
    """
    if variance < 0:
        raise ValueError("variance must be non-negative")
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both")
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    scale = (3.0 * variance) ** 0.5
    return (torch.rand(shape, generator=generator, device=device) * 2.0 - 1.0) * scale


def perturb_noise(
    z: torch.Tensor,
    variance: float = 0.03,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return fresh zero-mean Gaussian perturbation with the same shape/device."""
    if variance < 0:
        raise ValueError("variance must be non-negative")
    return torch.randn(
        z.shape,
        generator=generator,
        device=z.device,
        dtype=z.dtype,
    ) * variance**0.5
