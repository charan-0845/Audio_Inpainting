"""Single-sample deep-prior optimization."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from src.losses.masked_mse import masked_mse
from src.training.noise import perturb_noise


def _loss_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(0)
    return mask.expand_as(value)


def optimize_deep_prior(
    model: torch.nn.Module,
    noise: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    perturbation_variance: float = 0.03,
    log_every: int = 50,
    reference: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    include_model_state: bool = False,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> dict[str, Any]:
    """Fit a network to observed spectrogram regions for one sample.

    The reference is used only for a no-grad missing-region diagnostic.  It is
    deliberately rejected when it requires gradients to prevent leakage.

    Args:
        model: Network to optimise.
        noise: Fixed input noise tensor ``(C, M, L)``.
        observed: Observed (corrupted) STFT channels ``(C, M, L)`` or ``(2, M, L)``.
        mask: TF mask ``(M, L)`` or ``(C, M, L)`` with 1=observed, 0=missing.
        optimizer: Configured optimiser bound to *model*.
        epochs: Number of optimisation steps (must be >= 1).
        perturbation_variance: Variance of per-epoch Gaussian perturbation
            added to *noise*.
        log_every: Frequency (in epochs) at which to evaluate the
            missing-region diagnostic when *reference* is supplied.
        reference: Detached clean spectrogram channels for diagnostics only.
            Must not require gradients.
        generator: Optional :class:`torch.Generator` for reproducible noise.
        include_model_state: If True, snapshot ``model.state_dict()`` in
            the returned dict (useful for checkpointing).
        loss_fn: Optional callable ``(pred, target, mask) -> scalar``.
            When ``None`` (default), :func:`~src.losses.masked_mse.masked_mse`
            is used directly, giving bit-for-bit identical behaviour to the
            original implementation.  All existing tests pass with ``None``.

    Returns:
        Dict with keys:

        * ``final_output`` – detached final model output ``(C, M, L)``.
        * ``loss_history`` – list of per-epoch training loss floats.
        * ``val_loss_history`` – list of ``(epoch, loss)`` diagnostic tuples.
        * ``model_state`` – (optional) copy of model state dict.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    if reference is not None and reference.requires_grad:
        raise ValueError("reference must not require gradients")

    _loss_fn = loss_fn if loss_fn is not None else masked_mse

    device = next(model.parameters()).device
    noise = noise.to(device)
    observed = observed.to(device)
    mask = _loss_mask(mask.to(device), observed)
    if reference is not None:
        reference = reference.detach().to(device)

    model.train()
    loss_history: list[float] = []
    val_loss_history: list[tuple[int, float]] = []
    output: torch.Tensor | None = None
    for epoch in range(epochs):
        z_perturbed = noise + perturb_noise(
            torch.zeros_like(noise), perturbation_variance, generator
        )
        output = model(z_perturbed)
        loss = _loss_fn(output, observed, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        if epoch % 250 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1}/{epochs}  loss={loss_history[-1]:.4e}", flush=True)

        if reference is not None and (epoch % log_every == 0 or epoch == epochs - 1):
            with torch.no_grad():
                diagnostic = masked_mse(output.detach(), reference, 1.0 - mask)
            val_loss_history.append((epoch, float(diagnostic.cpu())))

    assert output is not None
    result: dict[str, Any] = {
        "final_output": output.detach(),
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
    }
    if include_model_state:
        result["model_state"] = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
    return result
