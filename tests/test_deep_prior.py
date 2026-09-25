import torch
from torch import nn

from src.models.unet import PlainUNet
from src.training.deep_prior import optimize_deep_prior
from src.training.noise import make_input_noise


def _run(observed: torch.Tensor, reference: torch.Tensor | None = None):
    torch.manual_seed(10)
    model = PlainUNet(base_filters=2)
    noise_generator = torch.Generator().manual_seed(20)
    noise = make_input_noise((2, 64, 64), seed=21)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    return optimize_deep_prior(
        model,
        noise,
        observed,
        torch.ones(64, 64),
        optimizer,
        epochs=20,
        log_every=5,
        reference=reference,
        generator=noise_generator,
    )


def test_loss_history_and_diagnostic_epochs() -> None:
    target = torch.randn(2, 64, 64)
    result = _run(target, target)
    assert len(result["loss_history"]) == 20
    assert [epoch for epoch, _ in result["val_loss_history"]] == [0, 5, 10, 15, 19]
    assert result["loss_history"][-1] < result["loss_history"][0]


def test_missing_region_garbage_does_not_change_training_loss() -> None:
    torch.manual_seed(5)
    target = torch.randn(2, 64, 64)
    mask = torch.zeros(64, 64)
    mask[:, :40] = 1
    first = target.clone()
    second = target.clone()
    first[:, :, 40:] = 100
    second[:, :, 40:] = -100

    def run(observed):
        torch.manual_seed(10)
        model = PlainUNet(base_filters=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        return optimize_deep_prior(
            model,
            make_input_noise((2, 64, 64), seed=21),
            observed,
            mask,
            optimizer,
            epochs=4,
            generator=torch.Generator().manual_seed(20),
        )["loss_history"]

    assert run(first) == run(second)


def test_reference_requires_no_grad() -> None:
    reference = torch.randn(2, 64, 64, requires_grad=True)
    try:
        _run(torch.randn(2, 64, 64), reference)
    except ValueError as error:
        assert "reference" in str(error)
    else:
        raise AssertionError("requires_grad reference should be rejected")


def test_same_seed_is_deterministic_and_perturbation_is_fresh() -> None:
    target = torch.randn(2, 64, 64)
    first = _run(target)["loss_history"]
    second = _run(target)["loss_history"]
    assert first == second

    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.inputs = []

        def forward(self, value):
            self.inputs.append(value.detach().clone())
            return value * self.weight

    model = RecordingModel()
    optimize_deep_prior(
        model,
        torch.zeros(2, 64, 64),
        torch.zeros(2, 64, 64),
        torch.ones(64, 64),
        torch.optim.Adam(model.parameters(), lr=0.01),
        epochs=3,
        generator=torch.Generator().manual_seed(2),
    )
    assert not torch.equal(model.inputs[0], model.inputs[1])
