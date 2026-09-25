import pytest
import torch

from src.models.unet import PlainUNet


def test_forward_preserves_shape_and_is_finite() -> None:
    model = PlainUNet()
    output = model(torch.randn(2, 64, 64))
    assert output.shape == (2, 64, 64)
    assert torch.isfinite(output).all()


def test_rejects_non_divisible_spatial_shape() -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        PlainUNet()(torch.randn(2, 65, 64))


def test_parameter_count_is_near_paper_target() -> None:
    count = PlainUNet().count_parameters()
    print(f"PlainUNet parameter count: {count}")
    assert abs(count - 2_158_578) / 2_158_578 < 0.05


def test_eval_forward_is_deterministic() -> None:
    model = PlainUNet().eval()
    input_tensor = torch.randn(2, 64, 64)
    assert torch.equal(model(input_tensor), model(input_tensor))
