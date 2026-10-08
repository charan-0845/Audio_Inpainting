"""Tests for MultiResBlock, ResPath, MultiResUNet, and architecture comparison script."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.models.multires_block import MultiResBlock
from src.models.multires_unet import MultiResUNet
from src.models.res_path import ResPath
from src.models.unet import PlainUNet
from src.training.deep_prior import optimize_deep_prior
from src.training.noise import make_input_noise


def test_multires_block_shape_and_channels() -> None:
    block = MultiResBlock(in_channels=2, out_channels=16, alpha=1.6)
    x = torch.randn(2, 64, 64)
    out = block(x)
    assert out.shape == (block.out_channels, 64, 64)
    assert block.out_channels > 0


def test_multires_block_finite() -> None:
    block = MultiResBlock(in_channels=4, out_channels=32)
    x = torch.randn(2, 4, 32, 32)
    out = block(x)
    assert torch.isfinite(out).all()


def test_multires_block_width_mismatch_error() -> None:
    block = MultiResBlock(in_channels=4, out_channels=16)
    with pytest.raises(ValueError, match="input channel mismatch"):
        block(torch.randn(2, 2, 32, 32))


def test_res_path_shape_and_length() -> None:
    for length in [1, 2, 3, 4]:
        path = ResPath(in_channels=16, length=length)
        x = torch.randn(2, 16, 32, 32)
        out = path(x)
        assert out.shape == (2, 16, 32, 32)
        assert torch.isfinite(out).all()


def test_multires_unet_forward() -> None:
    model = MultiResUNet()
    x = torch.randn(2, 64, 64)
    out = model(x)
    assert out.shape == (2, 64, 64)
    assert torch.isfinite(out).all()


def test_multires_unet_rejects_non_divisible_spatial_shape() -> None:
    model = MultiResUNet()
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(2, 65, 64))


def test_multires_unet_parameter_count_near_target() -> None:
    count = MultiResUNet().count_parameters()
    print(f"MultiResUNet parameter count: {count}")
    assert abs(count - 2_015_252) / 2_015_252 < 0.05





def test_multires_unet_eval_forward_is_deterministic() -> None:
    model = MultiResUNet().eval()
    x = torch.randn(2, 64, 64)
    assert torch.equal(model(x), model(x))


def test_multires_unet_deep_prior_integration() -> None:
    torch.manual_seed(42)
    model = MultiResUNet(channel_schedule=[8, 8, 8, 8, 8], res_path_lengths=[4, 3, 2, 1])
    observed = torch.randn(2, 64, 64)
    mask = torch.ones(64, 64)
    noise = make_input_noise((2, 64, 64), seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    result = optimize_deep_prior(
        model,
        noise,
        observed,
        mask,
        optimizer,
        epochs=30,
        log_every=10,
        generator=torch.Generator().manual_seed(42),
    )

    assert len(result["loss_history"]) == 30
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result["loss_history"])



