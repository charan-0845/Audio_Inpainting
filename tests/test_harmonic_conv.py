"""Tests for HarmonicConv2d and MultiResUNet with harmonic convolutions."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from src.models.harmonic_conv import HarmonicConv2d
from src.models.multires_block import MultiResBlock
from src.models.multires_unet import MultiResUNet
from src.models.res_path import ResPath
from src.training.deep_prior import optimize_deep_prior
from src.training.noise import make_input_noise


def test_harmonic_conv_output_shape() -> None:
    hconv = HarmonicConv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), padding="same")
    rconv = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), padding=1)
    x = torch.randn(2, 2, 64, 64)
    out_h = hconv(x)
    out_r = rconv(x)
    assert out_h.shape == out_r.shape == (2, 4, 64, 64)


def test_harmonic_conv_parameter_count() -> None:
    hconv = HarmonicConv2d(in_channels=4, out_channels=8, kernel_size=(3, 5), bias=True)
    rconv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 5), bias=True)
    h_params = sum(p.numel() for p in hconv.parameters())
    r_params = sum(p.numel() for p in rconv.parameters())
    assert h_params == r_params == 4 * 8 * 3 * 5 + 8


def test_harmonic_conv_anchor1_sanity_indices() -> None:
    M = 8
    Km = 3
    hconv = HarmonicConv2d(in_channels=2, out_channels=4, kernel_size=(Km, 3), anchor=1)
    indices = hconv._get_gather_indices(M, torch.device("cpu"))

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 3, 5, 7, 7, 7, 7, 7],
            [2, 5, 7, 7, 7, 7, 7, 7],
        ],
        dtype=torch.long,
    )
    assert torch.equal(indices, expected)


def test_harmonic_conv_nan_and_gradient() -> None:
    hconv = HarmonicConv2d(in_channels=2, out_channels=4, kernel_size=(3, 3))
    x = torch.randn(2, 2, 32, 32, requires_grad=True)
    out = hconv(x)
    assert torch.isfinite(out).all()

    loss = out.pow(2).sum()
    loss.backward()
    assert hconv.weight.grad is not None
    assert torch.isfinite(hconv.weight.grad).all()


def test_default_args_regression_bit_for_bit() -> None:
    """Critical regression check: default args MUST produce bit-for-bit identical output to regular conv."""
    torch.manual_seed(42)
    m_default = MultiResUNet(use_harmonic=False)

    torch.manual_seed(100)
    x = torch.randn(2, 2, 64, 64)

    m_default.eval()
    out_default = m_default(x)

    # Re-instantiate with default (use_harmonic not passed) under same seed
    torch.manual_seed(42)
    m_implicit = MultiResUNet()
    m_implicit.eval()
    out_implicit = m_implicit(x)

    assert torch.equal(out_default, out_implicit)


def test_multires_unet_harmonic_forward_and_params() -> None:
    m_regular = MultiResUNet(use_harmonic=False)
    m_harmonic = MultiResUNet(use_harmonic=True)

    assert m_regular.count_parameters() == m_harmonic.count_parameters()
    assert abs(m_harmonic.count_parameters() - 2_015_252) / 2_015_252 < 0.05

    x = torch.randn(2, 2, 64, 64)
    out_h = m_harmonic(x)
    assert out_h.shape == (2, 2, 64, 64)
    assert torch.isfinite(out_h).all()


def test_multires_unet_harmonic_deep_prior_integration() -> None:
    torch.manual_seed(42)
    model = MultiResUNet(base_filters=4, use_harmonic=True)
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
        epochs=20,
        log_every=5,
        generator=torch.Generator().manual_seed(42),
    )

    assert len(result["loss_history"]) == 20
    assert result["loss_history"][-1] < result["loss_history"][0]
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result["loss_history"])


def test_compare_with_harmonic_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "comparison_harmonic"
    cmd = [
        sys.executable,
        "scripts/compare_with_harmonic.py",
        "--clip_id",
        "synthetic_01",
        "--gap_ms",
        "400",
        "--epochs",
        "10",
        "--out_dir",
        str(out_dir),
        "--model",
        "all",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with output:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    assert (out_dir / "synthetic_01_400ms_plainunet_summary.json").exists()
    assert (out_dir / "synthetic_01_400ms_multiresunet_summary.json").exists()
    assert (out_dir / "synthetic_01_400ms_multiresunet_harmonic_summary.json").exists()
    assert (out_dir / "synthetic_01_400ms_comparison.png").exists()
