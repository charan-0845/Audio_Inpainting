"""Res Path implementation for MultiResUNet skip connections."""

from __future__ import annotations

import torch
from torch import nn

from src.models.harmonic_conv import HarmonicConv2d


class ResPath(nn.Module):
    """Res Path module to mitigate semantic gap between encoder and decoder skip features.

    Consists of a sequence of `length` mini-blocks, each combining a 3x3 convolution
    and a 1x1 residual shortcut connection while preserving channel counts.
    Supports optional HarmonicConv2d when use_harmonic=True.
    """

    def __init__(
        self,
        in_channels: int,
        length: int,
        negative_slope: float = 0.01,
        use_harmonic: bool = False,
        harmonic_anchors: dict = None,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.length = length
        self.use_harmonic = use_harmonic
        self.harmonic_anchors = harmonic_anchors or {}

        def _make_conv(in_c: int, out_c: int, kernel: int, anchor_val: int) -> nn.Module:
            if use_harmonic:
                return HarmonicConv2d(in_c, out_c, kernel, anchor=anchor_val, padding="same", bias=False)
            return nn.Conv2d(in_c, out_c, kernel, padding=kernel // 2, bias=False)

        def _make_norm(num_channels: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm2d(num_channels)
            elif norm == "group":
                return nn.GroupNorm(1, num_channels)
            raise ValueError(f"Unknown norm type: {norm}")

        a_c1 = self.harmonic_anchors.get("respath.conv1", 1)
        a_res = self.harmonic_anchors.get("respath.residual", 1)

        self.blocks = nn.ModuleList()
        for _ in range(length):
            block = nn.ModuleDict(
                {
                    "conv3x3": nn.Sequential(
                        _make_conv(in_channels, in_channels, 3, a_c1),
                        _make_norm(in_channels),
                        nn.LeakyReLU(negative_slope, inplace=True),
                    ),
                    "shortcut": nn.Sequential(
                        _make_conv(in_channels, in_channels, 1, a_res),
                        _make_norm(in_channels),
                    ),
                    "norm": _make_norm(in_channels),
                    "act": nn.LeakyReLU(negative_slope, inplace=True),
                }
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"ResPath expects 3D or 4D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"ResPath input channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )

        current = x
        for block in self.blocks:
            main = block["conv3x3"](current)
            shortcut = block["shortcut"](current)
            current = block["act"](block["norm"](main + shortcut))

        return current.squeeze(0) if was_unbatched else current
