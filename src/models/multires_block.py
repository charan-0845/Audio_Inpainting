"""MultiRes block implementation for MultiResUNet."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.models.harmonic_conv import HarmonicConv2d


class MultiResBlock(nn.Module):
    """MultiRes convolutional block as described in MultiResUNet (Ibtehaz & Rahman, 2020).

    Factorizes 5x5 and 7x7 convolutions into three consecutive 3x3 convolutions,
    concatenates intermediate multi-resolution features, and adds a 1x1 residual shortcut.
    Supports optional HarmonicConv2d when use_harmonic=True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 1.6,
        filter_ratios: Tuple[float, float, float] = (1 / 6, 2 / 6, 3 / 6),
        negative_slope: float = 0.01,
        use_harmonic: bool = False,
        harmonic_anchors: dict = None,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_harmonic = use_harmonic
        self.harmonic_anchors = harmonic_anchors or {}

        W = alpha * out_channels
        w1 = max(1, int(round(W * filter_ratios[0])))
        w2 = max(1, int(round(W * filter_ratios[1])))
        w3 = max(1, int(round(W * filter_ratios[2])))

        self.w1, self.w2, self.w3 = w1, w2, w3
        self.out_channels = w1 + w2 + w3

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

        a_c1 = self.harmonic_anchors.get("multires.conv1", 1)
        a_c2 = self.harmonic_anchors.get("multires.conv2", 2)
        a_c3 = self.harmonic_anchors.get("multires.conv3", 3)
        a_res = self.harmonic_anchors.get("multires.residual", 1)

        self.conv3x3_1 = nn.Sequential(
            _make_conv(in_channels, w1, 3, a_c1),
            _make_norm(w1),
            nn.LeakyReLU(negative_slope, inplace=True),
        )
        self.conv3x3_2 = nn.Sequential(
            _make_conv(w1, w2, 3, a_c2),
            _make_norm(w2),
            nn.LeakyReLU(negative_slope, inplace=True),
        )
        self.conv3x3_3 = nn.Sequential(
            _make_conv(w2, w3, 3, a_c3),
            _make_norm(w3),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.shortcut = nn.Sequential(
            _make_conv(in_channels, self.out_channels, 1, a_res),
            _make_norm(self.out_channels),
        )
        self.norm = _make_norm(self.out_channels)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"MultiResBlock expects 3D or 4D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"MultiResBlock input channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )

        c1 = self.conv3x3_1(x)
        c2 = self.conv3x3_2(c1)
        c3 = self.conv3x3_3(c2)

        concat_features = torch.cat([c1, c2, c3], dim=1)
        shortcut_features = self.shortcut(x)

        if concat_features.shape[-2:] != shortcut_features.shape[-2:]:
            raise ValueError(
                f"MultiResBlock spatial dimension mismatch: concat {concat_features.shape[-2:]} vs shortcut {shortcut_features.shape[-2:]}"
            )

        out = self.act(self.norm(concat_features + shortcut_features))
        return out.squeeze(0) if was_unbatched else out
