"""Res Path implementation for MultiResUNet skip connections."""

from __future__ import annotations

import torch
from torch import nn


class ResPath(nn.Module):
    """Res Path module to mitigate semantic gap between encoder and decoder skip features.

    Consists of a sequence of `length` mini-blocks, each combining a 3x3 convolution
    and a 1x1 residual shortcut connection while preserving channel counts.
    """

    def __init__(
        self,
        in_channels: int,
        length: int,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.length = length

        self.blocks = nn.ModuleList()
        for _ in range(length):
            block = nn.ModuleDict(
                {
                    "conv3x3": nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                        nn.GroupNorm(1, in_channels),
                        nn.LeakyReLU(negative_slope, inplace=True),
                    ),
                    "shortcut": nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, 1, bias=False),
                        nn.GroupNorm(1, in_channels),
                    ),
                    "norm": nn.GroupNorm(1, in_channels),
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
