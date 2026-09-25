"""Plain U-Net baseline used by the single-sample deep-prior experiment."""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlainUNet(nn.Module):
    """Five-level U-Net with nearest-neighbour upsampling and plain skips.

    GroupNorm is used instead of BatchNorm because deep-prior optimization
    operates on one spectrogram at a time and must also support 1x1
    bottleneck spatial maps without BatchNorm's batch-size-1 failure.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_filters: int = 7,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        channels = [base_filters * (2**i) for i in range(5)]
        self._depth = 5
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = in_channels
        for current in channels:
            self.encoders.append(_ConvBlock(previous, current, negative_slope))
            self.downsamples.append(
                nn.Conv2d(current, current * 2, 3, stride=2, padding=1, bias=False)
            )
            previous = current * 2

        self.bottleneck = _ConvBlock(previous, previous, negative_slope)
        self.decoders = nn.ModuleList()
        for current in reversed(channels):
            self.decoders.append(_ConvBlock(current * 2 + current, current, negative_slope))
        self.output = nn.Conv2d(base_filters, out_channels, 1)
        self._parameter_count = self.count_parameters()

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"PlainUNet expects (C, M, L) or (N, C, M, L), got {tuple(x.shape)}")
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(
                "PlainUNet spatial dimensions must be divisible by 32; "
                f"received {(x.shape[-2], x.shape[-1])}. Pad the spectrogram before the model."
            )

        skips = []
        current = x
        for encoder, downsample in zip(self.encoders, self.downsamples):
            current = encoder(current)
            skips.append(current)
            current = downsample(current)

        current = self.bottleneck(current)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            current = nn.functional.interpolate(current, scale_factor=2, mode="nearest")
            if current.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError("U-Net skip connection shapes differ after divisible-by-32 padding.")
            current = decoder(torch.cat((current, skip), dim=1))
        result = self.output(current)
        return result.squeeze(0) if was_unbatched else result
