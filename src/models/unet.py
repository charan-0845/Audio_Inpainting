
"""Plain U-Net baseline used by the single-sample deep-prior experiment."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float, norm: str = "batch") -> None:
        super().__init__()
        
        def _make_norm(num_channels: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm2d(num_channels)
            elif norm == "group":
                return nn.GroupNorm(1, num_channels)
            raise ValueError(f"Unknown norm type: {norm}")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _make_norm(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlainUNet(nn.Module):
    """Five-level U-Net with nearest-neighbour upsampling and plain skips."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        channel_schedule: Optional[List[int]] = None,
        negative_slope: float = 0.01,
        norm: str = "batch",
        pad_multiple: int = 32,
    ) -> None:
        super().__init__()
        if channel_schedule is None:
            # Default to the flat schedule that hits the 2M parameter count
            channel_schedule = [68, 68, 68, 68, 68]
            
        self.channel_schedule = channel_schedule
        self.pad_multiple = pad_multiple
        self._depth = len(channel_schedule)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = in_channels
        encoder_out_channels = []
        for i, current in enumerate(channel_schedule):
            self.encoders.append(_ConvBlock(previous, current, negative_slope, norm=norm))
            encoder_out_channels.append(current)
            if i < len(channel_schedule) - 1:
                next_ch = channel_schedule[i + 1]
                self.downsamples.append(
                    nn.Conv2d(current, next_ch, 3, stride=2, padding=1, bias=False)
                )
                previous = next_ch
            else:
                previous = current

        self.bottleneck = _ConvBlock(previous, channel_schedule[-1], negative_slope, norm=norm)
        self.decoders = nn.ModuleList()
        
        prev_dec_ch = channel_schedule[-1]
        for current_ch, skip_ch in zip(reversed(channel_schedule[:-1]), reversed(encoder_out_channels[:-1])):
            in_dec_ch = prev_dec_ch + skip_ch
            decoder = _ConvBlock(in_dec_ch, current_ch, negative_slope, norm=norm)
            self.decoders.append(decoder)
            prev_dec_ch = current_ch
        self.output = nn.Conv2d(channel_schedule[0], out_channels, 1)
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
        if x.shape[-2] % self.pad_multiple or x.shape[-1] % self.pad_multiple:
            raise ValueError(
                f"PlainUNet spatial dimensions must be divisible by {self.pad_multiple}; "
                f"received {(x.shape[-2], x.shape[-1])}. Pad the spectrogram before the model."
            )

        skips = []
        current = x
        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            skips.append(current)
            if i < len(self.downsamples):
                current = self.downsamples[i](current)

        current = self.bottleneck(current)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            current = nn.functional.interpolate(current, scale_factor=2, mode="nearest")
            if current.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(f"U-Net skip connection shapes differ after divisible-by-{self.pad_multiple} padding.")
            current = decoder(torch.cat((current, skip), dim=1))
        result = self.output(current)
        return result.squeeze(0) if was_unbatched else result
