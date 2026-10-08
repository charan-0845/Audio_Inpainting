"""MultiResUNet architecture."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn

from src.models.harmonic_conv import HarmonicConv2d
from src.models.multires_block import MultiResBlock
from src.models.res_path import ResPath


class MultiResUNet(nn.Module):
    """Five-level MultiResUNet with Res Path skip connections and nearest-neighbour upsampling."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        channel_schedule: Optional[List[int]] = None,
        res_path_lengths: Optional[List[int]] = None,
        alpha: float = 1.6,
        negative_slope: float = 0.01,
        use_harmonic: bool = False,
        harmonic_anchors: Optional[Dict[str, int]] = None,
        norm: str = "batch",
        pad_multiple: int = 32,
    ) -> None:
        super().__init__()
        
        if channel_schedule is None:
            channel_schedule = [68, 68, 68, 68, 68]  # default to paper size
        if res_path_lengths is None:
            res_path_lengths = [4, 3, 2, 1]

        self.channel_schedule = channel_schedule
        self.res_path_lengths = res_path_lengths
        self.pad_multiple = pad_multiple
        
        self._depth = len(channel_schedule)
        self.use_harmonic = use_harmonic
        self.harmonic_anchors = harmonic_anchors or {"default": 1, "res_path_residual": 1, "downsampling_conv1": 1}

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.res_paths = nn.ModuleList()

        previous = in_channels
        encoder_out_channels = []

        default_anchor = self.harmonic_anchors.get("default", 1)

        for i, current in enumerate(channel_schedule):
            encoder = MultiResBlock(
                previous,
                current,
                alpha=alpha,
                negative_slope=negative_slope,
                use_harmonic=use_harmonic,
                harmonic_anchors=self.harmonic_anchors,
                norm=norm,
            )
            self.encoders.append(encoder)
            out_ch = encoder.out_channels
            encoder_out_channels.append(out_ch)

            if i < len(channel_schedule) - 1:
                next_ch = channel_schedule[i + 1]
                if use_harmonic:
                    ds_anchor = self.harmonic_anchors.get("downsampling_conv1", default_anchor)
                    downsample = HarmonicConv2d(
                        out_ch,
                        next_ch,
                        3,
                        stride=2,
                        padding="same",
                        bias=False,
                        anchor=ds_anchor,
                    )
                else:
                    downsample = nn.Conv2d(out_ch, next_ch, 3, stride=2, padding=1, bias=False)
                self.downsamples.append(downsample)
                previous = next_ch
            else:
                previous = out_ch

            if i < len(res_path_lengths):
                rp_anchor = self.harmonic_anchors.get("res_path_residual", default_anchor)
                self.res_paths.append(
                    ResPath(
                        out_ch,
                        length=res_path_lengths[i],
                        negative_slope=negative_slope,
                        use_harmonic=use_harmonic,
                        harmonic_anchors=self.harmonic_anchors,
                        norm=norm,
                    )
                )

        # Bottleneck
        self.bottleneck = MultiResBlock(
            previous,
            channel_schedule[-1],
            alpha=alpha,
            negative_slope=negative_slope,
            use_harmonic=use_harmonic,
            harmonic_anchors=self.harmonic_anchors,
            norm=norm,
        )
        bottleneck_out = self.bottleneck.out_channels

        self.decoders = nn.ModuleList()
        prev_dec_ch = bottleneck_out

        # If length is 5, we have 4 decoders (upsampling back to level 0)
        # We need len(channel_schedule) - 1 decoders.
        for i in range(len(channel_schedule) - 1):
            # i=0 means upsampling from level 4 to 3
            current_idx = len(channel_schedule) - 2 - i
            current_ch = channel_schedule[current_idx]
            skip_ch = encoder_out_channels[current_idx]

            in_dec_ch = prev_dec_ch + skip_ch
            decoder = MultiResBlock(
                in_dec_ch,
                current_ch,
                alpha=alpha,
                negative_slope=negative_slope,
                use_harmonic=False,
                norm=norm,
            )
            self.decoders.append(decoder)
            prev_dec_ch = decoder.out_channels

        self.output = nn.Conv2d(prev_dec_ch, out_channels, 1)
        self._parameter_count = self.count_parameters()

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"MultiResUNet expects (C, M, L) or (N, C, M, L)")
        if x.shape[-2] % self.pad_multiple or x.shape[-1] % self.pad_multiple:
            raise ValueError(f"MultiResUNet spatial dimensions must be divisible by {self.pad_multiple}")

        skips = []
        current = x
        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            if i < len(self.res_paths):
                skips.append(self.res_paths[i](current))
            else:
                skips.append(current)
            if i < len(self.downsamples):
                current = self.downsamples[i](current)

        current = self.bottleneck(current)

        # We have len(channel_schedule)-1 decoders.
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            current = nn.functional.interpolate(current, scale_factor=2, mode="nearest")
            if current.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(f"MultiResUNet skip connection shapes differ")
            current = decoder(torch.cat((current, skip), dim=1))

        result = self.output(current)
        return result.squeeze(0) if was_unbatched else result
