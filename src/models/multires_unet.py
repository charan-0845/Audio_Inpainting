"""MultiResUNet architecture."""

from __future__ import annotations

import torch
from torch import nn

from src.models.multires_block import MultiResBlock
from src.models.res_path import ResPath


class MultiResUNet(nn.Module):
    """Five-level MultiResUNet with Res Path skip connections and nearest-neighbour upsampling.

    Replaces standard double convolutions with MultiResBlocks and processes skip connections
    with ResPaths of decreasing length (4, 3, 2, 1).
    GroupNorm is used instead of BatchNorm to match PlainUNet's deep-prior single-spectrogram setup.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_filters: int = 7,
        alpha: float = 1.6,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self._depth = 5
        channels = [base_filters * (2**i) for i in range(5)]

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.res_paths = nn.ModuleList()

        previous = in_channels
        encoder_out_channels = []
        res_lengths = [4, 3, 2, 1]

        for i, current in enumerate(channels):
            encoder = MultiResBlock(previous, current, alpha=alpha, negative_slope=negative_slope)
            self.encoders.append(encoder)
            out_ch = encoder.out_channels
            encoder_out_channels.append(out_ch)

            next_ch = out_ch * 2
            downsample = nn.Conv2d(out_ch, next_ch, 3, stride=2, padding=1, bias=False)
            self.downsamples.append(downsample)
            previous = next_ch

            if i < 4:
                self.res_paths.append(
                    ResPath(out_ch, length=res_lengths[i], negative_slope=negative_slope)
                )

        self.bottleneck = MultiResBlock(
            previous, base_filters * 32, alpha=alpha, negative_slope=negative_slope
        )
        bottleneck_out = self.bottleneck.out_channels

        self.decoders = nn.ModuleList()
        prev_dec_ch = bottleneck_out

        for i, current in enumerate(reversed(channels)):
            if i == 0:
                skip_ch = encoder_out_channels[4]
            else:
                skip_ch = encoder_out_channels[4 - i]

            in_dec_ch = prev_dec_ch + skip_ch
            decoder = MultiResBlock(in_dec_ch, current, alpha=alpha, negative_slope=negative_slope)
            self.decoders.append(decoder)
            prev_dec_ch = decoder.out_channels

        self.output = nn.Conv2d(prev_dec_ch, out_channels, 1)
        self._parameter_count = self.count_parameters()

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"MultiResUNet expects (C, M, L) or (N, C, M, L), got {tuple(x.shape)}")
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(
                "MultiResUNet spatial dimensions must be divisible by 32; "
                f"received {(x.shape[-2], x.shape[-1])}. Pad the spectrogram before the model."
            )

        skips = []
        current = x
        for i, (encoder, downsample) in enumerate(zip(self.encoders, self.downsamples)):
            current = encoder(current)
            if i < 4:
                skips.append(self.res_paths[i](current))
            else:
                skips.append(current)
            current = downsample(current)

        current = self.bottleneck(current)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            current = nn.functional.interpolate(current, scale_factor=2, mode="nearest")
            if current.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError("MultiResUNet skip connection shapes differ after divisible-by-32 padding.")
            current = decoder(torch.cat((current, skip), dim=1))

        result = self.output(current)
        return result.squeeze(0) if was_unbatched else result
