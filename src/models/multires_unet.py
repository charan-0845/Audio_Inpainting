"""MultiResUNet architecture."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from src.models.harmonic_conv import HarmonicConv2d
from src.models.multires_block import MultiResBlock
from src.models.res_path import ResPath


class MultiResUNet(nn.Module):
    """Five-level MultiResUNet with Res Path skip connections and nearest-neighbour upsampling.

    Replaces standard double convolutions with MultiResBlocks and processes skip connections
    with ResPaths of decreasing length (4, 3, 2, 1).
    GroupNorm is used instead of BatchNorm to match PlainUNet's deep-prior single-spectrogram setup.
    Supports optional HarmonicConv2d in the encoder when use_harmonic=True.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_filters: int = 7,
        alpha: float = 1.6,
        negative_slope: float = 0.01,
        use_harmonic: bool = False,
        harmonic_anchors: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self._depth = 5
        self.use_harmonic = use_harmonic
        self.harmonic_anchors = harmonic_anchors or {"default": 1, "res_path_residual": 1, "downsampling_conv1": 1}

        channels = [base_filters * (2**i) for i in range(5)]

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.res_paths = nn.ModuleList()

        previous = in_channels
        encoder_out_channels = []
        res_lengths = [4, 3, 2, 1]

        default_anchor = self.harmonic_anchors.get("default", 1)

        for i, current in enumerate(channels):
            encoder = MultiResBlock(
                previous,
                current,
                alpha=alpha,
                negative_slope=negative_slope,
                use_harmonic=use_harmonic,
                anchor=default_anchor,
            )
            self.encoders.append(encoder)
            out_ch = encoder.out_channels
            encoder_out_channels.append(out_ch)

            next_ch = out_ch * 2
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

            if i < 4:
                rp_anchor = self.harmonic_anchors.get("res_path_residual", default_anchor)
                self.res_paths.append(
                    ResPath(
                        out_ch,
                        length=res_lengths[i],
                        negative_slope=negative_slope,
                        use_harmonic=use_harmonic,
                        anchor=rp_anchor,
                    )
                )

        self.bottleneck = MultiResBlock(
            previous,
            base_filters * 32,
            alpha=alpha,
            negative_slope=negative_slope,
            use_harmonic=use_harmonic,
            anchor=default_anchor,
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
            # Decoder convolutions remain regular nn.Conv2d (use_harmonic=False)
            decoder = MultiResBlock(
                in_dec_ch,
                current,
                alpha=alpha,
                negative_slope=negative_slope,
                use_harmonic=False,
            )
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
