"""Harmonic 2D convolution layer with harmonic lowering."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


class HarmonicConv2d(nn.Module):
    """Harmonic 2D convolution layer.

    Reinterprets the frequency dimension of the kernel as weights on a harmonic series
    at each target frequency. Uses Harmonic Lowering (im2col-style gather) to restructure
    the computation into a 4D convolution over frequency harmonic taps and time.

    Out-of-bounds frequency indices exceeding [0, M-1] are clamped to valid range [0, M-1].

    Worked Numeric Example:
        For M = 8 frequency bins, kernel_size = (3, 3) (Km = 3), anchor = 1:
        - Output bin m = 0 (1-based bin 1):
          taps k in {1, 2, 3} -> 1-based source bins round(k * 1 / 1) in {1, 2, 3} -> 0-based [0, 1, 2].
        - Output bin m = 1 (1-based bin 2):
          taps k in {1, 2, 3} -> 1-based source bins round(k * 2 / 1) in {2, 4, 6} -> 0-based [1, 3, 5].
        - Output bin m = 3 (1-based bin 4):
          taps k in {1, 2, 3} -> 1-based source bins round(k * 4 / 1) in {4, 8, 12} -> 0-based [3, 7, 11]
          which clamps to valid range [0, 7] as [3, 7, 7].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        anchor: int = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = "same",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.Km, self.Kl = kernel_size
        self.anchor = max(1, int(anchor))

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.Km, self.Kl))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.register_buffer("_cached_gather_idx", None, persistent=False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _get_gather_indices(self, M: int, device: torch.device) -> torch.Tensor:
        stride_m = self.stride[0]
        m_idx = torch.arange(0, M, stride_m, device=device).unsqueeze(0)  # (1, M_out)
        M_out = m_idx.shape[1]

        if (
            self._cached_gather_idx is not None
            and self._cached_gather_idx.shape[1] == M_out
            and self._cached_gather_idx.device == device
        ):
            return self._cached_gather_idx

        # k_idx for harmonic taps k = 1..Km
        k_idx = torch.arange(1, self.Km + 1, device=device).unsqueeze(1)  # (Km, 1)

        # 1-based target frequency bin is (m_idx + 1)
        raw_source_1based = torch.round((k_idx * (m_idx + 1)) / float(self.anchor))
        raw_source_0based = (raw_source_1based - 1).to(torch.long)

        # Clamping out-of-bounds frequency indices to valid range [0, M - 1]
        clamped_idx = torch.clamp(raw_source_0based, 0, M - 1)

        self._cached_gather_idx = clamped_idx
        return clamped_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"HarmonicConv2d expects 3D or 4D tensor, got {tuple(x.shape)}")

        B, C, M, L = x.shape
        if C != self.in_channels:
            raise ValueError(f"HarmonicConv2d expected input channel {self.in_channels}, got {C}")

        # Time-dimension padding
        if self.padding == "same":
            pad_l = self.Kl // 2
            pad_r = self.Kl - 1 - pad_l
            x_padded = F.pad(x, (pad_l, pad_r, 0, 0))
        elif isinstance(self.padding, int):
            x_padded = F.pad(x, (self.padding, self.padding, 0, 0))
        elif isinstance(self.padding, tuple):
            x_padded = F.pad(x, (self.padding[1], self.padding[0], 0, 0))
        else:
            x_padded = x

        gather_idx = self._get_gather_indices(M, x.device)  # (Km, M_out)
        M_out = gather_idx.shape[1]

        # Vectorized gather over frequency axis
        flat_idx = gather_idx.reshape(-1)  # (Km * M_out)
        lowered = x_padded[:, :, flat_idx, :]  # (B, C, Km * M_out, L_padded)
        lowered = lowered.view(B, C, self.Km, M_out, -1)

        lowered_4d = lowered.permute(0, 1, 2, 3, 4).reshape(B, C * self.Km, M_out, -1)
        weight_4d = self.weight.reshape(self.out_channels, C * self.Km, 1, self.Kl)

        out = F.conv2d(lowered_4d, weight_4d, stride=(1, self.stride[1]))

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out.squeeze(0) if was_unbatched else out
