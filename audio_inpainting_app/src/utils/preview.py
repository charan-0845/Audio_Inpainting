"""In-progress spectrogram preview (moved out of src/web/server.py so workers can use it)."""

from __future__ import annotations

import base64
import io
import logging
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.audio.stft import channels_to_stft, crop_to_shape
from src.utils.visualization import plot_spectrogram

logger = logging.getLogger("dpai.preview")


def render_preview(
    output_padded: torch.Tensor,
    original_shape: Tuple[int, int],
    stft_cfg: Dict[str, int],
    sample_rate: int,
    epoch: int,
    total_epochs: int,
) -> str:
    """Render the current model output as a base64 PNG data URL ('' on failure)."""
    try:
        t = output_padded.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        out_ch = crop_to_shape(t, original_shape)
        stft_out = channels_to_stft(out_ch)
        fig = plot_spectrogram(
            stft_out,
            sample_rate=sample_rate,
            hop_length=int(stft_cfg["hop_length"]),
            n_fft=int(stft_cfg["n_fft"]),
            title=f"Inpainting Progress – Epoch {epoch}/{total_epochs}",
        )
        fig.set_size_inches(6, 2.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to render preview: %s", exc)
        return ""
