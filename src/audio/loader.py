"""Audio loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
    duration_seconds: Optional[float] = None,
    peak_normalise: bool = True,
) -> torch.Tensor:
    """Load an audio file and preprocess it.

    Loads with torchaudio, converts to mono by channel mean if *mono* is True,
    resamples to *sample_rate* if needed, trims or zero-pads to
    *duration_seconds*, and peak-normalises so that ``max|x| == 0.9``.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate in Hz. Defaults to 16000.
        mono: If True, average all channels to produce a single-channel signal.
        duration_seconds: If given, output length is exactly
            ``round(duration_seconds * sample_rate)`` samples (trim or
            zero-pad).
        peak_normalise: If True, scale so that ``max|x| = 0.9``.

    Returns:
        Waveform tensor of shape ``(N,)`` on CPU, dtype ``torch.float32``.
    """
    waveform, sr = torchaudio.load(str(path))  # (C, T) float32

    # --- mono conversion ---
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

    # --- resample ---
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # --- flatten to 1-D ---
    audio: torch.Tensor = waveform.squeeze(0)  # (T,)

    # --- trim / pad ---
    if duration_seconds is not None:
        target_len = round(duration_seconds * sample_rate)
        if audio.shape[0] >= target_len:
            audio = audio[:target_len]
        else:
            pad_len = target_len - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, pad_len))

    # --- peak normalise ---
    if peak_normalise:
        peak = audio.abs().max()
        if peak > 0:
            audio = audio * (0.9 / peak)

    return audio


def make_synthetic_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a deterministic harmonic test signal.

    Creates a mixture of several musical notes, each with 3-5 overtones,
    a slow amplitude envelope, and a small amount of Gaussian noise.  The
    result is peak-normalised to ``max|x| = 0.9`` and is fully reproducible
    from *seed*.

    Args:
        duration_seconds: Length of the output signal in seconds.
        sample_rate: Sample rate in Hz.
        seed: Integer seed for reproducibility.

    Returns:
        Waveform tensor of shape ``(N,)`` where ``N = round(duration_seconds *
        sample_rate)``, dtype ``torch.float32``.
    """
    rng = np.random.default_rng(seed)

    N = round(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, N, endpoint=False)

    # --- define notes: (fundamental Hz, n_overtones, onset_s, offset_s) ---
    notes = [
        (261.63, 4, 0.0, 2.5),   # C4
        (329.63, 5, 0.5, 3.0),   # E4
        (392.00, 3, 1.0, 3.5),   # G4
        (523.25, 4, 1.5, 4.0),   # C5
        (440.00, 5, 2.0, 5.0),   # A4
    ]

    signal = np.zeros(N, dtype=np.float64)
    for f0, n_ot, onset, offset in notes:
        # amplitude envelope: linear attack/release within the note window
        env = np.zeros(N)
        i_on = int(onset * sample_rate)
        i_off = min(int(offset * sample_rate), N)
        dur = i_off - i_on
        if dur <= 0:
            continue
        attack = min(int(0.05 * sample_rate), dur // 4)
        release = min(int(0.1 * sample_rate), dur // 4)
        env[i_on:i_on + attack] = np.linspace(0, 1, attack)
        env[i_on + attack:i_off - release] = 1.0
        env[i_off - release:i_off] = np.linspace(1, 0, release)

        # harmonics with decaying amplitudes
        for k in range(1, n_ot + 1):
            amp = rng.uniform(0.3, 1.0) / k
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * env * np.sin(2 * np.pi * f0 * k * t + phase)

    # small Gaussian noise
    signal += rng.normal(0, 0.01, N)

    # peak normalise to 0.9
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal * (0.9 / peak)

    return torch.from_numpy(signal.astype(np.float32))
