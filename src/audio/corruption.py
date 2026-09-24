"""Utilities for creating artificial missing portions in audio."""


def create_mask(num_samples, sample_rate, min_gap_ms=40, max_gap_ms=80,
                cumulative_gap_ms=200, seed=None):
    """Create a binary sample-level mask.

    1 = observed sample
    0 = missing sample

    TODO: implement random gap generation.
    """
    raise NotImplementedError


def apply_mask(audio, mask):
    """Apply a binary mask to an audio waveform."""
    return audio * mask
