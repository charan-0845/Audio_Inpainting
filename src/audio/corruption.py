"""Utilities for creating artificial missing portions in audio."""

from __future__ import annotations

from typing import Optional

import numpy as np


def create_mask(
    num_samples: int,
    sample_rate: int,
    min_gap_ms: float = 40.0,
    max_gap_ms: float = 80.0,
    cumulative_gap_ms: float = 200.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create a binary sample-level corruption mask.

    Returns a float32 array of length *num_samples* where ``1`` denotes an
    observed sample and ``0`` denotes a missing (corrupted) sample.

    Gap placement algorithm
    -----------------------
    1. Draw gap lengths uniformly from ``[min_gap_ms, max_gap_ms]``.
    2. Accumulate gaps until the cumulative target is met.
       Any leftover samples are merged into the last gap; if the last gap
       then becomes shorter than *min_gap_ms* the extra samples are still
       kept (documented exception).
    3. Gaps are placed uniformly at random inside the signal with at least
       one observed sample between consecutive gaps.
    4. The mask is fully reproducible given *seed*.

    Args:
        num_samples: Total number of samples in the signal.
        sample_rate: Sample rate in Hz (used to convert ms to samples).
        min_gap_ms: Minimum individual gap length in milliseconds.
        max_gap_ms: Maximum individual gap length in milliseconds.
        cumulative_gap_ms: Required total gap duration in milliseconds.
            The total missing samples will equal exactly
            ``round(cumulative_gap_ms * sample_rate / 1000)``.
        seed: Optional integer seed for reproducibility.

    Returns:
        Float32 ndarray of shape ``(num_samples,)`` with values in ``{0, 1}``.

    Raises:
        ValueError: If the cumulative gap cannot fit inside *num_samples* with
            the required spacing constraints.
    """
    rng = np.random.default_rng(seed)

    ms_to_samples = sample_rate / 1000.0
    min_gap = round(min_gap_ms * ms_to_samples)
    max_gap = round(max_gap_ms * ms_to_samples)
    total_missing = round(cumulative_gap_ms * ms_to_samples)

    if total_missing == 0:
        return np.ones(num_samples, dtype=np.float32)

    # --- build the list of gap lengths summing to total_missing ---
    gap_lengths: list[int] = []
    remaining = total_missing

    while remaining > 0:
        # how much can we still draw?
        if remaining <= max_gap:
            # last gap: clamp to remaining
            gap_lengths.append(remaining)
            remaining = 0
        else:
            g = int(rng.integers(min_gap, max_gap + 1))
            gap_lengths.append(g)
            remaining -= g

    # If the final gap is smaller than min_gap, merge it into the previous one
    # (documented exception: the merged gap may exceed max_gap).
    if len(gap_lengths) >= 2 and gap_lengths[-1] < min_gap:
        gap_lengths[-2] += gap_lengths[-1]
        gap_lengths.pop()

    n_gaps = len(gap_lengths)

    # --- place gaps randomly without overlap/touching ---
    # We need to fit n_gaps gaps with at least 1 observed sample between each
    # pair of consecutive gaps, and the gaps must be fully inside [0, num_samples).
    #
    # Minimum space required:
    #   sum(gap_lengths) + (n_gaps - 1) separators + 0 boundary margins
    # We allow gaps to start at sample 0 and end at sample num_samples.
    min_space = sum(gap_lengths) + max(n_gaps - 1, 0)
    if min_space > num_samples:
        raise ValueError(
            f"Cannot fit {n_gaps} gaps totalling {sum(gap_lengths)} samples "
            f"(+ {max(n_gaps-1,0)} separators) into {num_samples} samples."
        )

    # Use the "stars and bars" trick: distribute the remaining free samples
    # as margins around the gaps.
    #   free = num_samples - sum(gap_lengths) - (n_gaps - 1)
    # We need to split free into (n_gaps + 1) non-negative integers representing
    # leading margin, inter-gap spaces (>= 0 extra beyond the 1 forced), and
    # trailing margin.
    free = num_samples - sum(gap_lengths) - max(n_gaps - 1, 0)

    # Draw a uniform partition of `free` into n_gaps+1 non-negative parts.
    # Method: sort n_gaps random numbers in [0, free].
    cuts = np.sort(rng.integers(0, free + 1, size=n_gaps)).tolist()
    # margins[i] = extra free samples before gap i (beyond the forced separators)
    margins = []
    prev = 0
    for c in cuts:
        margins.append(c - prev)
        prev = c
    margins.append(free - prev)  # trailing margin

    # Build the mask
    mask = np.ones(num_samples, dtype=np.float32)
    cursor = 0
    for i, gl in enumerate(gap_lengths):
        cursor += margins[i]
        if i > 0:
            cursor += 1  # forced separator
        mask[cursor: cursor + gl] = 0.0
        cursor += gl

    return mask


def apply_mask(audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to an audio waveform.

    Args:
        audio: Audio array of shape ``(N,)``.
        mask: Binary mask of shape ``(N,)`` with 1 = observed, 0 = missing.

    Returns:
        Masked audio array of shape ``(N,)``.
    """
    return audio * mask
