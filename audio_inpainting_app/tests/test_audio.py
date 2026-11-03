"""Tests for audio loading / corruption mask utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.loader import make_synthetic_audio
from src.audio.corruption import create_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION = 5.0
NUM_SAMPLES = round(DURATION * SAMPLE_RATE)  # 80000

MIN_GAP_MS = 40.0
MAX_GAP_MS = 80.0


def _ms_to_samples(ms: float, sr: int = SAMPLE_RATE) -> int:
    return round(ms * sr / 1000.0)


def _gap_segments(mask: np.ndarray):
    """Return list of (start, length) for each run of zeros."""
    gaps = []
    in_gap = False
    start = 0
    for i, v in enumerate(mask):
        if v == 0.0 and not in_gap:
            start = i
            in_gap = True
        elif v != 0.0 and in_gap:
            gaps.append((start, i - start))
            in_gap = False
    if in_gap:
        gaps.append((start, len(mask) - start))
    return gaps


# ---------------------------------------------------------------------------
# create_mask tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cum_gap_ms", [200, 400, 600, 800, 1000])
def test_cumulative_gap_exact(cum_gap_ms):
    """Total missing samples must equal exactly round(cum_gap_ms * sr / 1000)."""
    mask = create_mask(
        NUM_SAMPLES, SAMPLE_RATE,
        min_gap_ms=MIN_GAP_MS, max_gap_ms=MAX_GAP_MS,
        cumulative_gap_ms=cum_gap_ms, seed=42,
    )
    total_missing = int((mask == 0).sum())
    expected = _ms_to_samples(cum_gap_ms)
    assert total_missing == expected, (
        f"Expected {expected} missing samples for {cum_gap_ms} ms, "
        f"got {total_missing}"
    )


@pytest.mark.parametrize("cum_gap_ms", [200, 400, 600])
def test_gap_lengths_within_bounds(cum_gap_ms):
    """Every individual gap must be in [min_gap, max_gap] (documented exception allowed)."""
    mask = create_mask(
        NUM_SAMPLES, SAMPLE_RATE,
        min_gap_ms=MIN_GAP_MS, max_gap_ms=MAX_GAP_MS,
        cumulative_gap_ms=cum_gap_ms, seed=7,
    )
    gaps = _gap_segments(mask)
    min_s = _ms_to_samples(MIN_GAP_MS)
    max_s = _ms_to_samples(MAX_GAP_MS)

    # At most ONE gap (the merged last one) is allowed to exceed max_s.
    violations_above = [g for (_, g) in gaps if g > max_s]
    assert len(violations_above) <= 1, (
        f"More than one gap exceeds max_gap_ms: {violations_above}"
    )
    # All gaps must be >= 1 (trivially true) and at most one may be < min_s
    violations_below = [g for (_, g) in gaps if g < min_s]
    assert len(violations_below) <= 1, (
        f"More than one gap is below min_gap_ms: {violations_below}"
    )


def test_no_overlapping_or_touching_gaps():
    """Gaps must not overlap or touch (at least 1 observed sample between them)."""
    mask = create_mask(
        NUM_SAMPLES, SAMPLE_RATE,
        min_gap_ms=MIN_GAP_MS, max_gap_ms=MAX_GAP_MS,
        cumulative_gap_ms=400, seed=99,
    )
    gaps = _gap_segments(mask)
    for i in range(len(gaps) - 1):
        start_a, len_a = gaps[i]
        start_b, _ = gaps[i + 1]
        end_a = start_a + len_a  # exclusive
        assert start_b > end_a, (
            f"Gap {i} ends at {end_a}, gap {i+1} starts at {start_b}: touching or overlapping"
        )


def test_gaps_inside_signal():
    """All gaps must be fully within [0, num_samples)."""
    mask = create_mask(
        NUM_SAMPLES, SAMPLE_RATE,
        min_gap_ms=MIN_GAP_MS, max_gap_ms=MAX_GAP_MS,
        cumulative_gap_ms=400, seed=1,
    )
    gaps = _gap_segments(mask)
    for start, length in gaps:
        assert start >= 0
        assert start + length <= NUM_SAMPLES


def test_same_seed_same_mask():
    """Same seed must produce identical masks."""
    m1 = create_mask(NUM_SAMPLES, SAMPLE_RATE, cumulative_gap_ms=400, seed=123)
    m2 = create_mask(NUM_SAMPLES, SAMPLE_RATE, cumulative_gap_ms=400, seed=123)
    np.testing.assert_array_equal(m1, m2)


def test_different_seeds_different_masks():
    """Different seeds should produce different masks."""
    m1 = create_mask(NUM_SAMPLES, SAMPLE_RATE, cumulative_gap_ms=400, seed=0)
    m2 = create_mask(NUM_SAMPLES, SAMPLE_RATE, cumulative_gap_ms=400, seed=1)
    assert not np.array_equal(m1, m2)


# ---------------------------------------------------------------------------
# extract_mask_from_audio tests
# ---------------------------------------------------------------------------

from src.audio.corruption import extract_mask_from_audio
import torch


def test_extract_mask_perfect_reconstruction():
    """Applying mask and then extracting mask should reconstruct the original mask."""
    audio = make_synthetic_audio(duration_seconds=1.0, sample_rate=16000, seed=42)
    orig_mask = create_mask(len(audio), 16000, cumulative_gap_ms=100.0, seed=42)
    corrupted_audio = audio * torch.from_numpy(orig_mask)

    extracted_mask_torch = extract_mask_from_audio(corrupted_audio, threshold=1e-6, min_gap_samples=5)
    extracted_mask_np = extracted_mask_torch.numpy()

    np.testing.assert_array_equal(extracted_mask_np, orig_mask)


def test_extract_mask_isolated_zero_crossings():
    """Isolated zero crossings (< min_gap_samples) should NOT be flagged as missing gaps."""
    # Create sine wave audio
    t = np.linspace(0, 1, 16000)
    sine = np.sin(2 * np.pi * 10 * t).astype(np.float32)  # 10 Hz sine, crosses 0 at exact samples

    # Extract mask with min_gap_samples=5
    extracted = extract_mask_from_audio(sine, threshold=1e-5, min_gap_samples=5)

    # Since there are no contiguous zero blocks >= 5 samples, all samples should be 1.0
    assert (extracted == 1.0).all()

