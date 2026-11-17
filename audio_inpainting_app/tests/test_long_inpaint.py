"""Tests for long-audio inpainting (src/pipeline/long_inpaint.py).

Verifies:
1. Window planner:
   - Minimal window count on handmade gap lists
   - Context margin respected (>= context_s on each side when inside signal)
   - Clamping at both clip edges by shifting inward (never shrinking)
   - Clip shorter than or equal to window length (single window cover)
   - Dense group splitting when span exceeds window length
2. LPC-skip rule:
   - Short gap inside planned window is NOT LPC-filled (assigned to DPAI)
   - Isolated short gap outside window IS LPC-filled
3. Merge & bit-exactness:
   - Observed samples stay bit-exact to original corrupted input
   - Only gap samples change
   - Crossfade produces no NaNs or infs and smooth transitions
4. Equivalence:
   - For a clip <= window length with smoke config (5 epochs), inpaint_long output
     equals DPAIInpainter.inpaint output exactly (with preserve_observed=True or bit-exact observed).
5. Determinism:
   - Same seed -> identical results; different seeds -> different results.
6. Cancellation:
   - cancel_check triggers cleanly between windows or during DPAI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

import src.pipeline.long_inpaint as long_pipeline
from src.pipeline.inpainter import DPAIInpainter
from src.pipeline.long_inpaint import (
    LongInpaintResult,
    assign_gaps_to_best_window,
    classify_short_gaps,
    inpaint_long,
    mask_to_gaps,
    plan_windows,
    split_gaps,
)

REPO = Path(__file__).resolve().parent.parent


def _smoke_cfg() -> Dict[str, Any]:
    base = yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))
    base["optimization"]["epochs"] = 5
    base["model"]["channel_schedule"] = [8, 8, 8, 8, 8]
    base["model"]["res_path_lengths"] = [4, 3, 2, 1]
    base["model"]["use_harmonic"] = False
    base["pipeline"] = base.get("pipeline", {})
    base["pipeline"]["log_every"] = 5
    base["pipeline"]["preserve_observed"] = True
    base["pipeline"]["save_checkpoint"] = False
    return base


def _make_synth_clip(length_samples: int, sr: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, length_samples / sr, length_samples, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(length_samples)
    peak = max(1e-6, np.abs(sig).max())
    return (sig * 0.9 / peak).astype(np.float32)


# ===========================================================================
# 1. Window planner tests
# ===========================================================================

def test_mask_to_gaps():
    mask = np.ones(100, dtype=np.float32)
    mask[10:20] = 0.0
    mask[30:35] = 0.0
    mask[90:100] = 0.0

    gaps = mask_to_gaps(mask)
    assert gaps == [(10, 20), (30, 35), (90, 100)]
    assert mask_to_gaps(np.ones(50)) == []
    assert mask_to_gaps(np.zeros(20)) == [(0, 20)]


def test_split_gaps():
    # sr=16000 -> 40ms = 640 samples
    sr = 16000
    gaps = [(0, 300), (1000, 1640), (2000, 3000)]
    short_gaps, long_gaps = split_gaps(gaps, sample_rate=sr, lpc_max_gap_ms=40.0)
    assert short_gaps == [(0, 300), (1000, 1640)]
    assert long_gaps == [(2000, 3000)]


def test_plan_windows_clip_shorter_than_window():
    sr = 16000
    sig_len = 3 * sr  # 3 seconds (less than 5s window)
    long_gaps = [(1000, 2000), (5000, 7000)]
    windows = plan_windows(long_gaps, signal_length=sig_len, sample_rate=sr, window_s=5.0, context_s=0.5)
    assert len(windows) == 1
    w_start, w_end, covered = windows[0]
    assert w_start == 0
    assert w_end == sig_len
    assert covered == long_gaps


def test_plan_windows_minimal_count_and_context_margin():
    sr = 16000
    sig_len = 20 * sr  # 20 seconds
    window_s = 5.0
    context_s = 0.5
    win_len = int(window_s * sr)  # 80000
    ctx_len = int(context_s * sr)  # 8000

    # Two long gaps close to each other (at 8s and 9s): span is 2s + 2*0.5s = 3s <= 5s -> 1 window
    gaps = [(8 * sr, int(8.2 * sr)), (int(9.5 * sr), int(9.7 * sr))]
    windows = plan_windows(gaps, signal_length=sig_len, sample_rate=sr, window_s=window_s, context_s=context_s)
    assert len(windows) == 1
    w_start, w_end, covered = windows[0]
    assert covered == gaps
    assert w_end - w_start == win_len
    # Check context margin on left and right of group
    assert w_start <= gaps[0][0] - ctx_len
    assert w_end >= gaps[-1][1] + ctx_len


def test_plan_windows_clamping_left_and_right():
    sr = 16000
    sig_len = 20 * sr  # 20s = 320000
    window_s = 5.0
    win_len = int(window_s * sr)  # 80000

    # Gap near left edge (at 0.2s)
    left_gap = [(int(0.2 * sr), int(0.5 * sr))]
    win_left = plan_windows(left_gap, signal_length=sig_len, sample_rate=sr, window_s=window_s, context_s=0.5)
    assert len(win_left) == 1
    assert win_left[0][0] == 0
    assert win_left[0][1] == win_len

    # Gap near right edge (at 19.5s)
    right_gap = [(int(19.5 * sr), int(19.8 * sr))]
    win_right = plan_windows(right_gap, signal_length=sig_len, sample_rate=sr, window_s=window_s, context_s=0.5)
    assert len(win_right) == 1
    assert win_right[0][0] == sig_len - win_len
    assert win_right[0][1] == sig_len


def test_plan_windows_dense_group_splitting():
    sr = 16000
    sig_len = 20 * sr
    window_s = 5.0
    context_s = 0.5

    # Gap 1 at 2s, Gap 2 at 8s -> distance between them is 6s > window (5s) -> must split into 2 windows
    gaps = [(2 * sr, int(2.5 * sr)), (8 * sr, int(8.5 * sr))]
    windows = plan_windows(gaps, signal_length=sig_len, sample_rate=sr, window_s=window_s, context_s=context_s)
    assert len(windows) == 2
    assert windows[0][2] == [gaps[0]]
    assert windows[1][2] == [gaps[1]]


def test_window_planning_ignores_short_gaps_in_mask():
    sr = 16000
    signal_length = 20 * sr
    mask = np.ones(signal_length, dtype=np.float32)
    large_gap = (8 * sr, int(8.2 * sr))
    small_gap = (int(8.8 * sr), int(8.82 * sr))
    mask[large_gap[0]:large_gap[1]] = 0.0
    mask[small_gap[0]:small_gap[1]] = 0.0

    _, large_gaps = split_gaps(mask_to_gaps(mask), sr, lpc_max_gap_ms=40.0)
    windows_with_small = plan_windows(
        large_gaps, signal_length, sr, window_s=5.0, context_s=0.5
    )
    windows_without_small = plan_windows(
        [large_gap], signal_length, sr, window_s=5.0, context_s=0.5
    )

    assert windows_with_small == windows_without_small


# ===========================================================================
# 2. LPC-skip rule tests
# ===========================================================================

def test_lpc_skip_if_in_window():
    sr = 16000
    # Window spans sample 16000 to 96000 (1s to 6s)
    windows = [(16000, 96000, [])]

    # Short gap inside window (2.0s to 2.02s)
    short_in = (32000, 32320)
    # Short gap outside window (10.0s to 10.02s)
    short_out = (160000, 160320)

    lpc_gaps, dpai_gaps = classify_short_gaps([short_in, short_out], windows, lpc_skip_if_in_window=True)
    assert dpai_gaps == [short_in]
    assert lpc_gaps == [short_out]

    # If lpc_skip_if_in_window=False, all short gaps go to LPC
    lpc_gaps_all, dpai_gaps_none = classify_short_gaps([short_in, short_out], windows, lpc_skip_if_in_window=False)
    assert lpc_gaps_all == [short_in, short_out]
    assert dpai_gaps_none == []


def test_lpc_skip_inside_window_e2e():
    """End-to-end test that inpaint_long routes a short gap inside a window to DPAI, not LPC."""
    sr = 16000
    sig_len = 6 * sr  # 6 seconds
    orig = _make_synth_clip(sig_len, sr=sr, seed=123)

    mask = np.ones(sig_len, dtype=np.float32)
    # Long gap at 2.0s (100ms = 1600 samples)
    long_gap = (32000, 33600)
    # Short gap at 2.5s (20ms = 320 samples, inside the 5s window centered around 2s)
    short_gap_inside = (40000, 40320)
    # Short gap at 5.5s (20ms = 320 samples, outside window 0: window 0 spans [0, 80000])
    # Actually window 0 will span [0, 80000] (0 to 5s), so at 5.5s it's outside!
    short_gap_outside = (88000, 88320)

    mask[long_gap[0]:long_gap[1]] = 0.0
    mask[short_gap_inside[0]:short_gap_inside[1]] = 0.0
    mask[short_gap_outside[0]:short_gap_outside[1]] = 0.0

    corrupted = orig * mask
    cfg = _smoke_cfg()

    res = inpaint_long(corrupted, mask, cfg, epochs=1, seed=5)

    # Confirm short_gap_inside is handled by DPAI and NOT LPC
    assert short_gap_inside in res.dpai_gaps
    assert short_gap_inside not in res.lpc_gaps

    # Confirm short_gap_outside is handled by LPC and NOT DPAI
    assert short_gap_outside in res.lpc_gaps
    assert short_gap_outside not in res.dpai_gaps

    # Confirm long gap is handled by DPAI
    assert long_gap in res.dpai_gaps
    assert long_gap not in res.lpc_gaps


# ===========================================================================
# 3. No long gaps -> LPC-fill only test
# ===========================================================================

def test_no_long_gaps_lpc_only():
    sr = 16000
    sig_len = 2 * sr
    orig = _make_synth_clip(sig_len, sr=sr, seed=42)

    # All gaps <= 40 ms (640 samples)
    mask = np.ones(sig_len, dtype=np.float32)
    mask[1000:1300] = 0.0  # 300 samples (~18ms)
    mask[5000:5400] = 0.0  # 400 samples (~25ms)
    corrupted = orig * mask

    cfg = _smoke_cfg()
    res = inpaint_long(corrupted, mask, cfg, epochs=5, seed=0)

    assert isinstance(res, LongInpaintResult)
    assert len(res.windows) == 0
    assert len(res.loss_histories) == 0
    assert res.n_params == 0
    assert set(res.lpc_gaps) == {(1000, 1300), (5000, 5400)}
    assert res.dpai_gaps == []
    # Observed samples must be strictly identical
    obs_idx = np.where(mask == 1.0)[0]
    np.testing.assert_array_equal(res.waveform[obs_idx], corrupted[obs_idx])
    # Gaps must be filled (not all zeros)
    assert np.abs(res.waveform[1000:1300]).sum() > 0.0
    assert np.abs(res.waveform[5000:5400]).sum() > 0.0


def test_windows_run_before_lpc_and_keep_short_gaps_in_dpai_mask(monkeypatch):
    """DPAI sees short gaps inside windows; LPC receives only leftovers."""
    sr = 16000
    signal_length = 12 * sr
    original = _make_synth_clip(signal_length, sr=sr, seed=3)
    mask = np.ones(signal_length, dtype=np.float32)
    long_gap = (2 * sr, int(2.1 * sr))
    short_inside = (int(2.5 * sr), int(2.52 * sr))
    short_outside = (10 * sr, int(10.02 * sr))
    for start, end in (long_gap, short_inside, short_outside):
        mask[start:end] = 0.0
    corrupted = original * mask
    calls = []
    seen_masks = []

    class FakeInpainter:
        def __init__(self, cfg, device=None):
            pass

        def inpaint(self, corrupted, sample_mask, reference, epochs, seed, callbacks=None):
            calls.append(("dpai", seed))
            seen_masks.append(sample_mask.copy())
            waveform = corrupted.copy()
            waveform[sample_mask == 0.0] = 0.25
            return type("FakeResult", (), {
                "waveform": waveform,
                "loss_history": [0.0],
                "n_params": 1,
            })()

    def fake_lpc(signal, mask, gaps, seed):
        calls.append(("lpc", tuple(gaps)))
        output = signal.copy()
        for start, end in gaps:
            output[start:end] = 0.75
        return output

    monkeypatch.setattr(long_pipeline, "DPAIInpainter", FakeInpainter)
    monkeypatch.setattr(long_pipeline, "lpc_inpaint", fake_lpc)

    result = long_pipeline.inpaint_long(
        corrupted, mask, _smoke_cfg(), epochs=5, seed=11, device="cpu"
    )

    assert len(result.windows) == 1
    assert any(np.any(window_mask[int(2.5 * sr):int(2.52 * sr)] == 0.0)
               for window_mask in seen_masks)
    assert result.dpai_gaps == [long_gap, short_inside]
    assert result.lpc_gaps == [short_outside]
    assert [call[0] for call in calls] == ["dpai", "lpc"]
    assert np.all(np.isfinite(result.waveform))
    assert np.all(result.waveform[mask == 0.0] != 0.0)


def test_partially_window_covered_small_gap_leaves_only_boundary_for_lpc(monkeypatch):
    sr = 16000
    signal_length = 12 * sr
    original = _make_synth_clip(signal_length, sr=sr, seed=4)
    mask = np.ones(signal_length, dtype=np.float32)
    long_gap = (2 * sr, int(2.1 * sr))
    partial_small = (int(4.99 * sr), int(5.01 * sr))
    for start, end in (long_gap, partial_small):
        mask[start:end] = 0.0
    corrupted = original * mask
    lpc_calls = []

    class FakeInpainter:
        def __init__(self, cfg, device=None):
            pass

        def inpaint(self, corrupted, sample_mask, reference, epochs, seed, callbacks=None):
            waveform = corrupted.copy()
            waveform[sample_mask == 0.0] = 0.5
            return type("FakeResult", (), {
                "waveform": waveform,
                "loss_history": [0.0],
                "n_params": 1,
            })()

    def fake_lpc(signal, mask, gaps, seed):
        lpc_calls.extend(gaps)
        output = signal.copy()
        for start, end in gaps:
            output[start:end] = 0.75
        return output

    monkeypatch.setattr(long_pipeline, "DPAIInpainter", FakeInpainter)
    monkeypatch.setattr(long_pipeline, "lpc_inpaint", fake_lpc)

    result = long_pipeline.inpaint_long(
        corrupted, mask, _smoke_cfg(), epochs=5, seed=12, device="cpu"
    )

    assert lpc_calls == [(5 * sr, int(5.01 * sr))]
    assert np.all(np.isfinite(result.waveform))
    assert np.all(result.waveform[mask == 0.0] != 0.0)


# ===========================================================================
# 4. Merge & bit-exactness of observed samples
# ===========================================================================

def test_merge_observed_bit_exact():
    sr = 16000
    sig_len = 8 * sr  # 8 seconds
    orig = _make_synth_clip(sig_len, sr=sr, seed=123)

    mask = np.ones(sig_len, dtype=np.float32)
    # Long gap (100ms = 1600 samples) at 2s
    mask[32000:33600] = 0.0
    # Short gap (20ms = 320 samples) at 7s
    mask[112000:112320] = 0.0

    corrupted = orig * mask
    cfg = _smoke_cfg()

    res = inpaint_long(corrupted, mask, cfg, epochs=2, seed=7)

    # Observed samples must stay 100% bit-exact
    obs_idx = np.where(mask == 1.0)[0]
    np.testing.assert_array_equal(res.waveform[obs_idx], corrupted[obs_idx])

    # No NaN or Inf
    assert np.all(np.isfinite(res.waveform))

    # Gap regions were actually filled
    assert np.abs(res.waveform[32000:33600]).sum() > 0.0
    assert np.abs(res.waveform[112000:112320]).sum() > 0.0


# ===========================================================================
# 5. Determinism test
# ===========================================================================

def test_determinism():
    sr = 16000
    sig_len = 6 * sr
    orig = _make_synth_clip(sig_len, sr=sr, seed=99)
    mask = np.ones(sig_len, dtype=np.float32)
    mask[16000:18000] = 0.0
    corrupted = orig * mask

    cfg = _smoke_cfg()

    res1 = inpaint_long(corrupted, mask, cfg, epochs=2, seed=42)
    res2 = inpaint_long(corrupted, mask, cfg, epochs=2, seed=42)
    np.testing.assert_array_equal(res1.waveform, res2.waveform)

    # Different seed produces different reconstruction inside gap
    res3 = inpaint_long(corrupted, mask, cfg, epochs=2, seed=999)
    assert not np.array_equal(res1.waveform[16000:18000], res3.waveform[16000:18000])


# ===========================================================================
# 6. Cancellation test
# ===========================================================================

def test_cancellation():
    from src.web.server import JobCancelledException

    sr = 16000
    sig_len = 12 * sr
    orig = _make_synth_clip(sig_len, sr=sr, seed=1)
    mask = np.ones(sig_len, dtype=np.float32)
    # Two windows
    mask[2 * sr : int(2.5 * sr)] = 0.0
    mask[9 * sr : int(9.5 * sr)] = 0.0
    corrupted = orig * mask
    cfg = _smoke_cfg()

    # Cancel check triggers after window 0
    call_count = 0
    def cancel_after_win0():
        nonlocal call_count
        call_count += 1
        return call_count > 1

    with pytest.raises(JobCancelledException):
        inpaint_long(
            corrupted,
            mask,
            cfg,
            epochs=2,
            seed=0,
            cancel_check=cancel_after_win0,
        )


# ===========================================================================
# 7. Equivalence test: clip <= window_s matches DPAIInpainter.inpaint
# ===========================================================================

def test_single_window_equivalence_with_dpai():
    """For a clip <= window length, inpaint_long output equals DPAIInpainter.inpaint."""
    sr = 16000
    sig_len = 1 * sr  # 1 second <= 5.0s window length
    orig = _make_synth_clip(sig_len, sr=sr, seed=77)

    mask = np.ones(sig_len, dtype=np.float32)
    # Long gap (100ms = 1600 samples)
    mask[4000:5600] = 0.0
    corrupted = orig * mask

    cfg = _smoke_cfg()
    cfg["pipeline"]["preserve_observed"] = True

    # Call DPAIInpainter directly
    inpainter = DPAIInpainter(cfg, device="cpu")
    direct_res = inpainter.inpaint(corrupted, mask, epochs=5, seed=10)

    # Call inpaint_long with same seed & epochs
    long_res = inpaint_long(
        corrupted,
        mask,
        cfg,
        epochs=5,
        seed=10,
        device="cpu",
    )

    # Waveforms must match closely across the whole clip
    # Inside observed regions: strictly identical
    obs_idx = np.where(mask == 1.0)[0]
    np.testing.assert_array_equal(long_res.waveform[obs_idx], direct_res.waveform[obs_idx])

    # Waveform across the entire clip must be bit-exact identical to direct DPAI
    np.testing.assert_array_equal(long_res.waveform, direct_res.waveform)
    max_diff = np.max(np.abs(long_res.waveform - direct_res.waveform))
    assert max_diff == 0.0, f"Expected exact equality (0.0 diff), got {max_diff}"

