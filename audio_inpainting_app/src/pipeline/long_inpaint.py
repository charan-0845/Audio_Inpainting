"""Long-audio inpainting pipeline (windowed DPAI + LPC leftovers).

Extends DPAIInpainter to arbitrary-length audio without modifying the core ML model.
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.audio.lpc_baseline import lpc_inpaint
from src.pipeline.inpainter import DPAIInpainter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LongInpaintResult:
    """Structured result returned by inpaint_long.

    Attributes:
        waveform: Full reconstructed waveform (N,) as float32 ndarray.
        loss_histories: Per-window loss histories (one list of float per window).
        windows: List of window dicts with keys:
            start (int), end (int), n_gaps (int), start_s (float), end_s (float).
        lpc_gaps: Gaps handled by LPC (start, end) in sample indices.
        dpai_gaps: Gaps handled by DPAI (start, end) in sample indices.
        n_params: Model parameter count (from DPAI model, or 0 if LPC only).
        runtime_s: Total wall-clock runtime in seconds.
        seed: Base seed provided.
        epochs: Number of epochs per window.
    """
    waveform: np.ndarray
    loss_histories: List[List[float]]
    windows: List[Dict[str, Any]]
    lpc_gaps: List[Tuple[int, int]]
    dpai_gaps: List[Tuple[int, int]]
    n_params: int
    runtime_s: float
    seed: int
    epochs: int
    context_infeasible_gaps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_processing_s: float = 0.0


@dataclass(frozen=True)
class AdaptiveWindow:
    """A mask-aware processing window and the DPAI gaps assigned to it."""
    start: int
    end: int
    gaps: List[Tuple[int, int]]
    individually_planned: bool
    left_context_available: bool
    right_context_available: bool


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def mask_to_gaps(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a sample-level binary mask (1=observed, 0=missing) to sorted (start, end) sample intervals.

    Args:
        mask: 1D array of shape (N,) with 1 = observed, 0 = missing.

    Returns:
        Sorted list of non-overlapping (start, end) tuples where mask is 0.
    """
    arr = np.asarray(mask).reshape(-1)
    if len(arr) == 0:
        return []
    z = (arr == 0).astype(int)
    d = np.diff(np.concatenate([[0], z, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def split_gaps(
    gaps: List[Tuple[int, int]],
    sample_rate: int,
    lpc_max_gap_ms: float = 40.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split gaps into short (<= lpc_max_gap_ms) and long (> lpc_max_gap_ms).

    Args:
        gaps: List of (start, end) sample intervals.
        sample_rate: Sample rate in Hz.
        lpc_max_gap_ms: Max gap length in ms to be considered short.

    Returns:
        (short_gaps, long_gaps)
    """
    ms_to_samples = sample_rate / 1000.0
    max_short_samples = round(lpc_max_gap_ms * ms_to_samples)

    short_gaps: List[Tuple[int, int]] = []
    long_gaps: List[Tuple[int, int]] = []

    for s, e in gaps:
        length = e - s
        if length <= max_short_samples:
            short_gaps.append((s, e))
        else:
            long_gaps.append((s, e))

    return short_gaps, long_gaps


def plan_windows(
    long_gaps: List[Tuple[int, int]],
    signal_length: int,
    sample_rate: int,
    window_s: float = 5.0,
    context_s: float = 0.25,
    mask: Optional[np.ndarray] = None,
    min_window_s: float = 1.0,
    max_window_s: float = 10.0,
    merge_strategy: str = "adaptive",
) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    """Plan adaptive windows, preserving the legacy tuple API.

    When ``mask`` is supplied, planning verifies original clean context and
    reports infeasible gaps through ``plan_adaptive_windows``.  The mask-less
    path remains for backwards-compatible callers and legacy tests.
    """
    if len(long_gaps) == 0:
        return []

    if mask is not None:
        windows, _ = plan_adaptive_windows(
            long_gaps=long_gaps,
            mask=mask,
            signal_length=signal_length,
            sample_rate=sample_rate,
            preferred_window_s=window_s,
            minimum_window_s=min_window_s,
            maximum_window_s=max_window_s,
            minimum_clean_context_s=context_s,
            merge_strategy=merge_strategy,
        )
        return [(w.start, w.end, w.gaps) for w in windows]

    # Legacy mode for callers that do not have a mask.  This never pretends
    # that an unknown context interval is clean.
    window_samples = round(window_s * sample_rate)
    context_samples = round(context_s * sample_rate)

    if signal_length <= window_samples:
        return [(0, signal_length, list(long_gaps))]

    sorted_gaps = sorted(long_gaps, key=lambda g: g[0])
    groups: List[List[Tuple[int, int]]] = []
    current_group: List[Tuple[int, int]] = [sorted_gaps[0]]

    for gap in sorted_gaps[1:]:
        group_first_start = current_group[0][0]
        group_last_end = gap[1]
        span_with_context = (group_last_end - group_first_start) + 2 * context_samples

        if span_with_context <= window_samples:
            current_group.append(gap)
        else:
            groups.append(current_group)
            current_group = [gap]

    if current_group:
        groups.append(current_group)

    windows: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for grp in groups:
        first_start = grp[0][0]
        last_end = grp[-1][1]
        mid = (first_start + last_end) / 2.0
        ideal_start = int(round(mid - window_samples / 2.0))

        # Clamp at clip edges by shifting inward (never shrink)
        w_start = max(0, min(ideal_start, signal_length - window_samples))
        w_end = w_start + window_samples
        windows.append((w_start, w_end, grp))

    return windows


def gap_has_clean_context(
    gap: Tuple[int, int],
    mask: np.ndarray,
    context_samples: int,
) -> bool:
    """Return whether a gap has C original observed samples on both sides."""
    s, e = gap
    n = len(mask)
    if s < context_samples or e + context_samples > n:
        return False
    left = np.asarray(mask[s - context_samples:s], dtype=np.float32)
    right = np.asarray(mask[e:e + context_samples], dtype=np.float32)
    return bool(len(left) == context_samples and len(right) == context_samples
                and np.all(left == 1.0) and np.all(right == 1.0))


def gap_metadata(
    gap: Tuple[int, int],
    sample_rate: int,
) -> Dict[str, Any]:
    """Serialize a gap for the web and JSON metadata formats."""
    s, e = int(gap[0]), int(gap[1])
    return {
        "start": s,
        "end": e,
        "start_sample": s,
        "end_sample": e,
        "start_s": round(float(s) / sample_rate, 4),
        "end_s": round(float(e) / sample_rate, 4),
        "duration_ms": round(float(e - s) * 1000.0 / sample_rate, 4),
    }


def plan_adaptive_windows(
    long_gaps: List[Tuple[int, int]],
    mask: np.ndarray,
    signal_length: int,
    sample_rate: int,
    preferred_window_s: float = 5.0,
    minimum_window_s: float = 1.0,
    maximum_window_s: float = 10.0,
    minimum_clean_context_s: float = 0.25,
    merge_strategy: str = "adaptive",
) -> Tuple[List[AdaptiveWindow], List[Dict[str, Any]]]:
    """Generate minimal, context-valid windows around DPAI gaps."""
    if not long_gaps:
        return [], []

    arr_mask = np.asarray(mask).reshape(-1)
    if len(arr_mask) != signal_length:
        raise ValueError("sample mask length must equal signal length")

    context_samples = max(1, round(minimum_clean_context_s * sample_rate))
    preferred_samples = round(preferred_window_s * sample_rate)
    minimum_samples = round(minimum_window_s * sample_rate)
    maximum_samples = max(1, round(maximum_window_s * sample_rate))

    sorted_gaps = sorted(set((int(s), int(e)) for s, e in long_gaps if e > s))
    feasible: List[Tuple[int, int]] = []
    infeasible: List[Dict[str, Any]] = []

    for gap in sorted_gaps:
        s, e = gap
        if s < 0 or e > signal_length or s >= e:
            infeasible.append({
                "start": s, "end": e, "reason": "out_of_bounds",
                "left_context_available": False, "right_context_available": False,
            })
            continue

        left_available = s >= context_samples and np.all(arr_mask[s - context_samples:s] == 1.0)
        right_available = e + context_samples <= signal_length and np.all(arr_mask[e:e + context_samples] == 1.0)
        if not (left_available and right_available):
            infeasible.append({
                "start": s, "end": e, "reason": "insufficient_clean_context",
                "left_context_available": bool(left_available),
                "right_context_available": bool(right_available),
            })
            continue

        minimum_span = (e + context_samples) - (s - context_samples)
        if minimum_span > maximum_samples:
            infeasible.append({
                "start": s, "end": e, "reason": "minimum_window_exceeds_maximum",
                "left_context_available": True, "right_context_available": True,
            })
            continue
        feasible.append(gap)

    if not feasible:
        return [], infeasible

    groups: List[List[Tuple[int, int]]] = []
    current = [feasible[0]]

    for gap in feasible[1:]:
        first_s = current[0][0]
        last_e = gap[1]
        combined_start = max(0, first_s - context_samples)
        combined_end = min(signal_length, last_e + context_samples)
        combined_length = combined_end - combined_start

        can_merge = (
            merge_strategy == "adaptive"
            and combined_length <= preferred_samples
            and combined_length <= maximum_samples
        )
        if can_merge:
            current.append(gap)
        else:
            groups.append(current)
            current = [gap]
    if current:
        groups.append(current)

    windows: List[AdaptiveWindow] = []
    for group in groups:
        minimum_start = max(0, group[0][0] - context_samples)
        minimum_end = min(signal_length, group[-1][1] + context_samples)
        minimum_span = minimum_end - minimum_start
        target_length = max(preferred_samples, minimum_samples, minimum_span)
        target_length = min(target_length, maximum_samples)
        target_length = min(target_length, signal_length)

        center = (minimum_start + minimum_end) / 2.0
        start = int(round(center - target_length / 2.0))
        start = max(0, min(start, signal_length - target_length))
        end = start + target_length
        start = min(start, minimum_start)
        end = max(end, minimum_end)
        if end - start > maximum_samples:
            start = minimum_start
            end = minimum_end
        start = max(0, start)
        end = min(signal_length, end)

        left_available = bool(group[0][0] >= context_samples
                             and np.all(arr_mask[group[0][0] - context_samples:group[0][0]] == 1.0))
        right_available = bool(group[-1][1] + context_samples <= signal_length
                              and np.all(arr_mask[group[-1][1]:group[-1][1] + context_samples] == 1.0))
        windows.append(AdaptiveWindow(
            start=start,
            end=end,
            gaps=list(group),
            individually_planned=len(group) == 1,
            left_context_available=left_available,
            right_context_available=right_available,
        ))

    return windows, infeasible


def classify_short_gaps(
    short_gaps: List[Tuple[int, int]],
    windows: List[Tuple[int, int, Any]],
    lpc_skip_if_in_window: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Classify short gaps into LPC-filled vs DPAI-filled.

    If lpc_skip_if_in_window is True:
        Short gaps that fall fully inside any planned window (w_start <= s and e <= w_end)
        are NOT LPC-filled; DPAI will fill them.
        Remaining short gaps are LPC-filled.
    Otherwise:
        All short gaps are LPC-filled.

    Returns:
        (lpc_short_gaps, dpai_short_gaps)
    """
    if not lpc_skip_if_in_window or not windows:
        return list(short_gaps), []

    lpc_short_gaps: List[Tuple[int, int]] = []
    dpai_short_gaps: List[Tuple[int, int]] = []

    for s, e in short_gaps:
        inside_window = False
        for w_start, w_end, _ in windows:
            if w_start <= s and e <= w_end:
                inside_window = True
                break
        if inside_window:
            dpai_short_gaps.append((s, e))
        else:
            lpc_short_gaps.append((s, e))

    return lpc_short_gaps, dpai_short_gaps


def assign_gaps_to_best_window(
    gaps: List[Tuple[int, int]],
    windows: List[Tuple[int, int, Any]],
) -> Dict[int, List[Tuple[int, int]]]:
    """Assign each DPAI gap to the window where it is most central.

    Returns:
        dict mapping window_index -> list of gaps handled by that window.
    """
    assignment: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(len(windows))}

    for s, e in gaps:
        gap_mid = (s + e) / 2.0
        best_win_idx = -1
        best_dist = float("inf")

        for w_idx, (w_start, w_end, _) in enumerate(windows):
            if w_start <= s and e <= w_end:
                w_mid = (w_start + w_end) / 2.0
                dist = abs(gap_mid - w_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_win_idx = w_idx

        if best_win_idx != -1:
            assignment[best_win_idx].append((s, e))

    return assignment


def paste_with_crossfade(
    target_waveform: np.ndarray,
    dpai_window_recon: np.ndarray,
    window_start: int,
    gaps: List[Tuple[int, int]],
    crossfade_samples: int,
) -> None:
    """Paste reconstructed samples into target_waveform strictly inside gap regions with equal-power crossfade.

    Observed samples outside gap regions stay bit-exact.

    At gap boundaries:
    - Left boundary (start of gap): crossfade from target_waveform[s : s + crossfade_samples]
      to dpai[s : s + crossfade_samples].
    - Right boundary (end of gap): crossfade from dpai[e - crossfade_samples : e]
      to target_waveform[e - crossfade_samples : e].
    - If gap length <= 2 * crossfade_samples: adjust crossfade length to half the gap.

    Crossfade uses equal-power weights: w_in = sin(theta), w_out = cos(theta).
    """
    for s, e in gaps:
        gap_len = e - s
        if gap_len <= 0:
            continue

        local_s = s - window_start
        local_e = e - window_start
        dpai_slice = dpai_window_recon[local_s:local_e].astype(np.float32)

        # Base paste inside the gap
        cf = min(crossfade_samples, gap_len // 2)

        if cf > 0:
            theta = np.linspace(0.0, np.pi / 2.0, cf, dtype=np.float32)
            w_sin = np.sin(theta)  # 0 -> 1
            w_cos = np.cos(theta)  # 1 -> 0

            # Left edge: fade from target (existing stitched audio) to DPAI if target has audio
            left_orig = target_waveform[s:s + cf]
            left_dpai = dpai_slice[:cf]
            if np.any(left_orig != 0):
                dpai_slice[:cf] = w_cos * left_orig + w_sin * left_dpai

            # Right edge: fade from DPAI to target (existing stitched audio) if target has audio
            right_dpai = dpai_slice[-cf:]
            right_orig = target_waveform[e - cf:e]
            if np.any(right_orig != 0):
                dpai_slice[-cf:] = w_cos * right_dpai + w_sin * right_orig

        target_waveform[s:e] = dpai_slice


# ---------------------------------------------------------------------------
# Main inpaint_long function
# ---------------------------------------------------------------------------

def inpaint_long(
    corrupted: np.ndarray,
    sample_mask: np.ndarray,
    cfg: Dict[str, Any],
    reference: Optional[np.ndarray] = None,
    epochs: Optional[int] = None,
    seed: int = 0,
    callbacks: Optional[Callable[[str, int, int, int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    device: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    executor: Optional[Any] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> LongInpaintResult:
    """Perform long-audio inpainting using windowed DPAI, LPC leftovers, and equal-power merging.

    Args:
        corrupted: Corrupted audio waveform (N,) as float32 ndarray.
        sample_mask: Binary sample mask (N,) with 1=observed, 0=missing.
        cfg: Fully-resolved DPAI config dict.
        reference: Optional clean reference waveform (N,) for diagnostic.
        epochs: Number of optimization epochs per DPAI window.
        seed: Base random seed.
        callbacks: Optional callback: callbacks(stage, window_index, n_windows, epoch, epochs)
        cancel_check: Optional callable returning True if job has been cancelled.
        device: Torch device string or None (auto-device).
        options: Optional dict overriding long_audio configuration.

    Returns:
        LongInpaintResult containing full reconstructed waveform and metadata.
    """
    t0 = time.perf_counter()
    N = len(corrupted)
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))

    # Resolve options from cfg["long_audio"] and overrides
    long_cfg = dict(cfg.get("long_audio", {}))
    if options:
        long_cfg.update(options)

    window_s = float(long_cfg.get("window_s", 5.0))
    lpc_max_gap_ms = float(long_cfg.get("lpc_max_gap_ms", 40.0))
    crossfade_ms = float(long_cfg.get("crossfade_ms", 8.0))
    minimum_window_s = float(long_cfg.get("min_window_s", 1.0))
    maximum_window_s = float(long_cfg.get("max_window_s", 10.0))
    minimum_clean_context_s = float(long_cfg.get("context_s", 0.25))
    merge_strategy = str(long_cfg.get("merge_strategy", "adaptive"))

    crossfade_samples = max(1, round(crossfade_ms * sample_rate / 1000.0))

    if epochs is None:
        epochs = int(cfg.get("optimization", {}).get("epochs", 100))

    # Guard 1: reject if total missing > 40% (existing project rule)
    total_missing = int((sample_mask == 0.0).sum())
    missing_ratio = total_missing / max(1, N)
    if missing_ratio >= 0.40:
        raise ValueError(
            f"Total missing audio ({missing_ratio * 100:.1f}%) exceeds the 40% limit."
        )

    # 1. mask_to_gaps
    all_gaps = mask_to_gaps(sample_mask)
    if not all_gaps:
        # Nothing missing
        return LongInpaintResult(
            waveform=np.asarray(corrupted, dtype=np.float32).copy(),
            loss_histories=[],
            windows=[],
            lpc_gaps=[],
            dpai_gaps=[],
            n_params=0,
            runtime_s=time.perf_counter() - t0,
            seed=seed,
            epochs=epochs,
        )

    # 2. Split short vs long gaps
    short_gaps, long_gaps = split_gaps(all_gaps, sample_rate, lpc_max_gap_ms=lpc_max_gap_ms)

    # 3. If there are no long gaps: LPC-fill everything and return (no DPAI)
    if not long_gaps:
        logger.info("Long-audio windows stage: 0 windows planned from 0 large gaps.")
        logger.info("Long-audio DPAI stage skipped: 0 small gaps included inside windows.")
        logger.info("Long-audio LPC stage: %d leftover gaps sent to LPC.", len(short_gaps))
        if callbacks:
            callbacks("lpc", 0, 0, 0, 0)
        filled = lpc_inpaint(
            corrupted,
            mask=sample_mask,
            gaps=all_gaps,
            seed=seed,
        )
        # Ensure observed samples stay bit-exact
        waveform = corrupted * sample_mask + filled * (1.0 - sample_mask)
        return LongInpaintResult(
            waveform=waveform.astype(np.float32),
            loss_histories=[],
            windows=[],
            lpc_gaps=list(all_gaps),
            dpai_gaps=[],
            n_params=0,
            runtime_s=time.perf_counter() - t0,
            seed=seed,
            epochs=epochs,
        )

    # 4. Plan mask-aware adaptive windows from the DPAI gap set.
    adaptive_windows, context_infeasible = plan_adaptive_windows(
        long_gaps=long_gaps,
        mask=sample_mask,
        signal_length=N,
        sample_rate=sample_rate,
        preferred_window_s=window_s,
        minimum_window_s=minimum_window_s,
        maximum_window_s=maximum_window_s,
        minimum_clean_context_s=minimum_clean_context_s,
        merge_strategy=merge_strategy,
    )
    windows = [(w.start, w.end, w.gaps) for w in adaptive_windows]

    logger.info(
        "Long-audio windows stage: %d windows planned from %d large gaps.",
        len(windows),
        len(long_gaps),
    )

    max_windows = int(long_cfg.get("max_windows", 24))
    if len(windows) > max_windows:
        raise ValueError(
            f"Long-audio window count ({len(windows)}) exceeds limit ({max_windows})."
        )

    # 5. Execute DPAI first. Every missing sample inside a window remains
    # missing in the local mask, including short gaps.
    current_signal = np.asarray(corrupted, dtype=np.float32).copy()
    dpai_inpainter = None if executor is not None else DPAIInpainter(cfg, device=device)
    cluster_results: Optional[List[Dict[str, Any]]] = None

    if executor is not None:
        segments: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
        for w_start, w_end, _ in windows:
            win_corrupted = corrupted[w_start:w_end].copy()
            win_mask = np.asarray(sample_mask[w_start:w_end], dtype=np.float32).copy()
            win_ref = reference[w_start:w_end].copy() if reference is not None else None
            segments.append((win_corrupted, win_mask, win_ref))

        cancel_event = threading.Event()
        if cancel_check and cancel_check():
            cancel_event.set()

        def cluster_event(event: Dict[str, Any]) -> None:
            if cancel_check and cancel_check():
                cancel_event.set()
                return
            if on_event:
                on_event(dict(event, stage="dpai"))
            if callbacks:
                k = int(event.get("window_index", event.get("window", 0)))
                ep = int(event.get("epoch", 0))
                callbacks(
                    "dpai", k, len(windows), ep,
                    int(event.get("epochs", ep)),
                    loss=event.get("loss"),
                )

        from src.cluster.dispatch import run_windows
        cluster_results = run_windows(
            executor, str(long_cfg.get("job_id", "")), cfg, segments, int(epochs), int(seed),
            cancel_event, cluster_event,
        )
    loss_histories: List[List[float]] = []
    n_params = 0
    covered_by_dpai = np.zeros(N, dtype=bool)
    small_gaps_inside_windows: set[Tuple[int, int]] = set()

    windows_meta: List[Dict[str, Any]] = []

    for k, (w_start, w_end, _) in enumerate(windows):
        if cancel_check and cancel_check():
            from src.web.server import JobCancelledException  # type: ignore
            raise JobCancelledException(f"Job cancelled before window {k}.")

        # Pass the original corrupted signal and mask unchanged for this slice.
        win_corrupted = corrupted[w_start:w_end].copy()
        win_mask = np.asarray(sample_mask[w_start:w_end], dtype=np.float32).copy()
        assigned_to_this_win: List[Tuple[int, int]] = []
        for s, e in all_gaps:
            overlap_start = max(s, w_start)
            overlap_end = min(e, w_end)
            if overlap_start < overlap_end:
                if gap_has_clean_context((s, e), sample_mask,
                                         max(1, round(minimum_clean_context_s * sample_rate))):
                    assigned_to_this_win.append((overlap_start, overlap_end))
                    covered_by_dpai[overlap_start:overlap_end] = True
                    if (s, e) in short_gaps:
                        small_gaps_inside_windows.add((s, e))

        # A window may include no missing samples only if its source mask is
        # inconsistent with the planned large-gap list; skip it safely.
        if not assigned_to_this_win:
            continue

        # Guard: check if window missing ratio > 40%
        win_missing = (win_mask == 0.0).sum()
        win_ratio = win_missing / max(1, len(win_mask))
        if win_ratio >= 0.40:
            # We warn or split; here log/continue or keep valid
            pass

        win_ref: Optional[np.ndarray] = None
        if reference is not None:
            win_ref = reference[w_start:w_end]

        def win_callback(ep: int, total_ep: int, loss_val: float, output: Any) -> None:
            if cancel_check and cancel_check():
                from src.web.server import JobCancelledException  # type: ignore
                raise JobCancelledException(f"Job cancelled during window {k} epoch {ep}.")
            if callbacks:
                try:
                    callbacks("dpai", k, len(windows), ep, total_ep, loss=loss_val, output=output)
                except TypeError:
                    callbacks("dpai", k, len(windows), ep, total_ep)

        win_seed = seed + k
        window_t0 = time.perf_counter()
        if cluster_results is not None:
            cluster_result = cluster_results[k]
            win_waveform = np.asarray(cluster_result["wave"], dtype=np.float32)
            win_loss = list(cluster_result.get("loss", []))
            win_n_params = int(cluster_result.get("n_params", 0))
        else:
            win_result = dpai_inpainter.inpaint(
                corrupted=win_corrupted,
                sample_mask=win_mask,
                reference=win_ref,
                epochs=epochs,
                seed=win_seed,
                callbacks=win_callback if callbacks or cancel_check else None,
            )
            win_waveform = win_result.waveform
            win_loss = win_result.loss_history
            win_n_params = win_result.n_params
        window_runtime_s = time.perf_counter() - window_t0

        loss_histories.append(win_loss)
        n_params = win_n_params

        # 7. Merge: paste only DPAI-handled gap samples into full signal with equal-power crossfade
        paste_with_crossfade(
            target_waveform=current_signal,
            dpai_window_recon=win_waveform,
            window_start=w_start,
            gaps=assigned_to_this_win,
            crossfade_samples=crossfade_samples,
        )

        windows_meta.append({
            "window_index": k,
            "start": int(w_start),
            "end": int(w_end),
            "start_sample": int(w_start),
            "end_sample": int(w_end),
            "start_s": round(float(w_start) / sample_rate, 4),
            "end_s": round(float(w_end) / sample_rate, 4),
            "duration_s": round(float(w_end - w_start) / sample_rate, 4),
            "n_gaps": len(assigned_to_this_win),
            "gaps": [gap_metadata(gap, sample_rate) for gap in assigned_to_this_win],
            "grouped": not adaptive_windows[k].individually_planned,
            "left_context_available": adaptive_windows[k].left_context_available,
            "right_context_available": adaptive_windows[k].right_context_available,
            "runtime_s": round(window_runtime_s, 4),
            "n_params": int(win_n_params),
            "final_loss": round(float(win_loss[-1]), 8) if win_loss else None,
            "epochs": int(epochs),
            "loss_history": [round(float(loss), 10) for loss in win_loss],
        })

    logger.info(
        "Long-audio DPAI stage: %d windows processed, %d small gaps included inside windows.",
        len(windows_meta),
        len(small_gaps_inside_windows),
    )

    # 6. LPC-fill only missing samples outside every DPAI window. Computing
    # this from coverage also handles gaps that cross a window boundary.
    remaining_missing = (np.asarray(sample_mask) == 0.0) & ~covered_by_dpai
    lpc_gaps = mask_to_gaps(np.where(remaining_missing, 0.0, 1.0))
    logger.info(
        "Long-audio LPC stage: %d leftover gaps sent to LPC.",
        len(lpc_gaps),
    )
    if lpc_gaps:
        if callbacks:
            callbacks("lpc", 0, len(windows), 0, 0)
        lpc_mask = np.ones(N, dtype=np.float32)
        lpc_mask[remaining_missing] = 0.0
        lpc_filled = lpc_inpaint(
            current_signal,
            mask=lpc_mask,
            gaps=lpc_gaps,
            seed=seed,
        )
        for s, e in lpc_gaps:
            current_signal[s:e] = lpc_filled[s:e]

    dpai_missing = np.ones(N, dtype=np.float32)
    dpai_missing[(sample_mask == 0.0) & covered_by_dpai] = 0.0
    all_dpai_gaps = mask_to_gaps(dpai_missing)

    # Final safeguard: verify observed samples stay strictly bit-exact to original input
    # Where sample_mask == 1, current_signal MUST equal corrupted
    obs_idx = np.where(sample_mask == 1.0)[0]
    current_signal[obs_idx] = corrupted[obs_idx]

    runtime_s = time.perf_counter() - t0

    return LongInpaintResult(
        waveform=current_signal.astype(np.float32),
        loss_histories=loss_histories,
        windows=windows_meta,
        lpc_gaps=lpc_gaps,
        dpai_gaps=all_dpai_gaps,
        n_params=n_params,
        runtime_s=runtime_s,
        seed=seed,
        epochs=epochs,
        context_infeasible_gaps=context_infeasible,
        estimated_processing_s=sum(
            (float(w["duration_s"]) for w in windows_meta),
            0.0,
        ),
    )
