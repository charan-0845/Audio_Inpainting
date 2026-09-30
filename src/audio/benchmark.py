"""Frozen benchmark: deterministic gap masks, manifest I/O, tier selection.

Design
------
* 15 clips (5 piano / 5 music / 5 speech), 5 s @ 16 kHz, mono.
* 5 cumulative gap levels; each gap is 40-80 ms.
* The "quick" tier is a subset of the "full" tier (same clips, same masks),
  so quick results are directly comparable with full results.
* Gaps are stored as (start, length) sample intervals inside the committed
  manifest, so masks can never silently change between runs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SR = 16000
DURATION_S = 5
N_SAMPLES = SR * DURATION_S

GAP_LEVELS_MS = [200, 400, 600, 800, 1000]
QUICK_LEVELS_MS = [200, 600, 1000]
MIN_GAP_MS, MAX_GAP_MS = 40, 80
MIN_SEP_MS = 50      # minimum distance between two gaps
EDGE_MS = 100        # keep gaps away from the start/end of the clip

CLIP_TYPES = ["piano", "music", "speech"]
N_PER_TYPE = 5
N_QUICK_PER_TYPE = 2
MASTER_SEED = 1234


def _seed(*parts):
    """Stable seed (Python's hash() is salted per process, so don't use it)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little")


def _ms(x):
    return int(round(x * SR / 1000))


def make_gaps(level_ms, clip_id, seed=MASTER_SEED, n_samples=N_SAMPLES):
    """Return [[start, length], ...] whose lengths sum to exactly level_ms."""
    rng = np.random.default_rng(_seed(seed, clip_id, level_ms))
    total = _ms(level_ms)
    lo, hi = _ms(MIN_GAP_MS), _ms(MAX_GAP_MS)
    sep, edge = _ms(MIN_SEP_MS), _ms(EDGE_MS)
    assert hi >= 2 * lo, "length sampler assumes max_gap >= 2 * min_gap"

    # 1) gap lengths in [lo, hi] that sum exactly to `total`
    lengths, remaining = [], total
    while remaining > hi:
        l = int(rng.integers(lo, min(hi, remaining - lo) + 1))
        lengths.append(l)
        remaining -= l
    lengths.append(remaining)          # lo <= remaining <= hi by construction
    rng.shuffle(lengths)

    # 2) random non-overlapping placement with minimum separation
    n = len(lengths)
    slack = (n_samples - 2 * edge) - sum(lengths) - (n - 1) * sep
    assert slack >= 0, "gaps do not fit in the clip"
    offsets = np.sort(rng.integers(0, slack + 1, size=n))
    gaps, used = [], 0
    for i, (l, o) in enumerate(zip(lengths, offsets)):
        gaps.append([int(edge + used + i * sep + o), int(l)])
        used += l
    return gaps


def mask_from_gaps(gaps, n_samples=N_SAMPLES):
    """1.0 = observed sample, 0.0 = missing sample."""
    mask = np.ones(n_samples, dtype=np.float32)
    for start, length in gaps:
        mask[start:start + length] = 0.0
    return mask


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))


def load_manifest(path="benchmark/manifest.json"):
    return json.loads(Path(path).read_text())


def iter_cases(manifest, tier="quick"):
    """Yield (clip_entry, level_ms) for a tier ('quick' or 'full')."""
    if tier not in ("quick", "full"):
        raise ValueError("tier must be 'quick' or 'full'")
    levels = QUICK_LEVELS_MS if tier == "quick" else GAP_LEVELS_MS
    for clip in manifest["clips"]:
        if tier == "quick" and not clip["quick"]:
            continue
        for level in levels:
            yield clip, level


def load_case(clip, level_ms, clean_dir="data/clean"):
    """Return (clean, mask, corrupted) as float32 arrays of length N_SAMPLES."""
    import soundfile as sf

    clean, sr = sf.read(str(Path(clean_dir) / f"{clip['id']}.wav"), dtype="float32")
    assert sr == SR and len(clean) == N_SAMPLES
    mask = mask_from_gaps(clip["gaps"][str(level_ms)])
    return clean, mask, clean * mask
