"""Build the frozen benchmark from raw audio.

Expected input layout (put source files here; names sort to a stable order):
    data/raw/piano/*.wav|flac|mp3      (MAESTRO)
    data/raw/music/*.wav|flac|mp3      (FMA)
    data/raw/speech/*.wav|flac|mp3     (MUSAN speech)

The first N_QUICK_PER_TYPE files of each type form the quick tier, so name your
files so that the ones you want in the quick tier sort first (piano_01, ...).

Outputs:
    benchmark/manifest.json   <- COMMIT THIS (clips + exact gap positions)
    data/clean/<id>.wav       (gitignored)
    data/corrupted/<id>_g<level>.wav (gitignored, convenience only)
"""
import argparse
import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio import benchmark as bm  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg"}
MAX_LOAD_S = 180          # long files (MAESTRO): only search the first 3 minutes
ACTIVE_DB = -40           # frame counts as "active" above this level vs. peak
MIN_ACTIVITY = 0.9        # required fraction of active 20 ms frames


def pick_window(y, rng):
    """Pick a 5 s window that is mostly non-silent (seeded, reproducible)."""
    win, hop, frame = bm.N_SAMPLES, bm.SR, int(0.02 * bm.SR)
    if len(y) < win:
        raise ValueError("source shorter than 5 s")
    peak = np.abs(y).max() + 1e-9
    starts = list(range(0, len(y) - win + 1, hop))

    def activity(s):
        seg = y[s:s + win][: (win // frame) * frame].reshape(-1, frame)
        rms = np.sqrt((seg ** 2).mean(axis=1)) + 1e-12
        return float((20 * np.log10(rms / peak) > ACTIVE_DB).mean())

    acts = np.array([activity(s) for s in starts])
    good = np.flatnonzero(acts >= MIN_ACTIVITY)
    idx = int(rng.choice(good)) if len(good) else int(acts.argmax())
    return starts[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--manifest", default="benchmark/manifest.json")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--corrupted-dir", default="data/corrupted")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    if Path(args.manifest).exists() and not args.force:
        sys.exit(f"{args.manifest} exists. The benchmark is frozen; use --force to rebuild.")

    Path(args.clean_dir).mkdir(parents=True, exist_ok=True)
    Path(args.corrupted_dir).mkdir(parents=True, exist_ok=True)
    clips = []

    for ctype in bm.CLIP_TYPES:
        files = sorted(p for p in (Path(args.raw) / ctype).glob("*")
                       if p.suffix.lower() in AUDIO_EXT)[: bm.N_PER_TYPE]
        if len(files) < bm.N_PER_TYPE:
            print(f"[warn] {ctype}: found {len(files)}/{bm.N_PER_TYPE} files")
        for i, path in enumerate(files):
            clip_id = f"{ctype}_{i + 1:02d}"
            y, _ = librosa.load(str(path), sr=bm.SR, mono=True, duration=MAX_LOAD_S)
            rng = np.random.default_rng(bm._seed(bm.MASTER_SEED, clip_id, "crop"))
            start = pick_window(y, rng)
            clean = y[start:start + bm.N_SAMPLES]
            clean = (0.9 * clean / (np.abs(clean).max() + 1e-9)).astype(np.float32)
            # quantise to 16-bit now so the stored wav IS the reference signal
            clean = (np.round(clean * 32767) / 32767).astype(np.float32)
            sf.write(str(Path(args.clean_dir) / f"{clip_id}.wav"), clean, bm.SR, subtype="PCM_16")

            gaps = {str(lv): bm.make_gaps(lv, clip_id) for lv in bm.GAP_LEVELS_MS}
            for lv, g in gaps.items():
                corrupted = clean * bm.mask_from_gaps(g)
                sf.write(str(Path(args.corrupted_dir) / f"{clip_id}_g{lv}.wav"),
                         corrupted, bm.SR, subtype="PCM_16")

            clips.append({
                "id": clip_id,
                "type": ctype,
                "source": path.name,
                "offset_s": round(start / bm.SR, 3),
                "sha256": hashlib.sha256(clean.tobytes()).hexdigest()[:16],
                "quick": i < bm.N_QUICK_PER_TYPE,
                "gaps": gaps,
            })
            print(f"built {clip_id} <- {path.name} @ {start / bm.SR:.1f}s")

    manifest = {
        "sr": bm.SR, "duration_s": bm.DURATION_S, "seed": bm.MASTER_SEED,
        "gap_levels_ms": bm.GAP_LEVELS_MS, "quick_levels_ms": bm.QUICK_LEVELS_MS,
        "gap_len_ms": [bm.MIN_GAP_MS, bm.MAX_GAP_MS],
        "clips": clips,
    }
    bm.save_manifest(manifest, args.manifest)
    n_quick = sum(c["quick"] for c in clips) * len(bm.QUICK_LEVELS_MS)
    print(f"\nwrote {args.manifest}: {len(clips)} clips | quick={n_quick} cases | "
          f"full={len(clips) * len(bm.GAP_LEVELS_MS)} cases")


if __name__ == "__main__":
    main()
