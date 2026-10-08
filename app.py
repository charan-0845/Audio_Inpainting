"""app.py – DPAI audio inpainting application (MultiResUNet only).

Subcommands
-----------
* ``inpaint``   – inpaint one audio file and save all artefacts.
* ``benchmark`` – run the benchmark suite with resumable CSV output.
* ``summarize`` – merge worker CSVs and print a summary table.

Quick-start examples::

    # Inpaint with a synthetic 200 ms gap, 10 epochs (smoke run):
    python app.py inpaint --gap_ms 200 --epochs 10 --config configs/smoke.yaml

    # Real inpaint run (5 000 epochs, GPU if available):
    python app.py inpaint --audio data/clean/piano_01.wav --gap_ms 400

    # Benchmark quick tier, seeds 0 and 1:
    python app.py benchmark --tier quick --seeds 0 1 --epochs 5000

    # Summarize results:
    python app.py summarize --csv experiments/multires_harmonic/results.csv

DEPRECATED SCRIPTS
------------------
scripts/run_deep_prior.py, scripts/compare_architectures.py,
scripts/compare_with_harmonic.py, and scripts/run_benchmark.py are
superseded by this application.  They are kept for reference only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load base YAML config.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed config dict.
    """
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (returns new dict).

    Args:
        base: Base config dict.
        override: Override values (nested dict supported).

    Returns:
        Merged dict (base is not mutated).
    """
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configs, then apply CLI overrides.

    Args:
        args: Parsed CLI namespace (must have ``config`` attribute).

    Returns:
        Fully resolved config dict.
    """
    cfg = _load_config(REPO / "configs" / "config.yaml")
    if hasattr(args, "config") and args.config:
        override_cfg = _load_config(args.config)
        cfg = _deep_merge(cfg, override_cfg)
    return cfg


def _auto_device(args: argparse.Namespace) -> str:
    """Resolve the device string from CLI args, defaulting to CUDA if available.

    Args:
        args: Parsed namespace (checked for ``device`` attribute).

    Returns:
        Device string, e.g. ``"cpu"`` or ``"cuda"``.
    """
    if hasattr(args, "device") and args.device:
        return args.device
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Git helper
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return short HEAD hash or empty string."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ``inpaint`` subcommand
# ---------------------------------------------------------------------------

def _cmd_inpaint(args: argparse.Namespace) -> None:
    """Run the inpaint subcommand.

    Loads (or synthesises) audio, derives/loads the mask, runs the pipeline,
    and saves all artefacts to ``outputs/runs/<run_id>/``.

    Args:
        args: Parsed CLI namespace for the ``inpaint`` subcommand.
    """
    import soundfile as sf
    from src.audio.corruption import create_mask, extract_mask_from_audio
    from src.audio.loader import load_audio, make_synthetic_audio
    from src.pipeline.inpainter import DPAIInpainter
    from src.pipeline.io import save_run

    cfg = _build_cfg(args)
    device = _auto_device(args)
    sr = int(cfg["audio"]["sample_rate"])
    opt_cfg = cfg["optimization"]

    # --- load or synthesise audio ---
    if args.audio:
        wav = load_audio(args.audio, sample_rate=sr).numpy().astype(np.float32)
        clip_id = Path(args.audio).stem
    else:
        # Synthetic signal for testing / demo
        wav = make_synthetic_audio(duration_seconds=5.0, sample_rate=sr,
                                   seed=int(args.seed)).numpy().astype(np.float32)
        clip_id = "synthetic"

    N = len(wav)

    # --- derive or load mask ---
    if args.mask:
        sample_mask = np.load(args.mask).astype(np.float32)
    elif args.detect_mask:
        from src.audio.corruption import extract_mask_from_audio as _emfa
        sample_mask = _emfa(wav).astype(np.float32)
    else:
        gap_ms = float(args.gap_ms)
        sample_mask = create_mask(
            num_samples=N,
            sample_rate=sr,
            cumulative_gap_ms=gap_ms,
            seed=int(args.seed),
        ).astype(np.float32)

    gap_ms = float(args.gap_ms) if not args.mask and not args.detect_mask else 0.0

    corrupted = wav * sample_mask

    epochs = int(args.epochs) if args.epochs else int(opt_cfg["epochs"])
    out_dir = args.out_dir or cfg.get("run", {}).get("out_dir", "outputs/runs")

    inpainter = DPAIInpainter(cfg, device=device)

    if args.dry_run:
        print("[dry-run] Skipping optimisation; returning corrupted input.")
        result_wav = corrupted
        # Build a minimal InpaintResult-like object
        from src.pipeline.inpainter import InpaintResult
        from src.audio.stft import compute_stft, stft_to_channels, channels_to_stft
        _stft = compute_stft(
            torch.from_numpy(corrupted),
            n_fft=cfg["stft"]["n_fft"],
            hop_length=cfg["stft"]["hop_length"],
            win_length=cfg["stft"]["win_length"],
        )
        result = InpaintResult(
            waveform=corrupted,
            loss_history=[0.0],
            val_loss_history=[],
            final_spectrogram=_stft,
            runtime_s=0.0,
            n_params=0,
            seed=int(args.seed),
            epochs=0,
        )
    else:
        result = inpainter.inpaint(
            corrupted=corrupted,
            sample_mask=sample_mask,
            epochs=epochs,
            seed=int(args.seed),
        )

    run_dir = save_run(
        result=result,
        corrupted=corrupted,
        sample_mask=sample_mask,
        cfg=cfg,
        clip_id=clip_id,
        gap_ms=gap_ms,
        device=device,
        out_dir=out_dir,
        clean=None,
        is_speech=False,
        sample_rate=sr,
    )
    print(f"\n[OK] Run saved to: {run_dir}")
    print(f"  Epochs: {result.epochs}  |  Params: {result.n_params:,}  |  "
          f"Runtime: {result.runtime_s:.1f}s")
    print("  Files produced:")
    for p in sorted(run_dir.iterdir()):
        print(f"    {p.name}")


# ---------------------------------------------------------------------------
# ``benchmark`` subcommand
# ---------------------------------------------------------------------------

BENCHMARK_CSV_FIELDS = [
    "variant", "tier", "clip", "type", "level_ms", "seed", "epochs",
    "nmse_tot_lin", "nmse_miss_lin", "pesq_delta", "pesq_mode", "runtime_s", "git", "note",
]


def _benchmark_nmse_lin(ref: np.ndarray, est: np.ndarray,
                        sel: Optional[np.ndarray] = None) -> float:
    """Compute NMSE in linear scale.

    Args:
        ref: Reference waveform.
        est: Estimated waveform.
        sel: Optional boolean array selecting a subset of samples.

    Returns:
        NMSE (linear scale).
    """
    if sel is not None:
        ref, est = ref[sel], est[sel]
    return float(((ref - est) ** 2).sum() / ((ref ** 2).sum() + 1e-12))


def _benchmark_pesq(clean: np.ndarray, corrupted: np.ndarray,
                     recon: np.ndarray, sr: int) -> str:
    """Compute PESQ delta; returns '' when unavailable.

    Args:
        clean: Clean reference ``(N,)``.
        corrupted: Corrupted input ``(N,)``.
        recon: Reconstructed waveform ``(N,)``.
        sr: Sample rate in Hz.

    Returns:
        PESQ delta string, or empty string on failure.
    """
    try:
        from pesq import pesq
        return str(round(
            float(pesq(sr, clean, recon, "wb")) -
            float(pesq(sr, clean, corrupted, "wb")),
            4,
        ))
    except Exception:
        return ""


def _done_keys(csv_path: Path):
    """Read already-completed (clip, level_ms, seed) triples from a CSV.

    Args:
        csv_path: Path to the results CSV.

    Returns:
        Set of ``(clip_id, level_ms_int, seed_int)`` triples.
    """
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {
            (r["clip"], int(r["level_ms"]), int(r["seed"]))
            for r in csv.DictReader(fh)
        }


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark subcommand.

    Iterates over ``(clip, level_ms, seed)`` triples for the requested tier,
    calls the inpainter for each, and appends one row per case to a CSV.
    Already-completed rows (identified by clip+level_ms+seed) are skipped so
    interrupted runs can be safely resumed.

    Args:
        args: Parsed namespace for the ``benchmark`` subcommand.
    """
    from src.audio import benchmark as bm
    from src.pipeline.inpainter import DPAIInpainter

    cfg = _build_cfg(args)
    device = _auto_device(args)
    sr = bm.SR

    # Config-driven variant label (default: from config, then CLI)
    variant = args.variant if args.variant else cfg.get("pipeline", {}).get(
        "variant", "multires_harmonic"
    )
    fname = f"results_{args.worker}.csv" if args.worker else "results.csv"
    csv_path = Path("experiments") / variant / fname
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = bm.load_manifest(args.manifest)
    finished = _done_keys(csv_path)
    new_file = not csv_path.exists()
    commit = _git_hash()
    ran = 0
    epochs = int(args.epochs) if args.epochs else int(cfg["optimization"]["epochs"])

    inpainter = DPAIInpainter(cfg, device=device)

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BENCHMARK_CSV_FIELDS)
        if new_file:
            writer.writeheader()

        for clip, level in bm.iter_cases(manifest, args.tier):
            if args.only and clip["type"] != args.only:
                continue
            for seed in args.seeds:
                if (clip["id"], level, seed) in finished:
                    continue
                if args.limit is not None and ran >= args.limit:
                    return

                clean, mask, corrupted = bm.load_case(clip, level)
                t0 = time.time()
                print(
                    f"\n=== {clip['id']} | {level} ms | seed {seed} | {variant} ===",
                    flush=True,
                )

                if args.dry_run:
                    recon = corrupted.copy()
                    runtime = 0.0
                else:
                    result = inpainter.inpaint(
                        corrupted=corrupted,
                        sample_mask=mask,
                        reference=clean,
                        epochs=epochs,
                        seed=seed,
                    )
                    recon = result.waveform
                    runtime = time.time() - t0

                missing = mask == 0
                row = {
                    "variant": variant,
                    "tier": args.tier,
                    "clip": clip["id"],
                    "type": clip["type"],
                    "level_ms": level,
                    "seed": seed,
                    "epochs": epochs,
                    "nmse_tot_lin": round(_benchmark_nmse_lin(clean, recon), 5),
                    "nmse_miss_lin": round(_benchmark_nmse_lin(clean, recon, missing), 5),
                    "pesq_delta": (
                        _benchmark_pesq(clean, corrupted, recon, sr)
                        if clip["type"] == "speech" and not args.dry_run
                        else ""
                    ),
                    "pesq_mode": "wb" if clip["type"] == "speech" and not args.dry_run else "",
                    "runtime_s": round(runtime, 1),
                    "git": commit,
                    "note": args.note + (" DUMMY" if args.dry_run else ""),
                }
                writer.writerow(row)
                fh.flush()
                ran += 1
                print(
                    f"{clip['id']} g{level} s{seed}: "
                    f"NMSE_tot={row['nmse_tot_lin']:.5f}, "
                    f"NMSE_miss={row['nmse_miss_lin']:.5f}, "
                    f"{runtime:.0f}s",
                )


# ---------------------------------------------------------------------------
# ``summarize`` subcommand
# ---------------------------------------------------------------------------

def _cmd_summarize(args: argparse.Namespace) -> None:
    """Merge worker CSVs, print a grouped summary, and save a plot.

    Args:
        args: Parsed namespace for the ``summarize`` subcommand.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dfs = [pd.read_csv(p) for p in args.csv]
    if not dfs:
        print("No CSV files provided.")
        return

    df = pd.concat(dfs, ignore_index=True)
    # De-duplicate: keep last occurrence if a (clip, level_ms, seed) appears in multiple worker files
    df = df.drop_duplicates(subset=["clip", "level_ms", "seed"], keep="last")

    for col in ("nmse_tot_lin", "nmse_miss_lin"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grp = df.groupby(["type", "level_ms"])[["nmse_tot_lin", "nmse_miss_lin"]]
    summary = grp.agg(["mean", "std"]).round(5)
    print("\n=== NMSE Summary (mean ± std, linear) ===")
    print(summary.to_string())
    print()

    # Save plot
    out_plot = Path(args.csv[0]).parent / "nmse_vs_gap.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    for clip_type in df["type"].unique():
        sub = df[df["type"] == clip_type].groupby("level_ms")["nmse_miss_lin"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=clip_type)
    ax.set_xlabel("Cumulative gap duration (ms)")
    ax.set_ylabel("NMSE_miss (linear)")
    ax.set_title("NMSE vs. gap duration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_plot), dpi=150)
    plt.close("all")
    print(f"Plot saved to: {out_plot}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="DPAI audio inpainting – MultiResUNet (single-model application)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- inpaint ---------------------------------------------------------------
    p_inpaint = sub.add_parser(
        "inpaint",
        help="Inpaint one audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_inpaint.add_argument("--audio", default=None,
                           help="Path to input WAV (omit to use synthetic signal).")
    group_mask = p_inpaint.add_mutually_exclusive_group()
    group_mask.add_argument("--mask", default=None,
                            help="Path to pre-computed mask .npy (1=observed, 0=missing).")
    group_mask.add_argument("--detect_mask", action="store_true",
                            help="Auto-detect gaps from near-zero regions.")
    p_inpaint.add_argument("--gap_ms", type=float, default=200.0,
                           help="Cumulative gap to create (ms) when mask not supplied.")
    p_inpaint.add_argument("--seed", type=int, default=0,
                           help="Seed for model init, input noise, and mask generation.")
    p_inpaint.add_argument("--epochs", type=int, default=None,
                           help="Optimisation epochs (overrides config).")
    p_inpaint.add_argument("--config", default=None,
                           help="Path to additional YAML config to merge over defaults.")
    p_inpaint.add_argument("--device", default=None,
                           help="Device: 'cpu' or 'cuda' (default: auto).")
    p_inpaint.add_argument("--out_dir", default=None,
                           help="Parent output directory (default: from config).")
    p_inpaint.add_argument("--dry_run", action="store_true",
                           help="Skip optimisation; return corrupted input (plumbing test).")

    # -- benchmark -------------------------------------------------------------
    p_bench = sub.add_parser(
        "benchmark",
        help="Run the frozen benchmark suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_bench.add_argument("--tier", default="quick", choices=["quick", "full"],
                         help="Benchmark tier.")
    p_bench.add_argument("--only", choices=["piano", "music", "speech"], default=None,
                         help="Restrict to one clip type.")
    p_bench.add_argument("--seeds", type=int, nargs="+", default=[0],
                         help="Seeds to run.")
    p_bench.add_argument("--epochs", type=int, default=None,
                         help="Epochs per run (overrides config).")
    p_bench.add_argument("--config", default=None,
                         help="Additional YAML config to merge.")
    p_bench.add_argument("--device", default=None,
                         help="Device: 'cpu' or 'cuda' (default: auto).")
    p_bench.add_argument("--manifest", default="benchmark/manifest.json",
                         help="Path to benchmark manifest JSON.")
    p_bench.add_argument("--variant", default=None,
                         help="Variant label (default: from config pipeline.variant).")
    p_bench.add_argument("--worker", default=None,
                         help="Worker letter; writes results_<worker>.csv.")
    p_bench.add_argument("--limit", type=int, default=None,
                         help="Run at most N new cases.")
    p_bench.add_argument("--note", default="",
                         help="Note appended to every row.")
    p_bench.add_argument("--dry_run", action="store_true",
                         help="Skip training (plumbing test).")

    # -- summarize -------------------------------------------------------------
    p_sum = sub.add_parser(
        "summarize",
        help="Merge worker CSVs and print summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_sum.add_argument("--csv", nargs="+", required=True,
                       help="Path(s) to results CSV file(s).")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "inpaint":
        _cmd_inpaint(args)
    elif args.cmd == "benchmark":
        _cmd_benchmark(args)
    elif args.cmd == "summarize":
        _cmd_summarize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
