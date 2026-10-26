"""precompute_examples.py – Precompute benchmark inpaintings for the web gallery.

Runs benchmark clips through DPAIInpainter and save_run, then copies the resulting
artefacts into src/web/static/examples/ and generates index.json for the web frontend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import _auto_device, _build_cfg, _deep_merge, _load_config
from src.audio.corruption import create_mask
from src.audio.loader import load_audio, make_synthetic_audio
from src.pipeline.inpainter import DPAIInpainter
from src.pipeline.io import save_run


DEFAULT_CLIPS = [
    {
        "id": "piano_01",
        "title": "Acoustic Piano Excerpt",
        "type": "piano",
        "source": "piano_01.wav",
        "gap_ms": 200.0,
        "seed": 0,
        "description": "Harmonic piano note with 200 ms artificial missing gaps restored via MultiResUNet.",
    },
    {
        "id": "speech_01",
        "title": "Speech Utterance",
        "type": "speech",
        "source": "speech_01.wav",
        "gap_ms": 150.0,
        "seed": 42,
        "description": "Spoken sentence with 150 ms missing intervals reconstructed without vocoder.",
    },
    {
        "id": "synthetic_01",
        "title": "Synthetic Harmonic Signal",
        "type": "synthetic",
        "source": "synthetic_01.wav",
        "gap_ms": 200.0,
        "seed": 123,
        "description": "Multi-tone synthetic waveform demonstrating reconstruction of harmonic carriers.",
    },
]


def precompute_examples(
    clips: List[Dict[str, Any]],
    epochs: int,
    cfg: Dict[str, Any],
    device: str,
    dest_dir: Path,
) -> None:
    """Run inpainting for each clip and populate the gallery directory.

    Args:
        clips: List of clip metadata dicts.
        epochs: Number of optimization epochs per clip.
        cfg: Resolved config dict.
        device: Torch device string.
        dest_dir: Destination path (e.g. src/web/static/examples).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    sr = int(cfg["audio"]["sample_rate"])

    temp_out = REPO / "outputs" / "_gallery_temp"
    temp_out.mkdir(parents=True, exist_ok=True)

    gallery_index: List[Dict[str, Any]] = []

    print(f"\n[DPAI Gallery Precompute] Generating {len(clips)} examples on {device} ({epochs} epochs)...")

    for clip_info in clips:
        clip_id = clip_info["id"]
        source_name = clip_info.get("source", f"{clip_id}.wav")
        audio_path = REPO / "data" / "clean" / source_name
        gap_ms = float(clip_info.get("gap_ms", 200.0))
        seed = int(clip_info.get("seed", 0))

        # 1. Load or synthesise audio
        if audio_path.is_file():
            clean = load_audio(audio_path, sample_rate=sr, duration_seconds=3.0).numpy().astype(np.float32)
        else:
            clean = make_synthetic_audio(duration_seconds=3.0, sample_rate=sr, seed=seed).numpy().astype(np.float32)

        N = len(clean)

        # 2. Build corruption mask
        mask = create_mask(num_samples=N, sample_rate=sr, cumulative_gap_ms=gap_ms, seed=seed).astype(np.float32)
        corrupted = clean * mask

        # 3. Optimize with DPAIInpainter
        print(f"  -> Inpainting '{clip_id}' ({gap_ms:.0f} ms gap, {epochs} epochs)...")
        inpainter = DPAIInpainter(cfg, device=device)
        result = inpainter.inpaint(
            corrupted=corrupted,
            sample_mask=mask,
            reference=clean,
            epochs=epochs,
            seed=seed,
        )

        # 4. Save run
        run_dir = save_run(
            result=result,
            corrupted=corrupted,
            sample_mask=mask,
            cfg=cfg,
            clip_id=clip_id,
            gap_ms=gap_ms,
            device=device,
            out_dir=temp_out,
            clean=clean,
            is_speech=(clip_info.get("type") == "speech"),
            sample_rate=sr,
        )

        # 5. Copy artefacts to static/examples/<clip_id>/
        example_target_dir = dest_dir / clip_id
        example_target_dir.mkdir(parents=True, exist_ok=True)

        files_to_copy = [
            "reconstruction.wav",
            "corrupted.wav",
            "clean.wav",
            "spec_reconstructed.png",
            "spec_corrupted.png",
            "spec_clean.png",
            "spectrogram.png",
            "loss_curve.png",
            "metrics.json",
        ]

        for fname in files_to_copy:
            src_file = run_dir / fname
            if src_file.is_file():
                shutil.copy2(src_file, example_target_dir / fname)

        # Read metrics if written
        metrics: Dict[str, Any] = {}
        metrics_file = run_dir / "metrics.json"
        if metrics_file.is_file():
            try:
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        spec_file = "spec_reconstructed.png" if (example_target_dir / "spec_reconstructed.png").is_file() else "spectrogram.png"

        gallery_index.append({
            "id": clip_id,
            "title": clip_info.get("title", clip_id),
            "clip_id": clip_id,
            "type": clip_info.get("type", "audio"),
            "gap_ms": gap_ms,
            "epochs": epochs,
            "seed": seed,
            "description": clip_info.get("description", ""),
            "spectrogram_url": f"examples/{clip_id}/{spec_file}",
            "spec_corrupted_url": f"examples/{clip_id}/spec_corrupted.png" if (example_target_dir / "spec_corrupted.png").is_file() else None,
            "audio_url": f"examples/{clip_id}/reconstruction.wav",
            "corrupted_url": f"examples/{clip_id}/corrupted.wav",
            "clean_url": f"examples/{clip_id}/clean.wav" if (example_target_dir / "clean.wav").is_file() else None,
            "metrics": metrics,
        })

    # 6. Write index.json
    index_file = dest_dir / "index.json"
    index_file.write_text(json.dumps({"examples": gallery_index}, indent=2), encoding="utf-8")
    print(f"\n[OK] Gallery index written to: {index_file}")
    print(f"Generated {len(gallery_index)} examples in: {dest_dir}")

    # Clean up temp
    if temp_out.is_dir():
        shutil.rmtree(temp_out, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute audio inpainting examples for DPAI web gallery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of optimisation epochs per clip.")
    parser.add_argument("--config", default="configs/smoke.yaml",
                        help="Configuration file to use.")
    parser.add_argument("--device", default=None,
                        help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--dest", default=str(REPO / "src" / "web" / "static" / "examples"),
                        help="Target directory for web gallery assets.")

    args = parser.parse_args()

    cfg = _build_cfg(args)
    device = _auto_device(args)
    dest_path = Path(args.dest)

    precompute_examples(
        clips=DEFAULT_CLIPS,
        epochs=args.epochs,
        cfg=cfg,
        device=device,
        dest_dir=dest_path,
    )


if __name__ == "__main__":
    main()
