"""Audio dataset utility functions and record structures."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


@dataclass
class ClipRecord:
    """Dataclass holding metadata for a raw source clip."""

    clip_id: str
    source_path: Path
    duration: float
    sample_rate: int


@dataclass
class PreparedSample:
    """Dataclass holding paths and parameters for a prepared clean/corrupted sample."""

    clip_id: str
    gap_ms: float
    seed: int
    clean_path: Path
    corrupted_path: Path
    mask_path: Path
    num_gaps: int
    cumulative_gap_samples: int


def list_raw_clips(raw_dir: str | Path) -> list[Path]:
    """Find all valid audio clips under raw_dir.

    Skips hidden files (starting with '.') and non-audio files.
    Returns a sorted list of Path objects for deterministic ordering.

    Args:
        raw_dir: Directory to search for audio files.

    Returns:
        Sorted list of Path objects pointing to raw audio files.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"Raw directory does not exist: {raw_path}")
        return []

    audio_files: list[Path] = []
    for p in raw_path.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                audio_files.append(p)

    audio_files.sort()
    logger.info(f"Found {len(audio_files)} audio clips in {raw_path}")
    return audio_files
