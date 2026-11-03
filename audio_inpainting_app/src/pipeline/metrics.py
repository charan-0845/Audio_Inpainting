"""Pipeline metrics: unified function for NMSE + optional PESQ evaluation.

Reuses :mod:`src.evaluation.nmse` so existing ``results.csv`` rows remain
comparable.  PESQ is optional (requires ``pip install pesq``); when the
package is not installed the corresponding metric is ``null`` / ``None``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from src.evaluation.nmse import missing_region_nmse, nmse

logger = logging.getLogger(__name__)


def _nmse_to_db(linear_nmse: float) -> Optional[float]:
    """Convert linear NMSE to dB (same formula as run_benchmark.nmse_db).

    Formula::

        dB = 10 * log10(nmse + 1e-12)

    The ``+1e-12`` floor matches the benchmark script so that rows from the
    old ``run_benchmark.py`` and the new pipeline are directly comparable.

    Args:
        linear_nmse: Linear NMSE value (NaN-safe).

    Returns:
        dB value, or ``None`` if *linear_nmse* is NaN.
    """
    import math
    if math.isnan(linear_nmse):
        return None
    return float(10.0 * math.log10(linear_nmse + 1e-12))


def compute_metrics(
    clean: np.ndarray,
    recon: np.ndarray,
    corrupted: np.ndarray,
    mask: np.ndarray,
    is_speech: bool = False,
    sample_rate: int = 16000,
) -> Dict[str, Optional[float]]:
    """Compute NMSE and optionally PESQ metrics for one inpainting result.

    NMSE is always computed using :func:`~src.evaluation.nmse.nmse` and
    :func:`~src.evaluation.nmse.missing_region_nmse`, then converted to dB
    using the same formula as ``run_benchmark.nmse_db``.

    PESQ delta (PESQ(clean, recon) - PESQ(clean, corrupted)) is computed
    only when *is_speech* is ``True`` and the ``pesq`` package is installed.

    Args:
        clean: Clean reference waveform ``(N,)``.
        recon: Reconstructed waveform ``(N,)`` from the inpainter.
        corrupted: Input corrupted waveform ``(N,)`` (used as PESQ baseline).
        mask: Sample-level binary mask ``(N,)`` with 1=observed, 0=missing.
        is_speech: If ``True``, attempt PESQ computation.
        sample_rate: Sample rate in Hz (required by PESQ).

    Returns:
        Dict with keys:

        * ``nmse_tot_db`` (float | None): NMSE over the whole clip in dB.
        * ``nmse_miss_db`` (float | None): NMSE over missing regions in dB.
        * ``pesq_delta`` (float | None): PESQ improvement; ``None`` if not
          computed.
    """
    nmse_tot_db = _nmse_to_db(nmse(clean, recon))
    nmse_miss_db = _nmse_to_db(missing_region_nmse(clean, recon, mask))

    pesq_delta: Optional[float] = None
    if is_speech:
        try:
            from pesq import pesq as pesq_fn
            pesq_recon = float(pesq_fn(sample_rate, clean.astype(np.float32),
                                       recon.astype(np.float32), "wb"))
            pesq_corr = float(pesq_fn(sample_rate, clean.astype(np.float32),
                                      corrupted.astype(np.float32), "wb"))
            pesq_delta = round(pesq_recon - pesq_corr, 4)
        except ImportError:
            logger.warning(
                "pesq package not installed; PESQ delta will be null. "
                "Install with: pip install pesq"
            )
        except Exception as exc:
            logger.warning("PESQ computation failed: %s", exc)

    return {
        "nmse_tot_db": nmse_tot_db,
        "nmse_miss_db": nmse_miss_db,
        "pesq_delta": pesq_delta,
    }
