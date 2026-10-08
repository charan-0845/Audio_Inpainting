"""Core DPAI inpainting pipeline (MultiResUNet only).

This module provides :class:`DPAIInpainter`, the single entry-point for
audio inpainting using the Deep Prior + MultiResUNet approach from:

  Miotello et al., "Deep Prior-Based Audio Inpainting Using
  Multi-Resolution Harmonic CNNs", IEEE/ACM TASLP 2024.

Usage example::

    import yaml
    from src.pipeline.inpainter import DPAIInpainter

    cfg = yaml.safe_load(open("configs/config.yaml"))
    inpainter = DPAIInpainter(cfg)
    result = inpainter.inpaint(corrupted, mask, reference=clean, epochs=5000)
    print(result.waveform.shape, result.runtime_s)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.audio.stft import (
    channels_to_stft,
    compute_stft,
    crop_to_shape,
    frame_mask,
    inverse_stft,
    pad_to_multiple,
    stft_to_channels,
    tf_mask,
)
from src.losses.total_loss import build_loss
from src.models.multires_unet import MultiResUNet
from src.training.deep_prior import optimize_deep_prior
from src.training.noise import make_input_noise


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InpaintResult:
    """Structured result returned by :meth:`DPAIInpainter.inpaint`.

    Attributes:
        waveform: Reconstructed waveform of shape ``(N,)`` as float32 ndarray.
        loss_history: Per-epoch training loss (observed region MSE).
        val_loss_history: ``(epoch, loss)`` tuples for the missing-region
            diagnostic.  Empty when no *reference* was supplied.
        final_spectrogram: Reconstructed complex STFT ``(M, L)`` (cropped,
            before inverse STFT).
        runtime_s: Wall-clock seconds spent inside :meth:`inpaint`.
        n_params: Number of trainable model parameters.
        seed: Seed used for model init and noise generation.
        epochs: Number of optimisation epochs actually run.
    """
    waveform: np.ndarray
    loss_history: List[float]
    val_loss_history: List[Tuple[int, float]]
    final_spectrogram: torch.Tensor
    runtime_s: float
    n_params: int
    seed: int
    epochs: int


# ---------------------------------------------------------------------------
# Main inpainter class
# ---------------------------------------------------------------------------

class DPAIInpainter:
    """Single-model audio inpainter using MultiResUNet and the deep-prior method.

    All hyperparameters are driven by *cfg*.  The class never contains any
    mutable state other than the config; every call to :meth:`inpaint` builds
    a fresh model from scratch so results are fully reproducible from *seed*.

    Args:
        cfg: Fully-resolved config dict (keys: ``model``, ``stft``,
            ``optimization``, ``loss``, ``pipeline``).
        device: Target device string or :class:`torch.device`.  When
            ``None`` (default) CUDA is used if available, else CPU.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: Optional[str | torch.device] = None,
    ) -> None:
        self._cfg = cfg
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Private helpers (each wraps one logical step)
    # ------------------------------------------------------------------

    def _stft_params(self) -> Dict[str, int]:
        """Return STFT params dict from config."""
        s = self._cfg["stft"]
        return {
            "n_fft": int(s["n_fft"]),
            "hop_length": int(s["hop_length"]),
            "win_length": int(s["win_length"]),
        }

    def _compute_observed_channels(
        self,
        corrupted: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """STFT -> 2-channel tensor for the corrupted waveform.

        Args:
            corrupted: Float32 waveform ``(N,)``.

        Returns:
            Tuple ``(channels, stft)`` where *channels* is ``(2, M, L)`` and
            *stft* is the complex STFT ``(M, L)``.
        """
        wav = torch.from_numpy(np.asarray(corrupted, dtype=np.float32))
        stft = compute_stft(wav, **self._stft_params())
        channels = stft_to_channels(stft)
        return channels, stft

    def _build_tf_mask(
        self,
        sample_mask: np.ndarray,
        num_bins: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derive per-frame and full TF mask from sample-level mask.

        Args:
            sample_mask: Float32 array ``(N,)`` with 1=observed, 0=missing.
            num_bins: Number of STFT frequency bins (M).

        Returns:
            Tuple ``(frame_mask_vec, tf_mask_mat)`` of shapes ``(L,)`` and
            ``(M, L)``.
        """
        sm = torch.from_numpy(np.asarray(sample_mask, dtype=np.float32))
        fmask = frame_mask(sm, **self._stft_params())
        tfmask = tf_mask(fmask, num_bins)
        return fmask, tfmask

    def _pad(
        self, observed_ch: torch.Tensor, tfmask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Pad both tensors to multiple of 32 (required by MultiResUNet).

        Args:
            observed_ch: ``(2, M, L)`` STFT channels.
            tfmask: ``(M, L)`` TF mask.

        Returns:
            Tuple ``(padded_channels, padded_mask, original_shape)`` where
            *original_shape* is ``(M, L)`` for later cropping.
        """
        padded_obs, original_shape = pad_to_multiple(observed_ch)
        padded_tf, _ = pad_to_multiple(tfmask.unsqueeze(0))
        padded_tf = padded_tf.squeeze(0)
        return padded_obs, padded_tf, original_shape

    def _make_noise(
        self,
        shape: Tuple[int, ...],
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Generator]:
        """Create fixed input noise and a seeded generator for perturbations.

        Args:
            shape: Shape of the noise tensor (matches padded observed channels).
            seed: Integer seed – controls both the input noise and all
                perturbation noise drawn during optimisation.

        Returns:
            Tuple ``(noise, generator)``.
        """
        generator = torch.Generator(device=self._device).manual_seed(seed)
        noise = make_input_noise(
            shape,
            variance=float(self._cfg["optimization"]["noise_variance"]),
            generator=generator,
            device=self._device,
        )
        return noise, generator

    def _build_model(self, seed: int) -> Tuple[torch.nn.Module, int]:
        """Instantiate a fresh MultiResUNet and return ``(model, n_params)``.

        Args:
            seed: Seed for model weight initialisation (via torch.manual_seed).

        Returns:
            Tuple ``(model, n_params)``.
        """
        torch.manual_seed(seed)
        model_cfg = self._cfg["model"]
        model = MultiResUNet(
            in_channels=int(model_cfg.get("input_channels", 2)),
            out_channels=int(model_cfg.get("output_channels", 2)),
            channel_schedule=model_cfg.get("channel_schedule"),
            res_path_lengths=model_cfg.get("res_path_lengths"),
            alpha=float(model_cfg.get("alpha", 1.6)),
            negative_slope=float(model_cfg.get("negative_slope", 0.01)),
            use_harmonic=bool(model_cfg.get("use_harmonic", True)),
            harmonic_anchors=model_cfg.get("harmonic_anchors"),
            norm=model_cfg.get("norm", "batch"),
        ).to(self._device)
        return model, model.count_parameters()

    def _run_optimisation(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        observed_padded: torch.Tensor,
        mask_padded: torch.Tensor,
        generator: torch.Generator,
        epochs: int,
        callbacks: Optional[List[Callable]] = None,
        reference_padded: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Wrap ``optimize_deep_prior`` with config-driven hyperparams.

        Args:
            model: Freshly built model already on the correct device.
            noise: Input noise tensor.
            observed_padded: Padded observed STFT channels on device.
            mask_padded: Padded TF mask on device.
            generator: Seeded generator for perturbation noise.
            epochs: Number of optimisation epochs.
            callbacks: Optional list of callables invoked after each epoch
                (not yet forwarded to the inner loop – reserved for future use).
            reference_padded: Optional clean reference channels (diagnostics
                only; never enters the loss).
            loss_fn: Optional custom loss callable.

        Returns:
            Dict from :func:`~src.training.deep_prior.optimize_deep_prior`.
        """
        opt_cfg = self._cfg["optimization"]
        lr = float(opt_cfg["learning_rate"])
        perturb_var = float(opt_cfg["perturbation_variance"])
        log_every = int(self._cfg.get("pipeline", {}).get("log_every", 50))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        include_state = bool(
            self._cfg.get("pipeline", {}).get("save_checkpoint", False)
        )
        return optimize_deep_prior(
            model=model,
            noise=noise,
            observed=observed_padded.to(self._device),
            mask=mask_padded.to(self._device),
            optimizer=optimizer,
            epochs=epochs,
            perturbation_variance=perturb_var,
            log_every=log_every,
            reference=reference_padded,
            generator=generator,
            include_model_state=include_state,
            loss_fn=loss_fn,
        )

    def _decode(
        self,
        final_output: torch.Tensor,
        original_shape: Tuple[int, int],
        signal_length: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Crop, invert STFT, and return waveform + final spectrogram.

        Args:
            final_output: Padded model output ``(2, M', L')``.
            original_shape: ``(M, L)`` before padding.
            signal_length: Original number of waveform samples ``N``.

        Returns:
            Tuple ``(waveform_np, spectrogram_complex)`` where *waveform_np*
            is float32 ndarray ``(N,)`` and *spectrogram_complex* is complex
            STFT ``(M, L)``.
        """
        sp = self._stft_params()
        out_ch = crop_to_shape(final_output.cpu(), original_shape)
        stft_out = channels_to_stft(out_ch)
        wav = inverse_stft(stft_out, **sp, length=signal_length)
        return wav.numpy(), stft_out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inpaint(
        self,
        corrupted: np.ndarray,
        sample_mask: np.ndarray,
        reference: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        seed: int = 0,
        callbacks: Optional[List[Callable]] = None,
    ) -> InpaintResult:
        """Inpaint one corrupted audio clip using the deep-prior method.

        The *reference* is used only for the no-grad missing-region diagnostic
        inside the optimisation loop.  It never enters the training loss.

        Args:
            corrupted: Corrupted waveform ``(N,)`` as float32 ndarray.
            sample_mask: Sample-level binary mask ``(N,)`` with 1=observed,
                0=missing.
            reference: Optional clean reference waveform ``(N,)`` for the
                missing-region diagnostic.  Detached before use.
            epochs: Number of optimisation epochs.  Defaults to
                ``optimization.epochs`` from the config.
            seed: Integer seed controlling model init, input noise, and
                perturbation noise sequence.
            callbacks: Reserved for future use (not yet forwarded).

        Returns:
            :class:`InpaintResult` containing the reconstructed waveform and
            all per-run metadata.
        """
        opt_cfg = self._cfg["optimization"]
        if epochs is None:
            epochs = int(opt_cfg["epochs"])

        N = len(corrupted)
        t0 = time.perf_counter()

        # a. STFT + channel split
        observed_ch, _ = self._compute_observed_channels(corrupted)

        # b. Frame mask + TF mask
        num_bins = observed_ch.shape[-2]
        fmask_vec, tfmask = self._build_tf_mask(sample_mask, num_bins)

        # c. Pad both to multiple of 32
        observed_padded, mask_padded, original_shape = self._pad(observed_ch, tfmask)

        # d. Input noise
        noise, generator = self._make_noise(tuple(observed_padded.shape), seed)

        # e. Fresh model + Adam
        model, n_params = self._build_model(seed)

        # Prepare reference channels (diagnostics only)
        reference_padded: Optional[torch.Tensor] = None
        if reference is not None:
            ref_wav = torch.from_numpy(np.asarray(reference, dtype=np.float32))
            ref_stft = compute_stft(ref_wav, **self._stft_params())
            ref_ch = stft_to_channels(ref_stft)
            ref_ch_padded, _ = pad_to_multiple(ref_ch)
            reference_padded = ref_ch_padded.detach().to(self._device)

        # Corrupted waveform for MSS reference
        corrupted_wav = torch.from_numpy(np.asarray(corrupted, dtype=np.float32)).detach()

        # f. Build loss function
        loss_fn = build_loss(
            self._cfg,
            stft_cfg=self._stft_params(),
            corrupted_wav=corrupted_wav,
            sample_mask=torch.from_numpy(np.asarray(sample_mask, dtype=np.float32)).detach().to(self._device),
            signal_length=N,
        )

        # g. Optimise
        dp_result = self._run_optimisation(
            model=model,
            noise=noise,
            observed_padded=observed_padded,
            mask_padded=mask_padded,
            generator=generator,
            epochs=epochs,
            callbacks=callbacks,
            reference_padded=reference_padded,
            loss_fn=loss_fn,
        )

        # h. Decode: crop -> iSTFT -> waveform
        waveform_np, final_stft = self._decode(
            dp_result["final_output"], original_shape, N
        )

        # Optional: preserve observed samples
        preserve = bool(self._cfg.get("pipeline", {}).get("preserve_observed", False))
        if preserve:
            obs_np = np.asarray(corrupted, dtype=np.float32)
            m_np = np.asarray(sample_mask, dtype=np.float32)
            waveform_np = waveform_np * (1.0 - m_np) + obs_np * m_np

        runtime_s = time.perf_counter() - t0

        return InpaintResult(
            waveform=waveform_np.astype(np.float32),
            loss_history=dp_result["loss_history"],
            val_loss_history=dp_result["val_loss_history"],
            final_spectrogram=final_stft,
            runtime_s=runtime_s,
            n_params=n_params,
            seed=seed,
            epochs=epochs,
        )
