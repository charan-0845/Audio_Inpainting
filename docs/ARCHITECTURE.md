# DPAI Architecture Notes

Short reference covering design decisions, deviations from the paper,
and open items.

---

## 1. Normalization: GroupNorm vs. BatchNorm

**Paper**: The original DPAI paper uses **Batch Normalization** (BatchNorm).

**This repo**: Uses `GroupNorm(1, C)` (also known as *Instance Norm* when
`num_groups=1`) in every `MultiResBlock`, `ResPath`, and encoder stage.

**Why**: The deep-prior method operates on a single spectrogram at a time
(effective batch size = 1).  BatchNorm statistics are undefined for batch
size 1; GroupNorm(1, C) is the standard substitute that gives equivalent
behaviour without batch-size constraints.

**Impact**: This is a documented deviation.  The paper's reported results
use BatchNorm, so numerical scores may differ slightly when reproduced here.

---

## 2. Parameter Count & Channel Schedule

The paper reports **2,015,252** parameters for MultiResUNet and **2,158,578** for PlainUNet.
A traditional U-Net with channel doubling per level vastly overshoots this target if scaled to match the "21 / 43 / 64" filter statement (which implies `out_channels=128` at level 0).

**Our Implementation**: We use a flat `channel_schedule` without doubling across all 5 levels.
- `channel_schedule=[68, 68, 68, 68, 68]` produces `2,013,662` parameters for MultiResUNet (error of just ~0.08%).
- The equivalent PlainUNet yields `2,127,042` parameters.
This flat schedule closely matches the paper's target sizes and resolves the ambiguity about channel doubling.

---

The decoder convolutions in `MultiResUNet` use regular `nn.Conv2d`, **not**
`HarmonicConv2d`, regardless of the `use_harmonic` flag.  This is an
intentional design choice.

However, the **bottleneck** block (the lowest level) uses `MultiResBlock` with `use_harmonic=True` because it acts as the final stage of the encoder path.

---

## 4. Harmonic Anchors

Table I of the paper lists per-block anchor frequencies, but the table is
partially illegible in the source PDF. Legible values (Res Path residual and downsampling convs) are set to 1. For the illegible MultiRes and Res Path convolutions, we use placeholder assumptions (e.g. 1, 2, 3 for conv1, conv2, conv3) that are explicitly separated in `config.yaml` to prevent confusion with actual paper values.

Out-of-bounds frequencies are gathered dynamically. Instead of clamping out-of-bounds indices, they are masked out (zero contribution) using the `out_of_bounds_strategy="zeros"` default, avoiding phantom harmonic artifacts at the spectral edges.

---

## 5. Multi-Scale Spectrogram Loss (MSS)

Enabled by default (`loss.use_mss: true`). The three STFT scales follow Parallel WaveGAN (verified against the standard implementation):

| Scale   | n_fft | win_length | hop_length |
|---------|-------|------------|------------|
| Fine    | 512   | 240        | 50         |
| Medium  | 1024  | 600        | 120        |
| Coarse  | 2048  | 1200       | 240        |

Each scale contributes: spectral convergence + log-STFT magnitude + linear
STFT magnitude + phase consistency. The loss uses the reconstructed corrupted waveform to strictly avoid clean-signal leakage into the optimisation loop. Masking is accurately recomputed per scale.


