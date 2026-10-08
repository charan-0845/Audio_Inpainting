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

## 2. Filter Counts per MultiRes Block

The paper (Table II) reports filter counts **21 / 43 / 64** per MultiRes
block at the first encoder level.  The relationship is:

    W  = alpha * out_channels = 1.6 * base_filters * 2^i
    w1 = round(W * 1/6)
    w2 = round(W * 2/6)
    w3 = round(W * 3/6)
    actual_out = w1 + w2 + w3

With `base_filters=7` and `alpha=1.6` the values per encoder level are:

| Level | Paper out | W    | w1 | w2 | w3 | actual_out |
|-------|-----------|------|----|----|----|------------|
| 0     | 128       | 11.2 |  2 |  4 |  6 |         12 |
| 1     | 256       | 22.4 |  4 |  7 | 11 |         22 |
| 2     | 512       | 44.8 |  7 | 15 | 22 |         44 |
| 3     | 1024      | 89.6 | 15 | 30 | 45 |         90 |
| 4     | 2048      | 179.2| 30 | 60 | 90 |        180 |

> **Mismatch detected**: The paper's Table II reports `21/43/64` for the
> first encoder block, suggesting `alpha*base_filters ≈ 21+43+64 = 128` i.e.
> `base_filters ≈ 128/alpha = 80`.  However `base_filters=7` with
> `alpha=1.6` gives `W=11.2` per block – far smaller.  The filter table above
> reflects the actual implementation.  We **do not tweak** these values.
>
> The paper's 2,015,252-parameter claim would require `base_filters ≈ 80` or
> a different `alpha`.  The `meta.json` file in every run directory records
> the actual parameter count for auditing.

---

## 3. Decoder Convolutions

The decoder convolutions in `MultiResUNet` use regular `nn.Conv2d`, **not**
`HarmonicConv2d`, regardless of the `use_harmonic` flag.  This is an
intentional design choice (harmonic structure is only meaningful along the
encoder's frequency axis; after upsampling the frequency-harmonic
relationship is no longer well-defined).

---

## 4. Harmonic Anchors (Open Item)

Table I of the paper lists per-block anchor frequencies, but the table is
only partially legible in the source PDF.  The current config uses
`anchor=1` everywhere, meaning the harmonic taps start at the fundamental
frequency bin.

To configure per-block anchors, pass a dict to `MultiResUNet` via
`harmonic_anchors` in `configs/config.yaml`:

```yaml
model:
  harmonic_anchors:
    default: 1            # used for all blocks unless overridden
    res_path_residual: 1
    downsampling_conv1: 1
```

A future extension could accept a list of anchors (one per encoder level).
This is tracked as an **open item**.

---

## 5. Multi-Scale Spectrogram Loss (MSS)

Disabled by default (`loss.use_mss: false`).  When enabled, the three
STFT scales are (Table II, 16 kHz):

| Scale   | n_fft | win_length | hop_length |
|---------|-------|------------|------------|
| Fine    | 256   | 240        | 60         |
| Medium  | 1024  | 600        | 120        |
| Coarse  | 2048  | 1200       | 240        |

Each scale contributes: spectral convergence + log-STFT magnitude + linear
STFT magnitude + phase consistency (eq. 14).

**Status**: Implemented and unit-tested but not yet validated against
held-out data.  Set `loss.use_mss: true` and `loss.alpha_mss: 0.1` to
enable.

---

## 6. Parameter Count

The actual parameter count is written to `meta.json` in every run
directory.  To inspect it before running:

```python
import yaml
from src.models.multires_unet import MultiResUNet

cfg = yaml.safe_load(open("configs/config.yaml"))["model"]
model = MultiResUNet(
    use_harmonic=cfg["use_harmonic"],
    harmonic_anchors=cfg.get("harmonic_anchors"),
    base_filters=cfg.get("base_filters", 7),
)
print(f"Parameters: {model.count_parameters():,}")
```

The paper reports **2,015,252** parameters.  This implementation with
`base_filters=7` produces a smaller model; see §2 for the filter-count
analysis.
