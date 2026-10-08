# Audio Inpainting (DPAI)

A from-scratch research implementation of **Deep Prior-Based Audio Inpainting
Using Multi-Resolution Harmonic Convolutional Neural Networks (DPAI)**.

Federico Miotello et al., *IEEE/ACM TASLP 2024*.

---

## Running the application

### 1. Inpaint a file

```bash
# With a synthetic 200 ms gap (no audio file needed), 10-epoch smoke run:
python app.py inpaint --gap_ms 200 --epochs 10 --config configs/smoke.yaml

# Real run (5 000 epochs, auto-selects CUDA if available):
python app.py inpaint --audio data/clean/piano_01.wav --gap_ms 400

# Provide a pre-computed mask:
python app.py inpaint --audio signal.wav --mask signal_mask.npy --epochs 5000

# Auto-detect gaps from near-zero regions:
python app.py inpaint --audio corrupted.wav --detect_mask
```

### 2. Run the benchmark suite

```bash
# Quick tier, seeds 0 and 1:
python app.py benchmark --tier quick --seeds 0 1 --epochs 5000

# Full tier, speech clips only, resumable (worker a):
python app.py benchmark --tier full --only speech --worker a --epochs 5000

# Dry-run (plumbing test, no training):
python app.py benchmark --tier quick --limit 3 --dry_run
```

### 3. Summarize results

```bash
# Merge all worker CSVs and print NMSE table + save a plot:
python app.py summarize --csv experiments/multires_harmonic/results*.csv
```

---

## Run-directory layout

Every `inpaint` run creates a timestamped directory:

```
outputs/runs/<clip>_<gap_ms>ms_s<seed>_<YYYYmmdd-HHMMSS>/
├── config_snapshot.yaml   # fully resolved config + CLI overrides
├── meta.json              # git hash, torch version, device, seed, epochs, n_params, runtime
├── reconstruction.wav     # reconstructed waveform (16-bit PCM)
├── corrupted.wav          # corrupted input
├── clean.wav              # clean reference (if provided)
├── metrics.json           # nmse_tot_db, nmse_miss_db, pesq_delta
├── loss_curve.png         # training + diagnostic loss curves
├── spec_corrupted.png     # spectrogram of corrupted input
├── spec_reconstructed.png # spectrogram of reconstruction
└── spec_clean.png         # spectrogram of clean signal (if provided)
```

An optional `checkpoint.pt` (model state dict) is saved when
`pipeline.save_checkpoint: true` is set in the config.

---

## Configuration

All hyperparameters are controlled via `configs/config.yaml`.  For fast
CI/smoke runs use `configs/smoke.yaml` (10 epochs, smaller model).

Key config sections added in this release:

| Key | Default | Description |
|-----|---------|-------------|
| `model.use_harmonic` | `true` | Enable HarmonicConv2d in encoder |
| `loss.use_mss` | `false` | Enable multi-scale spectrogram loss |
| `pipeline.preserve_observed` | `false` | Paste back observed samples post-inpaint |
| `pipeline.save_checkpoint` | `false` | Save model state dict as checkpoint.pt |
| `pipeline.log_every` | `50` | Missing-region diagnostic frequency (epochs) |
| `pipeline.variant` | `multires_harmonic` | Label for experiments/ CSV |
| `run.out_dir` | `outputs/runs` | Parent directory for run directories |

---

## Deprecated scripts

The following scripts are superseded by `app.py` and kept for reference only:

| Script | Replacement |
|--------|-------------|
| `scripts/run_deep_prior.py` | `python app.py inpaint` |
| `scripts/compare_architectures.py` | `python app.py inpaint` |
| `scripts/compare_with_harmonic.py` | `python app.py inpaint` |
| `scripts/run_benchmark.py` | `python app.py benchmark` |

---

## Project structure

```
audio-inpainting/
├── app.py                  # main CLI (inpaint / benchmark / summarize)
├── configs/
│   ├── config.yaml         # all hyperparameters (frozen interface)
│   └── smoke.yaml          # CI / fast-test overrides
├── docs/
│   └── ARCHITECTURE.md     # filter-count analysis, deviations, open items
├── src/
│   ├── audio/              # loader, STFT, corruption, benchmark
│   ├── evaluation/         # nmse, pesq
│   ├── losses/             # masked_mse, multiscale_loss, total_loss
│   ├── models/             # MultiResUNet, MultiResBlock, ResPath, HarmonicConv2d
│   ├── pipeline/           # inpainter, io, metrics  ← new
│   ├── training/           # deep_prior, noise
│   └── utils/              # visualization
├── tests/
│   ├── test_pipeline.py    # DPAIInpainter + CLI smoke tests  ← new
│   ├── test_multiscale_loss.py  ← new
│   └── ...                 # existing tests (unchanged)
├── benchmark/
│   └── manifest.json       # frozen benchmark manifest
└── experiments/            # CSV results per variant
```

---

## Reference

Federico Miotello et al., *Deep Prior-Based Audio Inpainting Using
Multi-Resolution Harmonic Convolutional Neural Networks*,
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.

---

## Status

🚧 Research implementation — under active development.
See `docs/ARCHITECTURE.md` for known deviations from the paper.
