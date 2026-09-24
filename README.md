# Audio Inpainting

A from-scratch research implementation of **Deep Prior-Based Audio Inpainting Using Multi-Resolution Harmonic Convolutional Neural Networks (DPAI)**.

The project aims to reproduce the core method described in the reference paper and provide a clean foundation for future research extensions.

## Current goal

Rebuild the pipeline step by step:

1. Audio loading and preprocessing
2. Artificial audio corruption / gap generation
3. STFT and inverse STFT
4. Time-frequency masking
5. MultiResUNet baseline
6. Deep-prior optimization
7. Harmonic convolution
8. Multi-scale spectrogram loss
9. Audio reconstruction
10. NMSE / PESQ evaluation
11. Research extensions

## Project structure

```text
audio-inpainting/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── clean/
│   ├── corrupted/
│   └── masks/
├── experiments/
│   ├── baseline/
│   ├── multires/
│   └── harmonic/
├── notebooks/
├── outputs/
│   ├── audio/
│   ├── spectrograms/
│   └── checkpoints/
├── src/
│   ├── audio/
│   ├── evaluation/
│   ├── losses/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

## Reference

Federico Miotello et al., *Deep Prior-Based Audio Inpainting Using Multi-Resolution Harmonic Convolutional Neural Networks*, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.

The initial reproduction target uses the paper's described deep-prior formulation, MultiResUNet architecture, harmonic convolution, masked reconstruction loss, and experimental setup.

## Development philosophy

This repository intentionally starts as a skeleton. Components will be implemented and tested independently before combining them into the complete inpainting system.

## Status

🚧 Research implementation — under development.
