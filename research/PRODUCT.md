# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are audio post-production users restoring audio clips.

Open: exact roles, team context, skill level, and whether restoration is performed individually or in batches.

## Product Purpose

The product supports restoration of clips with missing or corrupted audio through audio inpainting.

Open: the acceptable restoration-quality threshold, expected turnaround, and required output formats are not yet defined.

## Positioning

Open: the product's distinction from other audio-restoration tools has not been confirmed. The current implementation describes itself as a research implementation of Deep Prior-Based Audio Inpainting (DPAI); keep its documented limitations visible and do not imply broader validation.

## Operating Context

Repository-observed workflow: the dashboard summarizes experiment runs and links to a new reconstruction workspace. The workspace accepts audio, lets users mark gaps manually, generate random gaps, or detect silence, then runs a reconstruction and presents progress and results.

## Capabilities and Constraints

- Repository-observed implementation uses clip-specific deep-prior optimization and provides audio, waveform, spectrogram, and metric views.
- The server limits total missing audio to less than 40% and warns that gaps over 100 ms may degrade quality because the paper's method is designed for gaps of 100 ms or less.
- The interface documents deviations from the paper, including GroupNorm in place of BatchNorm, a flat channel schedule, and harmonic-anchor assumptions. The user confirmed these caveats should be preserved and explained.
- Current precomputed examples have gaps of approximately 929-1,743 ms and missing seed values. The user confirmed these should be explained rather than hidden.
- Open: whether the current NMSE/PESQ metrics and research-oriented run terminology meet post-production users' needs.

## Evidence on Hand

- The repository includes the cited DPAI research paper reference and architecture notes.
- Run artifacts can include reconstructed, corrupted, and optional clean-reference audio, metrics, spectrograms, configuration snapshots, and run metadata.
- Precomputed example audio, spectrograms, and metrics are under `src/web/static/examples/`; the examples index records the current gap durations and null seeds.
