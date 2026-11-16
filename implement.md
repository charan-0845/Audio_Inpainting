TASK: Add long-audio inpainting (LPC pre-fill + windowed DPAI + merge) to the Audio Inpainting Research Platform.

READ FIRST
- implementation.md (follow the OpenCode execution protocol, non-negotiable constraints and progress log).
- Inspect before changing: src/web/server.py, src/pipeline/inpainter.py, src/pipeline/io.py, src/audio/lpc_baseline.py, src/audio/corruption.py, frontend/src/pages/WorkspacePage.tsx, frontend/src/types/api.ts.
- Run the existing test suite first and record the baseline.

HARD CONSTRAINTS
1. Do NOT modify the ML core: DPAIInpainter.inpaint, MultiResUNet, HarmonicConv2d, STFT/frame_mask, losses, deep_prior, nmse. The new feature is a wrapper that calls DPAIInpainter.inpaint unchanged.
2. Do NOT change existing API contracts (POST /api/jobs, SSE event names/payloads, file whitelist). Extend only with optional fields. Clips <= max_seconds must behave exactly as today.
3. A single-window case must reproduce DPAIInpainter.inpaint output exactly (same seed, same inputs). Test this.
4. No fake metrics. Report only what is computed. Label this as an extension beyond the paper in docs.
5. Keep LPC-filled regions honest: record which gaps were LPC-filled and which were DPAI-filled in run metadata.

ALGORITHM (src/pipeline/long_inpaint.py)
Inputs: corrupted waveform (N,), sample mask (N,), cfg, epochs, seed, options.
Options (all in cfg under `long_audio:` with defaults, and overridable per job):
  enabled: auto  (auto = use when duration > window_s)
  window_s: 5.0
  context_s: 0.5            (min context kept on each side of a window's gaps)
  lpc_max_gap_ms: 40        (gaps <= this are LPC candidates)
  lpc_skip_if_in_window: true
  crossfade_ms: 8
Steps:
 1. mask_to_gaps(mask) -> sorted (start, end) sample intervals.
 2. Split: short = gaps with length <= lpc_max_gap_ms; long = the rest.
 3. If there are no long gaps: LPC-fill everything and finish (no DPAI).
 4. Plan windows from LONG gaps only, using the optimal greedy cover: sweep left to right, start a new window at the first uncovered long gap, add gaps while (last_end - first_start + 2*context) <= window length. Centre the group in a fixed-length window; clamp at clip edges by shifting inward (never shrink; if clip < window, use one window of the full clip). Fixed window length keeps STFT shapes identical across windows.
 5. If lpc_skip_if_in_window: short gaps that fall fully inside any planned window are NOT LPC-filled; they stay masked and DPAI fills them. Remaining short gaps are filled with lpc_inpaint (use existing function with gaps=..., seed=seed). Those regions become observed (mask=1) for DPAI and are tracked as "lpc" regions.
 6. For each window k: call DPAIInpainter.inpaint(corrupted_w, mask_w, reference=clean_w if provided, epochs=epochs, seed=seed + k, callbacks=[...]).
 7. Merge: start from the LPC-filled signal; for every gap that DPAI handled, write the DPAI samples into that gap only, with an equal-power (cos/sin) crossfade of crossfade_ms at gap boundaries. Observed samples stay bit-exact. If a gap appears in two windows (should not happen), prefer the window where it is more central.
 8. Return a LongInpaintResult: waveform, per-window loss histories, windows [{start, end, n_gaps}], lpc_gaps, dpai_gaps, n_params, total runtime, seed, epochs.
Guards: reject if total missing > 40% (existing rule); if a window would have > 40% missing, split that group; warn (not fail) if any gap > 100 ms (existing warning).

BACKEND INTEGRATION (src/web/server.py, src/pipeline/io.py)
- Raise/remove the hard max_seconds cap for the long path only: add `max_seconds_long` (default 120) in create_app and `app.py serve --max_seconds_long`. Clips <= max_seconds use the existing path untouched.
- In JobManager worker, if the clip qualifies, call inpaint_long instead of inpainter.inpaint. Keep LPC-preset behaviour unchanged.
- Progress/SSE: keep existing event shapes. Report overall progress as (window_idx + epoch/epochs)/n_windows by mapping into the existing `epoch`/`epochs` fields as a scaled total, and add OPTIONAL fields: window_index, n_windows, stage ("lpc" | "dpai" | "merge"). Check cancel_event between windows and inside the callback (existing JobCancelledException).
- Live preview spectrogram: render the current window only and say so in the payload (optional field `preview_window`).
- save_run: also write long_audio.json into the run dir (windows, lpc_gaps, dpai_gaps, params used) and add it to WHITELIST_FILES. Include the long_audio config in config_snapshot.yaml. Metrics: compute nmse_tot_db and nmse_miss_db on the full clip as today; add a note in metrics.json that nmse_tot is dominated by observed audio for long clips. PESQ unchanged.
- Add optional form fields to POST /api/jobs: long_audio (auto|on|off), window_s, lpc_max_gap_ms. Validate ranges (window_s 3-10, lpc_max_gap_ms 0-80, 0 disables LPC).
- /api/presets or /api/health: expose max_seconds_long and long_audio defaults only if truthfully available.

FRONTEND (frontend/)
- WorkspacePage: lift the duration check to max_seconds_long; show a "Long audio" panel in advanced settings (mode, window length, LPC threshold) with honest defaults read from the backend.
- Before submit, show a plan summary computed from the same rules: number of windows, number of gaps LPC-filled vs DPAI-filled, and estimated runtime clearly labelled as an estimate. Keep this logic in frontend/src/utils/longAudioPlan.ts with unit tests, and note that the backend plan is authoritative.
- Progress UI: show "Window k of n" and stage when those fields exist; fall back to current UI otherwise.
- Results page: if long_audio.json exists, show a Windows section (timeline bar with window extents, LPC vs DPAI region legend, not colour alone) and add it to Configuration tab. Missing file = section hidden.
- Update frontend/src/types/api.ts with the optional fields.

TESTS (must pass, add to existing suites)
- tests/test_long_inpaint.py:
  * window planner: minimal window count on hand-made gap lists; context margin respected; clamping at both clip edges; clip shorter than window; dense group splitting.
  * LPC-skip rule: short gap inside a window is not LPC-filled; isolated short gap is.
  * merge: observed samples bit-exact; only gap samples change; crossfade has no discontinuity at boundaries.
  * equivalence: for a clip <= window_s, inpaint_long output equals DPAIInpainter.inpaint output (same seed, 5 epochs, smoke config).
  * determinism: same seed -> identical result; per-window seeds differ.
  * cancellation between windows.
- tests/test_web.py: job with a 20 s synthetic clip using smoke config (few epochs) completes; SSE contains window_index/n_windows; long_audio.json produced; a <= 10 s job produces events identical in shape to today; over-limit clip rejected with clear 400.
- frontend: unit tests for longAudioPlan.ts.
- Run: pytest, npm run typecheck, npm run lint, npm run build. All must pass; no ML regressions.

EVALUATION SCRIPT (research use, not part of the web path)
- scripts/eval_long_audio.py: build 30-60 s test clips by concatenating benchmark clips of one type, apply the benchmark gap rule scaled to length, and compare (A) windowed DPAI only, (B) LPC + windowed DPAI, (C) LPC + windowed DPAI with skip-if-in-window. Write experiments/long_audio/results.csv with: variant, clip, level_ms, seed, epochs, n_windows, nmse_tot_db, nmse_miss_db, runtime_s, git, note. Support --dry_run.
- scripts/sweep_lpc_threshold.py: run lpc_inpaint alone on single gaps of 10/20/30/40/50/60 ms over the benchmark clips and write nmse_miss per length and clip type, so lpc_max_gap_ms is chosen from data.

DOCS
- docs/ARCHITECTURE.md: new section "Long-audio extension (not in the paper)" describing the algorithm, defaults, and that LPC-filled regions are treated as observed by DPAI.
- README: usage and the new flags. implementation.md: add a Phase 11 checklist and progress-log entry.

WORKFLOW
Work in this order, committing and logging after each: (1) planner + tests, (2) lpc/merge + tests, (3) inpaint_long + equivalence test, (4) server integration + tests, (5) frontend, (6) scripts, (7) docs. Mark checkboxes only after tests pass. If anything in the existing contracts blocks this, stop and record BLOCKED with evidence instead of changing the contract.

STATUS & PROGRESS:
- [x] (1) planner + tests (tests/test_long_inpaint.py passed)
- [x] (2) lpc/merge + tests (tests/test_long_inpaint.py passed)
- [x] (3) inpaint_long + equivalence test (tests/test_long_inpaint.py passed)
- [x] (4) server integration + tests (tests/test_web.py passed)
- [x] (5) frontend (vitest, typecheck, lint, build passed)
- [ ] (6) scripts
- [ ] (7) docs
