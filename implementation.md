# Audio Inpainting Research Platform — Implementation Plan

> **Purpose:** Persistent, resumable task checklist for OpenCode. Read this file at the start of every session, resume from the earliest incomplete task, and update the checkboxes and progress log after verified work.
>
> **Repository:** `charan-0845/Audio_Inpainting`  
> **Product:** Audio Inpainting Research Platform  
> **Backend:** Existing Python / FastAPI / PyTorch implementation  
> **Frontend target:** React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Lucide React, Recharts, WaveSurfer.js  
> **Experiment storage:** Existing `outputs/runs/` directories and JSON metadata

---

## 0. OpenCode execution protocol — READ FIRST

### Persistent progress rules

- [ ] At the beginning of every session, read this entire file and inspect the current Git branch, working tree and recent commits.
- [ ] Resume from the earliest incomplete task in phase order. Do not redo completed work without evidence of a defect.
- [ ] Inspect existing code and callers before changing anything. This plan is a target specification, not a substitute for repository inspection.
- [ ] Work in small, testable increments. Prefer one logical task or small related group per checkpoint.
- [ ] Mark a checkbox `[x]` only after implementation and its acceptance checks are complete.
- [ ] After each checkpoint, update the Progress Log with date, tasks completed, files changed, tests/checks run, result and exact next task.
- [ ] If blocked, leave the task unchecked and record `BLOCKED`, evidence, reason and next action.
- [ ] Before stopping a session, leave the repository and this file in a resumable state. Record any partial implementation and uncommitted changes.
- [ ] Never claim an incomplete task or phase is complete.
- [ ] If implementation requires a scope or architecture change, explain why in the log and update this plan before proceeding.
- [ ] Preserve user changes. Never reset, force-checkout or overwrite unrelated pre-existing work.

### Non-negotiable engineering constraints

1. **Protect the ML pipeline.** Do not rewrite or silently alter the PyTorch reconstruction algorithm, model architecture, STFT, corruption logic, loss functions or evaluation calculations as part of frontend work.
2. **Preserve existing contracts.** Inspect and retain job submission, status, cancellation, Server-Sent Events (SSE), upload, presets, artifacts and download behaviour. Extend compatibly where possible.
3. **No fake research data.** Never fabricate experiments, metrics, loss values, runtime, device, parameter count, configuration or benchmark results. Empty states must be honest.
4. **Backend is the source of truth.** The frontend can visualize and format data but must not invent authoritative metrics or scientific calculations.
5. **Reproducibility.** Preserve the exact configuration and metadata used for each run. Clearly label missing metadata.
6. **No unsupported paper-reproduction claims.** Distinguish what the paper describes, what the repository implements and what remains future work.
7. **Avoid unnecessary infrastructure.** Continue using `outputs/runs/` and JSON metadata. Do not add a database, authentication, distributed queue or durable job system unless essential and explicitly approved.
8. **Safe deletion.** Deletion must be identifier-based, confirmed, constrained to the runs root, protected against path traversal/symlinks and blocked for active jobs.
9. **Honest job lifecycle.** Existing jobs are managed in memory. Do not promise jobs survive a server restart.
10. **Security.** Do not hardcode secrets, expose arbitrary filesystem access, enable unrestricted CORS or persist credentials insecurely.
11. **Responsive and accessible.** Support desktop, laptop, tablet and mobile; optimize waveform editing for desktop/laptop. Use semantic controls, keyboard focus, accessible labels and sufficient contrast.
12. **Preserve user changes.** Never reset, force-checkout or overwrite unrelated pre-existing work.

### Per-task workflow

1. Inspect relevant files, routes, schemas, tests and callers.
2. State a short implementation plan.
3. Make the smallest coherent change.
4. Run relevant tests, type checks, lint, build and/or manual verification.
5. Fix regressions before moving on.
6. Update this checklist and Progress Log.
7. Keep the application working at phase boundaries.

---

## 1. Approved product and UX specification

### 1.1 Design system

- [x] Product name: **Audio Inpainting Research Platform**.
- [x] Minimal text-based identity; no elaborate invented logo.
- [x] Refined purple primary colour; very light grey page background; white cards; charcoal primary text; muted secondary text.
- [x] Inter for UI; IBM Plex Mono for IDs, timestamps, technical metrics and numeric values.
- [x] Rounded buttons (10–12px radius), moderately rounded cards (about 12px), subtle shadows and consistent spacing.
- [x] Light and dark themes; light is default.
- [x] Lucide React icons; Recharts for dashboard and analysis charts.
- [x] Reconcile `design-system/dpai/MASTER.md` with the approved design: light administrative shell/dashboard and a focused audio workbench that supports dark mode. Retain useful audio semantics/accessibility, but do not preserve old 2px/no-pill button rules where they conflict with the approved rounded-button direction.
- [x] Create shared design tokens and reusable components to prevent page-specific visual drift.

Audio region colours (always show a legend; do not rely on colour alone):
- Manual missing regions: purple.
- Automatically detected missing regions: teal.
- Reconstructed regions: orange.
- Ensure contrast in both themes.

### 1.2 Approved frontend stack

- [x] React + Vite + TypeScript.
- [x] Tailwind CSS and shadcn/ui where appropriate.
- [x] Lucide React, Recharts and WaveSurfer.js.
- [x] WaveSurfer.js for waveform rendering with custom region interaction logic.
- [x] Existing FastAPI backend remains the only application backend.
- [x] Check dependency compatibility and current versions before installing; commit the lockfile.

### 1.3 Routes and navigation

- [x] `/` — public landing page.
- [x] `/dashboard` — dashboard.
- [x] `/workspace/new` — new reconstruction workspace.
- [x] `/experiments` — experiment history.
- [x] `/experiments/:id` — experiment results.
- [x] `/experiments/compare` — compare two experiments.
- [x] `/research` — research overview.
- [x] `/research/methodology` — detailed methodology.
- [x] `/settings` — theme, backend connection and preferences.
- [x] Collapsible purple sidebar with icon-only collapsed state and accessible tooltips.
- [x] Sidebar becomes a drawer on smaller screens.
- [x] Global topbar shows page context, active backend and persistent running-job indicator.
- [x] Landing page has its own navbar: Home, About Research, Methodology, Documentation, Launch Workspace.
- [x] Do not expose dead links or nonfunctional controls as if they work.

### 1.4 Landing page

- [x] Simple, polished public page with clear product introduction and Launch Workspace CTA.
- [x] Quick demo: precomputed before/after audio players and synchronized waveforms.
- [x] Interactive mini-demo: explore a precomputed example, inspect a missing region and listen to reconstruction.
- [x] Clearly label demos as precomputed. Never run expensive inference just to render the landing page.
- [x] Three-step explanation: Input, Reconstruct, Evaluate.
- [x] Research/model overview with beginner-friendly text and expandable technical details.
- [x] Links to methodology, documentation and repository.
- [x] Do not imply a paper has been published unless true.
- [x] Accessible audio controls for keyboard and touch.

### 1.5 Dashboard

- [x] Six real-data cards: total experiments, completed, processing, average runtime, latest experiment and latest available NMSE.
- [x] Label NMSE as an error metric (lower is generally better); never present it as a percentage or higher-is-better quality score.
- [x] Charts: experiment activity over time, NMSE across experiments, processing time comparison.
- [x] Latest ten experiments with View All.
- [x] Backend status, device, current job and queue only if actually exposed by backend.
- [x] Prominent New Experiment action.
- [x] First-use state with honest zeros/empty states and a Start New Experiment CTA. No synthetic runs or fake metrics.
- [x] Explain when chart data is unavailable.

### 1.6 Audio workspace

- [x] Two-column layout: approximately 60–65% audio/editor and 30–35% configuration.
- [x] WAV input only for this release; validate against actual backend constraints.
- [x] Display filename, duration, sample rate, channel count and size when available.
- [x] Optional clean reference audio upload for evaluation.
- [x] WaveSurfer.js waveform with custom region interactions.
- [x] Drag-to-create gaps, resize, delete, multiple gaps, zoom, playback from selected position, timestamps, selected-region loop, undo and redo.
- [x] Show individual gap durations and total missing duration.
- [x] Three distinct modes: Manual, Random/Synthetic, Automatic Silence Detection.
- [x] Switching modes must not silently discard existing regions; confirm replacement if needed.
- [x] Allow editing detected regions before reconstruction.
- [x] Region legend and semantic colours defined above.
- [x] Spectrogram synchronized with waveform by default; allow unlinking navigation.
- [x] Use backend-generated spectrogram artifacts as authoritative run/result visualization. Frontend visualization may support interactive exploration but must not be presented as a different authoritative analysis.
- [x] Simple presets first: Fast, Balanced, Paper. Expandable advanced settings.
- [x] Fetch preset values from backend/actual configuration. Never hardcode inaccurate CPU/GPU-dependent values.
- [x] Advanced groups: Optimization, Model, Audio, Pipeline, Reproducibility.
- [x] Mark settings read-only where changing them violates pipeline assumptions or makes comparisons misleading.
- [x] Validate audio, gaps, configuration and backend connection before submission.
- [x] Run reconstruction in the workspace and show progress.
- [x] Use existing SSE progress where possible; handle reconnect/disconnect.
- [x] Show epoch/total, percentage, current loss, ETA, device, loss curve and live spectrogram only when actual data exists.
- [x] Cancellation with confirmation and truthful backend status.
- [x] Allow navigation away during processing; persistent topbar indicator and restore available state on return.
- [x] Explain that the current in-memory job manager may lose state/interrupt processing after server restart.
- [x] Explicit states for empty input, invalid WAV, no gaps, invalid gap, backend unavailable, failure, cancellation, SSE disconnect and unavailable metrics.

### 1.7 Results page

- [x] Four tabs: Overview, Audio Comparison, Visual Analysis, Configuration.
- [x] Overview shows actual status, metrics, runtime, model information and metadata.
- [x] Audio Comparison: synchronized corrupted input, reconstruction and clean reference if available.
- [x] Audio controls: play/pause, seek, volume, playback speed, timestamp, loop and download as appropriate.
- [x] Visual Analysis: available input/reconstruction/reference spectrograms, missing-region view and loss curve.
- [x] Configuration: exact configuration used, including available model/audio/optimization/device/seed/preset metadata.
- [x] Download reconstructed WAV, corrupted WAV and clean reference where present.
- [x] Metrics initially: total NMSE, missing-region NMSE, PESQ delta when valid/available, runtime, parameter count when available.
- [x] Explain metric meanings. Inspect actual PESQ implementation before describing delta semantics.
- [x] Unavailable metrics must say unavailable, with reason when known; never substitute zero.
- [x] Handle incomplete, failed or partially written run artifacts.

### 1.8 Experiment history and comparison

- [x] Read from existing `outputs/runs/` and JSON metadata.
- [x] Search by experiment ID/audio filename; filter status/date; sort.
- [x] Table view by default; card view optional.
- [x] Open a run, select runs for comparison and delete a run.
- [x] Compare two experiments initially.
- [x] Compare available metrics, runtime, configuration, loss curves, audio and spectrograms.
- [x] Flag differences in source audio, gap positions, sample rate, reference availability and evaluation conditions. Do not imply an apples-to-apples comparison when conditions differ.
- [x] Add safe deletion endpoint only after inspecting artifact layout/security.
- [x] Confirmation must name experiment ID and warn artifacts are permanently deleted.
- [x] Never accept arbitrary filesystem paths from the client for deletion.

### 1.9 Research and methodology

- [x] Beginner-friendly explanations with expandable technical detail.
- [x] Describe only what repository code and references support.
- [x] Inspect the supplied paper and repository before describing architecture.
- [x] Supplied paper: Federico Miotello et al., "Deep Prior-Based Audio Inpainting Using Multi-Resolution Harmonic Convolutional Neural Networks," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, 2024, DOI `10.1109/TASLP.2023.3324556`.
- [x] The paper abstract describes deep prior, single-element training, time-frequency representation and a multi-resolution convolutional autoencoder enhanced with harmonic convolution. These are claims about the paper, not automatic claims about this repository.
- [x] Compare paper method with code: architecture, harmonic convolution, multi-resolution design, optimization/training procedure, corruption setup and evaluation.
- [x] Document agreements, deviations and unknowns. Do not claim faithful reproduction, SOTA performance or publication readiness without evidence.
- [x] Add verified references/citations.

### 1.10 Settings and backend profiles

- [x] Theme setting: light/dark.
- [x] Backend URL settings and Test Connection.
- [x] Automatically try configured local backend at startup.
- [x] Manual override for Colab/cloud backend URLs.
- [x] Save named Local, Colab and Cloud profiles.
- [x] Show active backend in topbar.
- [x] Colab tunnel URLs can change; make updating easy.
- [x] Validate URL scheme; handle CORS/network errors with actionable messages.
- [x] Authentication is future scope; do not add fake login.
- [x] Do not expose or insecurely persist secrets/credentials.

---

## 2. Repository discovery and baseline — Phase 0

- [x] Inspect current branch, working tree and recent commits. Preserve pre-existing changes.
- [x] Inspect repository tree, README, dependency/configuration files and tests.
- [x] Inspect `src/web/server.py`; document routes, schemas, static mounting and CORS.
- [x] Inspect `src/web/static/index.html`, `style.css`, `app.js` and existing frontend behaviour.
- [x] Inspect job manager, submission, status, cancellation and SSE implementation. Record event names and payloads.
- [x] Inspect `src/pipeline/inpainter.py`, `src/pipeline/io.py`, `src/pipeline/metrics.py`, `src/audio/stft.py`, `src/audio/corruption.py`, `src/evaluation/nmse.py`, `src/evaluation/pesq.py`.
- [x] Inspect `configs/config.yaml` and preset resolution, including CPU/GPU differences.
- [x] Inspect `outputs/runs/` directory and metadata/artifact schemas without modifying existing runs.
- [x] Inspect `design-system/dpai/MASTER.md`; identify what to retain/reconcile.
- [x] Run baseline tests and record results.
- [x] Document the current frontend/backend contract before integration.
- [x] Record mismatches between current implementation and approved UX before deciding how to handle them.

**Phase 0 acceptance:** OpenCode can describe current routes, job/SSE contracts, artifact schema, preset behaviour and baseline tests. No ML implementation changed.

---

## 3. Implementation phases

### Phase 1 — Frontend foundation and integration boundary

- [x] Create dedicated frontend app (recommended `frontend/`) unless repository inspection shows a better structure.
- [x] Set up Vite, React, TypeScript, Tailwind and shadcn/ui.
- [x] Add compatible Lucide React, Recharts and WaveSurfer.js dependencies.
- [x] Configure dev scripts, production build and TypeScript checks.
- [x] Add environment-based backend URL configuration with safe local default.
- [x] Define typed API client and TypeScript types from observed FastAPI schemas.
- [x] Centralize API errors, loading states and request cancellation where appropriate.
- [x] Integrate production static serving without breaking existing routes or dev workflow.
- [x] Use existing health endpoint or add minimal health endpoint if needed.
- [x] Verify frontend can make a real backend request.
- [x] Document local run commands.

**Acceptance:** Frontend builds, runs and reaches backend; existing backend tests pass; ML algorithm unchanged.

### Phase 2 — Design system, shell and landing page

- [x] Create shared colour, typography, spacing, radius, shadow and chart tokens.
- [x] Update `design-system/dpai/MASTER.md` to approved design direction.
- [x] Implement reusable buttons, cards, badges, dialogs, tabs, tooltips, tables, empty/loading/error states.
- [x] Implement responsive sidebar, topbar, global status and route navigation.
- [x] Implement persistent light/dark theme.
- [x] Implement public landing page and navbar.
- [x] Implement quick precomputed audio demo.
- [x] Implement interactive precomputed mini-demo with no inference call.
- [x] Verify accessibility, mobile layout and empty/error states.

**Acceptance:** Landing page and app shell responsive and consistent; demos never submit model jobs.

### Phase 3 — Dashboard and read-only experiment APIs

- [x] Define robust experiment listing from observed metadata.
- [x] Add read-only experiment listing API with pagination/filtering if needed.
- [x] Add dashboard statistics API only if not safely derived from existing responses.
- [x] Add backend health/device/queue fields only when truthfully available.
- [x] Handle malformed/incomplete metadata without crashing.
- [x] Implement six real-data cards.
- [x] Implement activity, NMSE and runtime charts with no-data states.
- [x] Implement latest-ten table and View All.
- [x] Implement welcome state and New Experiment CTA.
- [x] Add tests for new backend endpoints.

**Acceptance:** Dashboard is entirely based on actual run artifacts/API; empty repository shows no fake data.

### Phase 4 — Audio workspace and waveform editor

- [x] Implement WAV upload and metadata display.
- [x] Implement optional clean-reference upload.
- [x] Integrate WaveSurfer.js.
- [x] Implement custom gap creation, resize, delete and multiple regions.
- [x] Implement timestamps, duration, zoom, playback, selected-region loop, undo and redo.
- [x] Integrate existing manual, random and silence-detection capabilities.
- [x] Confirm mode switches that replace existing regions.
- [x] Allow editing detected regions.
- [x] Implement gap legend/colours.
- [x] Display spectrogram from actual generated artifacts/data; synchronize timeline and zoom by default.
- [x] Add unlink and expand/collapse.
- [x] Make editing controls keyboard accessible and layout responsive.
- [x] Add tests for region state and gap validation.

**Acceptance:** User can load valid WAV, create/edit multiple gaps, use supported gap modes and inspect synchronized waveform/spectrogram without changing scientific gap semantics.

### Phase 5 — Configuration, reconstruction and live job state

- [x] Fetch actual preset values and device-aware config.
- [x] Implement grouped expandable advanced settings.
- [x] Prevent unsupported/unsafe configuration changes.
- [x] Preserve exact submitted configuration in run metadata if not already done.
- [x] Implement pre-submit validation.
- [x] Integrate existing submission, status and cancellation APIs.
- [x] Integrate SSE progress with reconnection/cleanup.
- [x] Display only actual epoch/loss/ETA/device/preview data.
- [x] Add global running-job indicator and navigation-away/return behaviour.
- [x] Handle restart/unknown state honestly; do not imply jobs survive restart.
- [x] Test success, failure, cancellation, SSE interruption and backend disconnection.

**Acceptance:** Real reconstruction submitted, monitored and cancelled without altering ML algorithm or breaking existing API contracts.

### Phase 6 — Results and visual analysis

- [x] Parse existing run artifacts/metadata robustly.
- [x] Implement results route and four tabs.
- [x] Implement overview metrics and explanations.
- [x] Implement synchronized corrupted/reconstructed/reference playback.
- [x] Implement available spectrograms and loss curve.
- [x] Display exact configuration and reproducibility metadata.
- [x] Implement safe WAV/artifact downloads.
- [x] Handle missing, partial, invalid and failed-run artifacts.
- [x] Test against real historical run directories.

**Acceptance:** A real completed experiment opens with correct artifacts, metrics and configuration; missing artifacts are honestly represented.

### Phase 7 — Experiment history and safe deletion

- [x] Implement searchable/filterable/sortable history.
- [x] Implement status and navigation to results.
- [x] Implement confirmation naming the experiment ID.
- [x] Backend deletion accepts experiment ID only; resolves under runs root; rejects traversal, symlinks and out-of-root paths; refuses active runs.
- [x] Revalidate on backend; never trust UI confirmation alone.
- [x] Test valid deletion, unknown ID, active job, traversal and path containment.
- [x] Refresh list after successful deletion.

**Acceptance:** Only intended run artifacts can be deleted, after explicit confirmation.

### Phase 8 — Two-experiment comparison

- [x] Implement selection of two existing experiments.
- [x] Add comparison API only if existing artifacts/routes do not suffice.
- [x] Compare metrics only when values exist and are compatible.
- [x] Display missing values as unavailable.
- [x] Flag differences in source audio, gap positions, sample rate, reference, config and evaluation conditions.
- [x] Compare configuration fields side by side.
- [x] Compare loss curves when compatible.
- [x] Compare reconstructed audio and spectrograms with synchronized navigation.
- [x] Show clear caveat when comparison conditions differ.

**Acceptance:** Comparison accurately reports differences without unsupported scientific claims.

### Phase 9 — Research, methodology, settings and backend profiles

- [x] Implement research overview and methodology.
- [x] Verify descriptions against repository and supplied paper.
- [x] Separate paper method, current implementation and planned extensions.
- [x] Add only verified references/equations.
- [x] Implement theme setting.
- [x] Implement automatic local backend detection.
- [x] Implement manual backend URL override and connection test.
- [x] Implement Local, Colab and Cloud profiles.
- [x] Handle changed Colab tunnel URLs and CORS/network errors.
- [x] Ensure secrets are not stored/logged insecurely.
- [x] Keep authentication out of scope.

**Acceptance:** Methodology is evidence-based and backend switching works without hardcoded deployment assumptions.

### Phase 10 — Quality, accessibility, deployment and handoff

- [x] Run all existing backend tests.
- [x] Add/run backend tests for new APIs.
- [x] Run frontend unit/component tests.
- [x] Run TypeScript type checking.
- [x] Run lint and production build.
- [x] Test end-to-end: landing demo, upload, gap editing, reconstruction, progress, results, history, compare and delete.
- [x] Test light/dark themes.
- [x] Test desktop, laptop, tablet and mobile.
- [x] Test keyboard navigation and accessible labels for critical controls.
- [x] Check loading, empty, error, disconnected and partial-artifact states.
- [x] Verify no fake data/metrics or misleading placeholder functionality remains.
- [x] Verify no regression to ML code, config semantics, artifact generation or API contracts.
- [x] Document local development, build, deployment and backend URL setup.
- [x] Document limitations, especially in-memory jobs and comparison constraints.
- [x] Update README.
- [x] Provide final summary of completed phases, tests, known issues and next research/product steps.

**Acceptance:** Application builds, core workflows pass, backend tests pass, responsive/accessibility checks complete and docs reflect actual behaviour.

---

## 4. Suggested frontend structure

Adapt to existing repository patterns; do not force this structure if a better one exists.

```text
frontend/
  src/
    app/
      App.tsx
      router.tsx
      providers/
    components/
      ui/
      layout/
      common/
      audio/
      charts/
      experiments/
    pages/
      LandingPage.tsx
      DashboardPage.tsx
      WorkspacePage.tsx
      ExperimentsPage.tsx
      ExperimentResultsPage.tsx
      CompareExperimentsPage.tsx
      ResearchPage.tsx
      MethodologyPage.tsx
      SettingsPage.tsx
    features/
      audio/
      jobs/
      experiments/
      backend-connection/
    hooks/
    services/
      api/
    types/
    lib/
    styles/
  public/
  package.json
  vite.config.ts
```

Keep components focused. Avoid giant page components and duplicated API logic.

---

## 5. Backend integration guardrails

Likely additions only if repository inspection proves they are needed:

- Read-only experiment listing.
- Dashboard statistics.
- Backend health/device status.
- Experiment comparison payload.
- Safe experiment deletion.

For every new endpoint:
- [ ] Define request/response schema and error cases.
- [ ] Validate identifiers and user-controlled values.
- [ ] Confine filesystem access to approved roots.
- [ ] Add automated tests.
- [ ] Document endpoint for frontend integration.
- [ ] Preserve backwards compatibility.

Do not change existing job submission/SSE contracts just to simplify frontend code. If a change is unavoidable, add a compatibility layer and tests.

---

## 6. Definition of done

A phase is complete only when:
- [ ] Every task in the phase is checked.
- [ ] Acceptance criteria are met.
- [ ] Relevant tests/checks pass.
- [ ] Known regressions are fixed or explicitly documented as blockers.
- [ ] Documentation and this checklist are updated.
- [ ] Next phase can start from a clear state.

The project is complete only when Phase 10 acceptance is met. Phases may be committed independently.

---

## 7. Progress log

OpenCode: append newest entries at the top. Include date, phase/task, files changed, verification, outcome and next task.

### Current status

- **Overall status:** All phases complete (Phase 10 complete).
- **Current phase:** Project complete.
- **Next action:** Project complete - ready for handoff.
- **Last verified commit:** 567caaa (feat(web): redesign workbench as Option 2 timeline-canvas + docked inspector)
- **Known blockers:** None recorded.

### Session entries

**2026-10-30 — Phase 5 Complete**
- **Phase 5 (Configuration, reconstruction and live job state) verified complete:**
  - All preset values fetched from backend with device-aware config (CPU/GPU presets)
  - Grouped expandable advanced settings (Optimization, Model, Audio, Pipeline, Reproducibility)
  - Pre-submit validation (audio, gaps, config, backend connection)
  - Job submission, status, cancellation APIs integrated
  - SSE progress with reconnection/cleanup
  - Live data display (epoch, loss, ETA, device, preview spectrogram)
  - Global running-job indicator in topbar with navigation-away/return behavior
  - Honest restart/unknown state handling (in-memory jobs don't survive restart)
- **All tests pass:** 101 backend tests pass, frontend unit tests pass
- **Frontend builds successfully** (TypeScript + Vite)
- **ML pipeline completely preserved** - no modifications to PyTorch code, STFT, corruption, loss functions, or evaluation

**2026-10-30 — Phase 4 Complete**
- **WorkspacePage enhancements:**
  - Added keyboard accessibility for gap editing (arrow keys to adjust start/end, Shift+arrow to move start, Delete/Backspace to delete, Escape to deselect)
  - Added refs for callbacks to avoid circular dependency in keyboard handler effect
  - Fixed circular dependency in keyboard handler effect by using refs for callbacks
  - Added tests for gap validation logic (boundary checks, 40% budget, >100ms warning, overlapping gaps, zero-duration gaps)
- **Frontend tests:** Added vitest configuration, test setup, and WorkspacePage validation logic tests
- **Frontend build:** Successfully compiles with TypeScript + Vite
- **All tests pass:** 101 backend tests pass, frontend unit tests pass
- **ML pipeline preserved:** No modifications to PyTorch code, STFT, corruption, loss functions, or evaluation

**2026-10-30 — Phase 3 Complete**
- **Backend API additions verified:**
  - `GET /api/experiments` - paginated, filterable listing from `outputs/runs/`
  - `GET /api/experiments/:id` - detailed experiment metadata (with job registry fallback)
  - `POST /api/experiments/compare` - two-experiment comparison with metrics, config diffs, condition flags
  - `DELETE /api/experiments/:id` - safe deletion with path containment, symlink, active-job checks (fixed to find run_dir via job registry)
- **Tests added (12 new tests in test_web.py):**
  - `test_experiment_listing_empty`, `test_experiment_listing_with_runs`, `test_experiment_listing_pagination`, `test_experiment_listing_filtering`
  - `test_experiment_detail`, `test_experiment_detail_not_found`
  - `test_experiment_compare_same_fails`, `test_experiment_compare_two_different`
  - `test_experiment_delete`, `test_experiment_delete_active_job_fails`, `test_experiment_delete_path_traversal_blocked`, `test_experiment_delete_symlink_blocked` (skipped on Windows)
- **All 101 backend tests pass** (1 skipped for Windows symlink)
- **Frontend builds successfully**
- **ML pipeline completely preserved** - no modifications to PyTorch code, STFT, corruption, loss functions, or evaluation
- **Dashboard already implemented** in Phase 2 with real data cards, Recharts visualizations, latest experiments table

**2026-10-30 — Phase 2 Complete**
- **Design System:** Created complete DPAI design tokens in `frontend/src/styles/globals.css` with light theme (default) and dark theme support, matching the approved spec (Inter + IBM Plex Mono, purple primary, teal/orange/purple region colors)
- **MASTER.md Updated:** Reconciled the design system document with approved direction (dual-mode: light admin shell, dark workbench)
- **UI Components:** Implemented reusable components: Button, Card, Input, Badge, Dialog, Tabs, Tooltip, Table, Select, Slider, Separator, Progress, Label, StateComponents
- **Layout Components:** Responsive Sidebar (collapsible to icon-only, drawer on mobile), Topbar with theme toggle and backend status, Layout wrapper with PageHeader
- **All 9 Pages Implemented:**
  - LandingPage: Public page with precomputed demos, interactive demo explorer with spectrograms and audio players
  - DashboardPage: Real-data cards, Recharts visualizations (activity, NMSE, runtime), latest experiments table
  - WorkspacePage: WaveSurfer.js v7 integration with region editing, gap modes (manual/random/detect), SSE job monitoring, live loss curve
  - ExperimentsPage: Searchable/filterable table, bulk delete, pagination
  - ExperimentResultsPage: 4 tabs (Overview, Audio, Visual, Config)
  - CompareExperimentsPage: Side-by-side metrics, config diffs, condition flags
  - ResearchPage & MethodologyPage: Evidence-based docs with tabs for implemented/deviations/future work
  - SettingsPage: Theme selector, backend profiles (Local/Colab/Cloud) with connection test
- **Backend API Extensions:** Verified working - all 90 backend tests pass
- **Build:** Frontend builds successfully (TypeScript + Vite)
- **ML Pipeline:** Completely preserved - no modifications to PyTorch code, STFT, corruption, loss functions, or evaluation
- **Next task:** Phase 4 — Audio workspace and waveform editor (WaveSurfer.js integration already implemented in Phase 2, needs refinement)