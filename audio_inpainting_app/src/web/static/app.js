/**
 * DPAI Web Demo Application Logic (Option 2: Timeline Canvas + Docked Inspector)
 * Adheres strictly to design-system/dpai/MASTER.md instrument specifications.
 */

(function () {
  'use strict';

  // --- Global State ---
  let audioContext = null;
  let rawAudioBuffer = null;
  let uploadedAudioFile = null;
  let uploadedReferenceFile = null;
  let audioDurationS = 0;
  let audioSampleRate = 16000;
  let gaps = []; // list of [start_ms, end_ms]
  let currentJobId = null;
  let eventSource = null;
  let isDragging = false;
  let dragStartX = 0;
  let dragCurrentX = 0;
  let isPlayingPreview = false;
  let previewSourceNode = null;
  let previewStartContextTime = 0;  // audioContext.currentTime when playback began
  let rafPlayhead = null;           // requestAnimationFrame handle for playhead
  let hoverCanvasX = -1;            // canvas-space X of mouse hover (-1 = not hovering)
  let lossHistory = [];

  // --- DOM Elements ---
  const tabBtnStudio = document.getElementById('tab-btn-studio');
  const tabBtnGallery = document.getElementById('tab-btn-gallery');
  const sectionStudio = document.getElementById('section-studio');
  const sectionGallery = document.getElementById('section-gallery');
  const serverBadge = document.getElementById('server-badge');
  const serverStatusText = document.getElementById('server-status-text');

  const dropZone = document.getElementById('drop-zone');
  const audioInput = document.getElementById('audio-input');
  const referenceInput = document.getElementById('reference-input');
  const refStatus = document.getElementById('ref-status');
  const btnSynthetic = document.getElementById('btn-synthetic');
  const btnPianoSample = document.getElementById('btn-piano-sample');

  const waveformContainer = document.getElementById('waveform-container');
  const waveformCanvas = document.getElementById('waveform-canvas');
  const clipNameDisplay = document.getElementById('clip-name-display');
  const clipDurDisplay = document.getElementById('clip-dur-display');
  const btnPlayPreview = document.getElementById('btn-play-preview');
  const playBtnText = document.getElementById('play-btn-text');
  const spectrogramStateTag = document.getElementById('spectrogram-state-tag');

  const btnRandomGaps = document.getElementById('btn-random-gaps');
  const btnDetectGaps = document.getElementById('btn-detect-gaps');
  const btnClearGaps = document.getElementById('btn-clear-gaps');
  const statMissingMs = document.getElementById('stat-missing-ms');
  const statMissingPct = document.getElementById('stat-missing-pct');
  const gapsList = document.getElementById('gaps-list');
  const validationBox = document.getElementById('validation-box');
  const validationMsg = document.getElementById('validation-msg');

  const presetRadios = document.querySelectorAll('input[name="preset"]');
  const epochsOverride = document.getElementById('epochs-override');
  const seedInput = document.getElementById('seed-input');
  const checkPreserve = document.getElementById('check-preserve-observed');
  const btnRun = document.getElementById('btn-run');
  const btnCancel = document.getElementById('btn-cancel');

  const progressContainer = document.getElementById('progress-container');
  const progressStatusBadge = document.getElementById('progress-status-badge');
  const progressEtaText = document.getElementById('progress-eta-text');
  const progressEpochCounter = document.getElementById('progress-epoch-counter');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const lossCanvas = document.getElementById('loss-canvas');
  const currentLossText = document.getElementById('current-loss-text');
  const liveSpecImg = document.getElementById('live-spec-img');
  const specPlaceholderText = document.getElementById('spec-placeholder-text');

  const resultsSection = document.getElementById('results-section');
  const resultRunId = document.getElementById('result-run-id');
  const btnDownloadWav = document.getElementById('btn-download-wav');
  const btnAbReconstructed = document.getElementById('btn-ab-reconstructed');
  const btnAbCorrupted = document.getElementById('btn-ab-corrupted');
  const btnAbClean = document.getElementById('btn-ab-clean');
  const audioPlayback = document.getElementById('audio-playback');

  const metricRuntime = document.getElementById('metric-runtime');
  const metricEpochs = document.getElementById('metric-epochs');
  const metricParams = document.getElementById('metric-params');
  const metricCardNmse = document.getElementById('metric-card-nmse');
  const metricNmseTot = document.getElementById('metric-nmse-tot');
  const metricCardMiss = document.getElementById('metric-card-miss');
  const metricNmseMiss = document.getElementById('metric-nmse-miss');

  const imgSpecCorrupted = document.getElementById('img-spec-corrupted');
  const imgSpecReconstructed = document.getElementById('img-spec-reconstructed');
  const colSpecClean = document.getElementById('col-spec-clean');
  const imgSpecClean = document.getElementById('img-spec-clean');
  const imgLossCurve = document.getElementById('img-loss-curve');
  const galleryGrid = document.getElementById('gallery-grid');

  let activeResultUrls = {
    reconstruction: '',
    corrupted: '',
    clean: ''
  };

  // --- Initialise ---
  function init() {
    setupTabNavigation();
    setupAudioContext();
    setupFileUpload();
    setupWaveformInteraction();
    setupPresetSelection();
    setupJobExecution();
    setupAbPlayer();
    setupKeyboardShortcuts();
    checkServerHealth();
    restorePreviousRun();
    setupCanvasResolution();
  }

  function setupCanvasResolution() {
    function adjust() {
      if (!waveformCanvas) return;
      const rect = waveformCanvas.getBoundingClientRect();
      if (rect.width > 0) {
        waveformCanvas.width = Math.round(rect.width * window.devicePixelRatio || 1);
        waveformCanvas.height = Math.round(136 * window.devicePixelRatio || 1);
        renderWaveform();
      }
    }
    window.addEventListener('resize', adjust);
    adjust();
  }

  function restorePreviousRun() {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const lastRun = urlParams.get('run_id') || localStorage.getItem('dpai_last_run_id');
      if (lastRun) {
        displayJobResults(lastRun, { run_id: lastRun });
      }
    } catch (_) {}
  }

  // --- Tab Navigation ---
  function setupTabNavigation() {
    tabBtnStudio.addEventListener('click', () => {
      tabBtnStudio.classList.add('active');
      tabBtnGallery.classList.remove('active');
      sectionStudio.classList.add('active');
      sectionGallery.classList.remove('active');
    });

    tabBtnGallery.addEventListener('click', () => {
      tabBtnGallery.classList.add('active');
      tabBtnStudio.classList.remove('active');
      sectionGallery.classList.add('active');
      sectionStudio.classList.remove('active');
      loadExamplesGallery();
    });
  }

  // --- Keyboard Shortcuts (Option 2) ---
  function setupKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
      const tag = document.activeElement ? document.activeElement.tagName : '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        // If results section is visible and audio is set, toggle audio playback
        if (!resultsSection.classList.contains('hidden') && audioPlayback.src) {
          if (audioPlayback.paused) audioPlayback.play().catch(() => {});
          else audioPlayback.pause();
        } else {
          togglePreviewPlayback();
        }
      } else if (e.key === '1') {
        if (!resultsSection.classList.contains('hidden') && activeResultUrls.reconstruction) {
          switchAbAudio('reconstruction');
        }
      } else if (e.key === '2') {
        if (!resultsSection.classList.contains('hidden') && activeResultUrls.corrupted) {
          switchAbAudio('corrupted');
        }
      } else if (e.key === '3') {
        if (!resultsSection.classList.contains('hidden') && activeResultUrls.clean && !colSpecClean.classList.contains('hidden')) {
          switchAbAudio('clean');
        }
      }
    });
  }

  // --- Server Health ---
  async function checkServerHealth() {
    try {
      const resp = await fetch('/api/health');
      if (resp.ok) {
        const data = await resp.json();
        if (data.demo_only) {
          serverStatusText.textContent = `DEMO MODE (${data.device.toUpperCase()})`;
          btnRun.title = 'Job submission is disabled in demo mode.';
        } else {
          serverStatusText.textContent = `ONLINE // ${data.device.toUpperCase()}`;
        }
        if (data.device === 'cpu') {
          const fastRadio = document.getElementById('preset-fast');
          if (fastRadio && (!document.querySelector('input[name="preset"]:checked') || document.querySelector('input[name="preset"]:checked').value === 'balanced')) {
            fastRadio.checked = true;
            document.querySelectorAll('.preset-choice').forEach((card) => card.classList.remove('selected'));
            fastRadio.closest('.preset-choice')?.classList.add('selected');
          }
        }
      } else {
        throw new Error('Health check failed');
      }
    } catch (e) {
      serverStatusText.textContent = 'SERVER OFFLINE';
      serverStatusText.style.color = '#E5484D';
      const dot = serverBadge.querySelector('.device-dot');
      if (dot) dot.style.backgroundColor = '#E5484D';
    }
  }

  // --- Audio Context Helper ---
  function setupAudioContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioContext = new AudioContextClass({ sampleRate: 16000 });
    }
  }

  function getAudioContext() {
    if (!audioContext) {
      setupAudioContext();
    }
    if (audioContext && audioContext.state === 'suspended') {
      audioContext.resume();
    }
    return audioContext;
  }

  // --- File Upload & Synthesis ---
  function setupFileUpload() {
    dropZone.addEventListener('click', () => audioInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleAudioFile(e.dataTransfer.files[0]);
      }
    });

    audioInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleAudioFile(e.target.files[0]);
      }
    });

    referenceInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        uploadedReferenceFile = e.target.files[0];
        refStatus.textContent = uploadedReferenceFile.name;
        refStatus.style.color = 'var(--recon)';
      } else {
        uploadedReferenceFile = null;
        refStatus.textContent = 'NONE';
        refStatus.style.color = 'var(--fg-muted)';
      }
    });

    btnSynthetic.addEventListener('click', () => generateSyntheticSine(440, 3.0));
    btnPianoSample.addEventListener('click', () => generatePianoTone(261.63, 2.5));
  }

  async function handleAudioFile(file) {
    uploadedAudioFile = file;
    clipNameDisplay.textContent = file.name.toUpperCase();

    try {
      const arrayBuffer = await file.arrayBuffer();
      const ctx = getAudioContext();
      const decoded = await ctx.decodeAudioData(arrayBuffer);
      setAudioBuffer(decoded, file.name);
    } catch (err) {
      alert(`Could not decode audio file: ${err.message}`);
    }
  }

  function generateSyntheticSine(freq = 440, durationS = 3.0) {
    const ctx = getAudioContext();
    const sr = 16000;
    const numSamples = Math.round(durationS * sr);
    const buffer = ctx.createBuffer(1, numSamples, sr);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < numSamples; i++) {
      const t = i / sr;
      data[i] = 0.5 * Math.sin(2 * Math.PI * freq * t) + 0.1 * (Math.random() * 2 - 1);
    }

    let maxVal = 0;
    for (let i = 0; i < numSamples; i++) {
      if (Math.abs(data[i]) > maxVal) maxVal = Math.abs(data[i]);
    }
    if (maxVal > 0) {
      for (let i = 0; i < numSamples; i++) data[i] = (data[i] * 0.9) / maxVal;
    }

    uploadedAudioFile = null;
    setAudioBuffer(buffer, `SYNTHETIC ${freq} HZ SINE`);
  }

  function generatePianoTone(freq = 261.63, durationS = 2.5) {
    const ctx = getAudioContext();
    const sr = 16000;
    const numSamples = Math.round(durationS * sr);
    const buffer = ctx.createBuffer(1, numSamples, sr);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < numSamples; i++) {
      const t = i / sr;
      const decay = Math.exp(-2.5 * t);
      const val =
        (0.6 * Math.sin(2 * Math.PI * freq * t) +
          0.3 * Math.sin(2 * Math.PI * 2 * freq * t) +
          0.15 * Math.sin(2 * Math.PI * 3 * freq * t) +
          0.08 * Math.sin(2 * Math.PI * 4 * freq * t)) *
        decay;
      data[i] = val;
    }

    let maxVal = 0;
    for (let i = 0; i < numSamples; i++) {
      if (Math.abs(data[i]) > maxVal) maxVal = Math.abs(data[i]);
    }
    if (maxVal > 0) {
      for (let i = 0; i < numSamples; i++) data[i] = (data[i] * 0.9) / maxVal;
    }

    uploadedAudioFile = null;
    setAudioBuffer(buffer, 'PIANO NOTE (C4)');
  }

  function setAudioBuffer(buffer, displayName) {
    rawAudioBuffer = buffer;
    audioDurationS = buffer.duration;
    audioSampleRate = buffer.sampleRate;

    clipNameDisplay.textContent = displayName.toUpperCase();
    clipDurDisplay.textContent = `${(audioSampleRate / 1000).toFixed(0)} kHz • ${audioDurationS.toFixed(2)} s`;

    if (gaps.length === 0) {
      const centerMs = (audioDurationS * 1000) / 2;
      const gapLen = Math.min(80, audioDurationS * 200);
      gaps = [[Math.round(centerMs - gapLen / 2), Math.round(centerMs + gapLen / 2)]];
    }

    if (spectrogramStateTag) {
      spectrogramStateTag.textContent = 'CORRUPTED INPUT READY';
    }
    if (specPlaceholderText) {
      specPlaceholderText.textContent = 'AUDIO LOADED // READY FOR OPTIMIZATION';
    }

    renderWaveform();
    updateGapStats();
  }

  // --- Waveform Rendering & Interaction (Lab Instrument Palette) ---
  function setupWaveformInteraction() {
    waveformCanvas.addEventListener('mousedown', onCanvasMouseDown);
    waveformCanvas.addEventListener('mousemove', onCanvasMouseMove);
    window.addEventListener('mouseup', onCanvasMouseUp);

    // Hover cursor: track mouse position in canvas-pixel space
    waveformCanvas.addEventListener('mousemove', onCanvasHover);
    waveformCanvas.addEventListener('mouseleave', () => {
      hoverCanvasX = -1;
      renderWaveform();
    });

    btnPlayPreview.addEventListener('click', togglePreviewPlayback);
    btnRandomGaps.addEventListener('click', addRandomGaps);
    btnDetectGaps.addEventListener('click', autoDetectSilence);
    btnClearGaps.addEventListener('click', () => {
      gaps = [];
      renderWaveform();
      updateGapStats();
    });
  }

  function renderWaveform() {
    if (!rawAudioBuffer) return;

    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const data = rawAudioBuffer.getChannelData(0);
    const N = data.length;
    // Use exact float ratio so gap overlay (which uses sample/N * width) aligns
    // perfectly. Math.ceil would compress the waveform leftward vs the markers.
    const samplesPerPx = N / width;
    const amp = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Canvas background: surface #16181B
    ctx.fillStyle = '#16181B';
    ctx.fillRect(0, 0, width, height);

    // Center baseline: rgba(255, 255, 255, 0.07)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();

    // Time markings with IBM Plex Mono
    ctx.fillStyle = 'rgba(232, 230, 225, 0.35)';
    ctx.font = '10px "IBM Plex Mono", monospace';
    const totalSecs = audioDurationS;
    const numMarkers = 6;
    for (let i = 0; i <= numMarkers; i++) {
      const frac = i / numMarkers;
      const x = frac * width;
      const timeVal = (frac * totalSecs).toFixed(2);
      ctx.fillText(`${timeVal}s`, Math.min(x + 4, width - 36), height - 6);
      ctx.beginPath();
      ctx.moveTo(x, height - 16);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Draw waveform bars using proportional mapping — pixel i covers samples
    // [floor(i * samplesPerPx), floor((i+1) * samplesPerPx)].
    // This keeps the waveform and gap overlay in the same coordinate space.
    ctx.fillStyle = 'rgba(232, 230, 225, 0.82)';
    for (let i = 0; i < width; i++) {
      const sStart = Math.floor(i * samplesPerPx);
      const sEnd = Math.floor((i + 1) * samplesPerPx);
      let min = 1.0;
      let max = -1.0;
      for (let s = sStart; s < sEnd; s++) {
        const datum = data[s] || 0;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      // Guarantee at least one sample contributes (boundary pixels)
      if (sStart === sEnd) {
        const datum = data[sStart] || 0;
        min = datum;
        max = datum;
      }
      const y1 = Math.max(0, (1 + min) * amp);
      const y2 = Math.min(height, (1 + max) * amp);
      ctx.fillRect(i, y1, 1, Math.max(1, y2 - y1));
    }

    // Draw Gaps Overlay:
    // Missing / gap: #3DD6C6 (--missing) at 25% fill with 1px solid edges and drag handles
    const durMs = audioDurationS * 1000;
    gaps.forEach(([startMs, endMs]) => {
      const x1 = (startMs / durMs) * width;
      const x2 = (endMs / durMs) * width;
      const w = Math.max(2, x2 - x1);

      // Translucent fill (25%)
      ctx.fillStyle = 'rgba(61, 214, 198, 0.25)';
      ctx.fillRect(x1, 0, w, height);

      // Solid 1px edge
      ctx.strokeStyle = '#3DD6C6';
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, 0, w, height);

      // Left & right drag handles
      ctx.fillStyle = '#3DD6C6';
      ctx.fillRect(x1 - 1, 0, 3, height);
      ctx.fillRect(x2 - 2, 0, 3, height);

      // Gap duration tag
      ctx.fillStyle = '#3DD6C6';
      ctx.font = '9px "IBM Plex Mono", monospace';
      const gapText = `${Math.round(endMs - startMs)}ms`;
      if (w > 30) {
        ctx.fillText(gapText, x1 + 4, 14);
      }
    });

    // Draw in-progress drag selection
    if (isDragging) {
      const startX = Math.min(dragStartX, dragCurrentX);
      const dragW = Math.abs(dragCurrentX - dragStartX);
      ctx.fillStyle = 'rgba(61, 214, 198, 0.40)';
      ctx.fillRect(startX, 0, dragW, height);
      ctx.strokeStyle = '#3DD6C6';
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(startX, 0, dragW, height);
      ctx.setLineDash([]);
    }

    // --- Hover cursor: thin teal line + time tooltip ---
    if (hoverCanvasX >= 0 && !isDragging) {
      const hx = hoverCanvasX;
      const hoverMs = canvasPxToMs(hx);
      const hoverS = (hoverMs / 1000).toFixed(3);

      // Dashed vertical line in --missing
      ctx.strokeStyle = 'rgba(61, 214, 198, 0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, 0);
      ctx.lineTo(hx, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Timestamp tooltip pill (follows cursor, flips at right edge)
      const label = `${hoverS}s`;
      ctx.font = '10px "IBM Plex Mono", monospace';
      const textW = ctx.measureText(label).width;
      const pillW = textW + 10;
      const pillH = 16;
      const pillX = (hx + 8 + pillW > width) ? hx - pillW - 6 : hx + 6;
      const pillY = 8;
      ctx.fillStyle = 'rgba(14, 15, 17, 0.88)';
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(pillX, pillY, pillW, pillH, 2);
      } else {
        ctx.rect(pillX, pillY, pillW, pillH);
      }
      ctx.fill();
      ctx.strokeStyle = 'rgba(61, 214, 198, 0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = '#3DD6C6';
      ctx.fillText(label, pillX + 5, pillY + 11);
    }

    // --- Playhead: solid amber line tracking live preview position ---
    if (isPlayingPreview && audioContext) {
      const elapsed = audioContext.currentTime - previewStartContextTime;
      const clampedElapsed = Math.min(elapsed, audioDurationS);
      const phX = (clampedElapsed / audioDurationS) * width;

      // Amber solid line (--recon)
      ctx.strokeStyle = '#FFB020';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(phX, 0);
      ctx.lineTo(phX, height);
      ctx.stroke();

      // Small triangle head at top
      ctx.fillStyle = '#FFB020';
      ctx.beginPath();
      ctx.moveTo(phX - 5, 0);
      ctx.lineTo(phX + 5, 0);
      ctx.lineTo(phX, 7);
      ctx.closePath();
      ctx.fill();

      // Elapsed time readout
      const phLabel = `${clampedElapsed.toFixed(2)}s`;
      ctx.font = '9px "IBM Plex Mono", monospace';
      const phTW = ctx.measureText(phLabel).width;
      const phPillX = (phX + 6 + phTW + 10 > width) ? phX - phTW - 14 : phX + 4;
      ctx.fillStyle = 'rgba(14, 15, 17, 0.85)';
      ctx.fillRect(phPillX, height - 20, phTW + 8, 14);
      ctx.fillStyle = '#FFB020';
      ctx.fillText(phLabel, phPillX + 4, height - 9);
    }
  }

  function onCanvasMouseDown(e) {
    if (!rawAudioBuffer) return;
    const rect = waveformCanvas.getBoundingClientRect();
    const scaleX = waveformCanvas.width / rect.width;
    dragStartX = (e.clientX - rect.left) * scaleX;
    dragCurrentX = dragStartX;
    isDragging = true;
  }

  function onCanvasMouseMove(e) {
    const rect = waveformCanvas.getBoundingClientRect();
    const scaleX = waveformCanvas.width / rect.width;
    const cx = Math.max(0, Math.min(waveformCanvas.width, (e.clientX - rect.left) * scaleX));
    if (isDragging) {
      dragCurrentX = cx;
    }
    hoverCanvasX = cx;
    renderWaveform();
  }

  /** Convert canvas-pixel X → display milliseconds. */
  function canvasPxToMs(px) {
    return (px / waveformCanvas.width) * audioDurationS * 1000;
  }

  /** Called only for hover; separate from drag handler. */
  function onCanvasHover(e) {
    if (!rawAudioBuffer) return;
    const rect = waveformCanvas.getBoundingClientRect();
    const scaleX = waveformCanvas.width / rect.width;
    hoverCanvasX = Math.max(0, Math.min(waveformCanvas.width, (e.clientX - rect.left) * scaleX));
    if (!isDragging) renderWaveform();
  }

  function onCanvasMouseUp() {
    if (!isDragging) return;
    isDragging = false;

    const minX = Math.min(dragStartX, dragCurrentX);
    const maxX = Math.max(dragStartX, dragCurrentX);
    const durMs = audioDurationS * 1000;
    const width = waveformCanvas.width;

    if (maxX - minX > 4) {
      const startMs = Math.round((minX / width) * durMs);
      const endMs = Math.round((maxX / width) * durMs);
      if (endMs > startMs) {
        gaps.push([startMs, endMs]);
        gaps.sort((a, b) => a[0] - b[0]);
      }
    }

    renderWaveform();
    updateGapStats();
  }

  function updateGapStats() {
    if (!rawAudioBuffer) return;

    const durMs = audioDurationS * 1000;
    let totalMissingMs = 0;
    let hasGapOver100 = false;

    const merged = [];
    gaps.forEach(([s, e]) => {
      if (e - s > 100) hasGapOver100 = true;
      if (merged.length === 0 || merged[merged.length - 1][1] < s) {
        merged.push([s, e]);
      } else {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], e);
      }
    });

    merged.forEach(([s, e]) => {
      totalMissingMs += e - s;
    });

    const missingPct = durMs > 0 ? (totalMissingMs / durMs) * 100 : 0;
    statMissingMs.textContent = `${Math.round(totalMissingMs)} ms`;
    statMissingPct.textContent = `${missingPct.toFixed(1)}%`;

    // Render chip list
    gapsList.innerHTML = '';
    gaps.forEach(([startMs, endMs], idx) => {
      const chip = document.createElement('div');
      chip.className = 'gap-chip';
      chip.innerHTML = `
        <span>${startMs} - ${endMs} ms (${endMs - startMs}ms)</span>
        <button type="button" class="btn-del-gap" data-idx="${idx}" title="Remove gap">×</button>
      `;
      chip.querySelector('.btn-del-gap').addEventListener('click', (e) => {
        const removeIdx = parseInt(e.target.dataset.idx, 10);
        gaps.splice(removeIdx, 1);
        renderWaveform();
        updateGapStats();
      });
      gapsList.appendChild(chip);
    });

    // Validation checks
    let warningMsg = '';
    let isBlocker = false;

    if (gaps.length === 0) {
      warningMsg = 'Mark at least one corrupted gap to inpaint.';
      isBlocker = true;
    } else if (missingPct >= 40.0) {
      warningMsg = `Total missing audio (${missingPct.toFixed(1)}%) exceeds maximum 40% budget.`;
      isBlocker = true;
    } else if (hasGapOver100) {
      warningMsg = 'Notice: One or more gaps exceed 100 ms. Paper method is calibrated for ≤ 100 ms.';
      isBlocker = false;
    }

    if (warningMsg) {
      validationBox.classList.remove('hidden');
      validationMsg.textContent = warningMsg;
      btnRun.disabled = isBlocker;
    } else {
      validationBox.classList.add('hidden');
      btnRun.disabled = false;
    }
  }

  function addRandomGaps() {
    if (!rawAudioBuffer) return;
    const durMs = audioDurationS * 1000;
    const numGaps = durMs > 3000 ? 2 : 1;
    const newGaps = [];

    for (let i = 0; i < numGaps; i++) {
      const gapDur = 50 + Math.random() * 40;
      const start = Math.random() * (durMs - gapDur - 100);
      newGaps.push([Math.round(start), Math.round(start + gapDur)]);
    }

    gaps = newGaps.sort((a, b) => a[0] - b[0]);
    renderWaveform();
    updateGapStats();
  }

  function autoDetectSilence() {
    if (!rawAudioBuffer) return;
    const data = rawAudioBuffer.getChannelData(0);
    const sr = rawAudioBuffer.sampleRate;
    const threshold = 1e-4;
    const minSamples = Math.round(0.02 * sr);
    const detected = [];

    let inRun = false;
    let runStart = 0;

    for (let i = 0; i < data.length; i++) {
      const isZero = Math.abs(data[i]) <= threshold;
      if (isZero && !inRun) {
        inRun = true;
        runStart = i;
      } else if (!isZero && inRun) {
        if (i - runStart >= minSamples) {
          detected.push([
            Math.round((runStart * 1000) / sr),
            Math.round((i * 1000) / sr)
          ]);
        }
        inRun = false;
      }
    }
    if (inRun && data.length - runStart >= minSamples) {
      detected.push([
        Math.round((runStart * 1000) / sr),
        Math.round((data.length * 1000) / sr)
      ]);
    }

    if (detected.length > 0) {
      gaps = detected;
    } else {
      alert('No silent intervals detected. Click and drag on waveform to mark corrupted gaps manually.');
    }

    renderWaveform();
    updateGapStats();
  }

  // --- Preview Playback (with gaps silenced) ---
  function togglePreviewPlayback() {
    if (!rawAudioBuffer) return;
    const ctx = getAudioContext();

    if (isPlayingPreview) {
      stopPreviewPlayback();
      return;
    }

    const sampleRate = rawAudioBuffer.sampleRate;
    const previewBuffer = ctx.createBuffer(1, rawAudioBuffer.length, sampleRate);
    const inData = rawAudioBuffer.getChannelData(0);
    const outData = previewBuffer.getChannelData(0);
    outData.set(inData);

    gaps.forEach(([startMs, endMs]) => {
      const s = Math.round((startMs * sampleRate) / 1000);
      const e = Math.round((endMs * sampleRate) / 1000);
      for (let i = s; i < Math.min(e, outData.length); i++) {
        outData[i] = 0.0;
      }
    });

    previewSourceNode = ctx.createBufferSource();
    previewSourceNode.buffer = previewBuffer;
    previewSourceNode.connect(ctx.destination);
    previewSourceNode.onended = () => stopPreviewPlayback();
    previewSourceNode.start(0);

    isPlayingPreview = true;
    previewStartContextTime = ctx.currentTime;
    playBtnText.textContent = 'STOP';
    btnPlayPreview.classList.add('active');

    // Drive playhead redraws at ~60 fps while audio plays
    function tickPlayhead() {
      if (!isPlayingPreview) return;
      renderWaveform();
      rafPlayhead = requestAnimationFrame(tickPlayhead);
    }
    rafPlayhead = requestAnimationFrame(tickPlayhead);
  }

  function stopPreviewPlayback() {
    if (previewSourceNode) {
      try {
        previewSourceNode.stop();
        previewSourceNode.disconnect();
      } catch (e) {}
      previewSourceNode = null;
    }
    if (rafPlayhead !== null) {
      cancelAnimationFrame(rafPlayhead);
      rafPlayhead = null;
    }
    isPlayingPreview = false;
    playBtnText.textContent = 'PREVIEW';
    btnPlayPreview.classList.remove('active');
    renderWaveform(); // clear playhead
  }

  // --- Presets UI ---
  function setupPresetSelection() {
    presetRadios.forEach((radio) => {
      radio.addEventListener('change', () => {
        document.querySelectorAll('.preset-choice').forEach((card) => card.classList.remove('selected'));
        radio.closest('.preset-choice').classList.add('selected');
      });
    });
  }

  // --- Job Execution & SSE ---
  function setupJobExecution() {
    btnRun.addEventListener('click', runInpainting);
    btnCancel.addEventListener('click', cancelInpainting);
  }

  function bufferToWavBlob(audioBuffer) {
    const numChannels = 1;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    const data = audioBuffer.getChannelData(0);
    const byteRate = (sampleRate * numChannels * bitDepth) / 8;
    const blockAlign = (numChannels * bitDepth) / 8;
    const dataSize = data.length * ((numChannels * bitDepth) / 8);
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < data.length; i++) {
      let sample = Math.max(-1, Math.min(1, data[i]));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, sample, true);
      offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  async function runInpainting() {
    stopPreviewPlayback();

    const selectedPreset = document.querySelector('input[name="preset"]:checked').value;
    const seed = parseInt(seedInput.value, 10) || 0;
    const preserve = checkPreserve.checked;
    const customEpochs = epochsOverride.value ? parseInt(epochsOverride.value, 10) : null;

    const formData = new FormData();
    formData.append('preset', selectedPreset);
    formData.append('seed', seed);
    formData.append('preserve_observed', preserve);
    if (customEpochs) {
      formData.append('epochs', customEpochs);
    }

    if (uploadedAudioFile) {
      formData.append('audio', uploadedAudioFile);
      formData.append('mode', 'gaps');
      formData.append('gaps_json', JSON.stringify(gaps));
    } else if (rawAudioBuffer) {
      const wavBlob = bufferToWavBlob(rawAudioBuffer);
      formData.append('audio', wavBlob, 'synthetic.wav');
      formData.append('mode', 'gaps');
      formData.append('gaps_json', JSON.stringify(gaps));
    } else {
      alert('Please upload or generate an audio clip first.');
      return;
    }

    if (uploadedReferenceFile) {
      formData.append('reference', uploadedReferenceFile);
    }

    // UI Updates for Running State
    btnRun.disabled = true;
    btnCancel.classList.remove('hidden');
    progressContainer.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    progressBarFill.style.width = '0%';
    progressEpochCounter.textContent = 'INIT...';
    lossHistory = [];
    clearLossCanvas();
    liveSpecImg.classList.remove('active');
    specPlaceholderText.classList.remove('hidden');
    specPlaceholderText.textContent = 'CONNECTING TO OPTIMIZATION STREAM...';
    if (spectrogramStateTag) spectrogramStateTag.textContent = 'INITIALIZING DEEP PRIOR';

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to submit inpainting job');
      }

      const jobData = await response.json();
      currentJobId = jobData.id;
      connectToJobEvents(currentJobId);
    } catch (err) {
      alert(`Job submission error: ${err.message}`);
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
      progressContainer.classList.add('hidden');
    }
  }

  function connectToJobEvents(jobId) {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(`/api/jobs/${jobId}/events`);

    eventSource.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        handleProgressEvent(event);
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    };

    eventSource.onerror = (err) => {
      console.warn('SSE connection interrupted:', err);
      if (eventSource && eventSource.readyState === EventSource.CLOSED) {
        progressStatusBadge.textContent = 'DISCONNECTED';
        progressStatusBadge.style.color = '#E5484D';
        specPlaceholderText.textContent = 'OPTIMIZATION STREAM DISCONNECTED';
        specPlaceholderText.classList.remove('hidden');
        btnRun.disabled = false;
        btnCancel.classList.add('hidden');
      }
    };
  }

  function handleProgressEvent(event) {
    if (event.type === 'status') {
      if (event.status === 'running') {
        progressStatusBadge.textContent = 'RUNNING';
        progressStatusBadge.style.color = 'var(--recon)';
        progressEpochCounter.textContent = `0 / ${event.epochs || '...'}`;
        specPlaceholderText.textContent = 'OPTIMIZING HARMONIC CONVOLUTIONS...';
      } else if (event.status === 'queued') {
        progressStatusBadge.textContent = 'QUEUED';
        progressStatusBadge.style.color = 'var(--fg-muted)';
        progressEpochCounter.textContent = 'IN QUEUE';
      }
    } else if (event.type === 'progress') {
      progressStatusBadge.textContent = 'RUNNING';
      progressStatusBadge.style.color = 'var(--recon)';
      const epoch = event.epoch;
      const totalEpochs = event.epochs;
      const pct = Math.min(100, (epoch / totalEpochs) * 100);

      progressBarFill.style.width = `${pct.toFixed(1)}%`;
      progressEpochCounter.textContent = `${epoch} / ${totalEpochs}`;
      progressEtaText.textContent = event.eta_s ? `${event.eta_s}s` : '--';
      currentLossText.textContent = `loss: ${event.loss.toExponential(4)}`;

      if (spectrogramStateTag) {
        spectrogramStateTag.textContent = `OPTIMIZING (EP ${epoch}/${totalEpochs})`;
      }

      // Update loss curve
      lossHistory.push(event.loss);
      drawLossCurve(lossHistory);

      // Update live preview spectrogram
      if (event.preview) {
        liveSpecImg.src = event.preview;
        liveSpecImg.classList.add('active');
        specPlaceholderText.classList.add('hidden');
      }
    } else if (event.type === 'completed' || event.status === 'completed') {
      if (eventSource) eventSource.close();
      progressBarFill.style.width = '100%';
      progressStatusBadge.textContent = 'COMPLETE';
      progressStatusBadge.style.color = 'var(--missing)';
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
      displayJobResults(currentJobId, event);
    } else if (event.type === 'cancelled' || event.status === 'cancelled') {
      if (eventSource) eventSource.close();
      progressStatusBadge.textContent = 'CANCELLED';
      progressStatusBadge.style.color = 'var(--danger)';
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
      if (spectrogramStateTag) spectrogramStateTag.textContent = 'CANCELLED';
      alert('Inpainting job was cancelled.');
    } else if (event.type === 'failed' || event.status === 'failed') {
      if (eventSource) eventSource.close();
      progressStatusBadge.textContent = 'FAILED';
      progressStatusBadge.style.color = 'var(--danger)';
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
      if (spectrogramStateTag) spectrogramStateTag.textContent = 'FAILED';
      alert(`Job failed: ${event.error || 'Unknown optimization error'}`);
    }
  }

  async function cancelInpainting() {
    if (!currentJobId) return;
    try {
      await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
    } catch (e) {
      console.error('Cancel request failed:', e);
    }
  }

  // --- Loss Curve Canvas Drawing (MASTER.md palette: #0E0F11 bg, #FFB020 recon line) ---
  function clearLossCanvas() {
    const ctx = lossCanvas.getContext('2d');
    ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
  }

  function drawLossCurve(history) {
    if (history.length < 2) return;
    const ctx = lossCanvas.getContext('2d');
    const w = lossCanvas.width;
    const h = lossCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0E0F11';
    ctx.fillRect(0, 0, w, h);

    // Subtle horizontal gridlines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.25); ctx.lineTo(w, h * 0.25);
    ctx.moveTo(0, h * 0.50); ctx.lineTo(w, h * 0.50);
    ctx.moveTo(0, h * 0.75); ctx.lineTo(w, h * 0.75);
    ctx.stroke();

    const minLoss = Math.min(...history);
    const maxLoss = Math.max(...history);
    const range = maxLoss - minLoss || 1e-6;

    // Line in --recon #FFB020
    ctx.strokeStyle = '#FFB020';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let i = 0; i < history.length; i++) {
      const x = (i / (history.length - 1)) * (w - 8) + 4;
      const normalized = (history[i] - minLoss) / range;
      const y = (1 - normalized) * (h - 14) + 7;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // --- Display Results ---
  async function displayJobResults(jobId, eventData) {
    resultsSection.classList.remove('hidden');
    const effectiveId = (eventData && eventData.run_id) ? eventData.run_id : jobId;
    resultRunId.textContent = effectiveId;
    try {
      localStorage.setItem('dpai_last_run_id', effectiveId);
    } catch (_) {}

    let job = null;
    try {
      let resp = await fetch(`/api/jobs/${effectiveId}`);
      if (!resp.ok && effectiveId !== jobId) {
        resp = await fetch(`/api/jobs/${jobId}`);
      }
      if (resp.ok) job = await resp.json();
    } catch (e) {
      console.warn('Could not fetch job metadata:', e);
    }

    const cacheBuster = Date.now();

    const reconUrl = `/api/jobs/${effectiveId}/files/reconstruction.wav`;
    const corrUrl = `/api/jobs/${effectiveId}/files/corrupted.wav`;
    const cleanUrl = `/api/jobs/${effectiveId}/files/clean.wav`;

    activeResultUrls = {
      reconstruction: reconUrl,
      corrupted: corrUrl,
      clean: cleanUrl
    };

    btnDownloadWav.href = `${reconUrl}?download=true`;

    // Default to reconstructed audio
    switchAbAudio('reconstruction');

    // Metrics
    if (job) {
      if (job.started_at && job.completed_at) {
        metricRuntime.textContent = `${(job.completed_at - job.started_at).toFixed(1)} s`;
      } else {
        metricRuntime.textContent = '--';
      }
      metricEpochs.textContent = job.progress?.epochs || '--';
      const nParams = job.n_params || (job.metrics && job.metrics.n_params) || (eventData && eventData.n_params);
      if (nParams) {
        metricParams.textContent = Number(nParams).toLocaleString();
      } else {
        metricParams.textContent = '--';
      }
      if (job.metrics) {
        if (job.metrics.nmse_tot_db !== undefined && job.metrics.nmse_tot_db !== null) {
          metricNmseTot.textContent = `${job.metrics.nmse_tot_db.toFixed(2)} dB`;
          metricCardNmse.classList.remove('hidden');
        } else {
          metricCardNmse.classList.add('hidden');
        }
        if (job.metrics.nmse_miss_db !== undefined && job.metrics.nmse_miss_db !== null) {
          metricNmseMiss.textContent = `${job.metrics.nmse_miss_db.toFixed(2)} dB`;
          metricCardMiss.classList.remove('hidden');
        } else {
          metricCardMiss.classList.add('hidden');
        }
      }
    } else if (eventData && eventData.metrics) {
      if (eventData.n_params) metricParams.textContent = Number(eventData.n_params).toLocaleString();
    }

    // Spectrogram Images with cache buster
    imgSpecCorrupted.src = `/api/jobs/${effectiveId}/files/spec_corrupted.png?t=${cacheBuster}`;
    imgSpecReconstructed.src = `/api/jobs/${effectiveId}/files/spec_reconstructed.png?t=${cacheBuster}`;
    imgLossCurve.src = `/api/jobs/${effectiveId}/files/loss_curve.png?t=${cacheBuster}`;

    // Update main scope view with reconstructed spectrogram
    liveSpecImg.src = imgSpecReconstructed.src;
    liveSpecImg.classList.add('active');
    specPlaceholderText.classList.add('hidden');
    if (spectrogramStateTag) spectrogramStateTag.textContent = 'RECONSTRUCTION SCOPE';

    if (job && job.files && job.files.includes('spec_clean.png')) {
      imgSpecClean.src = `/api/jobs/${effectiveId}/files/spec_clean.png?t=${cacheBuster}`;
      colSpecClean.classList.remove('hidden');
      btnAbClean.classList.remove('hidden');
    } else {
      colSpecClean.classList.add('hidden');
      btnAbClean.classList.add('hidden');
    }

    // Smooth scroll down to transport deck
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  // --- A/B Audio Switcher ---
  function setupAbPlayer() {
    btnAbReconstructed.addEventListener('click', () => switchAbAudio('reconstruction'));
    btnAbCorrupted.addEventListener('click', () => switchAbAudio('corrupted'));
    btnAbClean.addEventListener('click', () => switchAbAudio('clean'));
  }

  function switchAbAudio(type) {
    [btnAbReconstructed, btnAbCorrupted, btnAbClean].forEach((btn) => btn.classList.remove('active'));

    const currentTime = audioPlayback.currentTime || 0;
    const isPlaying = !audioPlayback.paused;

    if (type === 'reconstruction') {
      btnAbReconstructed.classList.add('active');
      audioPlayback.src = activeResultUrls.reconstruction;
      if (spectrogramStateTag) spectrogramStateTag.textContent = 'SOLO // RECONSTRUCTED';
    } else if (type === 'corrupted') {
      btnAbCorrupted.classList.add('active');
      audioPlayback.src = activeResultUrls.corrupted;
      if (spectrogramStateTag) spectrogramStateTag.textContent = 'SOLO // CORRUPTED INPUT';
    } else if (type === 'clean') {
      btnAbClean.classList.add('active');
      audioPlayback.src = activeResultUrls.clean;
      if (spectrogramStateTag) spectrogramStateTag.textContent = 'SOLO // CLEAN REFERENCE';
    }

    audioPlayback.currentTime = currentTime;
    if (isPlaying) {
      audioPlayback.play().catch(() => {});
    }
  }

  // --- Examples Gallery ---
  async function loadExamplesGallery() {
    galleryGrid.innerHTML = '<div class="rack-loader">SCANNING EXAMPLES DATABASE...</div>';

    try {
      const resp = await fetch('examples/index.json');
      if (!resp.ok) {
        galleryGrid.innerHTML = `
          <div class="rack-alert" style="grid-column: 1/-1;">
            <span class="alert-icon">!</span>
            <div>
              <strong>Examples Gallery Empty</strong>
              <p>Run: <code>python scripts/precompute_examples.py</code> to populate precomputed benchmark clips.</p>
            </div>
          </div>
        `;
        return;
      }

      const data = await resp.json();
      const examples = data.examples || [];

      if (examples.length === 0) {
        galleryGrid.innerHTML = '<p class="param-label">NO EXAMPLES FOUND IN INDEX.JSON</p>';
        return;
      }

      galleryGrid.innerHTML = '';
      examples.forEach((ex) => {
        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.innerHTML = `
          <div class="gallery-header">
            <span class="gallery-title">${ex.title || ex.clip_id}</span>
            <span class="gallery-meta">${ex.gap_ms} MS GAP // ${ex.epochs || 5000} EP</span>
          </div>
          <p class="gallery-desc">${ex.description || 'Precomputed benchmark reconstruction.'}</p>
          <div class="img-frame">
            <img src="${ex.spectrogram_url || `examples/${ex.id}/spectrogram.png`}" alt="${ex.title}">
          </div>
          <div class="param-item">
            <span class="param-label">RECONSTRUCTED SIGNAL:</span>
            <audio controls src="${ex.audio_url || `examples/${ex.id}/reconstruction.wav`}" class="native-player-docked"></audio>
          </div>
          <div class="param-item">
            <span class="param-label">CORRUPTED INPUT:</span>
            <audio controls src="${ex.corrupted_url || `examples/${ex.id}/corrupted.wav`}" class="native-player-docked"></audio>
          </div>
        `;
        galleryGrid.appendChild(card);
      });
    } catch (e) {
      galleryGrid.innerHTML = `
        <div class="rack-alert" style="grid-column: 1/-1;">
          <span class="alert-icon">!</span>
          <div>
            <strong>Examples Gallery Unavailable</strong>
            <p>Run: <code>python scripts/precompute_examples.py</code></p>
          </div>
        </div>
      `;
    }
  }

  // --- Run on DOM Ready ---
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
