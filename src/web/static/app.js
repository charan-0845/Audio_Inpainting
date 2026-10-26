/**
 * DPAI Web Demo Application Logic (Vanilla JavaScript)
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
    checkServerHealth();
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

  // --- Server Health ---
  async function checkServerHealth() {
    try {
      const resp = await fetch('/api/health');
      if (resp.ok) {
        const data = await resp.json();
        serverBadge.className = 'badge badge-idle';
        if (data.demo_only) {
          serverStatusText.textContent = `Demo Mode (${data.device.toUpperCase()})`;
          btnRun.title = 'Job submission is disabled in demo mode.';
        } else {
          serverStatusText.textContent = `Online • ${data.device.toUpperCase()}`;
        }
      } else {
        throw new Error('Health check failed');
      }
    } catch (e) {
      serverBadge.className = 'badge';
      serverBadge.style.background = 'rgba(244, 63, 94, 0.15)';
      serverBadge.style.color = '#f43f5e';
      serverStatusText.textContent = 'Server Offline';
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
        refStatus.textContent = `Attached: ${uploadedReferenceFile.name}`;
      } else {
        uploadedReferenceFile = null;
        refStatus.textContent = 'None';
      }
    });

    btnSynthetic.addEventListener('click', () => generateSyntheticSine(440, 3.0));
    btnPianoSample.addEventListener('click', () => generatePianoTone(261.63, 2.5));
  }

  async function handleAudioFile(file) {
    uploadedAudioFile = file;
    clipNameDisplay.textContent = file.name;

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

    // Peak normalise to 0.9
    let maxVal = 0;
    for (let i = 0; i < numSamples; i++) {
      if (Math.abs(data[i]) > maxVal) maxVal = Math.abs(data[i]);
    }
    if (maxVal > 0) {
      for (let i = 0; i < numSamples; i++) data[i] = (data[i] * 0.9) / maxVal;
    }

    uploadedAudioFile = null; // will use mode=synthetic or client wav
    setAudioBuffer(buffer, `Synthetic ${freq} Hz Sine`);
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
      // Fundamental + 3 harmonics
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
    setAudioBuffer(buffer, 'Piano Note (C4)');
  }

  function setAudioBuffer(buffer, displayName) {
    rawAudioBuffer = buffer;
    audioDurationS = buffer.duration;
    audioSampleRate = buffer.sampleRate;

    clipNameDisplay.textContent = displayName;
    clipDurDisplay.textContent = `${audioDurationS.toFixed(2)}s (${buffer.sampleRate} Hz)`;
    waveformContainer.classList.remove('hidden');

    // Default: seed with an initial gap if none exist
    if (gaps.length === 0) {
      const centerMs = (audioDurationS * 1000) / 2;
      const gapLen = Math.min(80, audioDurationS * 200);
      gaps = [[Math.round(centerMs - gapLen / 2), Math.round(centerMs + gapLen / 2)]];
    }

    renderWaveform();
    updateGapStats();
  }

  // --- Waveform Rendering & Interaction ---
  function setupWaveformInteraction() {
    waveformCanvas.addEventListener('mousedown', onCanvasMouseDown);
    waveformCanvas.addEventListener('mousemove', onCanvasMouseMove);
    window.addEventListener('mouseup', onCanvasMouseUp);

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
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Background & center grid line
    ctx.fillStyle = '#060911';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();

    // Time markings
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '9px JetBrains Mono, monospace';
    const totalSecs = audioDurationS;
    const numMarkers = 5;
    for (let i = 0; i <= numMarkers; i++) {
      const frac = i / numMarkers;
      const x = frac * width;
      const timeVal = (frac * totalSecs).toFixed(1);
      ctx.fillText(`${timeVal}s`, Math.min(x + 2, width - 24), height - 4);
      ctx.beginPath();
      ctx.moveTo(x, height - 12);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Draw waveform bars
    ctx.fillStyle = '#6366f1';
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[i * step + j] || 0;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      const y1 = Math.max(0, (1 + min) * amp);
      const y2 = Math.min(height, (1 + max) * amp);
      ctx.fillRect(i, y1, 1, Math.max(1, y2 - y1));
    }

    // Draw Gaps Overlay (Red translucent columns)
    const durMs = audioDurationS * 1000;
    gaps.forEach(([startMs, endMs]) => {
      const x1 = (startMs / durMs) * width;
      const x2 = (endMs / durMs) * width;
      const w = Math.max(2, x2 - x1);

      ctx.fillStyle = 'rgba(244, 63, 94, 0.35)';
      ctx.fillRect(x1, 0, w, height);

      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x1, 0, w, height);
    });

    // Draw in-progress drag selection
    if (isDragging) {
      const startX = Math.min(dragStartX, dragCurrentX);
      const dragW = Math.abs(dragCurrentX - dragStartX);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.5)';
      ctx.fillRect(startX, 0, dragW, height);
      ctx.strokeStyle = '#fff';
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(startX, 0, dragW, height);
      ctx.setLineDash([]);
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
    if (!isDragging) return;
    const rect = waveformCanvas.getBoundingClientRect();
    const scaleX = waveformCanvas.width / rect.width;
    dragCurrentX = Math.max(0, Math.min(waveformCanvas.width, (e.clientX - rect.left) * scaleX));
    renderWaveform();
  }

  function onCanvasMouseUp() {
    if (!isDragging) return;
    isDragging = false;

    const minX = Math.min(dragStartX, dragCurrentX);
    const maxX = Math.max(dragStartX, dragCurrentX);
    const durMs = audioDurationS * 1000;
    const width = waveformCanvas.width;

    if (maxX - minX > 3) {
      const startMs = Math.round((minX / width) * durMs);
      const endMs = Math.round((maxX / width) * durMs);
      if (endMs > startMs) {
        gaps.push([startMs, endMs]);
        // Sort gaps by start time
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

    // Merge overlapping intervals to compute total unique missing time
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
        <span>${startMs}ms - ${endMs}ms (${endMs - startMs}ms)</span>
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
      warningMsg = 'Please mark at least one corrupted gap to inpaint.';
      isBlocker = true;
    } else if (missingPct >= 40.0) {
      warningMsg = `Total missing audio (${missingPct.toFixed(1)}%) exceeds the 40% maximum limit. Reduce gap sizes.`;
      isBlocker = true;
    } else if (hasGapOver100) {
      warningMsg =
        'Notice: One or more gaps exceed 100 ms. The Deep Prior model is optimized for gaps ≤ 100 ms; longer gaps may show degraded fidelity.';
      isBlocker = false;
    }

    if (warningMsg) {
      validationBox.classList.remove('hidden');
      validationMsg.textContent = warningMsg;
      if (isBlocker) {
        validationBox.className = 'callout callout-warning';
        btnRun.disabled = true;
      } else {
        validationBox.className = 'callout callout-warning';
        btnRun.disabled = false;
      }
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
      const gapDur = 50 + Math.random() * 40; // 50 - 90 ms
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
    const minSamples = Math.round(0.02 * sr); // 20 ms
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
      alert(`Auto-detected ${detected.length} silent/corrupted gap(s).`);
    } else {
      alert('No silent/zeroed regions detected in this audio. Click and drag on the waveform to mark gaps manually.');
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

    // Clone buffer and zero out gaps
    const sampleRate = rawAudioBuffer.sampleRate;
    const durMs = rawAudioBuffer.duration * 1000;
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
    playBtnText.textContent = 'Stop';
    btnPlayPreview.classList.add('active');
  }

  function stopPreviewPlayback() {
    if (previewSourceNode) {
      try {
        previewSourceNode.stop();
        previewSourceNode.disconnect();
      } catch (e) {}
      previewSourceNode = null;
    }
    isPlayingPreview = false;
    playBtnText.textContent = 'Play';
    btnPlayPreview.classList.remove('active');
  }

  // --- Presets UI ---
  function setupPresetSelection() {
    presetRadios.forEach((radio) => {
      radio.addEventListener('change', () => {
        document.querySelectorAll('.preset-card').forEach((card) => card.classList.remove('selected'));
        radio.closest('.preset-card').classList.add('selected');
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
      // Export current synthetic or decoded buffer as a WAV
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
    progressEpochCounter.textContent = 'Starting job...';
    lossHistory = [];
    clearLossCanvas();
    liveSpecImg.classList.remove('active');
    specPlaceholderText.textContent = 'Connecting to optimization stream...';

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
    };
  }

  function handleProgressEvent(event) {
    if (event.type === 'progress') {
      const epoch = event.epoch;
      const totalEpochs = event.epochs;
      const pct = Math.min(100, (epoch / totalEpochs) * 100);

      progressBarFill.style.width = `${pct.toFixed(1)}%`;
      progressEpochCounter.textContent = `Epoch ${epoch} / ${totalEpochs}`;
      progressEtaText.textContent = `ETA: ${event.eta_s ? event.eta_s + 's' : 'Calculating...'}`;
      currentLossText.textContent = `loss: ${event.loss.toExponential(4)}`;

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
      progressStatusBadge.textContent = 'Completed';
      progressStatusBadge.className = 'badge badge-idle';
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
      displayJobResults(currentJobId, event);
    } else if (event.type === 'cancelled' || event.status === 'cancelled') {
      if (eventSource) eventSource.close();
      progressStatusBadge.textContent = 'Cancelled';
      progressStatusBadge.className = 'badge text-danger';
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
      alert('Inpainting job was cancelled.');
    } else if (event.type === 'failed' || event.status === 'failed') {
      if (eventSource) eventSource.close();
      progressStatusBadge.textContent = 'Failed';
      progressStatusBadge.className = 'badge text-danger';
      btnRun.disabled = false;
      btnCancel.classList.add('hidden');
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

  // --- Loss Curve Canvas Drawing ---
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
    ctx.fillStyle = '#04060b';
    ctx.fillRect(0, 0, w, h);

    const minLoss = Math.min(...history);
    const maxLoss = Math.max(...history);
    const range = maxLoss - minLoss || 1e-6;

    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let i = 0; i < history.length; i++) {
      const x = (i / (history.length - 1)) * (w - 10) + 5;
      const normalized = (history[i] - minLoss) / range;
      const y = (1 - normalized) * (h - 16) + 8;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // --- Display Results ---
  async function displayJobResults(jobId, eventData) {
    resultsSection.classList.remove('hidden');
    resultRunId.textContent = eventData.run_id || jobId;

    // Fetch full job data
    let job = null;
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      if (resp.ok) job = await resp.json();
    } catch (e) {
      console.warn('Could not fetch job metadata:', e);
    }

    // Set download WAV link
    const reconUrl = `/api/jobs/${jobId}/files/reconstruction.wav`;
    const corrUrl = `/api/jobs/${jobId}/files/corrupted.wav`;
    const cleanUrl = `/api/jobs/${jobId}/files/clean.wav`;

    activeResultUrls = {
      reconstruction: reconUrl,
      corrupted: corrUrl,
      clean: cleanUrl
    };

    btnDownloadWav.href = reconUrl;

    // Default to playing reconstructed audio
    switchAbAudio('reconstruction');

    // Metrics
    if (job) {
      if (job.started_at && job.completed_at) {
        metricRuntime.textContent = `${(job.completed_at - job.started_at).toFixed(1)} s`;
      } else {
        metricRuntime.textContent = '--';
      }
      metricEpochs.textContent = job.progress?.epochs || '--';
      if (job.metrics) {
        if (job.metrics.n_params) metricParams.textContent = Number(job.metrics.n_params).toLocaleString();
        if (job.metrics.nmse_tot_db !== undefined) {
          metricNmseTot.textContent = `${job.metrics.nmse_tot_db.toFixed(2)} dB`;
          metricCardNmse.classList.remove('hidden');
        } else {
          metricCardNmse.classList.add('hidden');
        }
        if (job.metrics.nmse_miss_db !== undefined) {
          metricNmseMiss.textContent = `${job.metrics.nmse_miss_db.toFixed(2)} dB`;
          metricCardMiss.classList.remove('hidden');
        } else {
          metricCardMiss.classList.add('hidden');
        }
      }
    }

    // Spectrogram Images
    imgSpecCorrupted.src = `/api/jobs/${jobId}/files/spec_corrupted.png`;
    imgSpecReconstructed.src = `/api/jobs/${jobId}/files/spec_reconstructed.png`;
    imgLossCurve.src = `/api/jobs/${jobId}/files/loss_curve.png`;

    if (job && job.files && job.files.includes('spec_clean.png')) {
      imgSpecClean.src = `/api/jobs/${jobId}/files/spec_clean.png`;
      colSpecClean.classList.remove('hidden');
      btnAbClean.classList.remove('hidden');
    } else {
      colSpecClean.classList.add('hidden');
      btnAbClean.classList.add('hidden');
    }

    // Smooth scroll down to results
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
    } else if (type === 'corrupted') {
      btnAbCorrupted.classList.add('active');
      audioPlayback.src = activeResultUrls.corrupted;
    } else if (type === 'clean') {
      btnAbClean.classList.add('active');
      audioPlayback.src = activeResultUrls.clean;
    }

    audioPlayback.currentTime = currentTime;
    if (isPlaying) {
      audioPlayback.play().catch(() => {});
    }
  }

  // --- Examples Gallery ---
  async function loadExamplesGallery() {
    galleryGrid.innerHTML = '<div class="loading-spinner">Loading gallery examples...</div>';

    try {
      const resp = await fetch('examples/index.json');
      if (!resp.ok) {
        galleryGrid.innerHTML = `
          <div class="callout callout-info" style="grid-column: 1/-1;">
            <div>
              <strong>Examples Gallery Empty</strong>
              <p>Precomputed example clips have not been generated yet. Run:
              <code>python scripts/precompute_examples.py</code> to populate the gallery.</p>
            </div>
          </div>
        `;
        return;
      }

      const data = await resp.json();
      const examples = data.examples || [];

      if (examples.length === 0) {
        galleryGrid.innerHTML = '<p class="text-secondary">No examples found in index.json.</p>';
        return;
      }

      galleryGrid.innerHTML = '';
      examples.forEach((ex) => {
        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.innerHTML = `
          <div class="gallery-header">
            <span class="gallery-title">${ex.title || ex.clip_id}</span>
            <span class="gallery-meta">${ex.gap_ms} ms Gap • ${ex.epochs || 5000} ep</span>
          </div>
          <p class="gallery-desc">${ex.description || 'Precomputed benchmark reconstruction.'}</p>
          <div class="img-frame">
            <img src="${ex.spectrogram_url || `examples/${ex.id}/spectrogram.png`}" alt="${ex.title}">
          </div>
          <div class="audio-player-ui">
            <label style="font-size:0.75rem; color:var(--text-muted); display:block; margin-bottom:4px;">Reconstruction:</label>
            <audio controls src="${ex.audio_url || `examples/${ex.id}/reconstruction.wav`}" class="native-player"></audio>
          </div>
          <div class="audio-player-ui">
            <label style="font-size:0.75rem; color:var(--text-muted); display:block; margin-bottom:4px;">Corrupted Input:</label>
            <audio controls src="${ex.corrupted_url || `examples/${ex.id}/corrupted.wav`}" class="native-player"></audio>
          </div>
        `;
        galleryGrid.appendChild(card);
      });
    } catch (e) {
      galleryGrid.innerHTML = `
        <div class="callout callout-info" style="grid-column: 1/-1;">
          <div>
            <strong>Examples Gallery</strong>
            <p>To populate this gallery with precomputed benchmark clips, run:</p>
            <p><code>python scripts/precompute_examples.py</code></p>
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
