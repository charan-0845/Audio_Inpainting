import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, X, Play, Pause, Volume2, VolumeX, RotateCcw, RotateCw, Minus, Plus, Trash2, Settings, Zap, Mic, FileAudio, CheckCircle, AlertCircle, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Badge } from '@/components/ui/Badge'
import { Select } from '@/components/ui/Select'
import { Dialog } from '@/components/ui/Dialog'
import { Tooltip } from '@/components/ui/Tooltip'
import { Separator } from '@/components/ui/Separator'
import { Progress } from '@/components/ui/Progress'
import { Label } from '@/components/ui/Label'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import { cn } from '@/lib/utils'
import { api } from '@/services/api/client'
import type { PresetsResponse } from '@/types/api'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js'
import { PageLayout } from '@/components/layout'
import { detectSilentRegions } from '@/utils/silenceDetection'

// Region type definitions
type RegionType = 'manual' | 'auto' | 'reconstructed'

interface GapRegion {
  id: string
  start: number
  end: number
  type: RegionType
  color: string
}

const REGION_COLORS = {
  manual: '#6D28D9',    // purple - manual missing regions
  auto: '#0D9488',      // teal - auto-detected missing regions
  reconstructed: '#EA580C', // orange - reconstructed regions
}

const PRESET_OPTIONS = [
  { value: 'fast', label: 'Fast', description: '100 epochs, 4 filters, ~30-60s on CPU' },
  { value: 'balanced', label: 'Balanced', description: '300 epochs, 16 filters, balanced quality/speed' },
  { value: 'paper', label: 'Paper', description: '1000 epochs, 32 filters, harmonic on, highest quality' },
  { value: 'lpc', label: 'LPC', description: 'Instant baseline, no neural optimization' },
]

export function WorkspacePage() {
  const navigate = useNavigate()
  
  // Audio state
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [referenceFile, setReferenceFile] = useState<File | null>(null)
  const [audioDuration, setAudioDuration] = useState(0)
  const [sampleRate, setSampleRate] = useState(16000)
  const [waveformData, setWaveformData] = useState<Float32Array | null>(null)
  
  // WaveSurfer
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const regionsPluginRef = useRef<RegionsPlugin | null>(null)
  const syncingRegionsRef = useRef(false)
  const waveformContainerRef = useRef<HTMLDivElement>(null)
  const spectrogramContainerRef = useRef<HTMLDivElement>(null)
  const [waveformReady, setWaveformReady] = useState(false)
  
  // Gap regions (managed locally, synced with WaveSurfer)
  const [regions, setRegions] = useState<GapRegion[]>([])
  const regionIdCounter = useRef(0)
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null)
  const [history, setHistory] = useState<GapRegion[][]>([[]])
  const [historyIndex, setHistoryIndex] = useState(0)
  const historyRef = useRef<GapRegion[][]>([[]])
  const historyIndexRef = useRef(0)
  
  // Refs for callbacks to avoid circular dependencies in keyboard handler
  const updateRegionInStateRef = useRef<((id: string, start: number, end: number) => void) | null>(null)
  const removeRegionFromStateRef = useRef<((id: string) => void) | null>(null)
  
  // Gap modes
  const [gapMode, setGapMode] = useState<'manual' | 'random' | 'detect'>('manual')
  const [gapDuration, setGapDuration] = useState(200)
  const [numGaps, setNumGaps] = useState(1)
  const [silenceThreshold, setSilenceThreshold] = useState(-60)
  const [minSilenceDuration, setMinSilenceDuration] = useState(50)
  
  // Configuration
  const [preset, setPreset] = useState('fast')
  const [epochs, setEpochs] = useState<number | null>(null)
  const [seed, setSeed] = useState(0)
  const [preserveObserved, setPreserveObserved] = useState(true)
  const [presets, setPresets] = useState<PresetsResponse['presets'] | null>(null)
  const [device, setDevice] = useState<'cpu' | 'cuda'>('cpu')
  
  // Job state
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'>('idle')
  const [jobProgress, setJobProgress] = useState({ epoch: 0, epochs: 0, loss: 0, eta: 0 })
  const [jobError, setJobError] = useState<string | null>(null)
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [liveSpectrogram, setLiveSpectrogram] = useState<string | null>(null)
  
  // Validation
  const [validationError, setValidationError] = useState<string | null>(null)
  const [validationWarning, setValidationWarning] = useState<string | null>(null)
  
  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showSpectrogram, setShowSpectrogram] = useState(true)
  const [linkedZoom, setLinkedZoom] = useState(true)

  // Region management callbacks (defined early to avoid hoisting issues)
  const addToHistory = useCallback((newRegions: GapRegion[]) => {
    const newHistory = [...historyRef.current.slice(0, historyIndexRef.current + 1), newRegions].slice(-50)
    const newIndex = newHistory.length - 1
    historyRef.current = newHistory
    historyIndexRef.current = newIndex
    setHistory(newHistory)
    setHistoryIndex(newIndex)
  }, [])
  
  const updateRegionInState = useCallback((id: string, start: number, end: number) => {
    setRegions(prev => {
      const updated = prev.map(r => r.id === id ? { ...r, start, end } : r)
      addToHistory(updated)
      return updated
    })
  }, [addToHistory])
  
  const removeRegionFromState = useCallback((id: string) => {
    setRegions(prev => {
      const updated = prev.filter(r => r.id !== id)
      addToHistory(updated)
      return updated
    })
  }, [addToHistory])
  
  // Update refs after callbacks are defined
  useEffect(() => {
    updateRegionInStateRef.current = updateRegionInState
    removeRegionFromStateRef.current = removeRegionFromState
  }, [updateRegionInState, removeRegionFromState])
  
  const createRegion = useCallback((start: number, end: number, type: RegionType = 'manual') => {
    const id = `region-${++regionIdCounter.current}`
    const region: GapRegion = { id, start, end, type, color: REGION_COLORS[type] }
    
    setRegions(prev => {
      const updated = [...prev, region]
      addToHistory(updated)
      return updated
    })
    
    return region
  }, [addToHistory])
  
  const undo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1
      historyIndexRef.current = newIndex
      setHistoryIndex(newIndex)
      setRegions(history[newIndex])
    }
  }, [history, historyIndex])
  
  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1
      historyIndexRef.current = newIndex
      setHistoryIndex(newIndex)
      setRegions(history[newIndex])
    }
  }, [history, historyIndex])
  
  const clearRegions = useCallback(() => {
    setRegions([])
    setHistory([[]])
    setHistoryIndex(0)
    historyRef.current = [[]]
    historyIndexRef.current = 0
    if (regionsPluginRef.current) {
      regionsPluginRef.current.clearRegions()
    }
  }, [])
  
  // Gap mode handlers
  const handleRandomGaps = useCallback(() => {
    if (!audioDuration) return
    clearRegions()
    
    const totalMs = audioDuration * 1000
    const newRegions: GapRegion[] = []
    
    for (let i = 0; i < numGaps; i++) {
      const gapMs = gapDuration
      const maxStart = totalMs - gapMs - 100
      const start = Math.random() * Math.max(100, maxStart)
      const end = start + gapMs
      newRegions.push({ id: `region-${Date.now()}-${i}`, start: start / 1000, end: end / 1000, type: 'auto', color: REGION_COLORS.auto })
    }
    
    addToHistory(newRegions)
    setRegions(newRegions)
  }, [audioDuration, numGaps, gapDuration, clearRegions, addToHistory])
  
  const handleDetectSilence = useCallback(async () => {
    if (!audioFile || !audioDuration) return
    try {
      const response = await api.detectSilence(audioFile, silenceThreshold, minSilenceDuration)
      const detected: GapRegion[] = response.regions.map((region, index) => ({
        id: `region-${Date.now()}-${index}`,
        start: region.start,
        end: Math.min(region.end, audioDuration),
        type: 'auto',
        color: REGION_COLORS.auto,
      }))

      if (detected.length === 0) {
        setValidationWarning('No silent intervals detected. Try adjusting threshold or mark gaps manually.')
        return
      }

      clearRegions()
      addToHistory(detected)
      setRegions(detected)
      setValidationError(null)
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : 'Silence detection failed')
    }
  }, [audioFile, audioDuration, silenceThreshold, minSilenceDuration, clearRegions, addToHistory])
  
  // Validation
  const validateGaps = useCallback(() => {
    if (!audioDuration) return { valid: false, error: 'No audio loaded' }
    if (regions.length === 0) return { valid: false, error: 'Mark at least one corrupted gap to inpaint' }
    
    const totalMs = audioDuration * 1000
    let totalMissingMs = 0
    let hasGapOver100 = false
    
    const merged: [number, number][] = []
    regions.forEach((r) => {
      if ((r.end - r.start) * 1000 > 100) hasGapOver100 = true
      if (merged.length === 0 || merged[merged.length - 1][1] < r.start) {
        merged.push([r.start, r.end])
      } else {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], r.end)
      }
    })
    
    merged.forEach(([s, e]) => {
      totalMissingMs += (e - s) * 1000
    })
    
    const missingPct = (totalMissingMs / totalMs) * 100
    if (missingPct >= 40) {
      return { valid: false, error: `Total missing audio (${missingPct.toFixed(1)}%) exceeds maximum 40% budget` }
    }
    if (hasGapOver100) {
      return { valid: true, warning: 'Notice: One or more gaps exceed 100 ms. Paper method is calibrated for ≤ 100 ms.' }
    }
    
    return { valid: true }
  }, [audioDuration, regions])
  
  // SSE connection
  const connectToSSE = useCallback((id: string) => {
    const es = new EventSource(`/api/jobs/${id}/events`)
    setEventSource(es)
    
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleSSEEvent(data)
      } catch (e) {
        console.error('Failed to parse SSE event', e)
      }
    }
    
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        setJobStatus('failed')
        setJobError('Connection lost to server')
      }
    }
  }, [])
  
  const handleSSEEvent = useCallback((event: any) => {
    if (event.type === 'status') {
      setJobStatus(event.status)
      if (event.status === 'running') {
        setJobProgress({ epoch: 0, epochs: event.epochs || 0, loss: 0, eta: 0 })
      }
    } else if (event.type === 'progress') {
      setJobStatus('running')
      setJobProgress({
        epoch: event.epoch,
        epochs: event.epochs,
        loss: event.loss,
        eta: event.eta_s || 0,
      })
      setLossHistory(prev => [...prev, event.loss])
      if (event.preview) {
        setLiveSpectrogram(event.preview)
      }
    } else if (event.type === 'completed') {
      setJobStatus('completed')
      setEventSource(null)
      navigate(`/experiments/${event.run_id}`)
    } else if (event.type === 'failed') {
      setJobStatus('failed')
      setJobError(event.error)
      setEventSource(null)
    } else if (event.type === 'cancelled') {
      setJobStatus('cancelled')
      setEventSource(null)
    }
  }, [navigate])
  
  const handleCancel = useCallback(async () => {
    if (!jobId) return
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' })
    } catch (e) {
      console.error('Cancel failed', e)
    }
  }, [jobId])
  
  // Job submission
  const handleSubmit = useCallback(async () => {
    const validation = validateGaps()
    if (!validation.valid) {
      setValidationError(validation.error || 'Invalid gaps')
      return
    }
    setValidationError(null)
    setValidationWarning(validation.warning || null)
    
    if (!audioFile) {
      setValidationError('Please upload an audio file first')
      return
    }
    
    setJobStatus('submitting')
    setJobError(null)
    setLossHistory([])
    setLiveSpectrogram(null)
    
    try {
      const formData = new FormData()
      formData.append('audio', audioFile)
      formData.append('preset', preset)
      formData.append('seed', String(seed))
      formData.append('preserve_observed', String(preserveObserved))
      formData.append('mode', 'gaps')
      formData.append('gaps_json', JSON.stringify(regions.map(r => [Math.round(r.start * 1000), Math.round(r.end * 1000)])))
      
      if (epochs) formData.append('epochs', String(epochs))
      if (referenceFile) formData.append('reference', referenceFile)
      
      const response = await fetch('/api/jobs', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Failed to submit job')
      }
      
      const jobData = await response.json()
      setJobId(jobData.id)
      setJobStatus('queued')
      connectToSSE(jobData.id)
      
    } catch (err) {
      setJobError(err instanceof Error ? err.message : 'Job submission failed')
      setJobStatus('failed')
    }
  }, [audioFile, referenceFile, preset, epochs, seed, preserveObserved, regions, validateGaps])
  
  const handleJobCancel = useCallback(async () => {
    if (!jobId) return
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' })
    } catch (e) {
      console.error('Cancel failed', e)
    }
  }, [jobId])
  
  // Compute gap stats
  const totalMissingMs = regions.reduce((sum, r) => sum + (r.end - r.start) * 1000, 0)
  const missingPct = audioDuration > 0 ? (totalMissingMs / (audioDuration * 1000)) * 100 : 0
  
  // Load presets on mount
  useEffect(() => {
    api.getPresets().then(data => {
      setPresets(data.presets)
      setDevice(data.device as 'cpu' | 'cuda')
      if (data.device === 'cpu') setPreset('fast')
    }).catch(console.error)
  }, [])
  
  // Initialize WaveSurfer
  useEffect(() => {
    if (!waveformContainerRef.current || wavesurferRef.current) return
    
    const regionPlugin = RegionsPlugin.create()
    const ws = WaveSurfer.create({
      container: waveformContainerRef.current,
      waveColor: '#6D28D9',
      progressColor: '#EA580C',
      cursorColor: '#EA580C',
      cursorWidth: 2,
      height: 120,
      barWidth: 2,
      barGap: 1,
      normalize: true,
      interact: true,
      hideScrollbar: true,
      autoCenter: false,
      plugins: [regionPlugin],
    })

    wavesurferRef.current = ws
    regionsPluginRef.current = regionPlugin
    
    ws.on('ready', () => {
      setAudioDuration(ws.getDuration())
      setWaveformReady(true)
    })
    
    regionPlugin.on('region-clicked', (region, e) => {
      e.stopPropagation()
      setSelectedRegionId(region.id)
    })

    regionPlugin.on('region-updated', (region) => {
      updateRegionInState(region.id, region.start, region.end)
    })

    regionPlugin.on('region-removed', (region) => {
      if (!syncingRegionsRef.current) {
        removeRegionFromState(region.id)
      }
    })
    
    ws.on('finish', () => {
      ws.pause()
    })
    
    return () => {
      ws.destroy()
      wavesurferRef.current = null
      regionsPluginRef.current = null
      setWaveformReady(false)
    }
  }, [])
  
  // Sync regions with WaveSurfer
  useEffect(() => {
    if (!regionsPluginRef.current || !waveformReady) return

    // Clear existing regions
    syncingRegionsRef.current = true
    regionsPluginRef.current.clearRegions()
    
    // Add current regions
    regions.forEach(region => {
      regionsPluginRef.current?.addRegion({
        id: region.id,
        start: region.start,
        end: region.end,
        color: region.color + '40', // 25% opacity
        drag: true,
        resize: true,
      })
    })
    syncingRegionsRef.current = false
  }, [regions, waveformReady])
  
  // Keyboard accessibility for region editing
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!selectedRegionId || !audioDuration) return
      
      const region = regions.find(r => r.id === selectedRegionId)
      if (!region) return
      
      const stepMs = 10 // 10ms step
      const minGapMs = 10
      
      let newStart = region.start
      let newEnd = region.end
      let shouldUpdate = false
      
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault()
          if (e.shiftKey) {
            // Shift+Left: move start earlier
            newStart = Math.max(0, region.start - stepMs / 1000)
          } else {
            // Left: move end earlier (shrink from right)
            newEnd = Math.max(region.start + minGapMs / 1000, region.end - stepMs / 1000)
          }
          shouldUpdate = true
          break
        case 'ArrowRight':
          e.preventDefault()
          if (e.shiftKey) {
            // Shift+Right: move start later
            newStart = Math.min(region.end - minGapMs / 1000, region.start + stepMs / 1000)
          } else {
            // Right: move end later (expand to right)
            newEnd = Math.min(audioDuration, region.end + stepMs / 1000)
          }
          shouldUpdate = true
          break
        case 'Delete':
        case 'Backspace':
          e.preventDefault()
          if (removeRegionFromStateRef.current) removeRegionFromStateRef.current(selectedRegionId)
          setSelectedRegionId(null)
          return
        case 'Escape':
          setSelectedRegionId(null)
          return
      }
      
      if (shouldUpdate && newStart < newEnd) {
        // Also update WaveSurfer region
        if (regionsPluginRef.current) {
          const wsRegion = regionsPluginRef.current.getRegions().find(region => region.id === selectedRegionId)
          if (wsRegion) {
            wsRegion.setOptions({ start: newStart, end: newEnd })
          }
        }
        if (updateRegionInStateRef.current) updateRegionInStateRef.current(selectedRegionId, newStart, newEnd)
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedRegionId, regions, audioDuration, updateRegionInStateRef, removeRegionFromStateRef])
  
  // Handle file upload
  const handleAudioUpload = useCallback(async (file: File) => {
    setAudioFile(file)
    setValidationError(null)
    setValidationWarning(null)
    
    const url = URL.createObjectURL(file)
    const audio = new Audio()
    audio.src = url
    
    audio.onloadedmetadata = () => {
      setAudioDuration(audio.duration)
      setSampleRate(16000)
      URL.revokeObjectURL(url)
    }
    
    audio.onerror = () => {
      setValidationError('Failed to load audio file')
      URL.revokeObjectURL(url)
    }
    
    // Load into WaveSurfer
    if (wavesurferRef.current) {
      wavesurferRef.current.loadBlob(file)
    }
    
    // Decode for waveform data
    const arrayBuffer = await file.arrayBuffer()
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
    const channelData = audioBuffer.getChannelData(0)
    setSampleRate(audioBuffer.sampleRate)
    setWaveformData(channelData)
  }, [])
  
  const handleReferenceUpload = useCallback((file: File) => {
    setReferenceFile(file)
  }, [])
  
  // Handle detect silence (needs to be after handleAudioUpload for clearRegions)
    
  // Compute gap stats
    
  // Load presets on mount
  useEffect(() => {
    api.getPresets().then(data => {
      setPresets(data.presets)
      setDevice(data.device as 'cpu' | 'cuda')
      if (data.device === 'cpu') setPreset('fast')
    }).catch(console.error)
  }, [])
  
  return (
    <PageLayout title="New Reconstruction" description="Upload audio, mark corrupted regions, and run inpainting">
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left panel - Waveform editor (65%) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Audio upload & info */}
          <Card variant="elevated">
            <CardContent className="p-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className={cn(
                    'w-12 h-12 rounded-lg flex items-center justify-center',
                    audioFile ? 'bg-[var(--primary)]/10' : 'bg-[var(--raised)]'
                  )}>
                    {audioFile ? (
                      <FileAudio className="h-6 w-6 text-[var(--primary)]" />
                    ) : (
                      <span className="text-[var(--fg-muted)]">+</span>
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-[var(--fg)]">
                      {audioFile ? audioFile.name : 'No audio loaded'}
                    </p>
                    {audioDuration && (
                      <p className="text-sm text-[var(--fg-muted)] font-mono">
                        {sampleRate / 1000} kHz • {audioDuration.toFixed(2)}s • {audioFile ? (audioFile.size / 1024).toFixed(1) : 0} KB
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="file"
                    accept="audio/*,.wav"
                    onChange={e => { const file = e.target.files?.[0]; if (file) handleAudioUpload(file); }}
                    className="hidden"
                    id="audio-upload"
                  />
                  <input
                    type="file"
                    accept="audio/*,.wav"
                    onChange={e => { const file = e.target.files?.[0]; if (file) handleReferenceUpload(file); }}
                    className="hidden"
                    id="reference-upload"
                  />
                  <Button variant="secondary" onClick={() => { const el = document.getElementById('audio-upload'); if (el) el.click(); }}>
                    <FileAudio className="h-4 w-4 mr-1" />
                    {audioFile ? 'Replace' : 'Upload WAV'}
                  </Button>
                  <Button variant="ghost" onClick={() => { const el = document.getElementById('reference-upload'); if (el) el.click(); }}>
                    <Mic className="h-4 w-4 mr-1" />
                    {referenceFile ? 'Replace Reference' : 'Add Clean Reference'}
                  </Button>
                  {audioFile && (
                    <Button variant="ghost" size="icon" onClick={() => { setAudioFile(null); setWaveformData(null); setAudioDuration(0); clearRegions(); }}>
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
              
              {referenceFile && (
                <div className="mt-3 flex items-center gap-2 text-sm text-[var(--fg-muted)]">
                  <Mic className="h-4 w-4" />
                  <span>Reference: {referenceFile.name}</span>
                  <Button variant="ghost" size="icon" onClick={() => setReferenceFile(null)}>
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Waveform with gap editor */}
          <Card variant="elevated">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Waveform & Gap Editor</CardTitle>
                <div className="flex items-center gap-2 text-xs text-[var(--fg-muted)]">
                  <span>Missing: <span className="font-mono text-[var(--manual)]">{Math.round(totalMissingMs)}ms</span></span>
                  <Separator orientation="vertical" className="h-4 mx-2" />
                  <span>{missingPct.toFixed(1)}% of total</span>
                  {validationError && (
                    <Badge variant="danger" className="ml-2"><AlertCircle className="h-3 w-3 mr-1" />{validationError}</Badge>
                  )}
                  {validationWarning && !validationError && (
                    <Badge variant="warning" className="ml-2"><AlertTriangle className="h-3 w-3 mr-1" />{validationWarning}</Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent className="pb-4">
              {/* Toolbar */}
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <div className="flex items-center gap-2">
                  <Button
                    variant={jobStatus === 'running' ? 'secondary' : 'ghost'}
                    size="sm"
                    onClick={() => { if (wavesurferRef.current) wavesurferRef.current.playPause(); }}
                    disabled={!audioFile || jobStatus === 'running'}
                  >
                    {wavesurferRef.current && wavesurferRef.current.isPlaying() ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </Button>
                  <Tooltip content="Random gaps" side="top">
                    <Button variant="ghost" size="sm" onClick={handleRandomGaps} disabled={!audioDuration || jobStatus === 'running'}>
                      <Zap className="h-4 w-4 mr-1" /> Random
                    </Button>
                  </Tooltip>
                  <Tooltip content="Auto-detect silence" side="top">
                    <Button variant="ghost" size="sm" onClick={handleDetectSilence} disabled={!audioFile || jobStatus === 'running'}>
                      <Mic className="h-4 w-4 mr-1" /> Detect Silence
                    </Button>
                  </Tooltip>
                  <Tooltip content="Clear all gaps" side="top">
                    <Button variant="ghost" size="sm" onClick={clearRegions} disabled={regions.length === 0 || jobStatus === 'running'}>
                      <Trash2 className="h-4 w-4 mr-1" /> Clear
                    </Button>
                  </Tooltip>
                  <Tooltip content="Undo (Ctrl+Z)" side="top">
                    <Button variant="ghost" size="icon" onClick={undo} disabled={historyIndex <= 0}>
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  </Tooltip>
                  <Tooltip content="Redo (Ctrl+Y)" side="top">
                    <Button variant="ghost" size="icon" onClick={redo} disabled={historyIndex >= history.length - 1}>
                      <RotateCw className="h-4 w-4" />
                    </Button>
                  </Tooltip>
                </div>
                
                <div className="flex-1" />
                
              </div>
              
              {/* Waveform container */}
              <div ref={waveformContainerRef} className="rounded-md bg-[var(--bg)] border border-[var(--border)] overflow-hidden" />
              
              {/* Region list */}
              {regions.length > 0 && (
                <div className="mt-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-medium text-[var(--fg-muted)] uppercase tracking-wider">Regions</span>
                    <div className="flex items-center gap-1 ml-auto">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: REGION_COLORS.manual }} />
                      <span className="text-xs text-[var(--fg-muted)]">Manual</span>
                      <span className="w-2 h-2 rounded-full ml-2" style={{ backgroundColor: REGION_COLORS.auto }} />
                      <span className="text-xs text-[var(--fg-muted)]">Auto-detected</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {regions.map((region, index) => (
                      <button
                        key={region.id}
                        type="button"
                        className={cn(
                          'flex w-full items-center justify-between rounded-md border px-3 py-2 text-left text-xs transition-colors',
                          selectedRegionId === region.id
                            ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                            : 'border-[var(--border)] hover:border-[var(--accent)]/60',
                        )}
                        onClick={() => {
                          setSelectedRegionId(region.id)
                          regionsPluginRef.current?.getRegions()
                            .find(waveRegion => waveRegion.id === region.id)
                            ?.setOptions({ start: region.start, end: region.end })
                        }}
                      >
                        {(() => {
                          const durationMs = (region.end - region.start) * 1000
                          const durationLabel = durationMs >= 1000
                            ? `${(durationMs / 1000).toFixed(2)} s`
                            : `${durationMs.toFixed(0)} ms`
                          return (
                            <>
                              <span className="flex items-center gap-2">
                                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: region.color }} />
                                <span>Region {index + 1}</span>
                                <span className="text-[var(--fg-muted)]">
                                  {region.type === 'auto' ? 'Auto-detected' : 'Manual'}
                                </span>
                              </span>
                              <span className="flex items-center gap-3 font-mono text-[var(--fg-muted)]">
                                <span>{(region.start * 1000).toFixed(0)}–{(region.end * 1000).toFixed(0)} ms</span>
                                <span className="font-semibold text-[var(--fg)]">{durationLabel}</span>
                              </span>
                            </>
                          )
                        })()}
                      </button>
                    ))}
                  </div>
                </div>
          )} 
          </CardContent>
            </Card>
            
            {/* Reconstruction monitoring */}
            <Card variant="elevated" className="mt-6">
              <CardHeader>
                <CardTitle className="text-base">Reconstruction</CardTitle>
              </CardHeader>
              <CardContent className="pt-0 space-y-4">
                {jobStatus === 'idle' && (
                  <Button
                    className="w-full justify-center gap-2"
                    size="lg"
                    onClick={handleSubmit}
                    disabled={!audioFile || regions.length === 0 || !validateGaps().valid}
                  >
                    <Zap className="h-5 w-5" />
                    Execute Inpainting
                  </Button>
                )}

                {jobStatus !== 'idle' && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Badge variant={jobStatus === 'running' ? 'primary' : jobStatus === 'completed' ? 'success' : jobStatus === 'failed' ? 'danger' : 'outline'}>
                        {jobStatus === 'running' && <Loader2 className="h-3 w-3 animate-spin mr-1" />}
                        {jobStatus.charAt(0).toUpperCase() + jobStatus.slice(1)}
                      </Badge>
                      {jobStatus === 'running' && (
                        <Button variant="danger" size="sm" onClick={handleJobCancel}>
                          <X className="h-3 w-3 mr-1" /> Cancel
                        </Button>
                      )}
                    </div>

                    {jobStatus === 'running' && (
                      <>
                        <Progress value={(jobProgress.epoch / jobProgress.epochs) * 100} className="h-2" />
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-[var(--fg-muted)]">Epoch</span>
                            <p className="font-mono text-[var(--fg)]">{jobProgress.epoch} / {jobProgress.epochs}</p>
                          </div>
                          <div>
                            <span className="text-[var(--fg-muted)]">ETA</span>
                            <p className="font-mono text-[var(--fg)]">{jobProgress.eta}s</p>
                          </div>
                          <div>
                            <span className="text-[var(--fg-muted)]">Loss</span>
                            <p className="font-mono text-[var(--fg)]">{jobProgress.loss ? jobProgress.loss.toExponential(4) : '—'}</p>
                          </div>
                          <div>
                            <span className="text-[var(--fg-muted)]">Device</span>
                            <p className="font-mono text-[var(--fg)] capitalize">{device}</p>
                          </div>
                        </div>

                        {lossHistory.length > 1 && (
                          <div className="h-64 bg-[var(--bg)] border border-[var(--border)] rounded-md p-3 relative">
                            <canvas
                              ref={el => {
                                if (el) {
                                  const ctx = el.getContext('2d')
                                  if (!ctx) return
                                  const w = el.width = el.offsetWidth
                                  const h = el.height = el.offsetHeight
                                  ctx.clearRect(0, 0, w, h)
                                  ctx.fillStyle = '#0E0F11'
                                  ctx.fillRect(0, 0, w, h)
                                  ctx.strokeStyle = '#EA580C'
                                  ctx.lineWidth = 2
                                  ctx.beginPath()
                                  const minLoss = Math.min(...lossHistory)
                                  const maxLoss = Math.max(...lossHistory)
                                  const range = maxLoss - minLoss || 1e-6
                                  lossHistory.forEach((loss, i) => {
                                    const x = (i / (lossHistory.length - 1)) * (w - 12) + 6
                                    const normalized = (loss - minLoss) / range
                                    const y = (1 - normalized) * (h - 20) + 10
                                    if (i === 0) ctx.moveTo(x, y)
                                    else ctx.lineTo(x, y)
                                  })
                                  ctx.stroke()
                                }
                              }}
                              className="w-full h-full"
                              aria-label="Live loss curve"
                            />
                          </div>
                        )}

                        {liveSpectrogram && (
                          <img src={liveSpectrogram} alt="Live spectrogram" className="w-full max-h-[28rem] object-contain rounded-md border border-[var(--border)]" />
                        )}
                      </>
                    )}

                    {jobError && (
                      <div className="p-3 rounded-md bg-[var(--danger)]/10 border border-[var(--danger)]/20 text-[var(--danger)] text-sm">
                        {jobError}
                      </div>
                    )}
                  </div>
                )}

                {validationError && jobStatus === 'idle' && (
                  <div className="p-3 rounded-md bg-[var(--danger)]/10 border border-[var(--danger)]/20 text-[var(--danger)] text-sm">
                    {validationError}
                  </div>
                )}
              </CardContent>
            </Card>
        </div>
        
        {/* Right panel - Configuration (35%) */}
        <div className="space-y-6">
          {/* Preset selection */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle className="text-base">Optimization Preset</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="grid grid-cols-2 gap-2">
                {PRESET_OPTIONS.map(p => (
                  <label
                    key={p.value}
                    className={cn(
                      'relative p-3 rounded-md border-2 text-center cursor-pointer transition-all',
                      'hover:border-[var(--border-strong)]',
                      preset === p.value
                        ? 'border-[var(--primary)] bg-[var(--primary)]/5'
                        : 'border-[var(--border)]'
                    )}
                  >
                    <input
                      type="radio"
                      name="preset"
                      value={p.value}
                      checked={preset === p.value}
                      onChange={e => setPreset(e.target.value)}
                      className="sr-only"
                    />
                    <div className="font-medium text-[var(--fg)]">{p.label}</div>
                    <div className="text-xs text-[var(--fg-muted)] mt-1">{p.description}</div>
                    {preset === p.value && (
                      <CheckCircle className="absolute top-1 right-1 h-4 w-4 text-[var(--primary)]" />
                    )}
                  </label>
                ))}
              </div>
            </CardContent>
          </Card>
          
          {/* Basic settings */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle className="text-base">Basic Settings</CardTitle>
            </CardHeader>
            <CardContent className="pt-0 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <Input
                  label="Epochs"
                  type="number"
                  placeholder={presets && presets[preset] && presets[preset].epochs ? presets[preset].epochs.toString() : '—'}
                  value={epochs || ''}
                  onChange={e => setEpochs(e.target.value ? parseInt(e.target.value) : null)}
                  min={1}
                  max={10000}
                />
                <Input
                  label="Seed"
                  type="number"
                  value={seed}
                  onChange={e => setSeed(parseInt(e.target.value) || 0)}
                  min={0}
                  max={999999}
                />
              </div>
              
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="preserve-observed"
                  checked={preserveObserved}
                  onChange={e => setPreserveObserved(e.target.checked)}
                  className="w-4 h-4 rounded border-[var(--border)] text-[var(--primary)] focus:ring-[var(--primary)]"
                />
                <Label htmlFor="preserve-observed" className="text-sm cursor-pointer">
                  Paste observed samples (bit-exact outside gaps)
                </Label>
              </div>
            </CardContent>
          </Card>
          
          {/* Advanced settings */}
          <Card variant="elevated">
            <CardHeader className="flex items-center justify-between">
              <CardTitle className="text-base">Advanced Settings</CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setShowAdvanced(!showAdvanced)}>
                {showAdvanced ? <Minus className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
              </Button>
            </CardHeader>
            {showAdvanced && (
              <CardContent className="pt-0 space-y-4">
                <div>
                  <Label>Gap Mode</Label>
                  <Select
                    value={gapMode}
                    onValueChange={(v) => setGapMode(v as 'manual' | 'random' | 'detect')}
                    options={[
                      { value: 'manual', label: 'Manual' },
                      { value: 'random', label: 'Random/Synthetic' },
                      { value: 'detect', label: 'Auto Silence Detection' },
                    ]}
                  />
                </div>
                
                {gapMode === 'random' && (
                  <div className="grid grid-cols-2 gap-4">
                    <Input label="Gap Duration (ms)" type="number" value={gapDuration} onChange={e => setGapDuration(parseInt(e.target.value) || 0)} min={10} max={1000} />
                    <Input label="Number of Gaps" type="number" value={numGaps} onChange={e => setNumGaps(parseInt(e.target.value) || 0)} min={1} max={10} />
                  </div>
                )}
                
                {gapMode === 'detect' && (
                  <div className="grid grid-cols-2 gap-4">
                    <Input label="Threshold (dB)" type="number" value={silenceThreshold} onChange={e => setSilenceThreshold(parseInt(e.target.value))} min={-100} max={-20} step={1} />
                    <Input label="Min Duration (ms)" type="number" value={minSilenceDuration} onChange={e => setMinSilenceDuration(parseInt(e.target.value) || 0)} min={10} max={500} />
                  </div>
                )}
                
                <Separator />
                
                <div className="text-xs text-[var(--fg-muted)]">
                  <p>Device: <span className="font-mono capitalize">{device}</span></p>
                  {presets && preset in presets && (
                    <p>Preset epochs: <span className="font-mono">{presets[preset as keyof typeof presets].epochs}</span></p>
                  )}
                </div>
              </CardContent>
            )}
          </Card>
          
        </div>
      </div>
    </PageLayout>
  )
}