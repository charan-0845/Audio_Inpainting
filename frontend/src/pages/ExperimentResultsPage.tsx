import { useState, useEffect, useRef, type CSSProperties, type MouseEvent } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Download, Volume2, VolumeX, RotateCcw, RotateCw, ChevronLeft, ChevronRight, ArrowRight, CheckCircle, XCircle, AlertTriangle, FileAudio, Settings, BarChart2, Music, Mic } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { EmptyState, LoadingState, ErrorState } from '@/components/ui/StateComponents'
import { Tooltip } from '@/components/ui/Tooltip'
import { Separator } from '@/components/ui/Separator'
import { api, ApiError } from '@/services/api/client'
import type { JobResponse, JobMetrics } from '@/types/api'
import { formatDuration, formatMs, formatNumber, formatDate } from '@/lib/utils'
import { PageLayout, PageHeader } from '@/components/layout'
import { cn } from '@/lib/utils'

export function ExperimentResultsPage() {
  const { id } = useParams<{ id: string }>()
  const [job, setJob] = useState<JobResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'audio' | 'visual' | 'config'>('overview')
  const [audioSources, setAudioSources] = useState<{ corrupted: string; reconstructed: string; clean?: string }>({
    corrupted: '', reconstructed: '', clean: undefined
  })
  const [activeAudio, setActiveAudio] = useState<'reconstructed' | 'corrupted' | 'clean'>('reconstructed')
  const audioElementsRef = useRef<Record<string, HTMLAudioElement>>({})
  const waveformCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const waveformBufferRef = useRef<AudioBuffer | null>(null)
  const redrawWaveformRef = useRef<(() => void) | null>(null)
  const hoverPointRef = useRef<{ x: number; time: number; amplitude: number } | null>(null)
  const [waveformHover, setWaveformHover] = useState<{ x: number; time: number; amplitude: number } | null>(null)
  const [playingAudio, setPlayingAudio] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    loadJob()
  }, [id])

  const loadJob = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getJob(id!)
      setJob(data)
      const baseUrl = `/api/jobs/${data.id}/files`
      setAudioSources({
        corrupted: `${baseUrl}/corrupted.wav`,
        reconstructed: `${baseUrl}/reconstruction.wav`,
        clean: data.files.includes('clean.wav') ? `${baseUrl}/clean.wav` : undefined,
      })
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError('Failed to load experiment')
      }
    } finally {
      setLoading(false)
    }
  }

  const downloadFile = (filename: string) => {
    window.open(`/api/jobs/${id}/files/${filename}?download=true`)
  }

  const audioFilename = (type: string) => type === 'reconstructed' ? 'reconstruction.wav' : `${type}.wav`

  useEffect(() => {
    const source = audioSources[activeAudio]
    const canvas = waveformCanvasRef.current
    if (!source || !canvas) return

    let cancelled = false
    const drawWaveform = async () => {
      try {
        const response = await fetch(source)
        if (!response.ok) throw new Error(`Unable to load ${activeAudio} audio`)
        const audioContext = new AudioContext()
        const buffer = await audioContext.decodeAudioData(await response.arrayBuffer())
        await audioContext.close()
        if (cancelled) return
        waveformBufferRef.current = buffer

        const context = canvas.getContext('2d')
        if (!context) return
        const draw = () => {
          const rect = canvas.getBoundingClientRect()
          const ratio = window.devicePixelRatio || 1
          canvas.width = Math.max(1, Math.floor(rect.width * ratio))
          canvas.height = Math.max(1, Math.floor(rect.height * ratio))
          context.setTransform(ratio, 0, 0, ratio, 0, 0)
          context.clearRect(0, 0, rect.width, rect.height)
          context.fillStyle = getComputedStyle(canvas).getPropertyValue('--waveform-background').trim() || '#111827'
          context.fillRect(0, 0, rect.width, rect.height)
          context.strokeStyle = getComputedStyle(canvas).getPropertyValue('--waveform-color').trim() || '#8b5cf6'
          context.lineWidth = 1.5
          context.beginPath()

          const samples = buffer.getChannelData(0)
          const mid = rect.height / 2
          const amplitude = rect.height * 0.42
          const step = Math.max(1, Math.floor(samples.length / rect.width))
          for (let x = 0; x < rect.width; x += 1) {
            const start = x * step
            const end = Math.min(samples.length, start + step)
            let peak = 0
            for (let i = start; i < end; i += 1) peak = Math.max(peak, Math.abs(samples[i]))
            context.moveTo(x + 0.5, mid - peak * amplitude)
            context.lineTo(x + 0.5, mid + peak * amplitude)
          }
          context.stroke()

          const hover = hoverPointRef.current
          if (hover) {
            context.strokeStyle = 'rgba(255, 255, 255, 0.75)'
            context.lineWidth = 1
            context.beginPath()
            context.moveTo(hover.x + 0.5, 0)
            context.lineTo(hover.x + 0.5, rect.height)
            context.stroke()
            context.fillStyle = getComputedStyle(canvas).getPropertyValue('--waveform-color').trim() || '#8b5cf6'
            context.beginPath()
            context.arc(hover.x, mid - hover.amplitude * amplitude, 3, 0, Math.PI * 2)
            context.fill()
          }
        }

        redrawWaveformRef.current = draw
        draw()
        window.addEventListener('resize', draw)
        return () => {
          window.removeEventListener('resize', draw)
          redrawWaveformRef.current = null
          waveformBufferRef.current = null
          hoverPointRef.current = null
        }
      } catch (error) {
        if (!cancelled) setError(error instanceof Error ? error.message : 'Unable to draw waveform')
      }
    }

    let cleanup: (() => void) | undefined
    drawWaveform().then(result => { cleanup = result })
    return () => {
      cancelled = true
      cleanup?.()
    }
  }, [activeAudio, activeTab, audioSources])

  const handleWaveformMove = (event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget
    const rect = canvas.getBoundingClientRect()
    const buffer = waveformBufferRef.current
    if (!buffer || rect.width <= 0) return

    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left))
    const time = (x / rect.width) * buffer.duration
    const sampleIndex = Math.min(buffer.length - 1, Math.floor((time / buffer.duration) * buffer.length))
    const amplitude = buffer.getChannelData(0)[sampleIndex] || 0
    const point = { x, time, amplitude }
    hoverPointRef.current = point
    setWaveformHover(point)
    redrawWaveformRef.current?.()
  }

  const clearWaveformHover = () => {
    hoverPointRef.current = null
    setWaveformHover(null)
    redrawWaveformRef.current?.()
  }

  const switchAudio = (type: 'reconstructed' | 'corrupted' | 'clean') => {
    Object.entries(audioElementsRef.current).forEach(([key, element]) => {
      if (key !== type) element.pause()
    })
    setActiveAudio(type)
  }

  const togglePlayback = async (type: string) => {
    const element = audioElementsRef.current[type]
    if (!element) return

    if (playingAudio === type) {
      element.pause()
      setPlayingAudio(null)
      return
    }

    Object.entries(audioElementsRef.current).forEach(([key, other]) => {
      if (key !== type) other.pause()
    })
    try {
      await element.play()
      setActiveAudio(type as 'reconstructed' | 'corrupted' | 'clean')
      setPlayingAudio(type)
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Unable to play audio')
    }
  }

  const statusBadges: Record<string, 'success' | 'primary' | 'danger' | 'warning' | 'outline'> = {
    completed: 'success',
    running: 'primary',
    queued: 'primary',
    failed: 'danger',
    cancelled: 'outline',
  }

  if (loading) {
    return (
      <PageLayout title="Experiment Results" description="Loading...">
        <LoadingState message="Loading experiment results..." />
      </PageLayout>
    )
  }

  if (error || !job) {
    return (
      <PageLayout title="Experiment Results" description="Error loading experiment">
        <ErrorState message={error || 'Experiment not found'} onRetry={loadJob} />
      </PageLayout>
    )
  }

  const isCompleted = job.status === 'completed'
  const hasClean = job.files.includes('clean.wav')
  const metrics = job.metrics

  return (
    <PageLayout
      title="Experiment Results"
      description={`Run: ${job.id}`}
      action={
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => downloadFile('reconstruction.wav')} disabled={!isCompleted}>
            <Download className="h-4 w-4 mr-1" /> Download WAV
          </Button>
          <Link to="/experiments">
            <Button variant="ghost">
              <ChevronLeft className="h-4 w-4 mr-1" /> Back
            </Button>
          </Link>
        </div>
      }
    >
      {/* Status header */}
      <Card variant="elevated" className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <Badge variant={statusBadges[job.status] || 'outline'} className="text-sm">
                  {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                </Badge>
                <span className="font-mono text-sm text-[var(--fg-muted)]">{job.id}</span>
              </div>
              <p className="mt-1 text-sm text-[var(--fg-muted)]">
                {job.clip_id} • {formatMs(job.gap_ms)} gap • {job.epochs} epochs • {job.seed ? `Seed: ${job.seed}` : 'No seed'}
              </p>
            </div>
            <div className="flex items-center gap-4 text-sm text-[var(--fg-muted)]">
              {job.started_at && job.completed_at && (
                <span>Runtime: <span className="font-mono text-[var(--fg)]">{formatDuration(job.completed_at - job.started_at)}</span></span>
              )}
              {job.n_params && (
                <span>Params: <span className="font-mono text-[var(--fg)]">{formatNumber(job.n_params)}</span></span>
              )}
              {job.device && (
                <span>Device: <span className="font-mono capitalize text-[var(--fg)]">{job.device}</span></span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} defaultValue="overview" onValueChange={(v) => setActiveTab(v as 'audio' | 'overview' | 'visual' | 'config')} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="audio">Audio Comparison</TabsTrigger>
          <TabsTrigger value="visual">Visual Analysis</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card variant="elevated">
              <CardContent className="pt-6">
                <p className="text-sm text-[var(--fg-muted)]">Overall NMSE</p>
                <p className="text-2xl font-bold text-[var(--recon)] mt-1">
                  {job.metrics && job.metrics.nmse_tot_db !== null ? `${job.metrics.nmse_tot_db.toFixed(2)} dB` : '—'}
                </p>
                <p className="text-xs text-[var(--fg-muted)] mt-1">Lower is better</p>
              </CardContent>
            </Card>
            <Card variant="elevated">
              <CardContent className="pt-6">
                <p className="text-sm text-[var(--fg-muted)]">Gap Region NMSE</p>
                <p className="text-2xl font-bold text-[var(--recon)] mt-1">
                  {job.metrics && job.metrics.nmse_miss_db !== null ? `${job.metrics.nmse_miss_db.toFixed(2)} dB` : '—'}
                </p>
                <p className="text-xs text-[var(--fg-muted)] mt-1">Lower is better</p>
              </CardContent>
            </Card>
            <Card variant="elevated">
              <CardContent className="pt-6">
                <p className="text-sm text-[var(--fg-muted)]">PESQ Delta</p>
                <p className="text-2xl font-bold text-[var(--missing)] mt-1">
                  {job.metrics && job.metrics.pesq_delta !== null ? job.metrics.pesq_delta.toFixed(3) : hasClean ? 'Computing...' : '—'}
                </p>
                <p className="text-xs text-[var(--fg-muted)] mt-1">{hasClean ? 'Higher is better' : 'No reference audio'}</p>
              </CardContent>
            </Card>
            <Card variant="elevated">
              <CardContent className="pt-6">
                <p className="text-sm text-[var(--fg-muted)]">Runtime</p>
                <p className="text-2xl font-bold text-[var(--fg)] mt-1">
                  {job.started_at && job.completed_at ? formatDuration(job.completed_at - job.started_at) : '—'}
                </p>
                <p className="text-xs text-[var(--fg-muted)] mt-1">Total optimization time</p>
              </CardContent>
            </Card>
          </div>

          {/* Metadata */}
          <Card variant="elevated">
            <CardHeader><CardTitle>Metadata</CardTitle></CardHeader>
            <CardContent>
              <div className="grid sm:grid-cols-2 gap-4 text-sm">
                <div><span className="text-[var(--fg-muted)]">Run ID:</span> <code className="ml-2 font-mono text-[var(--fg)]">{job.id}</code></div>
                <div><span className="text-[var(--fg-muted)]">Clip:</span> <span className="ml-2 font-mono text-[var(--fg)]">{job.clip_id}</span></div>
                <div><span className="text-[var(--fg-muted)]">Gap:</span> <span className="ml-2 font-mono text-[var(--fg)]">{formatMs(job.gap_ms)}</span></div>
                <div><span className="text-[var(--fg-muted)]">Seed:</span> <span className="ml-2 font-mono text-[var(--fg)]">{job.seed}</span></div>
                <div><span className="text-[var(--fg-muted)]">Epochs:</span> <span className="ml-2 font-mono text-[var(--fg)]">{job.epochs}</span></div>
                <div><span className="text-[var(--fg-muted)]">Parameters:</span> <span className="ml-2 font-mono text-[var(--fg)]">{job.n_params ? formatNumber(job.n_params) : '—'}</span></div>
                <div><span className="text-[var(--fg-muted)]">Created:</span> <span className="ml-2 font-mono text-[var(--fg)]">{formatDate(job.created_at)}</span></div>
                <div><span className="text-[var(--fg-muted)]">Status:</span> <Badge variant={statusBadges[job.status] || 'outline'} className="ml-2">{job.status}</Badge></div>
                {job.warnings.length > 0 && (
                  <div className="sm:col-span-2">
                    <span className="text-[var(--fg-muted)]">Warnings:</span>
                    <ul className="ml-2 list-disc text-[var(--danger)] text-sm mt-1">
                      {job.warnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Downloads */}
          <Card variant="elevated">
            <CardHeader><CardTitle>Downloads</CardTitle></CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                <Button variant="secondary" onClick={() => downloadFile('reconstruction.wav')} disabled={!isCompleted} className="gap-2">
                  <FileAudio className="h-4 w-4" /> Reconstruction (.wav)
                </Button>
                <Button variant="secondary" onClick={() => downloadFile('corrupted.wav')} disabled={!isCompleted} className="gap-2">
                  <FileAudio className="h-4 w-4" /> Corrupted Input (.wav)
                </Button>
                {hasClean && (
                  <Button variant="secondary" onClick={() => downloadFile('clean.wav')} className="gap-2">
                    <FileAudio className="h-4 w-4" /> Clean Reference (.wav)
                  </Button>
                )}
                <Button variant="secondary" onClick={() => downloadFile('config_snapshot.yaml')} className="gap-2">
                  <Settings className="h-4 w-4" /> Config (.yaml)
                </Button>
                <Button variant="secondary" onClick={() => downloadFile('metrics.json')} className="gap-2">
                  <BarChart2 className="h-4 w-4" /> Metrics (.json)
                </Button>
                <Button variant="secondary" onClick={() => downloadFile('loss_curve.png')} className="gap-2">
                  <Music className="h-4 w-4" /> Loss Curve (.png)
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Audio Comparison Tab */}
        <TabsContent value="audio" className="space-y-6">
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              { key: 'reconstructed', label: 'Reconstructed', icon: CheckCircle, color: 'text-[var(--recon)]' },
              { key: 'corrupted', label: 'Corrupted Input', icon: XCircle, color: 'text-[var(--danger)]' },
              ...(hasClean ? [{ key: 'clean', label: 'Clean Reference', icon: Music, color: 'text-[var(--missing)]' }] : []),
            ].map(({ key, label, icon: Icon, color }) => (
              <Card key={key} variant="elevated" className={cn(activeAudio === key && 'ring-2 ring-[var(--primary)]')}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Icon className={cn('h-4 w-4', color)} />
                      {label}
                    </CardTitle>
                    <Button
                      variant={playingAudio === key ? 'primary' : 'ghost'}
                      size="sm"
                      onClick={() => togglePlayback(key)}
                    >
                      {playingAudio === key ? 'Pause' : 'Play'}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <audio
                    ref={el => {
                      if (el) audioElementsRef.current[key] = el
                    }}
                    src={audioSources[key as keyof typeof audioSources]}
                    controls
                    onPlay={() => { setActiveAudio(key as 'reconstructed' | 'corrupted' | 'clean'); setPlayingAudio(key) }}
                    onPause={() => setPlayingAudio(current => current === key ? null : current)}
                    onEnded={() => setPlayingAudio(current => current === key ? null : current)}
                    className="w-full"
                  />
                  <div className="mt-2 flex items-center justify-between">
                    <Button variant="ghost" size="sm" onClick={() => downloadFile(audioFilename(key))} className="gap-1">
                      <Download className="h-3 w-3" /> Download
                    </Button>
                    <span className="text-xs text-[var(--fg-muted)]">16-bit PCM WAV</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {activeAudio && audioSources[activeAudio] && (
            <Card variant="elevated">
              <CardHeader><CardTitle>Waveform: {activeAudio.charAt(0).toUpperCase() + activeAudio.slice(1)}</CardTitle></CardHeader>
              <CardContent>
                <div className="h-48 bg-[var(--bg)] border border-[var(--border)] rounded-md relative overflow-hidden">
                  <canvas
                    ref={waveformCanvasRef}
                    id="result-waveform"
                    aria-label={`${activeAudio} audio waveform`}
                    className="w-full h-full"
                    onMouseMove={handleWaveformMove}
                    onMouseLeave={clearWaveformHover}
                    style={{
                      '--waveform-background': 'var(--bg)',
                      '--waveform-color': activeAudio === 'reconstructed' ? 'var(--recon)' : activeAudio === 'clean' ? 'var(--missing)' : 'var(--danger)',
                    } as CSSProperties}
                  />
                  {waveformHover && (
                    <div
                      className="pointer-events-none absolute top-2 z-10 rounded-md border border-[var(--border)] bg-[var(--surface)]/95 px-2.5 py-1.5 text-xs shadow-lg"
                      style={{
                        left: `clamp(70px, ${waveformHover.x}px, calc(100% - 70px))`,
                        transform: 'translateX(-50%)',
                      }}
                    >
                      <div className="font-mono text-[var(--fg)]">
                        {waveformHover.time.toFixed(3)} s
                      </div>
                      <div className="font-mono text-[var(--fg-muted)]">
                        Amplitude: {waveformHover.amplitude.toFixed(4)}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Visual Analysis Tab */}
        <TabsContent value="visual" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-4">
            <Card variant="elevated">
              <CardHeader><CardTitle>Corrupted Input</CardTitle></CardHeader>
              <CardContent>
                <img src={`/api/jobs/${job.id}/files/spec_corrupted.png`} alt="Corrupted spectrogram" className="w-full rounded-md border border-[var(--border)]" />
              </CardContent>
            </Card>
            <Card variant="elevated">
              <CardHeader><CardTitle>Reconstructed</CardTitle></CardHeader>
              <CardContent>
                <img src={`/api/jobs/${job.id}/files/spec_reconstructed.png`} alt="Reconstructed spectrogram" className="w-full rounded-md border border-[var(--border)]" />
              </CardContent>
            </Card>
            {hasClean && (
              <Card variant="elevated">
                <CardHeader><CardTitle>Clean Reference</CardTitle></CardHeader>
                <CardContent>
                  <img src={`/api/jobs/${job.id}/files/spec_clean.png`} alt="Clean spectrogram" className="w-full rounded-md border border-[var(--border)]" />
                </CardContent>
              </Card>
            )}
            <Card variant="elevated">
              <CardHeader><CardTitle>Loss Curve</CardTitle></CardHeader>
              <CardContent>
                <img src={`/api/jobs/${job.id}/files/loss_curve.png`} alt="Loss curve" className="w-full rounded-md border border-[var(--border)]" />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="config" className="space-y-6">
          <Card variant="elevated">
            <CardHeader><CardTitle>Full Configuration (config_snapshot.yaml)</CardTitle></CardHeader>
            <CardContent>
              <pre className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 overflow-x-auto text-sm font-mono text-[var(--fg)] max-h-96 overflow-y-auto">
                {`Loading configuration...`}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </PageLayout>
  )
}