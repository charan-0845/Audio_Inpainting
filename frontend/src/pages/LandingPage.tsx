import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Zap, BookOpen, GitBranch, Code, Play, Pause, Volume2, VolumeX, ChevronRight, Check, X, Music, Mic, Cpu } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import { Separator } from '@/components/ui/Separator'
import { cn } from '@/lib/utils'
import { api } from '@/services/api/client'

interface PrecomputedExample {
  id: string
  title: string
  clip_id: string
  type: 'piano' | 'speech' | 'synthetic'
  gap_ms: number
  epochs: number
  seed: number
  description: string
  spectrogram_url: string
  spec_corrupted_url: string | null
  audio_url: string
  corrupted_url: string
  clean_url: string | null
  metrics: {
    nmse_tot_db: number | null
    nmse_miss_db: number | null
    pesq_delta: number | null
  }
}

export function LandingPage() {
  const [examples, setExamples] = useState<PrecomputedExample[]>([])
  const [loading, setLoading] = useState(true)
  const [activeDemo, setActiveDemo] = useState<string | null>(null)
  const [playingDemo, setPlayingDemo] = useState<string | null>(null)
  const [audioElements, setAudioElements] = useState<Record<string, HTMLAudioElement>>({})

  const features = [
    {
      icon: Zap,
      title: 'Interactive Workspace',
      description: 'Drag-to-create gaps, real-time spectrogram, live loss curves, and A/B audio comparison.',
    },
    {
      icon: BookOpen,
      title: 'Research-Grade',
      description: 'MultiResUNet with HarmonicConv2d, deep prior optimization, NMSE & PESQ evaluation.',
    },
    {
      icon: GitBranch,
      title: 'Experiment Tracking',
      description: 'Full history, configuration snapshots, artifact downloads, and experiment comparison.',
    },
    {
      icon: Code,
      title: 'Reproducible',
      description: 'Exact config preservation, git hash tracking, seed control, and paper-aligned presets.',
    },
  ]

  const steps = [
    { number: '01', title: 'Input', description: 'Upload WAV audio or use synthetic signals. Optionally provide clean reference for evaluation.' },
    { number: '02', title: 'Reconstruct', description: 'Mark corrupted regions manually, generate random gaps, or auto-detect silence. Choose Fast, Balanced, or Paper preset.' },
    { number: '03', title: 'Evaluate', description: 'Listen to A/B comparison, inspect spectrograms, review NMSE/PESQ metrics, and download results.' },
  ]

  useEffect(() => {
    fetchExamples()
  }, [])

  const fetchExamples = async () => {
    try {
      const resp = await fetch('/examples/index.json')
      if (resp.ok) {
        const data = await resp.json()
        setExamples(data.examples || [])
      }
    } catch (e) {
      console.warn('Could not load precomputed examples:', e)
    } finally {
      setLoading(false)
    }
  }

  const handlePlay = (demoId: string, audioType: 'reconstructed' | 'corrupted' | 'clean') => {
    const key = `${demoId}-${audioType}`
    const audio = audioElements[key]
    if (!audio) return

    // Stop any currently playing
    Object.values(audioElements).forEach(a => { a.pause(); a.currentTime = 0 })
    setPlayingDemo(null)

    if (playingDemo === key) {
      setPlayingDemo(null)
      return
    }

    audio.play().then(() => setPlayingDemo(key)).catch(console.error)
  }

  const handlePause = (demoId: string, audioType: 'reconstructed' | 'corrupted' | 'clean') => {
    const key = `${demoId}-${audioType}`
    const audio = audioElements[key]
    if (audio) {
      audio.pause()
      setPlayingDemo(null)
    }
  }

  const formatMs = (ms: number) => `${Math.round(ms)}ms`

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-[var(--surface)] to-[var(--bg)] py-20 lg:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto">
            <Badge variant="primary" className="mb-6 inline-flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--primary)] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--primary)]"></span>
              </span>
              Research Implementation — IEEE/ACM TASLP 2024
            </Badge>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-[var(--fg)] tracking-tight mb-6">
              Audio Inpainting
              <br />
              <span className="text-[var(--primary)]">Research Platform</span>
            </h1>
            <p className="text-lg sm:text-xl text-[var(--fg-muted)] mb-10 max-w-2xl mx-auto">
              Deep Prior-Based Audio Inpainting using Multi-Resolution Harmonic Convolutional Neural Networks.
              Interactively reconstruct missing audio regions with clip-specific optimization.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/dashboard">
                <Button size="lg" className="gap-2" aria-label="Launch interactive workspace">
                  <Zap className="h-5 w-5" />
                  Launch Workspace
                  <ArrowRight className="h-5 w-5" />
                </Button>
              </Link>
              <Link to="/research">
                <Button variant="secondary" size="lg">
                  <BookOpen className="h-5 w-5 mr-1" />
                  View Research
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Three Steps */}
      <section className="py-20 lg:py-28 bg-[var(--bg)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--fg)] mb-4">How It Works</h2>
            <p className="text-[var(--fg-muted)] max-w-2xl mx-auto">
              Three steps from corrupted audio to reconstructed output with full scientific rigor.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {steps.map((step, index) => (
              <Card key={step.number} variant="elevated" className="relative">
                <div className="absolute -top-3 left-6 bg-[var(--bg)] px-2 text-xs font-mono text-[var(--primary)]">
                  {step.number}
                </div>
                <CardContent className="pt-6">
                  <h3 className="text-xl font-semibold text-[var(--fg)] mb-2">{step.title}</h3>
                  <p className="text-[var(--fg-muted)]">{step.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 lg:py-28 bg-[var(--surface)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--fg)] mb-4">Platform Capabilities</h2>
            <p className="text-[var(--fg-muted)] max-w-2xl mx-auto">
              Built for audio researchers with interactive tools and reproducible experiment tracking.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature) => (
              <Card key={feature.title} className="h-full">
                <CardContent className="pt-6">
                  <div className="w-12 h-12 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center mb-4">
                    <feature.icon className="h-6 w-6 text-[var(--primary)]" />
                  </div>
                  <h3 className="text-lg font-semibold text-[var(--fg)] mb-2">{feature.title}</h3>
                  <p className="text-[var(--fg-muted)]">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Precomputed Demos Section */}
      <section className="py-20 lg:py-28 bg-[var(--bg)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold text-[var(--fg)] mb-4">Precomputed Demos</h2>
            <p className="text-[var(--fg-muted)] max-w-2xl mx-auto mb-8">
              Explore precomputed reconstructions without waiting for optimization. 
              These examples use the Fast preset (10 epochs) for quick demonstration.
            </p>
            <Badge variant="outline" className="text-sm">
              <span className="relative flex h-2 w-2 mr-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-green-500 animate-ping opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              All demos are precomputed — no inference runs on this page
            </Badge>
          </div>

          {loading ? (
            <div className="grid md:grid-cols-3 gap-6">
              {[1,2,3].map(i => (
                <Card key={i} variant="elevated" className="overflow-hidden">
                  <div className="aspect-video bg-[var(--raised)] animate-pulse" />
                  <CardContent className="pt-4 space-y-3">
                    <div className="h-4 bg-[var(--raised)] rounded animate-pulse w-3/4" />
                    <div className="h-4 bg-[var(--raised)] rounded animate-pulse w-1/2" />
                    <div className="h-8 bg-[var(--raised)] rounded animate-pulse w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : examples.length === 0 ? (
            <Card variant="elevated" className="max-w-2xl mx-auto">
              <CardContent className="pt-8 pb-8 text-center">
                <Cpu className="h-12 w-12 mx-auto text-[var(--fg-muted)] mb-4" />
                <h3 className="text-lg font-semibold text-[var(--fg)] mb-2">No Precomputed Examples</h3>
                <p className="text-[var(--fg-muted)] mb-4">
                  Run <code className="bg-[var(--raised)] px-2 py-1 rounded font-mono text-sm">python scripts/precompute_examples.py</code> to generate demo assets.
                </p>
                <Link to="/dashboard">
                  <Button variant="primary" className="gap-2">
                    <Zap className="h-4 w-4" /> Try Live Workspace Instead
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Interactive Demo Explorer */}
              <div className="mb-12">
                <h3 className="text-xl font-semibold text-[var(--fg)] mb-6">Interactive Demo Explorer</h3>
                <Tabs defaultValue={examples[0]?.id || ''} onValueChange={(v) => setActiveDemo(v as string | null)} className="w-full">
                  <TabsList className="grid w-full grid-cols-3 gap-2">
                    {examples.map(ex => (
                      <TabsTrigger key={ex.id} value={ex.id} className="py-3">
                        <div className="flex flex-col items-center gap-1">
                          <span className="font-medium">{ex.title}</span>
                          <span className="text-xs text-[var(--fg-muted)] font-mono">{formatMs(ex.gap_ms)} gap</span>
                        </div>
                      </TabsTrigger>
                    ))}
                  </TabsList>

                  {examples.map(ex => (
                    <TabsContent key={ex.id} value={ex.id} className="mt-6">
                      <div className="grid lg:grid-cols-3 gap-6">
                        {/* Spectrograms */}
                        <div className="lg:col-span-2 space-y-6">
                          <Card variant="elevated">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-base flex items-center gap-2">
                                <Music className="h-4 w-4 text-[var(--danger)]" />
                                Corrupted Input
                              </CardTitle>
                            </CardHeader>
                            <CardContent>
                              <img src={`/${ex.spec_corrupted_url}`} alt="Corrupted spectrogram" className="w-full rounded-md border border-[var(--border)]" />
                            </CardContent>
                          </Card>

                          <Card variant="elevated">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-base flex items-center gap-2">
                                <Check className="h-4 w-4 text-[var(--primary)]" />
                                Reconstructed
                              </CardTitle>
                            </CardHeader>
                            <CardContent>
                              <img src={`/${ex.spectrogram_url}`} alt="Reconstructed spectrogram" className="w-full rounded-md border border-[var(--border)]" />
                            </CardContent>
                          </Card>

                          {ex.clean_url && (
                            <Card variant="elevated">
                              <CardHeader className="pb-2">
                                <CardTitle className="text-base flex items-center gap-2">
                                  <Mic className="h-4 w-4 text-[var(--missing)]" />
                                  Clean Reference
                                </CardTitle>
                              </CardHeader>
                              <CardContent>
                                <img src={`/${ex.clean_url.replace('clean.wav', 'spec_clean.png')}`} alt="Clean spectrogram" className="w-full rounded-md border border-[var(--border)]" />
                              </CardContent>
                            </Card>
                          )}
                        </div>

                        {/* Audio Players & Metrics */}
                        <div className="space-y-4">
                          <Card variant="elevated">
                            <CardHeader>
                              <CardTitle className="text-base">Audio Comparison</CardTitle>
                            </CardHeader>
                            <CardContent className="pt-0 space-y-4">
                              {[
                                { key: 'reconstructed', label: 'Reconstructed', icon: Check, color: 'text-[var(--primary)]', url: ex.audio_url },
                                { key: 'corrupted', label: 'Corrupted Input', icon: X, color: 'text-[var(--danger)]', url: ex.corrupted_url },
                                ...(ex.clean_url ? [{ key: 'clean', label: 'Clean Reference', icon: Mic, color: 'text-[var(--missing)]', url: ex.clean_url }] : []),
                              ].map(({ key, label, icon: Icon, color, url }) => (
                                <div key={key} className="space-y-2 p-3 rounded-md bg-[var(--raised)]/50 border border-[var(--border)]">
                                  <div className="flex items-center gap-2">
                                    <Icon className={cn('h-4 w-4', color)} />
                                    <span className="font-medium text-[var(--fg)]">{label}</span>
                                    <span className="ml-auto text-xs text-[var(--fg-muted)] font-mono">16-bit PCM WAV</span>
                                  </div>
                                  <audio
                                    ref={el => { if (el) { setAudioElements(prev => ({ ...prev, [`${ex.id}-${key}`]: el })) }} }
                                    src={`/${url}`}
                                    onPlay={() => setPlayingDemo(`${ex.id}-${key}`)}
                                    onPause={() => setPlayingDemo(null)}
                                    onEnded={() => setPlayingDemo(null)}
                                    className="w-full"
                                  />
                                  <div className="flex items-center gap-2">
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => handlePlay(ex.id, key as any)}
                                      disabled={playingDemo === `${ex.id}-${key}`}
                                    >
                                      {playingDemo === `${ex.id}-${key}` ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                                    </Button>
                                    <Button variant="ghost" size="sm" onClick={() => handlePause(ex.id, key as any)} disabled={playingDemo !== `${ex.id}-${key}`}>
                                      <Pause className="h-4 w-4" />
                                    </Button>
                                    <a href={`/${url}`} download className="ml-auto">
                                      <Button variant="ghost" size="sm" className="gap-1">
                                        <ArrowRight className="h-3 w-3" /> Download
                                      </Button>
                                    </a>
                                  </div>
                                </div>
                              ))}
                            </CardContent>
                          </Card>

                          <Card variant="elevated">
                            <CardHeader>
                              <CardTitle className="text-base">Metrics</CardTitle>
                            </CardHeader>
                            <CardContent className="pt-0 space-y-3">
                              <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 rounded-md bg-[var(--raised)]">
                                  <p className="text-xs text-[var(--fg-muted)]">Overall NMSE</p>
                                  <p className="font-mono text-[var(--recon)] text-lg">{ex.metrics.nmse_tot_db !== null ? ex.metrics.nmse_tot_db.toFixed(4) : '—'} dB</p>
                                </div>
                                <div className="p-3 rounded-md bg-[var(--raised)]">
                                  <p className="text-xs text-[var(--fg-muted)]">Gap Region NMSE</p>
                                  <p className="font-mono text-[var(--recon)] text-lg">{ex.metrics.nmse_miss_db !== null ? ex.metrics.nmse_miss_db.toFixed(4) : '—'} dB</p>
                                </div>
                                <div className="p-3 rounded-md bg-[var(--raised)]">
                                  <p className="text-xs text-[var(--fg-muted)]">PESQ Delta</p>
                                  <p className="font-mono text-[var(--missing)] text-lg">{ex.metrics.pesq_delta !== null ? ex.metrics.pesq_delta.toFixed(3) : ex.clean_url ? 'N/A (speech only)' : '—'}</p>
                                </div>
                                <div className="p-3 rounded-md bg-[var(--raised)]">
                                  <p className="text-xs text-[var(--fg-muted)]">Epochs / Seed</p>
                                  <p className="font-mono text-[var(--fg)] text-lg">{ex.epochs} / {ex.seed}</p>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                      </div>
                    </TabsContent>
                  ))}
                </Tabs>
              </div>

              {/* Quick Demo Cards */}
              <div>
                <h3 className="text-xl font-semibold text-[var(--fg)] mb-6">Quick Preview Cards</h3>
                <div className="grid md:grid-cols-3 gap-6">
                  {examples.map(ex => (
                    <Card key={ex.id} variant="elevated" className="overflow-hidden">
                      <div className="aspect-video bg-[var(--bg)] relative">
                        <img src={`/${ex.spectrogram_url}`} alt={`${ex.title} spectrogram`} className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <Button 
                            variant="secondary" 
                            size="lg" 
                            onClick={() => setActiveDemo(ex.id)}
                            className="gap-2"
                          >
                            <Play className="h-5 w-5" /> Explore
                          </Button>
                        </div>
                      </div>
                      <CardContent>
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-semibold text-[var(--fg)]">{ex.title}</h3>
                          <Badge variant="outline">{formatMs(ex.gap_ms)}</Badge>
                        </div>
                        <p className="text-sm text-[var(--fg-muted)] mb-3 capitalize">{ex.type} audio</p>
                        <div className="flex items-center gap-2 text-xs text-[var(--fg-muted)]">
                          <span className="flex items-center gap-1"><Cpu className="h-3 w-3" /> {ex.epochs} epochs</span>
                          <Separator orientation="vertical" className="h-4" />
                          <span className="flex items-center gap-1"><Music className="h-3 w-3" /> NMSE: {ex.metrics.nmse_tot_db !== null ? ex.metrics.nmse_tot_db.toFixed(2) : '—'} dB</span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Research Notice */}
      <section className="py-16 bg-[var(--surface)] border-t border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-[var(--primary)]/5 border border-[var(--primary)]/20 rounded-lg p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center flex-shrink-0">
                  <BookOpen className="h-5 w-5 text-[var(--primary)]" />
                </div>
                <div>
                  <h3 className="font-semibold text-[var(--fg)]">Research Implementation Notice</h3>
                  <p className="text-sm text-[var(--fg-muted)] mt-1">
                    This platform implements the method described in: Federico Miotello et al., "Deep Prior-Based Audio Inpainting
                    Using Multi-Resolution Harmonic Convolutional Neural Networks," IEEE/ACM TASLP 2024 (DOI: 10.1109/TASLP.2023.3324556).
                    This is a research implementation — numerical results may differ from the published paper due to documented
                    architectural deviations (GroupNorm vs BatchNorm, flat channel schedule, harmonic anchor assumptions).
                    See <Link to="/research/methodology" className="underline hover:text-[var(--primary)]">Methodology</Link> for details.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}