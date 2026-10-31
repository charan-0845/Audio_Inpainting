import { Link } from 'react-router-dom'
import { BookOpen, GitBranch, Code, ExternalLink, ArrowRight, ChevronRight, CheckCircle, AlertTriangle, Clock, Zap, Settings, Database, Mic, Music, FileAudio } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import { Separator } from '@/components/ui/Separator'
import { PageLayout } from '@/components/layout'
import { cn } from '@/lib/utils'

export function ResearchPage() {
  return (
    <PageLayout title="Research Overview" description="Deep Prior-Based Audio Inpainting using Multi-Resolution Harmonic CNNs">
      {/* Paper reference */}
      <Card variant="elevated" className="mb-8">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6">
            <div className="flex items-start gap-4">
              <div className="w-14 h-14 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center flex-shrink-0">
                <BookOpen className="h-7 w-7 text-[var(--primary)]" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-[var(--fg)]">IEEE/ACM TASLP 2024</h3>
                <p className="text-[var(--fg-muted)] mt-1">
                  "Deep Prior-Based Audio Inpainting Using Multi-Resolution Harmonic Convolutional Neural Networks"
                </p>
                <p className="text-sm text-[var(--fg-muted)] mt-2">
                  Federico Miotello et al. • DOI: <a href="https://doi.org/10.1109/TASLP.2023.3324556" target="_blank" rel="noopener noreferrer" className="text-[var(--primary)] hover:underline">10.1109/TASLP.2023.3324556</a>
                </p>
              </div>
            </div>
            <Link to="/research/methodology">
              <Button variant="primary" className="gap-2">
                View Detailed Methodology
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>

      {/* Key concepts */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Key Concepts</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              icon: Zap,
              title: 'Deep Prior',
              description: 'Clip-specific optimization without external corpus pre-training. Each audio sample trains its own network from random initialization.',
              color: 'text-[var(--primary)]',
            },
            {
              icon: GitBranch,
              title: 'Multi-Resolution Architecture',
              description: 'MultiResUNet with parallel multi-resolution blocks capturing features at different time-frequency scales simultaneously.',
              color: 'text-[var(--missing)]',
            },
            {
              icon: Code,
              title: 'Harmonic Convolution',
              description: 'HarmonicConv2d layers encode harmonic structure priors by constraining filters to harmonic frequency relationships.',
              color: 'text-[var(--recon)]',
            },
          ].map((item, i) => (
            <Card key={i} variant="elevated">
              <CardContent className="pt-6">
                <div className="w-12 h-12 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center mb-4">
                  <item.icon className="h-6 w-6" style={{ color: item.color }} />
                </div>
                <h3 className="text-lg font-semibold text-[var(--fg)] mb-2">{item.title}</h3>
                <p className="text-[var(--fg-muted)]">{item.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* What this repo implements */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Repository Implementation</h2>
        <Tabs defaultValue="implemented" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="implemented">Implemented</TabsTrigger>
            <TabsTrigger value="deviations">Deviations</TabsTrigger>
            <TabsTrigger value="future">Future Work</TabsTrigger>
          </TabsList>

          <TabsContent value="implemented" className="space-y-4 mt-4">
            <div className="grid md:grid-cols-2 gap-4">
              {[
                'MultiResUNet with flat channel schedule (68 filters × 5 levels)',
                'HarmonicConv2d in encoder (configurable anchors)',
                'GroupNorm (InstanceNorm) for batch-size-1 compatibility',
                'Deep prior optimization with Adam (lr=0.01)',
                'Input noise (σ=0.1) and perturbation (σ=0.03)',
                'Masked MSE loss + Multi-Scale Spectrogram loss (MSS)',
                'STFT: n_fft=1024, hop=120, win=600 @ 16kHz',
                'Observed sample preservation (configurable)',
                'NMSE (total & gap region) evaluation',
                'PESQ-WB for speech quality assessment',
                'Reproducible run artifacts (config, metrics, spectrograms)',
                'Web demo with real-time SSE progress',
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-md bg-[var(--raised)]/50">
                  <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                  <span className="text-sm text-[var(--fg)]">{item}</span>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="deviations" className="space-y-4 mt-4">
            <div className="grid md:grid-cols-2 gap-4">
              {[
                { title: 'Normalization', detail: 'GroupNorm(1, C) instead of BatchNorm. BatchNorm undefined for batch size 1 (deep prior). This is a documented deviation from the paper.' },
                { title: 'Channel Schedule', detail: 'Flat [68, 68, 68, 68, 68] instead of doubling. Matches paper\'s ~2.01M parameter target for MultiResUNet.' },
                { title: 'Decoder Harmonic', detail: 'Decoder uses regular Conv2d, not HarmonicConv2d. Only encoder (and bottleneck) uses harmonic convolutions.' },
                { title: 'Harmonic Anchors', detail: 'Table I partially illegible. Legible values set to 1; placeholder assumptions (1,2,3) for conv1/2/3 clearly separated in config.' },
                { title: 'OOB Frequencies', detail: 'Out-of-bounds harmonic frequencies masked to zero (not clamped), avoiding phantom artifacts at spectral edges.' },
                { title: 'MSS Loss Scales', detail: 'Three STFT scales (Fine/Medium/Coarse) from Parallel WaveGAN, verified against standard implementation.' },
              ].map((item, i) => (
                <Card key={i} variant="elevated" className="border-[var(--warning)]/50">
                  <CardContent className="pt-4">
                    <h4 className="font-semibold text-[var(--fg)] mb-1">{item.title}</h4>
                    <p className="text-sm text-[var(--fg-muted)]">{item.detail}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="future" className="space-y-4 mt-4">
            <div className="grid md:grid-cols-2 gap-4">
              {[
                'Full paper reproduction with verified harmonic anchors',
                'BatchNorm support for multi-sample training',
                'Additional corruption types (noise, clipping, packet loss)',
                'Real-time inference with distilled/fixed networks',
                'Extended evaluation (STOI, DNS-MOS, subjective)',
                'Multi-channel (stereo) inpainting support',
                'Conditional generation with text/audio prompts',
                'Integration with DAW plugins (VST/AU)',
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-md bg-[var(--raised)]/50">
                  <AlertTriangle className="h-5 w-5 text-[var(--warning)] flex-shrink-0 mt-0.5" />
                  <span className="text-sm text-[var(--fg)]">{item}</span>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </section>

      {/* Quick links */}
      <section>
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Quick Links</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <Link to="/research/methodology">
            <Card variant="elevated" className="hover:border-[var(--primary)] transition-colors">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center">
                    <GitBranch className="h-5 w-5 text-[var(--primary)]" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-[var(--fg)]">Methodology</h3>
                    <p className="text-sm text-[var(--fg-muted)]">Detailed architecture & training</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
          <Link to="/workspace/new">
            <Card variant="elevated" className="hover:border-[var(--primary)] transition-colors">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center">
                    <Zap className="h-5 w-5 text-[var(--primary)]" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-[var(--fg)]">Try It Live</h3>
                    <p className="text-sm text-[var(--fg-muted)]">Interactive workspace</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
          <Link to="/experiments">
            <Card variant="elevated" className="hover:border-[var(--primary)] transition-colors">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-[var(--primary)]/10 flex items-center justify-center">
                    <Database className="h-5 w-5 text-[var(--primary)]" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-[var(--fg)]">Experiment History</h3>
                    <p className="text-sm text-[var(--fg-muted)]">Browse past runs</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        </div>
      </section>
    </PageLayout>
  )
}