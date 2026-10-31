import { BookOpen, GitBranch, Code, ChevronRight, CheckCircle, AlertTriangle, Clock, Zap, Settings, Database, Mic, Music, FileAudio, ExternalLink, Check, X } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import { Separator } from '@/components/ui/Separator'
import { PageLayout } from '@/components/layout'
import { cn } from '@/lib/utils'

export function MethodologyPage() {
  return (
    <PageLayout title="Methodology" description="Detailed technical methodology of the DPAI implementation">
      {/* Disclaimer */}
      <Card variant="elevated" className="mb-8 border-[var(--warning)]/50 bg-[var(--warning)]/5">
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <AlertTriangle className="h-6 w-6 text-[var(--warning)] flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-[var(--fg)]">Implementation Status</h3>
              <p className="text-sm text-[var(--fg-muted)] mt-1">
                This page documents the <strong>actual repository implementation</strong>, not the paper verbatim.
                Key deviations are explicitly noted. See <a href="/research" className="text-[var(--primary)] hover:underline">Research Overview</a> for summary.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Architecture: MultiResUNet</h2>
        
        <Tabs defaultValue="encoder" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="encoder">Encoder</TabsTrigger>
            <TabsTrigger value="decoder">Decoder</TabsTrigger>
            <TabsTrigger value="bottleneck">Bottleneck</TabsTrigger>
          </TabsList>

          <TabsContent value="encoder" className="space-y-6 mt-4">
            <Card variant="elevated">
              <CardHeader><CardTitle>Encoder Structure</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-4 text-sm">
                  <p className="text-[var(--fg)]">The encoder consists of 5 levels with MultiResBlocks and ResPaths.</p>
                  
                  <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm overflow-x-auto">
                    {`Level 0: MultiResBlock(2 â†’ 68) + ResPath(4)
Level 1: MultiResBlock(68 â†’ 68) + ResPath(3)
Level 2: MultiResBlock(68 â†’ 68) + ResPath(2)
Level 3: MultiResBlock(68 â†’ 68) + ResPath(1)
Level 4: MultiResBlock(68 â†’ 68)  (bottleneck, no ResPath)

Each MultiResBlock: 3 parallel convs (3Ã—3, 5Ã—5, 7Ã—7) â†’ concat â†’ 1Ã—1 conv
ResPath: Series of 3Ã—3 convs for multi-scale feature propagation`}
                  </div>
                  
                  <h4 className="font-semibold text-[var(--fg)]">Key Parameters</h4>
                  <ul className="list-disc list-inside text-[var(--fg-muted)] space-y-1 ml-4">
                    <li>Channel schedule: [68, 68, 68, 68, 68] (flat, no doubling)</li>
                    <li>MultiRes width multiplier (Î±): 1.6</li>
                    <li>ResPath lengths: [4, 3, 2, 1] (decreasing per level)</li>
                    <li>Normalization: GroupNorm(1, C) = InstanceNorm</li>
                    <li>Activation: LeakyReLU(0.01)</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card variant="elevated">
              <CardHeader><CardTitle>Harmonic Convolution (Encoder Only)</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-4 text-sm">
                  <p className="text-[var(--fg)]">HarmonicConv2d constrains filters to harmonic frequency relationships.</p>
                  
                  <h4 className="font-semibold text-[var(--fg)]">Anchor Frequencies (from config.yaml)</h4>
                  <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm overflow-x-auto">
                    {`# Legible from paper Table I (set to 1):
respath.residual: 1
downsample.conv1: 1

# Placeholder assumptions (NOT paper values):
multires.residual: 1
multires.conv1: 1
multires.conv2: 2
multires.conv3: 3
respath.conv1: 1`}
                  </div>
                  
                  <div className="p-3 rounded-md bg-[var(--warning)]/10 border border-[var(--warning)]/50">
                    <p className="text-sm text-[var(--warning)]"><strong>Note:</strong> Placeholder anchors are explicitly separated in config to prevent confusion with actual paper values. This is a known deviation requiring verification.</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="decoder" className="space-y-6 mt-4">
            <Card variant="elevated">
              <CardHeader><CardTitle>Decoder Structure</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-4 text-sm">
                  <p className="text-[var(--fg)]">The decoder mirrors the encoder with 4 upsampling levels.</p>
                  
                  <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm overflow-x-auto">
                    {`Level 4â†’3: Upsample + MultiResBlock(68 â†’ 68)
Level 3â†’2: Upsample + MultiResBlock(68 â†’ 68)
Level 2â†’1: Upsample + MultiResBlock(68 â†’ 68)
Level 1â†’0: Upsample + MultiResBlock(68 â†’ 68)
Output: 1Ã—1 Conv(68 â†’ 2 channels) â†’ Complex STFT â†’ iSTFT

Upsampling: Nearest-neighbor 2Ã— + 1Ã—1 conv
Skip connections: Concatenate encoder features (channel-wise)
All decoder convolutions: Regular nn.Conv2d (NOT HarmonicConv2d)`}
                  </div>
                  
                  <h4 className="font-semibold text-[var(--fg)]">Design Decision</h4>
                  <p className="text-[var(--fg-muted)]">
                    Harmonic convolutions are only applied in the encoder path (including bottleneck).
                    The decoder uses regular convolutions regardless of the <code className="bg-[var(--bg)] px-1 rounded font-mono">use_harmonic</code> flag.
                    This is an intentional design choice documented in ARCHITECTURE.md.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bottleneck" className="space-y-6 mt-4">
            <Card variant="elevated">
              <CardHeader><CardTitle>Bottleneck (Level 4)</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-4 text-sm">
                  <p className="text-[var(--fg)]">The bottleneck is the deepest level of the encoder, acting as the final feature extraction stage.</p>
                  
                  <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm overflow-x-auto">
                    {`Bottleneck: MultiResBlock(68 â†’ 68)
- Uses HarmonicConv2d (since it&apos;s part of encoder)
- No ResPath at this level
- Receives features from Level 3 downsampling
- Output feeds into decoder Level 4 upsampling`}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </section>

      {/* STFT & Audio Processing */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">STFT & Audio Processing</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <Card variant="elevated">
            <CardHeader><CardTitle>STFT Parameters (Paper Table I)</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div className="grid grid-cols-2 gap-2 text-[var(--fg-muted)]">
                  <span>Sample Rate</span> <span className="font-mono text-[var(--fg)]">16 kHz</span>
                  <span>Window</span> <span className="font-mono text-[var(--fg)]">Hann</span>
                  <span>n_fft</span> <span className="font-mono text-[var(--fg)]">1024</span>
                  <span>hop_length</span> <span className="font-mono text-[var(--fg)]">120</span>
                  <span>win_length</span> <span className="font-mono text-[var(--fg)]">600</span>
                  <span>Overlap</span> <span className="font-mono text-[var(--fg)]">80% (480 samples)</span>
                </div>
                <p className="text-[var(--fg-muted)]">These values are driven from config.yaml and NOT hardcoded.</p>
              </div>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>Complex STFT Representation</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <p className="text-[var(--fg)]">Audio â†’ 2-channel complex STFT (Real, Imaginary)</p>
                <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm overflow-x-auto">
                  {`Input:  (2, F, T)  # 2 channels = Real + Imag
Output: (2, F, T)  # Predicted complex STFT
Reconstruction: iSTFT(complex_stft) â†’ waveform`}
                </div>
                <p className="text-[var(--fg-muted)]">Model operates entirely in complex time-frequency domain.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Corruption & Masking */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Corruption & Masking</h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          <Card variant="elevated">
            <CardHeader><CardTitle>Manual Gaps</CardTitle></CardHeader>
            <CardContent>
              <p className="text-[var(--fg-muted)]">User-defined time intervals marked on waveform.</p>
              <ul className="list-disc list-inside text-sm text-[var(--fg-muted)] mt-2 space-y-1">
                <li>Drag-to-create on waveform</li>
                <li>Multiple non-contiguous gaps</li>
                <li>Max 40% total missing</li>
                <li>Warning if any gap &gt; 100ms</li>
              </ul>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>Random/Synthetic</CardTitle></CardHeader>
            <CardContent>
              <p className="text-[var(--fg-muted)]">Programmatically generated gaps with controlled parameters.</p>
              <ul className="list-disc list-inside text-sm text-[var(--fg-muted)] mt-2 space-y-1">
                <li>Configurable gap duration</li>
                <li>Configurable number of gaps</li>
                <li>Random positions (seeded)</li>
                <li>Reproducible with same seed</li>
              </ul>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>Auto Silence Detection</CardTitle></CardHeader>
            <CardContent>
              <p className="text-[var(--fg-muted)]">Detects near-zero regions in audio.</p>
              <ul className="list-disc list-inside text-sm text-[var(--fg-muted)] mt-2 space-y-1">
                <li>Threshold: -60 dB (configurable)</li>
                <li>Min duration: 50ms (configurable)</li>
                <li>Operates on sample-level</li>
                <li>Editable before reconstruction</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Loss Function */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Loss Function</h2>
        
        <Card variant="elevated">
          <CardHeader><CardTitle>Total Loss = Î±_mse Ã— MSE_masked + Î±_mss Ã— MSS</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold text-[var(--fg)]">Masked MSE (Primary)</h4>
                <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm mt-2 overflow-x-auto">
                  {`L_mse = ||M âŠ™ (X_clean - X_recon)||Â² / ||M âŠ™ X_clean||Â²
M: binary mask (1=observed, 0=missing)
Only missing regions contribute to loss`}
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-[var(--fg)]">Multi-Scale Spectrogram Loss (MSS)</h4>
                <p className="text-[var(--fg-muted)] mt-1">Three STFT scales (Parallel WaveGAN configuration):</p>
                <div className="overflow-x-auto mt-2">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[var(--border)] text-left">
                        <th className="pb-2 text-[var(--fg-muted)]">Scale</th>
                        <th className="pb-2 text-[var(--fg-muted)]">n_fft</th>
                        <th className="pb-2 text-[var(--fg-muted)]">win_length</th>
                        <th className="pb-2 text-[var(--fg-muted)]">hop_length</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-[var(--border)]"><td className="py-2 font-mono">Fine</td><td className="py-2 font-mono">512</td><td className="py-2 font-mono">240</td><td className="py-2 font-mono">50</td></tr>
                      <tr className="border-b border-[var(--border)]"><td className="py-2 font-mono">Medium</td><td className="py-2 font-mono">1024</td><td className="py-2 font-mono">600</td><td className="py-2 font-mono">120</td></tr>
                      <tr><td className="py-2 font-mono">Coarse</td><td className="py-2 font-mono">2048</td><td className="py-2 font-mono">1200</td><td className="py-2 font-mono">240</td></tr>
                    </tbody>
                  </table>
                </div>
                <p className="text-[var(--fg-muted)] mt-2">Each scale: Spectral Convergence + Log-STFT Mag + Linear STFT Mag + Phase Consistency</p>
              </div>

              <div>
                <h4 className="font-semibold text-[var(--fg)]">Default Weights</h4>
                <div className="grid grid-cols-2 gap-4 mt-2">
                  <Card variant="elevated">
                    <CardContent className="pt-4 text-center">
                      <p className="text-2xl font-bold text-[var(--primary)]">Î±_mse = 1.0</p>
                      <p className="text-sm text-[var(--fg-muted)]">Masked MSE weight</p>
                    </CardContent>
                  </Card>
                  <Card variant="elevated">
                    <CardContent className="pt-4 text-center">
                      <p className="text-2xl font-bold text-[var(--missing)]">Î±_mss = 0.1</p>
                      <p className="text-sm text-[var(--fg-muted)]">MSS weight</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Optimization */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Deep Prior Optimization</h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          <Card variant="elevated">
            <CardHeader><CardTitle>Optimizer</CardTitle></CardHeader>
            <CardContent>
              <ul className="list-disc list-inside text-sm text-[var(--fg-muted)] space-y-2">
                <li>Adam (Î²â‚=0.9, Î²â‚‚=0.999)</li>
                <li>Learning rate: 0.01</li>
                <li>No learning rate scheduling</li>
                <li>Per-iteration Gaussian perturbation (Ïƒ=0.03)</li>
              </ul>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>Input Noise</CardTitle></CardHeader>
            <CardContent>
              <ul className="list-disc list-inside text-sm text-[var(--fg-muted)] space-y-2">
                <li>Uniform noise added to input</li>
                <li>Variance: 0.1</li>
                <li>Re-sampled each iteration</li>
                <li>Prevents trivial solutions</li>
              </ul>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>Presets (CPU-adapted)</CardTitle></CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-[var(--fg-muted)]">LPC</span> <span className="font-mono text-[var(--fg)]">0 epochs</span></div>
                <div className="flex justify-between"><span className="text-[var(--fg-muted)]">Fast</span> <span className="font-mono text-[var(--fg)]">100 epochs, 4 filters</span></div>
                <div className="flex justify-between"><span className="text-[var(--fg-muted)]">Balanced</span> <span className="font-mono text-[var(--fg)]">300 epochs, 16 filters</span></div>
                <div className="flex justify-between"><span className="text-[var(--fg-muted)]">Paper</span> <span className="font-mono text-[var(--fg)]">1000 epochs, 32 filters, harmonic</span></div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Evaluation */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Evaluation Metrics</h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          <Card variant="elevated">
            <CardHeader><CardTitle>NMSE (Total)</CardTitle></CardHeader>
            <CardContent>
              <p className="text-[var(--fg-muted)]">Normalized Mean Squared Error over entire signal.</p>
              <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm mt-2 overflow-x-auto">
                {`NMSE_tot = ||X_clean - X_recon||Â² / ||X_clean||Â²
Reported in dB: 10 Ã— log10(NMSE_tot)
Lower = better (negative dB = good)`}
              </div>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>NMSE (Gap Region)</CardTitle></CardHeader>
            <CardContent>
              <p className="text-[var(--fg-muted)]">NMSE computed only on missing regions.</p>
              <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm mt-2 overflow-x-auto">
                {`NMSE_miss = ||M_miss âŠ™ (X_clean - X_recon)||Â² / ||M_miss âŠ™ X_clean||Â²
M_miss: mask of missing samples only
Primary metric for inpainting quality`}
              </div>
            </CardContent>
          </Card>

          <Card variant="elevated">
            <CardHeader><CardTitle>PESQ-WB (Speech Only)</CardTitle></CardHeader>
            <CardContent>
              <p className="text-[var(--fg-muted)]">Perceptual Evaluation of Speech Quality (Wideband).</p>
              <div className="bg-[var(--bg)] border border-[var(--border)] rounded-md p-4 font-mono text-sm mt-2 overflow-x-auto">
                {`PESQ_delta = PESQ(clean, recon) - PESQ(clean, corrupted)
Range: -0.5 to 4.5 (higher = better)
Only computed for speech-type audio
Requires clean reference audio`}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* References */}
      <section>
        <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">References</h2>
        
        <Card variant="elevated">
          <CardContent>
            <div className="space-y-4 text-sm">
              <div>
                <h4 className="font-semibold text-[var(--fg)] mb-2">Primary Paper</h4>
                <p className="text-[var(--fg-muted)]">
                  Federico Miotello, Paolo Bestagini, Stefano Tubaro,
                  <em>"Deep Prior-Based Audio Inpainting Using Multi-Resolution Harmonic Convolutional Neural Networks"</em>,
                  IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, 2024.
                  DOI: <a href="https://doi.org/10.1109/TASLP.2023.3324556" target="_blank" rel="noopener noreferrer" className="text-[var(--primary)] hover:underline">10.1109/TASLP.2023.3324556</a>
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-[var(--fg)] mb-2">Related Work</h4>
                <ul className="list-disc list-inside text-[var(--fg-muted)] space-y-1">
                  <li>Deep Image Prior (Ulyanov et al., CVPR 2018) â€” Foundation for deep prior approach</li>
                  <li>MultiResUNet (Ibtehaz & Rahman, 2020) â€” Multi-resolution architecture</li>
                  <li>Parallel WaveGAN (Yamamoto et al., 2020) â€” MSS loss scales</li>
                  <li>Harmonic CNN (Wang et al.) â€” Harmonic convolution concept</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-[var(--fg)] mb-2">Repository Documentation</h4>
                <ul className="list-disc list-inside text-[var(--fg-muted)] space-y-1">
                  <li><a href="/docs/ARCHITECTURE.md" className="text-[var(--primary)] hover:underline">ARCHITECTURE.md</a> â€” Filter-count analysis, deviations, open items</li>
                  <li><a href="/README.md" className="text-[var(--primary)] hover:underline">README.md</a> â€” Usage, configuration, project structure</li>
                  <li><code className="bg-[var(--bg)] px-1 rounded font-mono">configs/config.yaml</code> â€” All hyperparameters (frozen interface)</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>
    </PageLayout>
  )
}


