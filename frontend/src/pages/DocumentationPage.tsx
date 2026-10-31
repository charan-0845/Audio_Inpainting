import { Link } from 'react-router-dom'
import { BookOpen, Code, ExternalLink, FileAudio, Play, Settings } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLayout } from '@/components/layout'

export function DocumentationPage() {
  return (
    <PageLayout
      title="Documentation"
      description="Run experiments, inspect reconstructions, and understand the platform workflow."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <Card variant="elevated">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Play className="h-5 w-5 text-[var(--primary)]" />
              Quick start
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-[var(--fg-muted)]">
            <p>Start the FastAPI backend on port 8000, then run the Vite frontend on port 5173.</p>
            <pre className="overflow-x-auto rounded-md bg-[var(--bg)] p-4 font-mono text-xs text-[var(--fg)]">
              {`python app.py serve --host 127.0.0.1 --port 8000
cd frontend
npm run dev`}
            </pre>
            <p>Open <code className="text-[var(--fg)]">http://localhost:5173</code> in your browser.</p>
          </CardContent>
        </Card>

        <Card variant="elevated">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileAudio className="h-5 w-5 text-[var(--primary)]" />
              Workspace workflow
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-[var(--fg-muted)]">
            <p>1. Upload a WAV or supported audio file.</p>
            <p>2. Mark gaps manually, generate random gaps, or use Detect Silence.</p>
            <p>3. Review each region's range and duration on the waveform.</p>
            <p>4. Choose a preset and execute inpainting.</p>
            <p>5. Monitor loss and spectrogram updates, then compare and download results.</p>
          </CardContent>
        </Card>

        <Card variant="elevated">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5 text-[var(--primary)]" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-[var(--fg-muted)]">
            <p><strong className="text-[var(--fg)]">Fast</strong> is intended for quick CPU checks.</p>
            <p><strong className="text-[var(--fg)]">Balanced</strong> provides a quality and runtime trade-off.</p>
            <p><strong className="text-[var(--fg)]">Paper</strong> uses the highest-quality research configuration.</p>
            <p><strong className="text-[var(--fg)]">LPC</strong> provides an instant classical baseline.</p>
          </CardContent>
        </Card>

        <Card variant="elevated">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Code className="h-5 w-5 text-[var(--primary)]" />
              Related references
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-[var(--fg-muted)]">
            <Link to="/research" className="flex items-center gap-2 text-[var(--primary)] hover:underline">
              <BookOpen className="h-4 w-4" /> About the research implementation
            </Link>
            <Link to="/research/methodology" className="flex items-center gap-2 text-[var(--primary)] hover:underline">
              <ExternalLink className="h-4 w-4" /> Detailed methodology
            </Link>
            <p>For command-line options and repository setup, see the project README.</p>
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  )
}
