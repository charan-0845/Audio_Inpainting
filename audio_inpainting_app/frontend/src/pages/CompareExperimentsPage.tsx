import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ChevronLeft, GitBranch, AlertTriangle, CheckCircle, XCircle, MinusCircle, Settings, BarChart2, Music, Mic, FileAudio, Download, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import { EmptyState, LoadingState, ErrorState } from '@/components/ui/StateComponents'
import { Tooltip } from '@/components/ui/Tooltip'
import { Separator } from '@/components/ui/Separator'
import { Select } from '@/components/ui/Select'
import { api, ApiError } from '@/services/api/client'
import type { ExperimentRun, ComparisonResult } from '@/types/api'
import { formatDuration, formatMs, formatNumber, formatDate } from '@/lib/utils'
import { PageLayout, PageHeader } from '@/components/layout'
import { cn } from '@/lib/utils'

export function CompareExperimentsPage() {
  const [runs, setRuns] = useState<ExperimentRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedA, setSelectedA] = useState<string | null>(null)
  const [selectedB, setSelectedB] = useState<string | null>(null)
  const [comparison, setComparison] = useState<ComparisonResult | null>(null)
  const [comparisonLoading, setComparisonLoading] = useState(false)

  useEffect(() => {
    loadExperiments()
  }, [])

  const loadExperiments = async () => {
    setLoading(true)
    try {
      const response = await api.getExperiments({ status: 'completed', page_size: 100, sort_by: 'created_at', sort_order: 'desc' })
      setRuns(response.runs || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load experiments')
    } finally {
      setLoading(false)
    }
  }

  const handleCompare = async () => {
    if (!selectedA || !selectedB || selectedA === selectedB) return
    setComparisonLoading(true)
    try {
      const data = await api.compareExperiments(selectedA, selectedB)
      setComparison(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compare experiments')
    } finally {
      setComparisonLoading(false)
    }
  }

  const statusBadges: Record<string, 'success' | 'primary' | 'danger' | 'warning' | 'outline'> = {
    completed: 'success',
    running: 'primary',
    queued: 'primary',
    failed: 'danger',
    cancelled: 'outline',
  }

  const completedRuns = runs.filter(r => r.status === 'completed')

  if (loading) {
    return (
      <PageLayout title="Compare Experiments" description="Select two completed experiments to compare">
        <LoadingState message="Loading experiments..." />
      </PageLayout>
    )
  }

  if (error) {
    return (
      <PageLayout title="Compare Experiments" description="Select two completed experiments to compare">
        <ErrorState message={error} onRetry={loadExperiments} />
      </PageLayout>
    )
  }

  if (completedRuns.length < 2) {
    return (
      <PageLayout title="Compare Experiments" description="Select two completed experiments to compare">
        <EmptyState
          icon={<GitBranch className="h-12 w-12" />}
          title="Need at least 2 completed experiments"
          description="Run more experiments to enable comparison"
          action={{ label: 'New Experiment', onClick: () => {}, variant: 'primary' }}
        />
      </PageLayout>
    )
  }

  return (
    <PageLayout
      title="Compare Experiments"
      description="Select two completed experiments to compare metrics, configurations, and results"
    >
      {/* Selection panel */}
      <Card variant="elevated" className="mb-6">
        <CardHeader><CardTitle>Select Experiments to Compare</CardTitle></CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="label">Experiment A</label>
              <Select
                value={selectedA || ''}
                onValueChange={setSelectedA}
                options={completedRuns.map(r => ({ value: r.id, label: `${r.clip_id} (${formatDate(r.created_at)})` }))}
                placeholder="Select first experiment"
              />
            </div>
            <div>
              <label className="label">Experiment B</label>
              <Select
                value={selectedB || ''}
                onValueChange={setSelectedB}
                options={completedRuns.map(r => ({ value: r.id, label: `${r.clip_id} (${formatDate(r.created_at)})` }))}
                placeholder="Select second experiment"
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end">
            <Button
              onClick={handleCompare}
              disabled={!selectedA || !selectedB || selectedA === selectedB || comparisonLoading}
              className="gap-2"
            >
              <GitBranch className="h-4 w-4" />
              {comparisonLoading ? 'Comparing...' : 'Compare'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Comparison results */}
      {comparison && (
        <div className="space-y-6">
          {/* Condition flags */}
          {comparison.condition_flags.length > 0 && (
            <Card variant="elevated" className="border-[var(--warning)]/50 bg-[var(--warning)]/5">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-[var(--warning)]" />
                  Comparison Caveats
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside text-sm text-[var(--fg-muted)] space-y-1">
                  {comparison.condition_flags.map((flag, i) => (
                    <li key={i}>{flag}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Metrics comparison */}
          <Card variant="elevated">
            <CardHeader><CardTitle>Metrics Comparison</CardTitle></CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead className="text-center">Experiment A</TableHead>
                    <TableHead className="text-center">Experiment B</TableHead>
                    <TableHead className="text-center">Difference</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>Overall NMSE</TableCell>
                    <TableCell className="text-center font-mono text-[var(--recon)]">
                      {comparison.metrics_comparison.nmse_tot_db.a !== null ? `${comparison.metrics_comparison.nmse_tot_db.a!.toFixed(2)} dB` : '—'}
                    </TableCell>
                    <TableCell className="text-center font-mono text-[var(--recon)]">
                      {comparison.metrics_comparison.nmse_tot_db.b !== null ? `${comparison.metrics_comparison.nmse_tot_db.b!.toFixed(2)} dB` : '—'}
                    </TableCell>
                    <TableCell className="text-center font-mono text-[var(--recon)]">
                      {comparison.metrics_comparison.nmse_tot_db.diff !== null ? `${comparison.metrics_comparison.nmse_tot_db.diff! > 0 ? '+' : ''}${comparison.metrics_comparison.nmse_tot_db.diff!.toFixed(2)} dB` : '—'}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Gap NMSE</TableCell>
                    <TableCell className="text-center font-mono text-[var(--recon)]">
                      {comparison.metrics_comparison.nmse_miss_db.a !== null ? `${comparison.metrics_comparison.nmse_miss_db.a!.toFixed(2)} dB` : '—'}
                    </TableCell>
                    <TableCell className="text-center font-mono text-[var(--recon)]">
                      {comparison.metrics_comparison.nmse_miss_db.b !== null ? `${comparison.metrics_comparison.nmse_miss_db.b!.toFixed(2)} dB` : '—'}
                    </TableCell>
                    <TableCell className="text-center font-mono text-[var(--recon)]">
                      {comparison.metrics_comparison.nmse_miss_db.diff !== null ? `${comparison.metrics_comparison.nmse_miss_db.diff! > 0 ? '+' : ''}${comparison.metrics_comparison.nmse_miss_db.diff!.toFixed(2)} dB` : '—'}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Runtime</TableCell>
                    <TableCell className="text-center font-mono">{formatDuration(comparison.metrics_comparison.runtime_s.a)}</TableCell>
                    <TableCell className="text-center font-mono">{formatDuration(comparison.metrics_comparison.runtime_s.b)}</TableCell>
                    <TableCell className="text-center font-mono">
                      {comparison.metrics_comparison.runtime_s.diff > 0 ? '+' : ''}{formatDuration(comparison.metrics_comparison.runtime_s.diff)}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Epochs</TableCell>
                    <TableCell className="text-center font-mono">{comparison.metrics_comparison.epochs.a}</TableCell>
                    <TableCell className="text-center font-mono">{comparison.metrics_comparison.epochs.b}</TableCell>
                    <TableCell className="text-center font-mono">{comparison.metrics_comparison.epochs.a - comparison.metrics_comparison.epochs.b}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Parameters</TableCell>
                    <TableCell className="text-center font-mono">{comparison.metrics_comparison.n_params.a !== null ? formatNumber(comparison.metrics_comparison.n_params.a!) : '—'}</TableCell>
                    <TableCell className="text-center font-mono">{comparison.metrics_comparison.n_params.b !== null ? formatNumber(comparison.metrics_comparison.n_params.b!) : '—'}</TableCell>
                    <TableCell className="text-center">—</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Config differences */}
          <Card variant="elevated">
            <CardHeader><CardTitle>Configuration Differences</CardTitle></CardHeader>
            <CardContent>
              {Object.keys(comparison.config_differences).length === 0 ? (
                <p className="text-[var(--fg-muted)]">No configuration differences</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Setting</TableHead>
                      <TableHead className="text-center">Experiment A</TableHead>
                      <TableHead className="text-center">Experiment B</TableHead>
                      <TableHead className="text-center">Match</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(comparison.config_differences).map(([key, diff]) => (
                      <TableRow key={key}>
                        <TableCell className="font-mono text-sm">{key}</TableCell>
                        <TableCell className="text-center font-mono text-sm">{JSON.stringify(diff.a)}</TableCell>
                        <TableCell className="text-center font-mono text-sm">{JSON.stringify(diff.b)}</TableCell>
                        <TableCell className="text-center">
                          {diff.same ? (
                            <CheckCircle className="h-5 w-5 text-green-500 mx-auto" />
                          ) : (
                            <XCircle className="h-5 w-5 text-[var(--danger)] mx-auto" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </PageLayout>
  )
}