import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { Plus, Clock, CheckCircle, Loader2, TrendingUp, Database, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { EmptyState, LoadingState, ErrorState } from '@/components/ui/StateComponents'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { api, ApiError } from '@/services/api/client'
import type { ExperimentRun } from '@/types/api'
import { formatDuration, formatNumber, formatDate, formatMs } from '@/lib/utils'
import { PageLayout } from '@/components/layout'
import { cn } from '@/lib/utils'

const CHART_COLORS = {
  primary: 'hsl(var(--primary))',
  missing: 'hsl(var(--missing))',
  recon: 'hsl(var(--recon))',
  muted: 'hsl(var(--fg-muted))',
}

export function DashboardPage() {
  const [runs, setRuns] = useState<ExperimentRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState({
    total: 0,
    completed: 0,
    processing: 0,
    avgRuntime: 0,
    latestExperiment: null as ExperimentRun | null,
    latestNmse: null as number | null,
  })

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.getExperiments({ page_size: 50, sort_by: 'created_at', sort_order: 'desc' })
      const runsData = response.runs || []
      setRuns(runsData)

      const completed = runsData.filter(r => r.status === 'completed')
      const processing = runsData.filter(r => r.status === 'running' || r.status === 'queued')
      const avgRuntime = completed.length > 0
        ? completed.reduce((sum, r) => sum + (r.runtime_s || 0), 0) / completed.length
        : 0
      const latestNmse = completed.length > 0 && completed[0].metrics?.nmse_tot_db !== null
        ? completed[0].metrics!.nmse_tot_db
        : null

      setStats({
        total: runsData.length,
        completed: completed.length,
        processing: processing.length,
        avgRuntime,
        latestExperiment: runsData[0] || null,
        latestNmse,
      })
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError('Failed to load dashboard data')
      }
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <PageLayout title="Dashboard" description="Overview of your experiments and system status">
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <Card key={i}><CardContent className="h-24"><LoadingState /></CardContent></Card>
          ))}
        </div>
        <Card><CardContent className="h-64"><LoadingState /></CardContent></Card>
      </PageLayout>
    )
  }

  if (error) {
    return (
      <PageLayout title="Dashboard" description="Overview of your experiments and system status">
        <ErrorState message={error} onRetry={loadDashboardData} />
      </PageLayout>
    )
  }

  const statsCards = [
    { label: 'Total Experiments', value: stats.total, icon: Database, color: 'text-[var(--fg)]' },
    { label: 'Completed', value: stats.completed, icon: CheckCircle, color: 'text-green-500' },
    { label: 'Processing', value: stats.processing, icon: Loader2, color: 'text-[var(--primary)]' },
    { label: 'Avg Runtime', value: formatDuration(stats.avgRuntime), icon: Clock, color: 'text-[var(--fg)]' },
    { label: 'Latest NMSE', value: stats.latestNmse !== null ? `${stats.latestNmse.toFixed(2)} dB` : '—', icon: TrendingUp, color: 'text-[var(--recon)]' },
    { label: 'Latest Experiment', value: stats.latestExperiment?.clip_id || '—', icon: Database, color: 'text-[var(--fg-muted)]' },
  ]

  const latestRuns = runs.slice(0, 10)

  // Prepare chart data
  const activityData = runs
    .filter(r => r.created_at)
    .sort((a, b) => (a.created_at || 0) - (b.created_at || 0))
    .slice(-30)
    .map(r => ({
      date: new Date((r.created_at || 0) * 1000).toLocaleDateString(),
      count: 1,
    }))
    .reduce((acc, curr) => {
      const existing = acc.find(d => d.date === curr.date)
      if (existing) existing.count++
      else acc.push(curr)
      return acc
    }, [] as { date: string; count: number }[])

  const nmseData = runs
    .filter(r => r.status === 'completed' && r.metrics?.nmse_tot_db !== null)
    .slice(-20)
    .map((r, i) => ({
      experiment: r.clip_id.substring(0, 12),
      nmse: r.metrics!.nmse_tot_db!,
    }))

  const runtimeData = runs
    .filter(r => r.status === 'completed')
    .slice(-10)
    .map(r => ({
      experiment: r.clip_id.substring(0, 12),
      runtime: r.runtime_s,
    }))

  return (
    <PageLayout
      title="Dashboard"
      description="Overview of your experiments and system status"
      action={
        <Link to="/workspace/new">
          <Button className="gap-2">
            <Plus className="h-4 w-4" />
            New Experiment
          </Button>
        </Link>
      }
    >
      {/* Stats Cards */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {statsCards.map((stat, index) => (
          <Card key={index} variant="elevated">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-[var(--fg-muted)]">{stat.label}</p>
                  <p className="text-2xl font-bold text-[var(--fg)] mt-1">{stat.value}</p>
                </div>
                <div className={cn('w-12 h-12 rounded-lg flex items-center justify-center', stat.color)}>
                  <stat.icon className="h-6 w-6" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <Card variant="elevated">
          <CardHeader>
            <CardTitle>Experiment Activity (Last 30 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            {activityData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={activityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="date" tick={{ fill: 'hsl(var(--fg-muted))', fontSize: 12 }} />
                  <YAxis tick={{ fill: 'hsl(var(--fg-muted))', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--surface))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState title="No activity data" description="Run experiments to see activity chart" />
            )}
          </CardContent>
        </Card>

        <Card variant="elevated">
          <CardHeader>
            <CardTitle>NMSE Across Experiments</CardTitle>
          </CardHeader>
          <CardContent>
            {nmseData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={nmseData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="experiment" tick={{ fill: 'hsl(var(--fg-muted))', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'hsl(var(--fg-muted))', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--surface))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${value.toFixed(2)} dB`, 'NMSE (lower is better)']}
                  />
                  <Bar dataKey="nmse" fill="hsl(var(--recon))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState title="No NMSE data" description="Completed experiments with reference audio will appear here" />
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <Card variant="elevated">
          <CardHeader>
            <CardTitle>Processing Time Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            {runtimeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={runtimeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="experiment" tick={{ fill: 'hsl(var(--fg-muted))', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'hsl(var(--fg-muted))', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--surface))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${formatDuration(value)}`, 'Runtime']}
                  />
                  <Bar dataKey="runtime" fill="hsl(var(--missing))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState title="No runtime data" description="Completed experiments will appear here" />
            )}
          </CardContent>
        </Card>

        <Card variant="elevated">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Latest Experiments</CardTitle>
            <Link to="/experiments" className="text-sm text-[var(--primary)] hover:underline flex items-center gap-1">
              View All
              <ArrowRight className="h-4 w-4" />
            </Link>
          </CardHeader>
          <CardContent>
            {latestRuns.length > 0 ? (
              <div className="space-y-3">
                {latestRuns.map((run) => (
                  <Link key={run.id} to={`/experiments/${run.id}`} className="block">
                    <div className="flex items-center justify-between p-3 rounded-md hover:bg-[var(--raised)] transition-colors">
                      <div className="flex items-center gap-3 min-w-0">
                        <div className={cn('w-2 h-2 rounded-full', run.status === 'completed' ? 'bg-green-500' : run.status === 'running' || run.status === 'queued' ? 'bg-[var(--primary)]' : run.status === 'failed' ? 'bg-[var(--danger)]' : 'bg-gray-500')} />
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-[var(--fg)] truncate">{run.clip_id}</p>
                          <p className="text-xs text-[var(--fg-muted)] font-mono">{formatMs(run.gap_ms)} gap • {run.epochs} epochs</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 text-sm text-[var(--fg-muted)]">
                        <span className="font-mono">{formatDate(run.created_at)}</span>
                        <Badge
                          variant={
                            run.status === 'completed' ? 'success' :
                            run.status === 'running' || run.status === 'queued' ? 'primary' :
                            run.status === 'failed' ? 'danger' : 'outline'
                          }
                          size="sm"
                        >
                          {run.status}
                        </Badge>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <EmptyState
                title="No experiments yet"
                description="Start your first reconstruction to see results here"
                action={{ label: 'New Experiment', onClick: () => {}, variant: 'primary' }}
              />
            )}
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  )
}