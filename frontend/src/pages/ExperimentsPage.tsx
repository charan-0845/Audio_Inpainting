import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { Search, Filter, Download, Trash2, RefreshCw, ChevronLeft, ChevronRight, CheckSquare, Square, MinusSquare } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Badge } from '@/components/ui/Badge'
import { Dialog } from '@/components/ui/Dialog'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { EmptyState, LoadingState, ErrorState } from '@/components/ui/StateComponents'
import { Tooltip } from '@/components/ui/Tooltip'
import { api, ApiError } from '@/services/api/client'
import type { ExperimentRun, ExperimentFilters } from '@/types/api'
import { formatDuration, formatMs, formatDate, formatNumber } from '@/lib/utils'
import { PageLayout } from '@/components/layout'
import { cn } from '@/lib/utils'

export function ExperimentsPage() {
  const [runs, setRuns] = useState<ExperimentRun[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const pageSize = 20
  
  const [filters, setFilters] = useState<ExperimentFilters>({
    search: '',
    status: 'all',
    sort_by: 'created_at',
    sort_order: 'desc',
    page: 1,
    page_size: pageSize,
  })
  
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [deleteId, setDeleteId] = useState<string | null>(null)

  const loadExperiments = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.getExperiments(filters)
      setRuns(response.runs || [])
      setTotal(response.total || 0)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError('Failed to load experiments')
      }
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    loadExperiments()
  }, [loadExperiments])

  const handleSearch = (value: string) => {
    setFilters(prev => ({ ...prev, search: value, page: 1 }))
    setPage(1)
  }

  const handleStatusChange = (status: string) => {
    setFilters(prev => ({ ...prev, status: status as ExperimentFilters['status'], page: 1 }))
    setPage(1)
  }

  const handleSort = (sortBy: ExperimentFilters['sort_by']) => {
    setFilters(prev => {
      const newOrder = prev.sort_by === sortBy && prev.sort_order === 'desc' ? 'asc' : 'desc'
      return { ...prev, sort_by: sortBy, sort_order: newOrder }
    })
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === runs.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(runs.map(r => r.id)))
    }
  }

  const handleDelete = async (id: string) => {
    setDeleteId(id)
    setDeleteDialogOpen(true)
  }

  const confirmDelete = async () => {
    if (!deleteId) return
    try {
      await api.deleteExperiment(deleteId)
      loadExperiments()
      setSelectedIds(prev => { const next = new Set(prev); next.delete(deleteId); return next })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete experiment')
    }
    setDeleteDialogOpen(false)
    setDeleteId(null)
  }

  const handleBulkDelete = async () => {
    for (const id of selectedIds) {
      try {
        await api.deleteExperiment(id)
      } catch (err) {
        console.error(`Failed to delete ${id}`, err)
      }
    }
    loadExperiments()
    setSelectedIds(new Set())
  }

  const statusBadges: Record<string, 'success' | 'primary' | 'danger' | 'warning' | 'outline'> = {
    completed: 'success',
    running: 'primary',
    queued: 'primary',
    failed: 'danger',
    cancelled: 'outline',
  }

  return (
    <PageLayout
      title="Experiment History"
      description="Browse, search, and manage your inpainting experiments"
      action={
        <div className="flex items-center gap-2">
          <Link to="/workspace/new">
            <Button className="gap-2">
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
              New Experiment
            </Button>
          </Link>
        </div>
      }
    >
      {/* Toolbar */}
      <Card variant="elevated" className="mb-6">
        <CardContent className="pt-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--fg-muted)]" />
              <Input
                placeholder="Search by clip ID..."
                value={filters.search || ''}
                onChange={e => handleSearch(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select
              value={filters.status || 'all'}
              onValueChange={handleStatusChange}
              options={[
                { value: 'all', label: 'All Status' },
                { value: 'completed', label: 'Completed' },
                { value: 'running', label: 'Running' },
                { value: 'queued', label: 'Queued' },
                { value: 'failed', label: 'Failed' },
                { value: 'cancelled', label: 'Cancelled' },
              ]}
              className="w-40"
            />
            <Select
              value={`${filters.sort_by}:${filters.sort_order}`}
              onValueChange={v => {
                const [sort_by, sort_order] = v.split(':') as [ExperimentFilters['sort_by'], ExperimentFilters['sort_order']]
                setFilters(prev => ({ ...prev, sort_by, sort_order }))
              }}
              options={[
                { value: 'created_at:desc', label: 'Newest First' },
                { value: 'created_at:asc', label: 'Oldest First' },
                { value: 'gap_ms:desc', label: 'Largest Gap' },
                { value: 'gap_ms:asc', label: 'Smallest Gap' },
                { value: 'runtime_s:desc', label: 'Longest Runtime' },
                { value: 'runtime_s:asc', label: 'Shortest Runtime' },
              ]}
              className="w-40"
            />
            <Button variant="secondary" onClick={loadExperiments} disabled={loading}>
              <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results table */}
      <Card variant="elevated">
        {loading ? (
          <CardContent className="py-12"><LoadingState message="Loading experiments..." /></CardContent>
        ) : error ? (
          <CardContent className="py-12"><ErrorState message={error} onRetry={loadExperiments} /></CardContent>
        ) : runs.length === 0 ? (
          <CardContent className="py-12">
            <EmptyState
              icon={<svg className="h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>}
              title="No experiments found"
              description={filters.search || filters.status !== 'all' ? 'Try adjusting your filters' : 'Start your first reconstruction to see results here'}
              action={{ label: 'New Experiment', onClick: () => {}, variant: 'primary' }}
            />
          </CardContent>
        ) : (
          <>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">
                      <input
                        type="checkbox"
                        ref={el => { if (el) el.indeterminate = selectedIds.size > 0 && selectedIds.size < runs.length; }}
                        checked={selectedIds.size === runs.length && runs.length > 0}
                        onChange={toggleSelectAll}
                        className="w-4 h-4 rounded border-[var(--border)] text-[var(--primary)] focus:ring-[var(--primary)]"
                      />
                    </TableHead>
                    <TableHead onClick={() => handleSort('created_at')} className="cursor-pointer select-none">
                      Date <span className="ml-1">{filters.sort_by === 'created_at' ? (filters.sort_order === 'desc' ? '↓' : '↑') : ''}</span>
                    </TableHead>
                    <TableHead className="cursor-pointer select-none">
                      Clip
                    </TableHead>
                    <TableHead onClick={() => handleSort('gap_ms')} className="cursor-pointer select-none">
                      Gap <span className="ml-1">{filters.sort_by === 'gap_ms' ? (filters.sort_order === 'desc' ? '↓' : '↑') : ''}</span>
                    </TableHead>
                    <TableHead className="hidden md:table-cell">Status</TableHead>
                    <TableHead className="hidden lg:table-cell">Epochs</TableHead>
                    <TableHead className="hidden lg:table-cell cursor-pointer select-none" onClick={() => handleSort('runtime_s')}>
                      Runtime <span className="ml-1">{filters.sort_by === 'runtime_s' ? (filters.sort_order === 'desc' ? '↓' : '↑') : ''}</span>
                    </TableHead>
                    <TableHead className="hidden lg:table-cell">NMSE</TableHead>
                    <TableHead className="w-24">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runs.map(run => (
                    <TableRow key={run.id} className={selectedIds.has(run.id) ? 'bg-[var(--primary)]/5' : ''}>
                      <TableCell>
                        <input
                          type="checkbox"
                          checked={selectedIds.has(run.id)}
                          onChange={e => { e.stopPropagation(); toggleSelect(run.id) }}
                          className="w-4 h-4 rounded border-[var(--border)] text-[var(--primary)] focus:ring-[var(--primary)]"
                        />
                      </TableCell>
                      <TableCell className="font-mono text-sm text-[var(--fg-muted)]">
                        {formatDate(run.created_at)}
                      </TableCell>
                      <TableCell>
                        <Link to={`/experiments/${run.id}`} className="font-medium text-[var(--fg)] hover:text-[var(--primary)]">
                          {run.clip_id}
                        </Link>
                      </TableCell>
                      <TableCell className="font-mono text-[var(--fg)]">
                        {formatMs(run.gap_ms)}
                      </TableCell>
                      <TableCell className="hidden md:table-cell">
                        <Badge variant={statusBadges[run.status] || 'outline'} size="sm">
                          {run.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="hidden lg:table-cell font-mono">{run.epochs}</TableCell>
                      <TableCell className="hidden lg:table-cell font-mono text-[var(--fg)]">
                        {formatDuration(run.runtime_s)}
                      </TableCell>
                      <TableCell className="hidden lg:table-cell font-mono text-[var(--recon)]">
                        {run.metrics && run.metrics.nmse_tot_db !== null ? `${run.metrics.nmse_tot_db.toFixed(2)} dB` : '—'}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <Tooltip content="View results" side="top">
                            <Link to={`/experiments/${run.id}`}>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7z" /></svg>
                              </Button>
                            </Link>
                          </Tooltip>
                          <Tooltip content="Download reconstruction" side="top">
                            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={e => { e.preventDefault(); e.stopPropagation(); window.open(`/api/jobs/${run.id}/files/reconstruction.wav?download=true`) }}>
                              <Download className="h-4 w-4" />
                            </Button>
                          </Tooltip>
                          <Tooltip content="Delete" side="top">
                            <Button variant="ghost" size="icon" className="h-8 w-8 text-[var(--danger)] hover:bg-[var(--danger)]/10" onClick={e => { e.preventDefault(); e.stopPropagation(); handleDelete(run.id) }}>
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </Tooltip>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {/* Pagination */}
            {total > pageSize && (
              <div className="flex items-center justify-between px-6 py-4 border-t border-[var(--border)]">
                <p className="text-sm text-[var(--fg-muted)]">
                  Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, total)} of {total} experiments
                </p>
                <div className="flex items-center gap-2">
                  <Button variant="secondary" size="sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => setPage(p => Math.min(Math.ceil(total / pageSize), p + 1))} disabled={page >= Math.ceil(total / pageSize)}>
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}

            {/* Bulk actions */}
            {selectedIds.size > 0 && (
              <div className="px-6 py-4 border-t border-[var(--border)] bg-[var(--raised)]/50 flex items-center justify-between">
                <span className="text-sm text-[var(--fg-muted)]">{selectedIds.size} selected</span>
                <Button variant="danger" size="sm" onClick={handleBulkDelete}>
                  <Trash2 className="h-4 w-4 mr-1" /> Delete Selected
                </Button>
              </div>
            )}
          </>
        )}
      </Card>

      {/* Delete confirmation dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen} title="Delete Experiment" description="This action cannot be undone.">
        <p className="text-sm text-[var(--fg-muted)] mb-6">
          Are you sure you want to delete experiment <code className="font-mono bg-[var(--raised)] px-1 rounded">{deleteId}</code>? All associated artifacts will be permanently removed.
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button variant="danger" onClick={confirmDelete}>Delete</Button>
        </div>
      </Dialog>
    </PageLayout>
  )
}