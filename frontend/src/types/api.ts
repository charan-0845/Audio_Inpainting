export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface HealthResponse {
  status: 'ok'
  device: string
  queue_length: number
  running_job: string | null
  demo_only: boolean
  max_seconds: number
}

export interface PresetInfo {
  epochs: number
  base_filters: number
  harmonic: boolean
  description: string
}

export interface PresetsResponse {
  presets: {
    fast: PresetInfo
    balanced: PresetInfo
    paper: PresetInfo
    [key: string]: PresetInfo
  }
  default: string
  device: string
}

export interface JobSubmitRequest {
  audio?: File
  gaps_json?: string
  mode: 'gaps' | 'synthetic' | 'detect'
  gap_ms?: number
  preset: string
  epochs?: number
  seed: number
  preserve_observed: boolean
  reference?: File
}

export interface JobSubmitResponse {
  id: string
  status: JobStatus
  clip_id: string
  gap_ms: number
  warnings: string[]
  queue_position: number
}

export interface JobProgress {
  epoch: number
  epochs: number
  loss: number | null
  eta_s: number | null
}

export interface JobMetrics {
  nmse_tot_db: number | null
  nmse_miss_db: number | null
  pesq_delta: number | null
}

export interface JobResponse {
  id: string
  status: JobStatus
  clip_id: string
  gap_ms: number
  seed: number
  epochs: number
  progress: JobProgress
  warnings: string[]
  error: string | null
  created_at: number | null
  started_at: number | null
  completed_at: number | null
  metrics: JobMetrics | null
  n_params: number | null
  device: string
  runtime_s: number
  files: string[]
}

export type SSEEventType = 'status' | 'progress' | 'completed' | 'failed' | 'cancelled'

export interface SSEBaseEvent {
  type: SSEEventType
  status?: JobStatus
}

export interface SSEStatusEvent extends SSEBaseEvent {
  type: 'status'
  status: JobStatus
  epoch?: number
  epochs?: number
}

export interface SSEProgressEvent extends SSEBaseEvent {
  type: 'progress'
  epoch: number
  epochs: number
  loss: number
  eta_s: number
  preview?: string
}

export interface SSECompletedEvent extends SSEBaseEvent {
  type: 'completed'
  run_id: string
  metrics: JobMetrics
  n_params: number | null
  files: string[]
}

export interface SSEFailedEvent extends SSEBaseEvent {
  type: 'failed'
  error: string
}

export interface SSECanceledEvent extends SSEBaseEvent {
  type: 'cancelled'
}

export type SSEEvent =
  | SSEStatusEvent
  | SSEProgressEvent
  | SSECompletedEvent
  | SSEFailedEvent
  | SSECanceledEvent

export interface ExperimentRun {
  id: string
  clip_id: string
  gap_ms: number
  seed: number
  epochs: number
  runtime_s: number
  device: string
  status: JobStatus
  metrics: JobMetrics | null
  n_params: number | null
  created_at: number | null
  files: string[]
}

export interface ExperimentListResponse {
  runs: ExperimentRun[]
  total: number
}

export interface ExperimentFilters {
  search?: string
  status?: JobStatus | 'all'
  date_from?: string
  date_to?: string
  sort_by?: 'created_at' | 'gap_ms' | 'runtime_s' | 'nmse_tot_db'
  sort_order?: 'asc' | 'desc'
  page?: number
  page_size?: number
}

export interface ComparisonResult {
  experiment_a: ExperimentRun
  experiment_b: ExperimentRun
  metrics_comparison: {
    nmse_tot_db: { a: number | null; b: number | null; diff: number | null }
    nmse_miss_db: { a: number | null; b: number | null; diff: number | null }
    runtime_s: { a: number; b: number; diff: number }
    epochs: { a: number; b: number }
    n_params: { a: number | null; b: number | null }
  }
  config_differences: Record<string, { a: unknown; b: unknown; same: boolean }>
  condition_flags: string[]
}

export interface BackendProfile {
  id: string
  name: string
  url: string
  type: 'local' | 'colab' | 'cloud'
  is_active: boolean
}

export interface SettingsState {
  theme: 'light' | 'dark' | 'system'
  backend_profiles: BackendProfile[]
  active_backend_id: string | null
}