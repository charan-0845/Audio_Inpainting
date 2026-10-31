import type {
  HealthResponse,
  PresetsResponse,
  JobSubmitRequest,
  JobSubmitResponse,
  JobResponse,
  SSEEvent,
  ExperimentRun,
  ExperimentListResponse,
  ExperimentFilters,
  ComparisonResult,
} from '@/types/api'

const API_BASE = '/api'

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers)
  if (!(options?.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (!response.ok) {
    const responseText = await response.text()
    let errorData: unknown = responseText
    try {
      errorData = responseText ? JSON.parse(responseText) : undefined
    } catch {
      // Keep the plain-text response as the error payload.
    }
    throw new ApiError(
      typeof errorData === 'object' && errorData && 'detail' in errorData
        ? String((errorData as Record<string, unknown>).detail)
        : `HTTP ${response.status}: ${response.statusText}`,
      response.status,
      errorData
    )
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

async function fetchBlob(url: string, options?: RequestInit): Promise<Blob> {
  const response = await fetch(url, options)
  if (!response.ok) {
    throw new ApiError(`HTTP ${response.status}: ${response.statusText}`, response.status)
  }
  return response.blob()
}

export const api = {
  async health(): Promise<HealthResponse> {
    return fetchJson(`${API_BASE}/health`)
  },

  async getPresets(): Promise<PresetsResponse> {
    return fetchJson(`${API_BASE}/presets`)
  },

  async detectSilence(
    audio: File,
    thresholdDb: number,
    minGapMs: number,
  ): Promise<{ sample_rate: number; duration_seconds: number; regions: Array<{ start: number; end: number }> }> {
    const formData = new FormData()
    formData.append('audio', audio)
    formData.append('threshold_db', String(thresholdDb))
    formData.append('min_gap_ms', String(minGapMs))
    return fetchJson(`${API_BASE}/detect-silence`, {
      method: 'POST',
      body: formData,
    })
  },

  async submitJob(request: JobSubmitRequest): Promise<JobSubmitResponse> {
    const formData = new FormData()

    if (request.audio) {
      formData.append('audio', request.audio)
    }
    if (request.reference) {
      formData.append('reference', request.reference)
    }
    if (request.gaps_json) {
      formData.append('gaps_json', request.gaps_json)
    }
    formData.append('mode', request.mode)
    if (request.gap_ms !== undefined) {
      formData.append('gap_ms', String(request.gap_ms))
    }
    formData.append('preset', request.preset)
    if (request.epochs !== undefined) {
      formData.append('epochs', String(request.epochs))
    }
    formData.append('seed', String(request.seed))
    formData.append('preserve_observed', String(request.preserve_observed))

    const response = await fetch(`${API_BASE}/jobs`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const responseText = await response.text()
      let errorData: unknown = responseText
      try {
        errorData = responseText ? JSON.parse(responseText) : undefined
      } catch {
        // Keep the plain-text response as the error payload.
      }
      throw new ApiError(
        typeof errorData === 'object' && errorData && 'detail' in errorData
          ? String((errorData as Record<string, unknown>).detail)
          : `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      )
    }

    return response.json()
  },

  async getJob(jobId: string): Promise<JobResponse> {
    return fetchJson(`${API_BASE}/jobs/${jobId}`)
  },

  async cancelJob(jobId: string): Promise<{ id: string; status: string }> {
    return fetchJson(`${API_BASE}/jobs/${jobId}/cancel`, { method: 'POST' })
  },

  async *streamJobEvents(jobId: string): AsyncGenerator<SSEEvent> {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/events`)
    if (!response.ok) {
      throw new ApiError(`HTTP ${response.status}: ${response.statusText}`, response.status)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new ApiError('No response body', 500)
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as SSEEvent
              yield event
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  async downloadFile(jobId: string, filename: string, download = false): Promise<Blob> {
    const url = `${API_BASE}/jobs/${jobId}/files/${filename}${download ? '?download=true' : ''}`
    return fetchBlob(url)
  },

  async getExperiments(filters?: ExperimentFilters): Promise<ExperimentListResponse> {
    const params = new URLSearchParams()
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
          params.append(key, String(value))
        }
      })
    }
    const query = params.toString() ? `?${params.toString()}` : ''
    return fetchJson(`${API_BASE}/experiments${query}`)
  },

  async getExperiment(id: string): Promise<ExperimentRun> {
    return fetchJson(`${API_BASE}/experiments/${id}`)
  },

  async deleteExperiment(id: string): Promise<void> {
    return fetchJson(`${API_BASE}/experiments/${id}`, { method: 'DELETE' })
  },

  async compareExperiments(idA: string, idB: string): Promise<ComparisonResult> {
    return fetchJson(`${API_BASE}/experiments/compare`, {
      method: 'POST',
      body: JSON.stringify({ experiment_a: idA, experiment_b: idB }),
    })
  },

  async testConnection(url: string): Promise<HealthResponse> {
    return fetchJson(`${url}/api/health`)
  },
}

export { ApiError }