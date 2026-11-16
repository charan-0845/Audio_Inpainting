/**
 * Long-audio window planner and gap classifier utility.
 * Mirrors the logic of src/pipeline/long_inpaint.py.
 * Note: The backend plan is authoritative.
 */

export interface LongAudioPlanInput {
  gaps: [number, number][] // Gaps in milliseconds [start_ms, end_ms]
  duration_s: number
  sample_rate?: number
  window_s?: number
  context_s?: number
  lpc_max_gap_ms?: number
  lpc_skip_if_in_window?: boolean
  preset?: string
  device?: string
  epochs?: number
  long_audio_mode?: 'auto' | 'on' | 'off'
}

export interface PlannedWindow {
  window_index: number
  start_s: number
  end_s: number
  start_sample: number
  end_sample: number
  n_gaps: number
  gaps: [number, number][] // in samples [start, end]
}

export interface LongAudioPlanResult {
  is_long_audio: boolean
  windows: PlannedWindow[]
  n_windows: number
  lpc_gaps: [number, number][] // in ms
  dpai_gaps: [number, number][] // in ms
  lpc_gap_count: number
  dpai_gap_count: number
  estimated_runtime_s: number
  runtime_estimate_label: string
}

export function planLongAudio(input: LongAudioPlanInput): LongAudioPlanResult {
  const {
    gaps,
    duration_s,
    sample_rate = 16000,
    window_s = 5.0,
    context_s = 0.5,
    lpc_max_gap_ms = 40.0,
    lpc_skip_if_in_window = true,
    preset = 'balanced',
    device = 'cpu',
    epochs,
    long_audio_mode = 'auto',
  } = input

  if (!duration_s || duration_s <= 0 || gaps.length === 0) {
    return {
      is_long_audio: false,
      windows: [],
      n_windows: 0,
      lpc_gaps: [],
      dpai_gaps: [],
      lpc_gap_count: 0,
      dpai_gap_count: 0,
      estimated_runtime_s: 0,
      runtime_estimate_label: '0s (estimate)',
    }
  }

  // Determine if long audio should be active
  let is_long_audio = false
  if (long_audio_mode === 'on') {
    is_long_audio = true
  } else if (long_audio_mode === 'auto') {
    is_long_audio = duration_s > 10.0
  } else {
    is_long_audio = false
  }

  const sr = sample_rate
  const window_samples = Math.round(window_s * sr)
  const context_samples = Math.round(context_s * sr)
  const lpc_max_samples = Math.round((lpc_max_gap_ms / 1000) * sr)
  const sig_len = Math.round(duration_s * sr)

  // Convert gaps to samples and sort
  const sample_gaps: [number, number][] = gaps
    .map(([s_ms, e_ms]) => [
      Math.round((s_ms / 1000) * sr),
      Math.round((e_ms / 1000) * sr),
    ] as [number, number])
    .sort((a, b) => a[0] - b[0])

  // Split into short and long
  const short_gaps: [number, number][] = []
  const long_gaps: [number, number][] = []

  for (const gap of sample_gaps) {
    if (gap[1] - gap[0] <= lpc_max_samples) {
      short_gaps.push(gap)
    } else {
      long_gaps.push(gap)
    }
  }

  // If long audio mode is off, all gaps are processed via direct DPAI or LPC preset
  if (!is_long_audio) {
    const isLpcPreset = preset === 'lpc' || epochs === 0
    return {
      is_long_audio: false,
      windows: [],
      n_windows: 1,
      lpc_gaps: isLpcPreset ? gaps : [],
      dpai_gaps: isLpcPreset ? [] : gaps,
      lpc_gap_count: isLpcPreset ? gaps.length : 0,
      dpai_gap_count: isLpcPreset ? 0 : gaps.length,
      estimated_runtime_s: isLpcPreset ? 0.2 : estimateRuntime(1, preset, device, epochs),
      runtime_estimate_label: `~${isLpcPreset ? '0.2' : estimateRuntime(1, preset, device, epochs)}s (estimate)`,
    }
  }

  // If there are no long gaps: LPC-fill everything, no DPAI windows
  if (long_gaps.length === 0) {
    return {
      is_long_audio: true,
      windows: [],
      n_windows: 0,
      lpc_gaps: gaps,
      dpai_gaps: [],
      lpc_gap_count: gaps.length,
      dpai_gap_count: 0,
      estimated_runtime_s: 0.2,
      runtime_estimate_label: '~0.2s (estimate, LPC only)',
    }
  }

  // Plan windows from LONG gaps
  const windows: PlannedWindow[] = []
  if (sig_len <= window_samples) {
    windows.push({
      window_index: 0,
      start_s: 0,
      end_s: duration_s,
      start_sample: 0,
      end_sample: sig_len,
      n_gaps: long_gaps.length,
      gaps: long_gaps,
    })
  } else {
    // Greedy left-to-right cover
    const groups: [number, number][][] = []
    let current_group: [number, number][] = []

    for (const gap of long_gaps) {
      if (current_group.length === 0) {
        current_group.push(gap)
      } else {
        const first_start = current_group[0][0]
        const last_end = gap[1]
        if (last_end - first_start + 2 * context_samples <= window_samples) {
          current_group.push(gap)
        } else {
          groups.push(current_group)
          current_group = [gap]
        }
      }
    }
    if (current_group.length > 0) {
      groups.push(current_group)
    }

    groups.forEach((grp, idx) => {
      const first_start = grp[0][0]
      const last_end = grp[grp.length - 1][1]
      const mid = (first_start + last_end) / 2
      const ideal_start = Math.round(mid - window_samples / 2)
      const w_start = Math.max(0, Math.min(ideal_start, sig_len - window_samples))
      const w_end = w_start + window_samples

      windows.push({
        window_index: idx,
        start_s: Number((w_start / sr).toFixed(4)),
        end_s: Number((w_end / sr).toFixed(4)),
        start_sample: w_start,
        end_sample: w_end,
        n_gaps: grp.length,
        gaps: grp,
      })
    })
  }

  // Classify short gaps
  const lpc_short_gaps: [number, number][] = []
  const dpai_short_gaps: [number, number][] = []

  for (const [s, e] of short_gaps) {
    let inside = false
    if (lpc_skip_if_in_window && windows.length > 0) {
      for (const win of windows) {
        if (win.start_sample <= s && e <= win.end_sample) {
          inside = true
          break
        }
      }
    }
    if (inside) {
      dpai_short_gaps.push([s, e])
    } else {
      lpc_short_gaps.push([s, e])
    }
  }

  // Convert back to ms for output
  const lpc_gaps_ms: [number, number][] = lpc_short_gaps.map(([s, e]) => [
    Number(((s / sr) * 1000).toFixed(1)),
    Number(((e / sr) * 1000).toFixed(1)),
  ])

  const dpai_all_samples = [...long_gaps, ...dpai_short_gaps].sort((a, b) => a[0] - b[0])
  const dpai_gaps_ms: [number, number][] = dpai_all_samples.map(([s, e]) => [
    Number(((s / sr) * 1000).toFixed(1)),
    Number(((e / sr) * 1000).toFixed(1)),
  ])

  const n_win = windows.length
  const est_s = estimateRuntime(n_win, preset, device, epochs)

  return {
    is_long_audio: true,
    windows,
    n_windows: n_win,
    lpc_gaps: lpc_gaps_ms,
    dpai_gaps: dpai_gaps_ms,
    lpc_gap_count: lpc_gaps_ms.length,
    dpai_gap_count: dpai_gaps_ms.length,
    estimated_runtime_s: est_s,
    runtime_estimate_label: `~${est_s}s (estimate)`,
  }
}

function estimateRuntime(
  n_windows: number,
  preset: string,
  device: string,
  epochs?: number,
): number {
  if (n_windows === 0) return 0.2
  const isCpu = device === 'cpu'
  let ep = epochs
  if (ep === undefined || ep === null) {
    if (preset === 'fast') ep = isCpu ? 100 : 300
    else if (preset === 'paper') ep = 5000
    else ep = isCpu ? 300 : 1500
  }
  if (ep === 0) return 0.2

  // Approximate seconds per epoch per window
  const secPerEpoch = isCpu ? 0.12 : 0.015
  const est = Math.round(n_windows * ep * secPerEpoch)
  return Math.max(1, est)
}

