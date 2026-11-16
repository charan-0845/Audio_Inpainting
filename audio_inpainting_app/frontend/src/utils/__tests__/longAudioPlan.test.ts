import { describe, it, expect } from 'vitest'
import { planLongAudio } from '../longAudioPlan'

describe('longAudioPlan', () => {
  it('handles empty input gracefully', () => {
    const res = planLongAudio({ gaps: [], duration_s: 0 })
    expect(res.is_long_audio).toBe(false)
    expect(res.n_windows).toBe(0)
    expect(res.windows).toEqual([])
  })

  it('respects long_audio_mode = off', () => {
    const res = planLongAudio({
      gaps: [[1000, 1100], [4000, 4100]],
      duration_s: 20.0,
      long_audio_mode: 'off',
    })
    expect(res.is_long_audio).toBe(false)
    expect(res.dpai_gap_count).toBe(2)
  })

  it('routes to LPC-only when there are no long gaps', () => {
    // Both gaps <= 40 ms
    const res = planLongAudio({
      gaps: [[1000, 1020], [5000, 5030]],
      duration_s: 20.0,
      long_audio_mode: 'on',
      lpc_max_gap_ms: 40.0,
    })
    expect(res.is_long_audio).toBe(true)
    expect(res.n_windows).toBe(0)
    expect(res.lpc_gap_count).toBe(2)
    expect(res.dpai_gap_count).toBe(0)
  })

  it('plans a single window when clip <= window length', () => {
    const res = planLongAudio({
      gaps: [[1000, 1200]],
      duration_s: 4.0, // <= 5.0s window
      window_s: 5.0,
      long_audio_mode: 'on',
    })
    expect(res.n_windows).toBe(1)
    expect(res.windows[0].start_s).toBe(0)
    expect(res.windows[0].end_s).toBe(4.0)
    expect(res.dpai_gap_count).toBe(1)
  })

  it('plans minimal windows with context margin and clamping', () => {
    // 20s audio, window_s=5s, context_s=0.5s
    // Gap 1 near left edge (200ms - 500ms) -> clamped at 0
    // Gap 2 at 18s (18000ms - 18300ms) -> clamped at 20s
    const res = planLongAudio({
      gaps: [[200, 500], [18000, 18300]],
      duration_s: 20.0,
      window_s: 5.0,
      context_s: 0.5,
      long_audio_mode: 'on',
    })
    expect(res.n_windows).toBe(2)
    // Left window clamped at 0
    expect(res.windows[0].start_s).toBe(0)
    expect(res.windows[0].end_s).toBe(5.0)
    // Right window clamped at 20
    expect(res.windows[1].start_s).toBe(15.0)
    expect(res.windows[1].end_s).toBe(20.0)
  })

  it('correctly implements the LPC-skip rule', () => {
    // 20s audio
    // Long gap at 2.0s (2000-2200ms) -> window 0 centered around 2s ([0, 5s])
    // Short gap inside window 0 at 2.5s (2500-2520ms) -> skipped by LPC, goes to DPAI
    // Short gap outside at 12s (12000-12020ms) -> LPC-filled
    const res = planLongAudio({
      gaps: [
        [2000, 2200], // long (200ms)
        [2500, 2520], // short (20ms, inside win 0)
        [12000, 12020], // short (20ms, outside win 0)
      ],
      duration_s: 20.0,
      window_s: 5.0,
      lpc_max_gap_ms: 40.0,
      lpc_skip_if_in_window: true,
      long_audio_mode: 'on',
    })

    expect(res.n_windows).toBe(1)
    expect(res.dpai_gap_count).toBe(2) // long gap + short gap inside window
    expect(res.lpc_gap_count).toBe(1) // isolated short gap
  })

  it('provides an estimated runtime with estimate label', () => {
    const res = planLongAudio({
      gaps: [[2000, 2500]],
      duration_s: 20.0,
      long_audio_mode: 'on',
      epochs: 50,
      device: 'cpu',
    })
    expect(res.estimated_runtime_s).toBeGreaterThan(0)
    expect(res.runtime_estimate_label).toContain('estimate')
  })
})

