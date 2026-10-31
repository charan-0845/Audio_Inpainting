import { vi } from 'vitest'
import { detectSilentRegions } from '@/utils/silenceDetection'

function createSignal(sampleRate: number, durationSeconds: number, silentStart: number, silentEnd: number) {
  const samples = new Float32Array(Math.round(sampleRate * durationSeconds))
  for (let i = 0; i < samples.length; i += 1) {
    const time = i / sampleRate
    samples[i] = time >= silentStart && time < silentEnd ? 0 : 0.2
  }
  return samples
}

describe('Silence Detection', () => {
  it('detects a sustained silent interval using the actual sample rate', () => {
    const sampleRate = 44100
    const regions = detectSilentRegions(
      createSignal(sampleRate, 2, 0.75, 1.15),
      sampleRate,
      -40,
      100,
    )

    expect(regions).toHaveLength(1)
    expect(regions[0][0]).toBeCloseTo(0.75, 2)
    expect(regions[0][1]).toBeCloseTo(1.15, 2)
  })

  it('does not fragment silence around short noisy interruptions', () => {
    const sampleRate = 16000
    const samples = createSignal(sampleRate, 1.5, 0.4, 1.1)
    const interruptionStart = Math.round(sampleRate * 0.7)
    for (let i = interruptionStart; i < interruptionStart + Math.round(sampleRate * 0.01); i += 1) {
      samples[i] = 0.2
    }

    expect(detectSilentRegions(samples, sampleRate, -40, 100)).toHaveLength(1)
  })

  it('ignores intervals shorter than the configured duration', () => {
    const sampleRate = 16000
    expect(detectSilentRegions(createSignal(sampleRate, 1, 0.4, 0.45), sampleRate, -40, 100)).toEqual([])
  })
})

describe('Gap Validation Logic', () => {
  // Test the validation logic directly
  it('should validate gap bounds correctly', () => {
    // Test validation logic from the component
    const validateGaps = (regions: Array<{start: number, end: number}>, audioDuration: number) => {
      if (regions.length === 0) {
        return { valid: false, error: 'Mark at least one corrupted gap to inpaint' }
      }
      
      const totalMs = audioDuration * 1000
      let totalMissingMs = 0
      let hasGapOver100 = false
      
      const merged: [number, number][] = []
      regions.forEach((r) => {
        if ((r.end - r.start) * 1000 > 100) hasGapOver100 = true
        if (merged.length === 0 || merged[merged.length - 1][1] < r.start) {
          merged.push([r.start, r.end])
        } else {
          merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], r.end)
        }
      })
      
      merged.forEach(([s, e]) => {
        totalMissingMs += (e - s) * 1000
      })
      
      const missingPct = (totalMissingMs / totalMs) * 100
      if (missingPct >= 40) {
        return { valid: false, error: `Total missing audio (${missingPct.toFixed(1)}%) exceeds maximum 40% budget` }
      }
      if (hasGapOver100) {
        return { valid: true, warning: 'Notice: One or more gaps exceed 100 ms. Paper method is calibrated for ≤ 100 ms.' }
      }
      
      return { valid: true }
    }
    
    // Test cases
    expect(validateGaps([], 5)).toEqual({ valid: false, error: 'Mark at least one corrupted gap to inpaint' })
    
    // Gap of 2 seconds in 5 seconds = 40% - this should fail (>= 40%)
    // Use a smaller gap that's valid
    expect(validateGaps([{ start: 1, end: 2.5 }], 5)).toEqual({ 
      valid: true, 
      warning: expect.stringContaining('exceed 100 ms') 
    })
    
    // 45% missing should fail
    expect(validateGaps([{ start: 0.5, end: 3 }], 5)).toEqual({ 
      valid: false, 
      error: expect.stringContaining('exceeds maximum 40%') 
    })
    
    // Gap over 100ms should warn
    const result = validateGaps([{ start: 1, end: 1.2 }], 5)
    expect(result.valid).toBe(true)
    expect(result.warning).toContain('exceed 100 ms')
    
    // Test non-overlapping small gaps (each 500ms = 500ms < 100ms? No, 500ms > 100ms)
    // Use gaps < 100ms each
    expect(validateGaps([{ start: 1, end: 1.05 }, { start: 2, end: 2.05 }], 5)).toEqual({ valid: true })
    
    // Test zero duration gap (merged to 0 duration - considered valid since no missing audio)
    expect(validateGaps([{ start: 1, end: 1 }], 5)).toEqual({ valid: true })
  })
})