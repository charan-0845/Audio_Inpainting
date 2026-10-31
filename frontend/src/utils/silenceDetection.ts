export function detectSilentRegions(
  samples: Float32Array,
  sampleRate: number,
  thresholdDb: number,
  minDurationMs: number,
): Array<[number, number]> {
  const windowSize = Math.max(1, Math.round(sampleRate * 0.01))
  const hopSize = Math.max(1, Math.round(sampleRate * 0.005))
  const threshold = Math.pow(10, thresholdDb / 20)
  const minDuration = minDurationMs / 1000
  const silentFrames: Array<[number, number]> = []

  for (let start = 0; start < samples.length; start += hopSize) {
    const end = Math.min(samples.length, start + windowSize)
    let energy = 0
    for (let i = start; i < end; i += 1) {
      energy += samples[i] * samples[i]
    }
    const rms = Math.sqrt(energy / Math.max(1, end - start))
    if (rms <= threshold) {
      silentFrames.push([start / sampleRate, end / sampleRate])
    }
  }

  const regions: Array<[number, number]> = []
  for (const [start, end] of silentFrames) {
    const previous = regions[regions.length - 1]
    if (!previous || start - previous[1] > 0.02) {
      regions.push([start, end])
    } else {
      previous[1] = Math.max(previous[1], end)
    }
  }

  return regions.filter(([start, end]) => end - start >= minDuration)
}
