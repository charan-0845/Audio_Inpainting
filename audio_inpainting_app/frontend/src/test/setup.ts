import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock AudioContext
const mockAudioContext = {
  decodeAudioData: vi.fn().mockResolvedValue({
    getChannelData: () => new Float32Array(80000),
    length: 80000,
    sampleRate: 16000,
  }),
  createBuffer: vi.fn(),
  destination: {},
  sampleRate: 16000,
  state: 'running',
  resume: vi.fn(),
  suspend: vi.fn(),
  close: vi.fn(),
}

global.AudioContext = vi.fn().mockImplementation(() => mockAudioContext)
global.webkitAudioContext = global.AudioContext