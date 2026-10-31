// WaveSurfer v7 type extensions for regions plugin
import WaveSurfer from 'wavesurfer.js'

interface Region {
  id: string
  start: number
  end: number
  color?: string
  [key: string]: any
  
  update(options: { start?: number; end?: number; [key: string]: any }): void
  remove(): void
}

declare module 'wavesurfer.js' {
  interface WaveSurfer {
    // Region methods (built-in in v7)
    addRegion(options: {
      id: string
      start: number
      end: number
      color?: string
      drag?: boolean
      resize?: boolean
      [key: string]: any
    }): Region
    
    clearRegions(): void
    
    // Region events
    on(event: 'region-click', listener: (region: Region, e: Event) => void): void
    on(event: 'region-update-end', listener: (region: Region) => void): void
    on(event: 'region-removed', listener: (region: Region) => void): void
    on(event: 'region-update', listener: (region: Region) => void): void
    on(event: 'region-in', listener: (region: Region) => void): void
    on(event: 'region-out', listener: (region: Region) => void): void
  }
}