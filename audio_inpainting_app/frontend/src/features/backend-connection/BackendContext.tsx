import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import type { HealthResponse, BackendProfile, SettingsState } from '@/types/api'
import { api, ApiError } from '@/services/api/client'

interface BackendContextType {
  backendUrl: string
  health: HealthResponse | null
  isConnected: boolean
  isConnecting: boolean
  connectionError: string | null
  profiles: BackendProfile[]
  activeProfile: BackendProfile | null
  setBackendUrl: (url: string) => void
  testConnection: (url: string) => Promise<boolean>
  addProfile: (profile: Omit<BackendProfile, 'id' | 'is_active'>) => void
  removeProfile: (id: string) => void
  setActiveProfile: (id: string) => void
  refreshHealth: () => Promise<void>
}

const DEFAULT_LOCAL_URL = 'http://127.0.0.1:8000'

const BackendContext = createContext<BackendContextType | undefined>(undefined)

const STORAGE_KEY = 'dpai-backend-settings'

function loadSettings(): SettingsState {
  if (typeof window === 'undefined') {
    return { theme: 'system', backend_profiles: [], active_backend_id: null }
  }
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) return JSON.parse(stored)
  } catch {
    // Ignore parse errors
  }
  return { theme: 'system', backend_profiles: [], active_backend_id: null }
}

function saveSettings(settings: SettingsState) {
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  }
}

export function BackendProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<SettingsState>(() => loadSettings())
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)

  const activeProfile = settings.backend_profiles.find(p => p.id === settings.active_backend_id) || null
  const backendUrl = activeProfile?.url || DEFAULT_LOCAL_URL
  const isConnected = health?.status === 'ok'

  const refreshHealth = useCallback(async () => {
    try {
      const h = await api.health()
      setHealth(h)
      setConnectionError(null)
    } catch (err) {
      setHealth(null)
      if (err instanceof ApiError) {
        setConnectionError(err.message)
      } else {
        setConnectionError('Failed to connect to backend')
      }
    }
  }, [])

  useEffect(() => {
    refreshHealth()
    const interval = setInterval(refreshHealth, 30000)
    return () => clearInterval(interval)
  }, [refreshHealth])

  const setBackendUrl = useCallback((url: string) => {
    const normalizedUrl = url.replace(/\/$/, '')
    setSettings(prev => {
      const existingIndex = prev.backend_profiles.findIndex(p => p.url === normalizedUrl)
      let newProfiles: BackendProfile[]
      let newActiveId: string

      if (existingIndex >= 0) {
        newProfiles = [...prev.backend_profiles]
        newActiveId = newProfiles[existingIndex].id
      } else {
        const newProfile: BackendProfile = {
          id: crypto.randomUUID(),
          name: new URL(normalizedUrl).host || 'Custom',
          url: normalizedUrl,
          type: 'local',
          is_active: true,
        }
        newProfiles = [...prev.backend_profiles, newProfile]
        newActiveId = newProfile.id
      }

      const updated = { ...prev, backend_profiles: newProfiles, active_backend_id: newActiveId }
      saveSettings(updated)
      return updated
    })
  }, [])

  const testConnection = useCallback(async (url: string): Promise<boolean> => {
    setIsConnecting(true)
    setConnectionError(null)
    try {
      const normalizedUrl = url.replace(/\/$/, '')
      const h = await api.testConnection(normalizedUrl)
      setHealth(h)
      setIsConnecting(false)
      return h.status === 'ok'
    } catch (err) {
      setIsConnecting(false)
      if (err instanceof ApiError) {
        setConnectionError(err.message)
      } else {
        setConnectionError('Connection failed')
      }
      return false
    }
  }, [])

  const addProfile = useCallback((profile: Omit<BackendProfile, 'id' | 'is_active'>) => {
    setSettings(prev => {
      const newProfile: BackendProfile = {
        ...profile,
        id: crypto.randomUUID(),
        is_active: true,
      }
      const updated = {
        ...prev,
        backend_profiles: [...prev.backend_profiles, newProfile],
        active_backend_id: newProfile.id,
      }
      saveSettings(updated)
      return updated
    })
  }, [])

  const removeProfile = useCallback((id: string) => {
    setSettings(prev => {
      const newProfiles = prev.backend_profiles.filter(p => p.id !== id)
      const newActiveId = prev.active_backend_id === id
        ? (newProfiles[0]?.id || null)
        : prev.active_backend_id
      const updated = { ...prev, backend_profiles: newProfiles, active_backend_id: newActiveId }
      saveSettings(updated)
      return updated
    })
  }, [])

  const setActiveProfile = useCallback((id: string) => {
    setSettings(prev => {
      const updated = { ...prev, active_backend_id: id }
      saveSettings(updated)
      return updated
    })
  }, [])

  return (
    <BackendContext.Provider
      value={{
        backendUrl,
        health,
        isConnected,
        isConnecting,
        connectionError,
        profiles: settings.backend_profiles,
        activeProfile,
        setBackendUrl,
        testConnection,
        addProfile,
        removeProfile,
        setActiveProfile,
        refreshHealth,
      }}
    >
      {children}
    </BackendContext.Provider>
  )
}

export function useBackend() {
  const context = useContext(BackendContext)
  if (!context) {
    throw new Error('useBackend must be used within a BackendProvider')
  }
  return context
}