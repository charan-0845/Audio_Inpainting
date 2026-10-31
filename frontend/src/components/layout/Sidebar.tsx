import { ReactNode, useState, useEffect } from 'react'
import { Link, useLocation, NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { Tooltip } from '@/components/ui/Tooltip'
import {
  LayoutDashboard,
  FilePlus,
  History,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  Menu,
  Zap,
  GitBranch,
  Code,
} from 'lucide-react'
import { useBackend } from '@/features/backend-connection/BackendContext'

const navigation = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/workspace/new', label: 'New Reconstruction', icon: FilePlus },
  { path: '/experiments', label: 'Experiment History', icon: History },
  { path: '/research', label: 'Research', icon: BookOpen },
  { path: '/settings', label: 'Settings', icon: Settings },
]

const landingNavigation = [
  { path: '/', label: 'Home', icon: LayoutDashboard },
  { path: '/research', label: 'About Research', icon: BookOpen },
  { path: '/research/methodology', label: 'Methodology', icon: GitBranch },
  { path: '/documentation', label: 'Documentation', icon: Code },
  { path: '/workspace/new', label: 'Launch Workspace', icon: Zap },
]

interface SidebarProps {
  isLanding?: boolean
}

export function Sidebar({ isLanding = false }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const { isConnected, health, activeProfile } = useBackend()
  const navItems = isLanding ? landingNavigation : navigation

  useEffect(() => {
    if (window.innerWidth >= 1024) {
      setMobileOpen(false)
    }
  }, [])

  const handleResize = () => {
    if (window.innerWidth >= 1024) {
      setMobileOpen(false)
    }
  }

  useEffect(() => {
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  if (isLanding) {
    return (
      <nav
        className={cn(
          'fixed top-16 left-0 right-0 z-40 bg-[var(--surface)] border-b border-[var(--border)]',
          'transition-transform duration-200'
        )}
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-6">
              {navItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    cn(
                      'flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                      'hover:bg-[var(--raised)] hover:text-[var(--fg)]',
                      isActive
                        ? 'bg-[var(--primary)]/10 text-[var(--primary)]'
                        : 'text-[var(--fg-muted)]'
                    )
                  }
                >
                  <item.icon className="h-4 w-4" aria-hidden="true" />
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <LandingBackendStatus isConnected={isConnected} health={health} activeProfile={activeProfile} />
              <MobileMenuButton onClick={() => setMobileOpen(true)} />
            </div>
          </div>
        </div>
      </nav>
    )
  }

  return (
    <>
      <button
        className="fixed bottom-6 left-6 z-50 lg:hidden p-2 rounded-full bg-[var(--surface)] border border-[var(--border)] shadow-elevated"
        onClick={() => setMobileOpen(true)}
        aria-label="Open navigation menu"
      >
        <Menu className="h-5 w-5" />
      </button>

      <aside
        className={cn(
          'fixed top-16 left-0 bottom-0 z-40 bg-[var(--surface)] border-r border-[var(--border)] transition-all duration-200',
          'lg:translate-x-0',
          mobileOpen ? 'translate-x-0 w-64' : '-translate-x-full',
          collapsed ? 'w-16' : 'w-64'
        )}
        role="navigation"
        aria-label="Main navigation"
        aria-expanded={mobileOpen}
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between h-14 px-4 border-b border-[var(--border)]">
            {!collapsed && (
              <span className="font-semibold text-[var(--fg)] truncate">DPAI Platform</span>
            )}
            <Tooltip content={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} side="right">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCollapsed(!collapsed)}
                aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                className={cn('h-8 w-8', collapsed && 'ml-auto')}
              >
                {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
              </Button>
            </Tooltip>
          </div>

          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto" aria-label="Main navigation">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path
              return (
                <Tooltip key={item.path} content={collapsed ? item.label : ''} side="right" disabled={!collapsed}>
                  <NavLink
                    to={item.path}
                    className={({ isActive: active }) =>
                      cn(
                        'flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors',
                        'hover:bg-[var(--raised)] hover:text-[var(--fg)]',
                        active
                          ? 'bg-[var(--primary)]/10 text-[var(--primary)]'
                          : 'text-[var(--fg-muted)]',
                        collapsed && 'justify-center px-2'
                      )
                    }
                    onClick={() => setMobileOpen(false)}
                    aria-current={isActive ? 'page' : undefined}
                  >
                    <item.icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                  </NavLink>
                </Tooltip>
              )
            })}
          </nav>

          <div className="p-4 border-t border-[var(--border)]">
            {!collapsed && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-[var(--raised)]">
                  <div className="flex items-center gap-1.5">
                    <span className={cn('w-2 h-2 rounded-full', isConnected ? 'bg-green-500' : 'bg-[var(--danger)]')} />
                    <span className="text-xs text-[var(--fg-muted)]">
                      {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                  </div>
                  {health && (
                    <span className="ml-auto text-xs font-mono text-[var(--fg-muted)]">
                      {health.device.toUpperCase()}
                    </span>
                  )}
                </div>
                {activeProfile && (
                  <div className="text-xs text-[var(--fg-muted)] truncate" title={activeProfile.name}>
                    {activeProfile.name}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </aside>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}
    </>
  )
}

function LandingBackendStatus({ isConnected, health, activeProfile }: { isConnected: boolean; health: { device: string } | null; activeProfile: { name: string } | null }) {
  return (
    <Tooltip content={activeProfile?.name || 'No backend configured'} side="bottom">
      <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-[var(--raised)] border border-[var(--border)]">
        <span className={cn('w-2 h-2 rounded-full', isConnected ? 'bg-green-500' : 'bg-[var(--danger)]')} />
        {health && (
          <span className="text-xs font-mono text-[var(--fg-muted)]">
            {health.device.toUpperCase()}
          </span>
        )}
      </div>
    </Tooltip>
  )
}

function MobileMenuButton({ onClick }: { onClick: () => void }) {
  return (
    <Button variant="ghost" size="icon" onClick={onClick} aria-label="Open menu" className="lg:hidden">
      <Menu className="h-5 w-5" />
    </Button>
  )
}