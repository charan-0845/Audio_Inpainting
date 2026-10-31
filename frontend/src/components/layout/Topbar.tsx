import { ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { Tooltip } from '@/components/ui/Tooltip'
import { Badge } from '@/components/ui/Badge'
import { useBackend } from '@/features/backend-connection/BackendContext'
import { useTheme } from '@/features/backend-connection/ThemeContext'
import { Sun, Moon, Monitor, Bell, Zap, BookOpen, Code, GitBranch, Home } from 'lucide-react'

interface TopbarProps {
  title?: string
  subtitle?: string
  children?: ReactNode
  showBackLink?: boolean
  backHref?: string
}

export function Topbar({ title, subtitle, children, showBackLink = false, backHref = '/' }: TopbarProps) {
  const location = useLocation()
  const { theme, resolvedTheme, setTheme } = useTheme()
  const { isConnected, health, activeProfile, backendUrl } = useBackend()
  const isLanding = location.pathname === '/'

  return (
    <header
      className={cn(
        'sticky top-0 z-30 bg-[var(--surface)]/80 backdrop-blur-sm border-b border-[var(--border)]',
        'transition-colors duration-140'
      )}
      role="banner"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          <div className="flex items-center gap-4 flex-1 min-w-0">
            {showBackLink && (
              <Link
                to={backHref}
                className="p-2 rounded-md text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--raised)] transition-colors lg:hidden"
                aria-label="Go back"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </Link>
            )}

            <div className="min-w-0">
              {title && (
                <h1 className="text-lg font-semibold text-[var(--fg)] truncate">{title}</h1>
              )}
              {subtitle && (
                <p className="text-sm text-[var(--fg-muted)] truncate">{subtitle}</p>
              )}
            </div>

            {!isLanding && (
              <nav className="hidden lg:flex items-center gap-1" aria-label="Platform navigation">
                {[
                  { path: '/', label: 'Home', icon: Home },
                  { path: '/research', label: 'About Research', icon: BookOpen },
                  { path: '/research/methodology', label: 'Methodology', icon: GitBranch },
                  { path: '/documentation', label: 'Documentation', icon: Code },
                ].map(item => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      cn(
                        'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors',
                        'hover:bg-[var(--raised)] hover:text-[var(--fg)]',
                        isActive
                          ? 'bg-[var(--primary)]/10 text-[var(--primary)]'
                          : 'text-[var(--fg-muted)]'
                      )
                    }
                  >
                    <item.icon className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </nav>
            )}
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            {isLanding && (
              <Link to="/dashboard" className="hidden sm:inline-flex">
                <Button variant="primary" size="sm">
                  <Zap className="h-4 w-4 mr-1" />
                  Launch Workspace
                </Button>
              </Link>
            )}

            {!isLanding && (
              <>
                <Tooltip content={resolvedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'} side="bottom">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
                    aria-label={resolvedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                  >
                    {resolvedTheme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                  </Button>
                </Tooltip>

                <Tooltip content={`Backend: ${activeProfile?.name || backendUrl}`} side="bottom">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[var(--raised)] border border-[var(--border)]">
                    <span className={cn('w-2 h-2 rounded-full', isConnected ? 'bg-green-500' : 'bg-[var(--danger)]')} />
                    <span className="text-xs font-mono text-[var(--fg-muted)]">
                      {health?.device?.toUpperCase() || 'CPU'}
                    </span>
                  </div>
                </Tooltip>
              </>
            )}

            {children}
          </div>
        </div>
      </div>

      {!isLanding && (
        <RunningJobIndicator />
      )}
    </header>
  )
}

function RunningJobIndicator() {
  // This would connect to a job context to show active job status
  // For now, return null - will be implemented when job context exists
  return null
}

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 border-b border-[var(--border)] bg-[var(--surface)]/50">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[var(--fg)]">{title}</h1>
          {description && <p className="mt-1 text-[var(--fg-muted)]">{description}</p>}
        </div>
        {action && <div className="flex-shrink-0">{action}</div>}
      </div>
    </div>
  )
}