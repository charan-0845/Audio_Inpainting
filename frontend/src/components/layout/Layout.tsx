import { ReactNode } from 'react'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { Topbar, PageHeader } from './Topbar'
import { cn } from '@/lib/utils'

interface LayoutProps {
  children?: ReactNode
  isLanding?: boolean
}

export function Layout({ children, isLanding = false }: LayoutProps) {
  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Sidebar isLanding={isLanding} />
      <div className={cn('transition-all duration-200', isLanding ? 'lg:pl-0' : 'lg:pl-64')}>
        <Topbar />
        <main className="min-h-[calc(100vh-4rem)]" id="main-content" role="main">
          {children}
          <Outlet />
        </main>
        <footer className="border-t border-[var(--border)] bg-[var(--surface)]/50">
          <div className="max-w-7xl mx-auto px-4 py-4 text-center text-sm text-[var(--fg-muted)]">
            Audio Inpainting Research Platform — Deep Prior-Based Audio Inpainting using Multi-Resolution Harmonic CNNs
          </div>
        </footer>
      </div>
    </div>
  )
}

export function PageLayout({
  title,
  description,
  action,
  children,
}: {
  title: string
  description?: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <PageHeader title={title} description={description} action={action} />
      <div className="p-4 sm:p-6 lg:p-8">{children}</div>
    </div>
  )
}