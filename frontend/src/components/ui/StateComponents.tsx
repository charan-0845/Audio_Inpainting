import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Button } from './Button'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: {
    label: string
    onClick: () => void
    variant?: 'primary' | 'secondary'
  }
  className?: string
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center py-12 px-4',
        className
      )}
    >
      {icon && (
        <div className="mb-4 text-[var(--fg-muted)]" aria-hidden="true">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-medium text-[var(--fg)]">{title}</h3>
      {description && (
        <p className="mt-2 text-sm text-[var(--fg-muted)] max-w-sm">{description}</p>
      )}
      {action && (
        <Button
          variant={action.variant || 'primary'}
          onClick={action.onClick}
          className="mt-4"
        >
          {action.label}
        </Button>
      )}
    </div>
  )
}

interface LoadingStateProps {
  message?: string
  className?: string
}

export function LoadingState({ message, className }: LoadingStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12 px-4 gap-4',
        className
      )}
    >
      <svg className="animate-spin h-8 w-8 text-[var(--primary)]" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
      </svg>
      {message && <p className="text-sm text-[var(--fg-muted)]">{message}</p>}
    </div>
  )
}

interface ErrorStateProps {
  title?: string
  message: string
  onRetry?: () => void
  className?: string
}

export function ErrorState({ title = 'Something went wrong', message, onRetry, className }: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center py-12 px-4 gap-4',
        className
      )}
    >
      <svg className="h-10 w-10 text-[var(--danger)]" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
      <div>
        <h3 className="text-lg font-medium text-[var(--fg)]">{title}</h3>
        <p className="mt-1 text-sm text-[var(--fg-muted)] max-w-sm">{message}</p>
      </div>
      {onRetry && (
        <Button variant="primary" onClick={onRetry} size="sm">
          Try Again
        </Button>
      )}
    </div>
  )
}