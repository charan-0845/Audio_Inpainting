import { Fragment, ReactNode, KeyboardEvent } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from './Button'

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: ReactNode
  title?: string
  description?: string
}

export function Dialog({ open, onOpenChange, children, title, description }: DialogProps) {
  if (!open) return null

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') onOpenChange(false)
  }

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onOpenChange(false)
  }

  return (
    <Fragment>
      <div
        className="fixed inset-0 bg-black/50 z-50 animate-fade-in"
        onClick={handleOverlayClick}
        aria-hidden="true"
      />
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onKeyDown={handleKeyDown}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'dialog-title' : undefined}
        aria-describedby={description ? 'dialog-description' : undefined}
      >
        <div
          className={cn(
            'bg-[var(--surface)] border border-[var(--border)] rounded-lg shadow-elevated w-full max-w-lg animate-slide-up',
            'max-h-[90vh] overflow-hidden flex flex-col'
          )}
        >
          {(title || description) && (
            <div className="px-6 py-4 border-b border-[var(--border)] flex items-start justify-between gap-4">
              <div>
                {title && (
                  <h2 id="dialog-title" className="text-lg font-semibold text-[var(--fg)]">
                    {title}
                  </h2>
                )}
                {description && (
                  <p id="dialog-description" className="mt-1 text-sm text-[var(--fg-muted)]">
                    {description}
                  </p>
                )}
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onOpenChange(false)}
                aria-label="Close dialog"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-6">{children}</div>
        </div>
      </div>
    </Fragment>
  )
}

interface DialogTriggerProps {
  children: ReactNode
}

export function DialogTrigger({ children }: DialogTriggerProps) {
  return <>{children}</>
}

export function DialogContent({ children }: { children: ReactNode }) {
  return <>{children}</>
}