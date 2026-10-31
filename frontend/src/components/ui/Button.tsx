import { forwardRef, ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg' | 'icon'
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, disabled, children, ...props }, ref) => {
    const baseStyles = 'inline-flex items-center justify-center gap-2 font-medium transition-colors duration-140 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)] focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none'

    const variants = {
      primary: 'bg-[var(--primary)] text-[var(--primary-foreground)] hover:bg-[var(--primary-hover)]',
      secondary: 'bg-[var(--surface)] text-[var(--fg)] border border-[var(--border)] hover:bg-[var(--raised)] hover:border-[var(--border-strong)]',
      danger: 'bg-[var(--danger)] text-white hover:bg-red-700',
      ghost: 'bg-transparent text-[var(--fg)] hover:bg-[var(--raised)]',
    }

    const sizes = {
      sm: 'rounded-md px-3 py-1.5 text-sm',
      md: 'rounded-md px-4 py-2 text-sm',
      lg: 'rounded-lg px-6 py-3 text-base',
      icon: 'rounded-md p-2',
    }

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        )}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'