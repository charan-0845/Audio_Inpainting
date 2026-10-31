import { forwardRef, InputHTMLAttributes, LabelHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  helperText?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, helperText, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-')

    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="label">
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={cn(
            'w-full rounded-md bg-[var(--bg)] border px-3 py-2 text-sm text-[var(--fg)] placeholder-[var(--fg-muted)] transition-colors duration-140',
            'focus:border-[var(--ring)] focus:outline-none focus:ring-1 focus:ring-[var(--ring)]',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            error && 'border-[var(--danger)] focus:border-[var(--danger)] focus:ring-[var(--danger)]',
            !error && 'border-[var(--border)] hover:border-[var(--border-strong)]',
            className
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={error ? `${inputId}-error` : helperText ? `${inputId}-helper` : undefined}
          {...props}
        />
        {error && (
          <p id={`${inputId}-error`} className="mt-1 text-sm text-[var(--danger)]" role="alert">
            {error}
          </p>
        )}
        {helperText && !error && (
          <p id={`${inputId}-helper`} className="mt-1 text-sm text-[var(--fg-muted)]">
            {helperText}
          </p>
        )}
      </div>
    )
  }
)

Input.displayName = 'Input'

export interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {}

export const Label = forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <label ref={ref} className={cn('label', className)} {...props} />
  )
)
Label.displayName = 'Label'