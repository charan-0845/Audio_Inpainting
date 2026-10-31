import { forwardRef, SelectHTMLAttributes, ChangeEvent } from 'react'
import { cn } from '@/lib/utils'

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  error?: string
  helperText?: string
  options: { value: string; label: string; disabled?: boolean }[]
  placeholder?: string
  onValueChange?: (value: string) => void
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, error, helperText, options, placeholder, id, onValueChange, onChange, ...props }, ref) => {
    const selectId = id || label?.toLowerCase().replace(/\s+/g, '-')

    const handleChange = (e: ChangeEvent<HTMLSelectElement>) => {
      onValueChange?.(e.target.value)
      onChange?.(e)
    }

    return (
      <div className="w-full">
        {label && (
          <label htmlFor={selectId} className="label">
            {label}
          </label>
        )}
        <select
          ref={ref}
          id={selectId}
          className={cn(
            'w-full rounded-md bg-[var(--bg)] border px-3 py-2 text-sm text-[var(--fg)] transition-colors duration-140',
            'focus:border-[var(--ring)] focus:outline-none focus:ring-1 focus:ring-[var(--ring)]',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            error && 'border-[var(--danger)] focus:border-[var(--danger)] focus:ring-[var(--danger)]',
            !error && 'border-[var(--border)] hover:border-[var(--border-strong)]',
            className
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={error ? `${selectId}-error` : helperText ? `${selectId}-helper` : undefined}
          onChange={handleChange}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((option) => (
            <option key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
            </option>
          ))}
        </select>
        {error && (
          <p id={`${selectId}-error`} className="mt-1 text-sm text-[var(--danger)]" role="alert">
            {error}
          </p>
        )}
        {helperText && !error && (
          <p id={`${selectId}-helper`} className="mt-1 text-sm text-[var(--fg-muted)]">
            {helperText}
          </p>
        )}
      </div>
    )
  }
)

Select.displayName = 'Select'