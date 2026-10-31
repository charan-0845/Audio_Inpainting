import { HTMLAttributes, forwardRef } from 'react'
import { cn } from '@/lib/utils'

export interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  value: number
  max?: number
}

export const Progress = forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, max = 100, ...props }, ref) => {
    const percentage = Math.min(100, Math.max(0, (value / max) * 100))

    return (
      <div
        ref={ref}
        className={cn(
          'relative h-2 w-full overflow-hidden rounded-full bg-[var(--raised)]',
          className
        )}
        {...props}
      >
        <div
          className="h-full bg-[var(--primary)] transition-all duration-300 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
    )
  }
)

Progress.displayName = 'Progress'