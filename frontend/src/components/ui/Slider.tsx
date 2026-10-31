import { forwardRef, InputHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export interface SliderProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  min?: number
  max?: number
  step?: number
  value: number
  onValueChange: (value: number) => void
}

export const Slider = forwardRef<HTMLInputElement, SliderProps>(
  ({ className, label, min = 0, max = 100, step = 1, value, onValueChange, id, ...props }, ref) => {
    const sliderId = id || label?.toLowerCase().replace(/\s+/g, '-')
    const percentage = ((value - min) / (max - min)) * 100

    return (
      <div className="w-full">
        {label && (
          <label htmlFor={sliderId} className="label">
            {label}
          </label>
        )}
        <div className="relative">
          <input
            ref={ref}
            type="range"
            id={sliderId}
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={e => onValueChange(Number(e.target.value))}
            className={cn(
              'w-full h-2 bg-[var(--raised)] rounded-lg appearance-none cursor-pointer',
              'focus:outline-none focus:ring-2 focus:ring-[var(--ring)] focus:ring-offset-2',
              className
            )}
            style={{
              background: `linear-gradient(to right, hsl(var(--primary)) ${percentage}%, hsl(var(--raised)) ${percentage}%)`,
            }}
            {...props}
          />
        </div>
      </div>
    )
  }
)

Slider.displayName = 'Slider'