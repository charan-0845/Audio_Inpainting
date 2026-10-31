import { ReactNode, useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'

interface TooltipProps {
  content: ReactNode
  children: ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
  disabled?: boolean
}

export function Tooltip({ content, children, side = 'top', delay = 200, disabled }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const triggerRef = useRef<HTMLDivElement>(null)

  const show = () => {
    if (disabled) return
    timeoutRef.current = setTimeout(() => setIsVisible(true), delay)
  }

  const hide = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    setIsVisible(false)
  }

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [])

  const sideClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  }

  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-[var(--fg)] border-transparent border-x-4',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-[var(--fg)] border-transparent border-x-4',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-[var(--fg)] border-transparent border-y-4',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-[var(--fg)] border-transparent border-y-4',
  }

  return (
    <div
      ref={triggerRef}
      className="inline-block relative"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {isVisible && (
        <div
          role="tooltip"
          className={cn(
            'fixed z-50 px-3 py-1.5 text-xs font-medium text-[var(--bg)] bg-[var(--fg)] rounded-md shadow-elevated',
            'whitespace-nowrap animate-fade-in',
            sideClasses[side]
          )}
        >
          {content}
          <div
            className={cn(
              'absolute w-0 h-0 border-4',
              arrowClasses[side]
            )}
          />
        </div>
      )}
    </div>
  )
}