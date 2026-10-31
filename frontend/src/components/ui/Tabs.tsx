import { ReactNode, createContext, useContext, useState, useCallback } from 'react'
import { cn } from '@/lib/utils'

interface TabsContextType {
  value: string
  onValueChange: (value: string | ((prev: string) => string)) => void
}

const TabsContext = createContext<TabsContextType | null>(null)

function useTabsContext() {
  const context = useContext(TabsContext)
  if (!context) {
    throw new Error('Tabs components must be used within Tabs')
  }
  return context
}

interface TabsProps {
  value?: string
  defaultValue: string
  onValueChange?: (value: string | ((prev: string) => string)) => void
  children: ReactNode
  className?: string
}

export function Tabs({ value, defaultValue, onValueChange, children, className }: TabsProps) {
  const isControlled = value !== undefined
  const [uncontrolledValue, setUncontrolledValue] = useState(defaultValue)
  const currentValue = isControlled ? value : uncontrolledValue
  
  const handleValueChange = useCallback((newValue: string | ((prev: string) => string)) => {
    if (!isControlled) {
      setUncontrolledValue(newValue)
    }
    onValueChange?.(newValue)
  }, [onValueChange, isControlled])

  return (
    <TabsContext.Provider value={{ value: currentValue, onValueChange: handleValueChange }}>
      <div className={cn('data-[orientation=horizontal]:flex', className)}>
        {children}
      </div>
    </TabsContext.Provider>
  )
}

interface TabsListProps {
  children: ReactNode
  className?: string
}

export function TabsList({ children, className }: TabsListProps) {
  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className={cn(
        'inline-flex items-center justify-center gap-1 bg-[var(--raised)] p-1 rounded-md',
        className
      )}
    >
      {children}
    </div>
  )
}

interface TabsTriggerProps {
  value: string
  children: ReactNode
  disabled?: boolean
  className?: string
}

export function TabsTrigger({ value, children, disabled, className, ...props }: TabsTriggerProps) {
  const { value: currentValue, onValueChange } = useTabsContext()
  const isActive = currentValue === value
  
  return (
    <button
      role="tab"
      aria-selected={isActive}
      aria-disabled={disabled}
      disabled={disabled}
      type="button"
      className={cn(
        'px-4 py-2 text-sm font-medium rounded-md transition-all duration-140',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]',
        'disabled:opacity-50 disabled:pointer-events-none',
        'data-[state=active]:bg-[var(--surface)] data-[state=active]:shadow-card data-[state=active]:text-[var(--fg)]',
        'data-[state=inactive]:text-[var(--fg-muted)] hover:data-[state=inactive]:text-[var(--fg)]',
        className
      )}
      data-state={isActive ? 'active' : 'inactive'}
      data-value={value}
      onClick={() => !disabled && onValueChange(value)}
      {...props}
    >
      {children}
    </button>
  )
}

interface TabsContentProps {
  value: string
  children: ReactNode
  className?: string
  forceMount?: boolean
}

export function TabsContent({ value, children, className, forceMount }: TabsContentProps) {
  const { value: currentValue } = useTabsContext()
  const isActive = currentValue === value
  
  if (!isActive && !forceMount) return null

  return (
    <div
      role="tabpanel"
      className={cn('mt-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]', className)}
      data-value={value}
      data-state={isActive ? 'active' : 'inactive'}
    >
      {children}
    </div>
  )
}