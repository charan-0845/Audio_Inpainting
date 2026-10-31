import { HTMLAttributes, forwardRef, ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface TableProps extends HTMLAttributes<HTMLTableElement> {}

export const Table = forwardRef<HTMLTableElement, TableProps>(
  ({ className, ...props }, ref) => (
    <div className="overflow-x-auto">
      <table
        ref={ref}
        className={cn('w-full caption-bottom text-sm', className)}
        {...props}
      />
    </div>
  )
)
Table.displayName = 'Table'

interface TableHeaderProps extends HTMLAttributes<HTMLTableSectionElement> {}

export const TableHeader = forwardRef<HTMLTableSectionElement, TableHeaderProps>(
  ({ className, ...props }, ref) => (
    <thead ref={ref} className={cn('[&_tr]:border-b', className)} {...props} />
  )
)
TableHeader.displayName = 'TableHeader'

interface TableBodyProps extends HTMLAttributes<HTMLTableSectionElement> {}

export const TableBody = forwardRef<HTMLTableSectionElement, TableBodyProps>(
  ({ className, ...props }, ref) => (
    <tbody ref={ref} className={cn('[&_tr:last-child]:border-0', className)} {...props} />
  )
)
TableBody.displayName = 'TableBody'

interface TableFooterProps extends HTMLAttributes<HTMLTableSectionElement> {}

export const TableFooter = forwardRef<HTMLTableSectionElement, TableFooterProps>(
  ({ className, ...props }, ref) => (
    <tfoot ref={ref} className={cn('border-t bg-[var(--raised)]/50 font-medium [&>tr]:last:border-b-0', className)} {...props} />
  )
)
TableFooter.displayName = 'TableFooter'

interface TableRowProps extends HTMLAttributes<HTMLTableRowElement> {}

export const TableRow = forwardRef<HTMLTableRowElement, TableRowProps>(
  ({ className, ...props }, ref) => (
    <tr
      ref={ref}
      className={cn(
        'border-b border-[var(--border)] transition-colors hover:bg-[var(--raised)]/50 data-[state=selected]:bg-[var(--primary)]/10',
        className
      )}
      {...props}
    />
  )
)
TableRow.displayName = 'TableRow'

interface TableHeadProps extends HTMLAttributes<HTMLTableCellElement> {}

export const TableHead = forwardRef<HTMLTableCellElement, TableHeadProps>(
  ({ className, ...props }, ref) => (
    <th
      ref={ref}
      className={cn(
        'h-12 px-4 text-left align-middle font-medium text-[var(--fg-muted)] uppercase tracking-wider',
        className
      )}
      {...props}
    />
  )
)
TableHead.displayName = 'TableHead'

interface TableCellProps extends HTMLAttributes<HTMLTableCellElement> {}

export const TableCell = forwardRef<HTMLTableCellElement, TableCellProps>(
  ({ className, ...props }, ref) => (
    <td
      ref={ref}
      className={cn('p-4 align-middle text-[var(--fg)]', className)}
      {...props}
    />
  )
)
TableCell.displayName = 'TableCell'

interface TableCaptionProps extends HTMLAttributes<HTMLTableCaptionElement> {}

export const TableCaption = forwardRef<HTMLTableCaptionElement, TableCaptionProps>(
  ({ className, ...props }, ref) => (
    <caption ref={ref} className={cn('mt-4 text-sm text-[var(--fg-muted)]', className)} {...props} />
  )
)
TableCaption.displayName = 'TableCaption'