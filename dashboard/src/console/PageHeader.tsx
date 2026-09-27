import type { ReactNode } from 'react'

/** One header shape for every Studio view: a title, a line of context, optional actions. */
export function PageHeader({ title, meta, actions }: { title: ReactNode; meta?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex shrink-0 flex-col gap-3 border-b border-neutral-900 px-5 py-5 sm:flex-row sm:items-end sm:justify-between sm:px-7">
      <div className="min-w-0">
        <h1 className="text-2xl font-medium text-white">{title}</h1>
        {meta && <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-400">{meta}</div>}
      </div>
      {actions}
    </header>
  )
}
