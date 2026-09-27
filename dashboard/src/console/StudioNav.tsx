import type { MouseEvent } from 'react'
import { formatRoute, VIEWS, type StudioView } from './studioRoute'
import { navigate } from './useStudioRoute'

/**
 * Page navigation, rendered once by the Studio shell. These are links, not ARIA
 * tabs: each one is its own address, and they have no tab panels to control.
 */
export function StudioNav({ view }: { view: StudioView }) {
  function open(event: MouseEvent<HTMLAnchorElement>, id: StudioView) {
    // Keep new-tab and new-window clicks native; a plain click re-enters the
    // view even when it is already open, which resets any arrival filter.
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    navigate({ view: id })
  }
  return (
    <nav aria-label="Studio" className="flex h-9 max-w-full shrink-0 overflow-x-auto border border-neutral-800 bg-neutral-900/70 p-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      {VIEWS.map(item => (
        <a
          key={item.id}
          href={formatRoute({ view: item.id })}
          aria-current={view === item.id ? 'page' : undefined}
          onClick={event => open(event, item.id)}
          className={`flex min-w-[68px] shrink-0 items-center justify-center px-2 text-[11px] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-500 sm:min-w-20 sm:px-3 sm:text-xs ${view === item.id ? 'bg-neutral-700 text-white' : 'text-neutral-400 hover:text-neutral-200'}`}
        >
          {item.label}
        </a>
      ))}
    </nav>
  )
}
