import { useEffect, useRef, type MouseEvent } from 'react'
import { ChevronDown, LayoutGrid } from 'lucide-react'
import { formatRoute, VIEWS, type StudioView } from './studioRoute'
import { navigate } from './useStudioRoute'

/**
 * Page navigation, rendered once by the Studio shell. These are links, not ARIA
 * tabs: each one is its own address, and they have no tab panels to control.
 */
export function StudioNav({ view }: { view: StudioView }) {
  const menu = useRef<HTMLDetailsElement>(null)
  const primary: StudioView[] = ['switchboard', 'jobs', 'assignments', 'projects']
  const advanced = VIEWS.filter(item => !primary.includes(item.id))
  const activeAdvanced = advanced.find(item => item.id === view)
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (menu.current && !menu.current.contains(event.target as Node)) menu.current.open = false
    }
    document.addEventListener('pointerdown', close)
    return () => document.removeEventListener('pointerdown', close)
  }, [])
  useEffect(() => { if (menu.current) menu.current.open = false }, [view])
  function open(event: MouseEvent<HTMLAnchorElement>, id: StudioView) {
    // Keep new-tab and new-window clicks native; a plain click re-enters the
    // view even when it is already open, which resets any arrival filter.
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    if (menu.current) {
      if (menu.current.contains(event.currentTarget)) menu.current.querySelector('summary')?.focus()
      menu.current.open = false
    }
    navigate({ view: id })
  }
  return (
    <nav aria-label="Studio" className="flex w-full min-w-0 items-center gap-1">
      <div className="grid min-w-0 flex-1 grid-cols-4 gap-1 sm:flex">
      {primary.map(id => <a key={id} href={formatRoute({ view: id })} aria-current={view === id ? 'page' : undefined} onClick={event => open(event, id)} className={`flex min-h-11 items-center justify-center rounded text-sm sm:px-5 focus-visible:outline-2 focus-visible:outline-emerald-300 ${view === id ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'}`}>
        {VIEWS.find(item => item.id === id)?.label}
      </a>)}
      </div>
      <details ref={menu} className="relative shrink-0" onKeyDown={event => {
        if (event.key === 'Escape' && menu.current?.open) {
          menu.current.open = false
          menu.current.querySelector('summary')?.focus()
        }
      }}>
        <summary aria-label="Studio tools" title={activeAdvanced?.label || 'Studio tools'} className={`flex min-h-11 min-w-11 cursor-pointer list-none items-center justify-center gap-2 rounded px-2 text-sm focus-visible:outline-2 focus-visible:outline-emerald-300 [&::-webkit-details-marker]:hidden ${activeAdvanced ? 'bg-neutral-800 text-white' : 'text-neutral-300'}`}><LayoutGrid size={18} aria-hidden="true" /><span className="hidden sm:inline">{activeAdvanced?.label || 'Studio'}</span><ChevronDown size={14} className="hidden sm:block" aria-hidden="true" /></summary>
        <div className="absolute right-0 top-full z-50 mt-1 w-52 max-w-[80vw] rounded border border-neutral-700 bg-neutral-950 p-1 shadow-lg">
          {advanced.map(item => <a key={item.id} href={formatRoute({ view: item.id })} onClick={event => open(event, item.id)} aria-current={view === item.id ? 'page' : undefined} className={`flex min-h-11 items-center px-3 text-sm focus-visible:outline-2 focus-visible:outline-cyan-300 ${view === item.id ? 'bg-neutral-800 text-white' : 'text-neutral-300 hover:bg-neutral-900'}`}>{item.label}</a>)}
        </div>
      </details>
    </nav>
  )
}
