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
  const primary: StudioView[] = ['switchboard', 'agents', 'assignments']
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
    <>
    <nav aria-label="Studio" className="grid w-full grid-cols-4 gap-1 lg:hidden">
      {primary.map(id => <a key={id} href={formatRoute({ view: id })} aria-current={view === id ? 'page' : undefined} onClick={event => open(event, id)} className={`flex min-h-11 items-center justify-center rounded text-sm focus-visible:outline-2 focus-visible:outline-cyan-300 ${view === id ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'}`}>
        {id === 'switchboard' ? 'Today' : id === 'agents' ? 'Agents' : 'Work'}
      </a>)}
      <details ref={menu} className="relative" onKeyDown={event => {
        if (event.key === 'Escape' && menu.current?.open) {
          menu.current.open = false
          menu.current.querySelector('summary')?.focus()
        }
      }}>
        <summary aria-label="Studio tools" className={`flex min-h-11 cursor-pointer list-none items-center justify-center gap-1 rounded text-sm focus-visible:outline-2 focus-visible:outline-cyan-300 [&::-webkit-details-marker]:hidden ${activeAdvanced ? 'bg-neutral-800 text-white' : 'text-neutral-400'}`}><LayoutGrid size={14} aria-hidden="true" />Studio<ChevronDown size={12} aria-hidden="true" /></summary>
        <div className="absolute right-0 top-full z-50 mt-1 w-52 max-w-[80vw] rounded border border-neutral-700 bg-neutral-950 p-1 shadow-lg">
          {advanced.map(item => <a key={item.id} href={formatRoute({ view: item.id })} onClick={event => open(event, item.id)} aria-current={view === item.id ? 'page' : undefined} className={`flex min-h-11 items-center px-3 text-sm focus-visible:outline-2 focus-visible:outline-cyan-300 ${view === item.id ? 'bg-neutral-800 text-white' : 'text-neutral-300 hover:bg-neutral-900'}`}>{item.label}</a>)}
        </div>
      </details>
    </nav>
    <nav aria-label="Studio" className="hidden h-9 max-w-full shrink-0 overflow-x-auto border border-neutral-800 bg-neutral-900/70 p-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden lg:flex">
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
    </>
  )
}
