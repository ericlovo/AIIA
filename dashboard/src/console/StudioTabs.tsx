export type StudioView = 'activity' | 'agents' | 'assignments' | 'handoffs' | 'world'

const VIEWS: { id: StudioView; label: string }[] = [
  { id: 'activity', label: 'Overview' },
  { id: 'agents', label: 'Agents' },
  { id: 'assignments', label: 'Assignments' },
  { id: 'handoffs', label: 'Handoffs' },
  { id: 'world', label: 'Map' },
]

export function StudioTabs({ view, onChange }: { view: StudioView; onChange: (view: StudioView) => void }) {
  return (
    <div role="tablist" aria-label="Agent Studio views" className="flex h-9 max-w-full shrink-0 overflow-x-auto border border-neutral-800 bg-neutral-900/70 p-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      {VIEWS.map(item => (
        <button
          key={item.id}
          role="tab"
          aria-selected={view === item.id}
          onClick={() => onChange(item.id)}
          className={`min-w-[68px] shrink-0 px-2 text-[11px] transition-colors sm:min-w-20 sm:px-3 sm:text-xs ${view === item.id ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-200'}`}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}
