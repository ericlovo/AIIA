export type StudioView = 'agents' | 'assignments' | 'handoffs'

const VIEWS: { id: StudioView; label: string }[] = [
  { id: 'agents', label: 'Agents' },
  { id: 'assignments', label: 'Assignments' },
  { id: 'handoffs', label: 'Handoffs' },
]

export function StudioTabs({ view, onChange }: { view: StudioView; onChange: (view: StudioView) => void }) {
  return (
    <div role="tablist" aria-label="Agent Studio views" className="flex h-9 shrink-0 border border-neutral-800 bg-neutral-900/70 p-0.5">
      {VIEWS.map(item => (
        <button
          key={item.id}
          role="tab"
          aria-selected={view === item.id}
          onClick={() => onChange(item.id)}
          className={`min-w-20 px-3 text-xs transition-colors ${view === item.id ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-200'}`}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}
