import type { Page } from '../App'

const NAV: { id: Page; label: string; icon: string }[] = [
  { id: 'today', label: 'Today', icon: '◉' },
  { id: 'stories', label: 'Stories', icon: '▦' },
  { id: 'ops', label: 'Ops', icon: '⚙' },
]

export function Sidebar({ page, onNavigate }: { page: Page; onNavigate: (p: Page) => void }) {
  return (
    <nav className="w-16 bg-[#111] border-r border-[#222] flex flex-col items-center pt-4 gap-1 shrink-0">
      <div className="text-xs font-bold text-purple-400 mb-4 tracking-wider">AIIA</div>
      {NAV.map(n => (
        <button
          key={n.id}
          onClick={() => onNavigate(n.id)}
          className={`w-12 h-12 rounded-lg flex flex-col items-center justify-center gap-0.5 text-xs transition-colors cursor-pointer ${
            page === n.id
              ? 'bg-[#1e1e1e] text-white'
              : 'text-[#666] hover:text-[#999] hover:bg-[#181818]'
          }`}
        >
          <span className="text-base">{n.icon}</span>
          <span>{n.label}</span>
        </button>
      ))}
    </nav>
  )
}
