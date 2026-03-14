import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export function Topbar({ onBrainToggle, brainOpen }: { onBrainToggle: () => void; brainOpen: boolean }) {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000 })

  const aiiaStatus = health?.aiia?.status ?? 'unknown'
  const ollamaStatus = health?.ollama?.status ?? 'unknown'
  const allOnline = aiiaStatus === 'online' && ollamaStatus === 'online'

  return (
    <header className="h-12 bg-[#111] border-b border-[#222] flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-3">
        <span className="text-sm text-[#666]">Command Center</span>
        <span className="text-xs bg-[#1a1a1a] text-[#888] px-2 py-0.5 rounded border border-[#2a2a2a]">local</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-xs">
          <span className={`w-2 h-2 rounded-full ${allOnline ? 'bg-green-500' : 'bg-amber-500'}`} />
          <span className="text-[#888]">Brain {allOnline ? 'online' : 'degraded'}</span>
        </div>
        <button
          onClick={onBrainToggle}
          className={`text-xs px-3 py-1.5 rounded-md transition-colors cursor-pointer ${
            brainOpen
              ? 'bg-purple-600 text-white'
              : 'bg-[#1a1a1a] text-[#888] border border-[#2a2a2a] hover:border-purple-500/50 hover:text-purple-400'
          }`}
        >
          Brain
        </button>
      </div>
    </header>
  )
}
