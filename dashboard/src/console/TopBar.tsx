import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export function TopBar() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 15_000,
  })
  const { data: monitor } = useQuery({
    queryKey: ['monitor'],
    queryFn: api.monitor,
    refetchInterval: 15_000,
  })
  const { data: autonomy } = useQuery({
    queryKey: ['autonomyStatus'],
    queryFn: () => fetch('/api/autonomy/status').then(r => r.ok ? r.json() : null).catch(() => null),
    refetchInterval: 30_000,
  })

  const services = monitor?.services ? Object.values(monitor.services) : []
  const online = services.filter(s => s.status === 'online').length
  const allOnline = online === services.length && services.length > 0
  // Tri-state for status so we don't lie by defaulting to "down" during load
  const brainState = !health ? 'loading' : health.aiia?.status === 'online' ? 'up' : 'down'
  const ollamaState = !health ? 'loading' : health.ollama?.status === 'online' ? 'up' : 'down'

  const phase = (autonomy as { level?: string } | null)?.level ?? 'phase1'
  const phase2 = phase === 'phase2'

  return (
    <header className="h-14 shrink-0 border-b border-neutral-900 flex items-center justify-between px-3 sm:px-6 bg-neutral-950">
      <div className="flex min-w-0 items-center gap-3 sm:gap-8">
        {/* Brand */}
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-bold text-purple-400 tracking-[0.3em]">AIIA</span>
          <span className="text-[10px] text-neutral-700">console</span>
        </div>

        {/* Status pills */}
        <div className="hidden items-center gap-4 text-xs sm:flex">
          <Pill
            dot={brainState === 'up' ? 'green' : brainState === 'loading' ? 'gray' : 'red'}
            label={brainState === 'loading' ? 'checking brain…' : brainState === 'up' ? 'brain online' : 'brain down'}
          />
          <Pill
            dot={ollamaState === 'up' ? 'green' : ollamaState === 'loading' ? 'gray' : 'amber'}
            label={ollamaState === 'loading' ? 'checking ollama…' : ollamaState === 'up' ? 'ollama ready' : 'ollama warming'}
          />
          <Pill
            dot={!monitor ? 'gray' : allOnline ? 'green' : 'amber'}
            label={!monitor ? 'checking services…' : `${online}/${services.length} services`}
          />
          <Pill
            dot={phase2 ? 'purple' : 'gray'}
            label={phase2 ? 'phase2 active' : 'phase1'}
          />
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs text-neutral-500">
        <div className="w-6 h-6 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-300 font-medium">
          A
        </div>
        <span className="hidden sm:inline">shared workspace</span>
      </div>
    </header>
  )
}

function Pill({ dot, label }: { dot: 'green' | 'amber' | 'red' | 'purple' | 'gray'; label: string }) {
  const dotColor = {
    green: 'bg-green-500',
    amber: 'bg-amber-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
    gray: 'bg-neutral-700',
  }[dot]
  return (
    <div className="flex items-center gap-1.5 text-neutral-500">
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      <span>{label}</span>
    </div>
  )
}
