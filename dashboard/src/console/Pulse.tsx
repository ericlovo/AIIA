import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

type LoopState = 'healthy' | 'stale' | 'failing' | 'idle'

function loopState(t: {
  run_count: number
  fail_count: number
  last_status: string | null
  last_run: string | null
  interval_seconds: number
}): LoopState {
  if (t.run_count === 0) return 'idle'
  // Failing if last result was error OR fail_count > 10% of runs
  if (t.last_status === 'error') return 'failing'
  if (t.fail_count > 0 && t.fail_count / Math.max(t.run_count, 1) > 0.15) return 'failing'
  // Stale if it hasn't run in 3x its interval
  if (t.last_run && t.interval_seconds > 0) {
    const ago = (Date.now() - new Date(t.last_run).getTime()) / 1000
    if (ago > t.interval_seconds * 3) return 'stale'
  }
  return 'healthy'
}

const DOT_COLOR: Record<LoopState, string> = {
  healthy: 'bg-green-500',
  stale: 'bg-amber-500',
  failing: 'bg-red-500',
  idle: 'bg-[#333]',
}

function timeAgo(iso: string | null): string {
  if (!iso) return 'never'
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

export function Pulse() {
  const { data } = useQuery({
    queryKey: ['pulse-tasks'],
    queryFn: api.tasks,
    refetchInterval: 5_000,
  })
  const tasks = data ?? []
  const [hovered, setHovered] = useState<string | null>(null)

  const counts = useMemo(() => {
    const c: Record<LoopState, number> = { healthy: 0, stale: 0, failing: 0, idle: 0 }
    tasks.forEach(t => { c[loopState(t)] += 1 })
    return c
  }, [tasks])

  const hoveredTask = tasks.find(t => t.task_id === hovered)

  return (
    <footer className="h-14 shrink-0 border-t border-[#181818] bg-[#0a0a0a] flex items-center px-6 gap-4 relative">
      <div className="flex items-center gap-3 text-xs shrink-0">
        <span className="text-[10px] tracking-[0.25em] text-[#666] font-semibold">PULSE</span>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
          <span className="text-[11px] text-[#888]">{counts.healthy}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
          <span className="text-[11px] text-[#888]">{counts.stale}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          <span className="text-[11px] text-[#888]">{counts.failing}</span>
        </div>
      </div>

      {/* Loop dots */}
      <div className="flex-1 flex items-center gap-1.5 overflow-x-auto">
        {tasks.map(t => {
          const state = loopState(t)
          return (
            <button
              key={t.task_id}
              onMouseEnter={() => setHovered(t.task_id)}
              onMouseLeave={() => setHovered(null)}
              className="relative group flex flex-col items-center gap-1 cursor-pointer"
            >
              <span
                className={`w-2.5 h-2.5 rounded-full ${DOT_COLOR[state]} ${state === 'healthy' ? '' : state === 'failing' ? 'animate-pulse' : ''} ring-1 ring-black/40 transition-transform group-hover:scale-125`}
              />
              <span className="text-[9px] text-[#444] group-hover:text-[#888] tracking-tight max-w-[80px] truncate">
                {t.task_id.replace(/_/g, ' ')}
              </span>
            </button>
          )
        })}
      </div>

      {/* Hover tooltip */}
      {hoveredTask && (
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 bg-[#111] border border-[#2a2a2a] rounded-lg px-4 py-3 text-xs shadow-2xl z-20 max-w-[480px]">
          <div className="flex items-baseline gap-3 mb-1">
            <span className={`w-1.5 h-1.5 rounded-full ${DOT_COLOR[loopState(hoveredTask)]}`} />
            <span className="text-[#ddd] font-medium">{hoveredTask.name}</span>
            <span className="text-[10px] text-[#555]">{timeAgo(hoveredTask.last_run)}</span>
          </div>
          <p className="text-[11px] text-[#888] mb-1">{hoveredTask.description}</p>
          <p className="text-[11px] text-[#666]">
            {hoveredTask.run_count} runs · {hoveredTask.fail_count} fails
          </p>
        </div>
      )}
    </footer>
  )
}
