import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { loopState, DOT_COLOR, type LoopState } from './taskStatus'

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
  const tasks = useMemo(() => data ?? [], [data])
  const [hovered, setHovered] = useState<string | null>(null)

  const counts = useMemo(() => {
    const c: Record<LoopState, number> = { healthy: 0, stale: 0, failing: 0, idle: 0, running: 0 }
    tasks.forEach(t => { c[loopState(t)] += 1 })
    return c
  }, [tasks])

  const hoveredTask = tasks.find(t => t.task_id === hovered)

  return (
    <footer className="h-14 shrink-0 border-t border-neutral-900 bg-neutral-950 flex items-center px-3 sm:px-6 gap-4 relative">
      <div className="flex items-center gap-3 text-xs shrink-0">
        <span className="text-[10px] tracking-[0.25em] text-neutral-500 font-semibold">PULSE</span>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
          <span className="text-[11px] text-neutral-500">{counts.healthy}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
          <span className="text-[11px] text-neutral-500">{counts.stale}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          <span className="text-[11px] text-neutral-500">{counts.failing}</span>
        </div>
      </div>

      {/* Loop dots */}
      <div className="flex-1 flex items-center gap-1.5 overflow-x-auto">
        {tasks.map(t => {
          const state = loopState(t)
          return (
            <button
              key={t.task_id}
              aria-label={`Open ${t.name}: ${state}`}
              title={`${t.name}: ${state}`}
              onClick={() => window.dispatchEvent(new CustomEvent('studio:switchboard', { detail: { taskId: t.task_id } }))}
              onFocus={() => setHovered(t.task_id)}
              onBlur={() => setHovered(null)}
              onMouseEnter={() => setHovered(t.task_id)}
              onMouseLeave={() => setHovered(null)}
              className="relative group flex shrink-0 flex-col items-center gap-1 cursor-pointer px-1"
            >
              <span
                className={`w-2.5 h-2.5 rounded-full ${DOT_COLOR[state]} ${state === 'healthy' ? '' : state === 'failing' ? 'animate-pulse' : ''} ring-1 ring-black/40 transition-transform group-hover:scale-125`}
              />
              <span className="text-[9px] text-neutral-700 group-hover:text-neutral-500 tracking-tight max-w-[80px] truncate">
                {t.task_id.replace(/_/g, ' ')}
              </span>
            </button>
          )
        })}
      </div>

      {/* Hover tooltip */}
      {hoveredTask && (
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 bg-neutral-900 border border-neutral-800 rounded-lg px-4 py-3 text-xs shadow-2xl z-20 max-w-[480px]">
          <div className="flex items-baseline gap-3 mb-1">
            <span className={`w-1.5 h-1.5 rounded-full ${DOT_COLOR[loopState(hoveredTask)]}`} />
            <span className="text-neutral-300 font-medium">{hoveredTask.name}</span>
            <span className="text-[10px] text-neutral-600">{timeAgo(hoveredTask.last_run)}</span>
          </div>
          <p className="text-[11px] text-neutral-500 mb-1">{hoveredTask.description}</p>
          <p className="text-[11px] text-neutral-500">
            {hoveredTask.run_count} runs · {hoveredTask.fail_count} fails
          </p>
        </div>
      )}
    </footer>
  )
}
