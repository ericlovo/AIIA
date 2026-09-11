export type LoopState = 'healthy' | 'stale' | 'failing' | 'idle' | 'running'

export function loopState(t: { status?: string; run_history?: { status: string }[]; last_status?: string | null; last_run: string | null; interval_seconds?: number }): LoopState {
  if (t.status === 'running') return 'running'
  const last = t.last_status ?? t.run_history?.[0]?.status ?? t.status
  if (last === 'error' || last === 'failed') return 'failing'
  if (!t.last_run) return 'idle'
  if (t.interval_seconds && Date.now() - Date.parse(t.last_run) > t.interval_seconds * 3_000) return 'stale'
  return last === 'done' || last === 'completed' ? 'healthy' : 'idle'
}

export const DOT_COLOR: Record<LoopState, string> = {
  healthy: 'bg-green-500', stale: 'bg-amber-500', failing: 'bg-red-500', idle: 'bg-neutral-700', running: 'bg-cyan-400',
}
