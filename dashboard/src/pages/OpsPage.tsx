import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'

function timeAgo(iso: string | null): string {
  if (!iso) return 'never'
  const ms = Date.now() - new Date(iso).getTime()
  const m = Math.floor(ms / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

function formatInterval(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`
  return `${Math.round(seconds / 86400)}d`
}

const STATUS_DOT: Record<string, string> = {
  online: 'bg-green-500',
  degraded: 'bg-amber-500',
  offline: 'bg-red-500',
}

export function OpsPage() {
  return (
    <div className="max-w-[1400px] mx-auto">
      <h1 className="text-xl font-semibold text-white mb-6">Operations</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Service Health */}
        <ServiceHealth />

        {/* Execution Engine */}
        <ExecutionPanel />

        {/* Scheduled Jobs */}
        <JobsPanel />

        {/* Token Usage */}
        <TokenPanel />
      </div>
    </div>
  )
}

function ServiceHealth() {
  const { data } = useQuery({ queryKey: ['monitor'], queryFn: api.monitor, refetchInterval: 15_000 })
  const services = data?.services ? Object.values(data.services) : []

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <span className="text-green-400 text-sm font-medium block mb-4">Service Health</span>
      <div className="space-y-0">
        {services.map(s => (
          <div key={s.id} className="flex items-center gap-3 py-3 border-b border-[#1a1a1a] last:border-0">
            <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${STATUS_DOT[s.status] ?? STATUS_DOT.offline}`} />
            <div className="flex-1">
              <div className="text-sm text-[#ddd]">{s.name}</div>
              <div className="text-xs text-[#555]">{s.category} · {s.consecutive_up.toLocaleString()} checks up</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-[#888] font-mono">{Math.round(s.response_time_ms)}ms</div>
              <div className="text-xs text-[#555]">{s.uptime_pct.toFixed(1)}%</div>
            </div>
            {/* Sparkline */}
            <div className="flex items-end gap-px h-4 shrink-0">
              {(s.sparkline ?? []).slice(-20).map((p, i) => (
                <div
                  key={i}
                  className={`w-1 rounded-t ${p.ok ? 'bg-green-500/40' : 'bg-red-500/60'}`}
                  style={{ height: `${Math.max(2, Math.min(16, p.ms / 2))}px` }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ExecutionPanel() {
  const { data } = useQuery({ queryKey: ['execution'], queryFn: api.executionStatus, refetchInterval: 10_000 })

  if (!data) return null

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-purple-400 text-sm font-medium">Execution Engine</span>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${data.is_running ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs text-[#888]">{data.is_running ? 'Running' : 'Stopped'}</span>
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex gap-4 mb-4">
        <div className="text-center">
          <div className="text-lg font-mono text-white">{data.stats.total}</div>
          <div className="text-[10px] text-[#555] uppercase">Total</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono text-green-400">{data.stats.by_status.success ?? 0}</div>
          <div className="text-[10px] text-[#555] uppercase">Success</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono text-red-400">{data.stats.by_status.failed ?? 0}</div>
          <div className="text-[10px] text-[#555] uppercase">Failed</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono text-[#888]">{(data.stats.success_rate * 100).toFixed(0)}%</div>
          <div className="text-[10px] text-[#555] uppercase">Rate</div>
        </div>
      </div>

      {/* Recent executions */}
      <div className="text-xs text-[#666] uppercase tracking-wide mb-2">Recent</div>
      {data.recent.slice(0, 5).map(r => (
        <div key={r.id} className="flex items-center gap-2 py-1.5 border-b border-[#1a1a1a] last:border-0">
          <span className={`w-1.5 h-1.5 rounded-full ${r.status === 'success' ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs text-[#ccc] flex-1 truncate">{r.input_summary || r.action_type}</span>
          <span className="text-[10px] text-[#555] font-mono">{r.duration_ms}ms</span>
          <span className="text-[10px] text-[#555]">{timeAgo(r.started_at)}</span>
        </div>
      ))}
    </div>
  )
}

function JobsPanel() {
  const qc = useQueryClient()
  const { data: tasks } = useQuery({ queryKey: ['tasks'], queryFn: api.tasks })
  const runTask = useMutation({
    mutationFn: (id: string) => api.runTask(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  })

  const jobList = tasks ?? []

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <span className="text-blue-400 text-sm font-medium block mb-4">Scheduled Jobs</span>
      <div className="space-y-0">
        {jobList.map(t => (
          <div key={t.id} className="flex items-center gap-3 py-2.5 border-b border-[#1a1a1a] last:border-0">
            <div className="flex-1 min-w-0">
              <div className="text-sm text-[#ddd]">{t.name}</div>
              <div className="text-xs text-[#555]">
                Every {formatInterval(t.interval_seconds)} · last: {timeAgo(t.last_run)} · runs: {t.run_count}
              </div>
            </div>
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${
              t.last_status === 'success' ? 'bg-green-500/20 text-green-400'
              : t.last_status === 'error' ? 'bg-red-500/20 text-red-400'
              : 'bg-[#2a2a2a] text-[#888]'
            }`}>
              {t.last_status ?? 'pending'}
            </span>
            <button
              onClick={() => runTask.mutate(t.id)}
              className="text-[10px] px-2 py-1 rounded bg-[#1a1a1a] text-[#888] border border-[#2a2a2a] hover:border-blue-500/30 hover:text-blue-400 cursor-pointer"
            >
              Run
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function TokenPanel() {
  const { data } = useQuery({ queryKey: ['tokens'], queryFn: api.tokensToday })

  if (!data) return null

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <span className="text-amber-400 text-sm font-medium block mb-4">Token Usage</span>
      <pre className="text-xs text-[#888] font-mono whitespace-pre-wrap overflow-auto max-h-[300px]">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  )
}
