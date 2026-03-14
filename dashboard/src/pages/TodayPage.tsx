import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { Action, Commit, Story } from '../lib/api'

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime()
  const m = Math.floor(ms / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

const PRIORITY_COLORS: Record<string, string> = {
  P0: 'bg-red-500/20 text-red-400 border-red-500/30',
  P1: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  P2: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  P3: 'bg-[#2a2a2a] text-[#888] border-[#333]',
}

const SEVERITY_COLORS: Record<string, string> = {
  error: 'bg-red-500/20 text-red-400',
  warn: 'bg-amber-500/20 text-amber-400',
  info: 'bg-blue-500/20 text-blue-400',
}

const TYPE_ICONS: Record<string, string> = {
  feat: '+',
  fix: '~',
  refactor: '↻',
  chore: '·',
  docs: '◔',
}

export function TodayPage() {
  const { data: checkin, isLoading } = useQuery({ queryKey: ['checkin'], queryFn: api.checkin })
  const { data: work } = useQuery({ queryKey: ['workContext'], queryFn: api.workContext })
  const { data: actionData } = useQuery({ queryKey: ['actions'], queryFn: () => api.actions() })

  if (isLoading) return <div className="text-[#666] text-sm">Loading...</div>

  const wip = checkin?.wip?.[0]
  const activeStories = checkin?.active_stories ?? []
  const blockedStories = checkin?.blocked_stories ?? []
  const commits = work?.today?.products
    ? Object.values(work.today.products).flatMap(p => p.commits)
    : []
  // Dedupe commits by hash
  const uniqueCommits = [...new Map(commits.map(c => [c.hash, c])).values()]
  const pendingActions = (actionData?.actions ?? []).filter((a: Action) => a.status === 'pending').slice(0, 8)
  const summary = work?.today?.summary

  return (
    <div className="max-w-[1400px] mx-auto">
      <div className="flex items-baseline gap-3 mb-6">
        <h1 className="text-xl font-semibold text-white">Today</h1>
        <span className="text-sm text-[#666]">{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}</span>
        {summary && (
          <span className="text-xs text-[#555] ml-auto">
            {summary.total_commits} commits · {summary.total_files_changed} files · +{summary.total_additions} -{summary.total_deletions}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* LEFT: My Focus (3 cols) */}
        <div className="lg:col-span-3 flex flex-col gap-5">
          {/* NOW card */}
          <NowCard wip={wip} activeStories={activeStories} blockedStories={blockedStories} />

          {/* What to work on next */}
          <PriorityList />

          {/* Pending actions */}
          {pendingActions.length > 0 && (
            <ActionsList actions={pendingActions} />
          )}
        </div>

        {/* RIGHT: Context (2 cols) */}
        <div className="lg:col-span-2 flex flex-col gap-5">
          {/* Timeline */}
          <Timeline commits={uniqueCommits} actions={pendingActions} />

          {/* System status */}
          <SystemStatus />
        </div>
      </div>
    </div>
  )
}

function NowCard({ wip, activeStories, blockedStories }: {
  wip?: { fact: string; created_at: string } | null;
  activeStories: Story[];
  blockedStories: Story[];
}) {
  // Parse WIP fact into useful pieces
  const wipLines = wip?.fact?.split(';').filter(Boolean) ?? []
  const wipSummary = wipLines.length > 0 ? wipLines[0].replace('Auto-WIP: ', '') : null

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-purple-400 text-sm font-medium">Now</span>
        {wip && <span className="text-[#555] text-xs">{timeAgo(wip.created_at)}</span>}
      </div>

      {wipSummary ? (
        <p className="text-sm text-[#ccc] mb-3 leading-relaxed">{wipSummary}</p>
      ) : (
        <p className="text-sm text-[#555] mb-3 italic">No active WIP. Start a session to track your work.</p>
      )}

      {activeStories.length > 0 && (
        <div className="mb-3">
          <div className="text-xs text-[#666] mb-1.5 uppercase tracking-wide">Active</div>
          {activeStories.map(s => (
            <StoryRow key={s.id} story={s} />
          ))}
        </div>
      )}

      {blockedStories.length > 0 && (
        <div>
          <div className="text-xs text-red-400/70 mb-1.5 uppercase tracking-wide">Blocked</div>
          {blockedStories.map(s => (
            <StoryRow key={s.id} story={s} />
          ))}
        </div>
      )}
    </div>
  )
}

function StoryRow({ story }: { story: Story }) {
  return (
    <div className="flex items-center gap-2 py-1.5">
      <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${PRIORITY_COLORS[story.priority] ?? PRIORITY_COLORS.P3}`}>
        {story.priority}
      </span>
      <span className="text-sm text-[#ddd] truncate">{story.title}</span>
      <span className="text-xs text-[#555] ml-auto shrink-0">{story.product}</span>
    </div>
  )
}

function PriorityList() {
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['prioritize'],
    queryFn: () => api.prioritize(5),
    enabled: false, // Manual trigger
  })

  const stories = data?.stories ?? []

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-purple-400 text-sm font-medium">What to build next</span>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="text-xs px-2.5 py-1 rounded-md bg-purple-600/20 text-purple-400 border border-purple-500/30 hover:bg-purple-600/30 transition-colors cursor-pointer disabled:opacity-50"
        >
          {isFetching ? 'Scoring...' : isLoading && !data ? 'Score backlog' : 'Re-score'}
        </button>
      </div>

      {stories.length === 0 && !isFetching && (
        <p className="text-sm text-[#555] italic">Click "Score backlog" to rank stories by business impact.</p>
      )}

      {stories.map((s, i) => (
        <div key={s.id} className="flex items-center gap-3 py-2 border-b border-[#1a1a1a] last:border-0">
          <span className="text-[#555] text-xs w-4 text-right">{i + 1}</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${PRIORITY_COLORS[s.suggested_priority ?? s.priority] ?? PRIORITY_COLORS.P3}`}>
            {s.suggested_priority ?? s.priority}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-sm text-[#ddd] truncate">{s.title}</div>
            {s.priority_reasoning && (
              <div className="text-xs text-[#555] truncate mt-0.5">{s.priority_reasoning}</div>
            )}
          </div>
          <div className="text-right shrink-0">
            <div className="text-xs text-[#888] font-mono">{s.composite_score != null ? s.composite_score : s.priority_score ?? '—'}</div>
            <div className="text-[10px] text-[#555]">{s.composite_score != null ? '/100' : '/150'}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

function ActionsList({ actions }: { actions: Action[] }) {
  const qc = useQueryClient()
  const approve = useMutation({
    mutationFn: (id: string) => api.approveAction(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['actions'] }); qc.invalidateQueries({ queryKey: ['checkin'] }) },
  })
  const reject = useMutation({
    mutationFn: (id: string) => api.rejectAction(id, 'rejected from dashboard'),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['actions'] }); qc.invalidateQueries({ queryKey: ['checkin'] }) },
  })

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-amber-400 text-sm font-medium">Actions needing approval</span>
        <span className="text-xs bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded-full">{actions.length}</span>
      </div>
      {actions.map(a => (
        <div key={a.id} className="flex items-start gap-3 py-2.5 border-b border-[#1a1a1a] last:border-0">
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium mt-0.5 ${SEVERITY_COLORS[a.severity] ?? SEVERITY_COLORS.info}`}>
            {a.severity}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-sm text-[#ddd]">{a.title}</div>
            <div className="text-xs text-[#555] mt-0.5">{a.type} · {a.source_task} · {timeAgo(a.created_at)}</div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            <button
              onClick={() => approve.mutate(a.id)}
              className="text-xs px-2 py-1 rounded bg-green-600/20 text-green-400 border border-green-500/30 hover:bg-green-600/30 cursor-pointer"
            >
              Approve
            </button>
            <button
              onClick={() => reject.mutate(a.id)}
              className="text-xs px-2 py-1 rounded bg-[#1a1a1a] text-[#888] border border-[#333] hover:border-red-500/30 hover:text-red-400 cursor-pointer"
            >
              Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

function Timeline({ commits, actions }: { commits: Commit[]; actions: Action[] }) {
  // Merge commits and actions into a single timeline
  type TimelineItem = { type: 'commit'; data: Commit; time: string } | { type: 'action'; data: Action; time: string }
  const items: TimelineItem[] = [
    ...commits.map(c => ({ type: 'commit' as const, data: c, time: new Date().toISOString() })),
    ...actions.map(a => ({ type: 'action' as const, data: a, time: a.created_at })),
  ]

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <span className="text-blue-400 text-sm font-medium block mb-3">Activity</span>
      {items.length === 0 ? (
        <p className="text-sm text-[#555] italic">No activity yet today.</p>
      ) : (
        <div className="space-y-0">
          {items.slice(0, 15).map((item, i) => (
            <div key={i} className="flex gap-3 py-2 border-b border-[#1a1a1a] last:border-0">
              {item.type === 'commit' ? (
                <>
                  <span className="text-green-400 text-xs font-mono w-5 text-center mt-0.5">
                    {TYPE_ICONS[item.data.type] ?? '·'}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-[#ccc] truncate">{item.data.subject}</div>
                    <div className="text-xs text-[#555]">{item.data.product} · {item.data.files.length} files</div>
                  </div>
                </>
              ) : (
                <>
                  <span className={`text-xs w-5 text-center mt-0.5 ${
                    item.data.severity === 'error' ? 'text-red-400' : 'text-amber-400'
                  }`}>!</span>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-[#ccc] truncate">{item.data.title}</div>
                    <div className="text-xs text-[#555]">{item.data.type} · {timeAgo(item.time)}</div>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function SystemStatus() {
  const { data } = useQuery({ queryKey: ['monitor'], queryFn: api.monitor, refetchInterval: 15_000 })
  const services = data?.services ? Object.values(data.services) : []

  const STATUS_DOT: Record<string, string> = {
    online: 'bg-green-500',
    degraded: 'bg-amber-500',
    offline: 'bg-red-500',
  }

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl p-5">
      <span className="text-[#888] text-sm font-medium block mb-3">System</span>
      {services.map(s => (
        <div key={s.id} className="flex items-center gap-2.5 py-2 border-b border-[#1a1a1a] last:border-0">
          <span className={`w-2 h-2 rounded-full shrink-0 ${STATUS_DOT[s.status] ?? STATUS_DOT.offline}`} />
          <span className="text-sm text-[#ccc] flex-1">{s.name}</span>
          <span className="text-xs text-[#555] font-mono">{Math.round(s.response_time_ms)}ms</span>
          <span className="text-xs text-[#555]">{s.uptime_pct.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}
