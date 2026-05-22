import { useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { Action, Story, TaskInfo } from '../lib/api'
import type { TeamUser } from '../lib/chatStore'

// Loops whose output is genuinely "AIIA is watching/working" activity.
// The scheduled dev loops (test_runner, health_journal) are infrastructure —
// they belong on the bottom pulse strip, not in the live activity feed.
const LIVE_LOOP_IDS = new Set([
  'proactive_story_eval',
  'gated_downgrade_check',
  'self_healing_monitor',
  'memory_quality_loop',
  'ci_monitor',
  'repo_sync',
  'security_scan',
  'daily_brief',
])

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return ''
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

// Skip noise: CI failures on ephemeral branches
const NOISE_BRANCH_RE = /\bon (claude|dependabot|fix|feat|security|chore)\//i

// Author convention — tags like "from:paul" or "@paul" mean Paul wrote it.
// "@paul" tag also means Paul is the current owner (added when someone picks up).
function parseAuthor(story: Story): string | null {
  const tags = story.tags ?? []
  for (const t of tags) {
    const m = t.match(/^from:(\w+)/i)
    if (m) return m[1].toLowerCase()
  }
  // Fallback: parse from source_type "direct:paul"
  if (story.source_type) {
    const m = story.source_type.match(/^direct:(\w+)/i)
    if (m) return m[1].toLowerCase()
  }
  return null
}

function parseOwner(story: Story): string | null {
  const tags = story.tags ?? []
  for (const t of tags) {
    const m = t.match(/^@(\w+)/)
    if (m) return m[1].toLowerCase()
  }
  return null
}

export function RightNow({ user }: { user: TeamUser }) {
  const me = user.toLowerCase()
  const { data: actions } = useQuery({
    queryKey: ['rn-actions'],
    queryFn: () => api.actions(),
    refetchInterval: 10_000,
  })
  const { data: execution } = useQuery({
    queryKey: ['rn-execution'],
    queryFn: api.executionStatus,
    refetchInterval: 5_000,
  })
  const { data: stories } = useQuery({
    queryKey: ['rn-stories'],
    queryFn: () => api.stories(),
    refetchInterval: 30_000,
  })
  const { data: work } = useQuery({
    queryKey: ['rn-workcontext'],
    queryFn: api.workContext,
    refetchInterval: 30_000,
  })
  const { data: tasks } = useQuery({
    queryKey: ['rn-tasks'],
    queryFn: api.tasks,
    refetchInterval: 10_000,
  })

  // Pending actions worth your attention (dedup + main branch only)
  const pending = useMemo(() => {
    const allPending = (actions?.actions ?? []).filter(a => a.status === 'pending')
    const mainOnly = allPending.filter(a => !NOISE_BRANCH_RE.test(a.title))
    const seen = new Set<string>()
    return mainOnly.filter(a => {
      const key = a.title.replace(/[a-f0-9]{7,}.*$/, '').trim()
      if (seen.has(key)) return false
      seen.add(key)
      return true
    }).slice(0, 6)
  }, [actions])

  // Currently running (execution engine subprocesses)
  const active = execution?.recent?.filter(r => r.status === 'running') ?? []

  // Active / in-progress stories
  const activeStories = (stories?.stories ?? []).filter(
    s => s.status === 'active' || s.status === 'in_progress'
  )

  // Team inbox: unowned stories in backlog written by someone else in the last 3 days.
  // "Unowned" = no @user tag yet. "Someone else" = author is not me.
  const threeDaysAgo = Date.now() - 3 * 24 * 3600 * 1000
  const fromTeam = (stories?.stories ?? [])
    .filter(s => {
      if (s.status !== 'backlog') return false
      const author = parseAuthor(s)
      const owner = parseOwner(s)
      if (!author || author === me) return false
      if (owner) return false
      const created = new Date(s.created_at).getTime()
      return created >= threeDaysAgo
    })
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 5)

  // My stories — things I wrote OR picked up that aren't shipped
  const mine = (stories?.stories ?? [])
    .filter(s => {
      if (['shipped', 'cancelled'].includes(s.status)) return false
      const author = parseAuthor(s)
      const owner = parseOwner(s)
      return author === me || owner === me
    })
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
    .slice(0, 4)

  // Recent shipped — commits today
  const commits = work?.today?.products
    ? Object.values(work.today.products).flatMap(p => p.commits)
    : []
  const uniqueCommits = [...new Map(commits.map(c => [c.hash, c])).values()].slice(0, 5)

  // Recent executions (completed today)
  const recentCompleted = (execution?.recent ?? [])
    .filter(r => r.status !== 'running')
    .slice(0, 4)

  // "Watching" — live feed of recent loop outcomes. Pick the loops that
  // represent real work (not test_runner/code_health) and sort by last_run.
  const watching = useMemo(() => {
    const liveLoops = (tasks ?? [])
      .filter(t => LIVE_LOOP_IDS.has(t.task_id) && t.last_run)
      .sort((a, b) => {
        const ar = a.last_run ? new Date(a.last_run).getTime() : 0
        const br = b.last_run ? new Date(b.last_run).getTime() : 0
        return br - ar
      })
      .slice(0, 5)
    return liveLoops
  }, [tasks])

  return (
    <section className="bg-neutral-950 flex flex-col min-h-0">
      <div className="px-6 pt-5 pb-3 shrink-0 border-b border-neutral-900">
        <h2 className="text-[11px] font-semibold tracking-[0.25em] text-neutral-500 uppercase">Right Now</h2>
        <p className="text-[11px] text-neutral-600">What I'm doing, what I need, what I shipped</p>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-4 space-y-5">
        {/* FROM THE TEAM — stories written by other team members, awaiting pickup */}
        <TeamInboxSection stories={fromTeam} me={me} />

        {/* MY WORK — stories you wrote or picked up */}
        <MyWorkSection stories={mine} />

        {/* IN FLIGHT — what AIIA is actively doing */}
        <LiveSection
          title="In flight"
          count={active.length}
          emptyText="Nothing running right now."
          accent="purple"
        >
          {active.map(r => (
            <LiveCard
              key={r.id}
              icon="▶"
              title={r.input_summary || r.action_type}
              subtitle={`${r.strategy} · started ${timeAgo(r.started_at)}`}
              accent="purple"
              pulse
            />
          ))}
          {activeStories.map(s => {
            const author = parseAuthor(s)
            const parts = [s.product || 'unassigned', s.priority, s.status.replace('_', ' ')]
            if (author) parts.push(`from ${author}`)
            return (
              <LiveCard
                key={s.id}
                icon="◉"
                title={s.title}
                subtitle={parts.join(' · ')}
                accent="blue"
              />
            )
          })}
        </LiveSection>

        {/* NEEDS YOU — decisions blocking progress */}
        <PendingSection actions={pending} />

        {/* WATCHING — ambient autonomous loops doing their thing */}
        <WatchingSection tasks={watching} />

        {/* SHIPPED — recent outcomes */}
        <LiveSection
          title="Recently shipped"
          count={uniqueCommits.length + recentCompleted.filter(r => r.status === 'success').length}
          emptyText="Nothing shipped yet today."
          accent="green"
        >
          {uniqueCommits.map(c => (
            <LiveCard
              key={c.hash}
              icon="✓"
              title={c.subject}
              subtitle={`${c.product ?? 'repo'} · ${c.files.length} files · ${c.hash.slice(0, 7)}`}
              accent="green"
            />
          ))}
          {recentCompleted.filter(r => r.status === 'success').slice(0, 2).map(r => (
            <LiveCard
              key={r.id}
              icon="✓"
              title={r.output_summary || r.input_summary || 'Completed'}
              subtitle={`${r.action_type} · ${r.duration_ms}ms · ${timeAgo(r.started_at)}`}
              accent="green"
            />
          ))}
        </LiveSection>
      </div>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────

function LiveSection({
  title,
  count,
  emptyText,
  accent,
  children,
}: {
  title: string
  count: number
  emptyText: string
  accent: 'purple' | 'green' | 'amber'
  children: React.ReactNode
}) {
  const accentClass = {
    purple: 'text-purple-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
  }[accent]

  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className={`text-xs font-medium ${accentClass}`}>{title}</h3>
        <span className="text-[10px] text-neutral-600">{count}</span>
      </div>
      {count === 0 ? (
        <p className="text-xs text-neutral-700 italic">{emptyText}</p>
      ) : (
        <div className="space-y-1.5">{children}</div>
      )}
    </div>
  )
}

function LiveCard({
  icon,
  title,
  subtitle,
  accent,
  pulse,
}: {
  icon: string
  title: string
  subtitle: string
  accent: 'purple' | 'green' | 'blue' | 'amber'
  pulse?: boolean
}) {
  const accentBg = {
    purple: 'bg-purple-500/10 border-purple-500/20 hover:border-purple-500/40',
    green: 'bg-green-500/5 border-neutral-800 hover:border-green-500/30',
    blue: 'bg-blue-500/5 border-neutral-800 hover:border-blue-500/30',
    amber: 'bg-amber-500/10 border-amber-500/20 hover:border-amber-500/40',
  }[accent]
  const iconColor = {
    purple: 'text-purple-400',
    green: 'text-green-400',
    blue: 'text-blue-400',
    amber: 'text-amber-400',
  }[accent]
  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg border ${accentBg} transition-colors`}>
      <span className={`text-sm ${iconColor} ${pulse ? 'animate-pulse' : ''} mt-0.5 shrink-0`}>
        {icon}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-sm text-neutral-300 truncate">{title}</div>
        <div className="text-[11px] text-neutral-600 truncate mt-0.5">{subtitle}</div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// Team inbox — stories written by other team members, awaiting pickup
// ─────────────────────────────────────────────────────────────

function AuthorBadge({ name }: { name: string }) {
  const initial = name[0].toUpperCase()
  const colors: Record<string, string> = {
    eric: 'bg-purple-500/20 border-purple-500/50 text-purple-300',
    paul: 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300',
    tony: 'bg-amber-500/20 border-amber-500/50 text-amber-300',
  }
  const color = colors[name.toLowerCase()] ?? 'bg-neutral-800 border-neutral-700 text-neutral-400'
  return (
    <div className={`w-5 h-5 rounded-full border flex items-center justify-center text-[10px] font-medium shrink-0 ${color}`} title={name}>
      {initial}
    </div>
  )
}

function TeamInboxSection({ stories, me }: { stories: Story[]; me: string }) {
  const qc = useQueryClient()
  const pickUp = useMutation({
    mutationFn: async (story: Story) => {
      // Add @me tag (keep from: tag for authorship history) and move to active
      const existingTags = (story.tags ?? []).filter(t => !t.startsWith('@'))
      const newTags = [...existingTags, `@${me}`]
      return api.updateStory(story.id, { tags: newTags, status: 'active' })
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rn-stories'] }),
  })

  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className="text-xs font-medium text-cyan-400">From the team</h3>
        <span className="text-[10px] text-neutral-600">{stories.length}</span>
      </div>
      {stories.length === 0 ? (
        <p className="text-xs text-neutral-700 italic">No new requests from the team.</p>
      ) : (
        <div className="space-y-1.5">
          {stories.map(s => {
            const author = parseAuthor(s) ?? 'someone'
            return (
              <div
                key={s.id}
                className="flex items-start gap-3 p-3 rounded-lg bg-cyan-500/5 border border-cyan-500/20 hover:border-cyan-500/40 transition-colors"
              >
                <AuthorBadge name={author} />
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-neutral-300">{s.title}</div>
                  <div className="text-[11px] text-neutral-500 mt-0.5">
                    {author} · {s.priority} · {s.product || 'unassigned'} · {timeAgo(s.created_at)}
                  </div>
                </div>
                <button
                  onClick={() => pickUp.mutate(s)}
                  disabled={pickUp.isPending}
                  className="text-[10px] px-2.5 py-1 rounded bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 cursor-pointer shrink-0 disabled:opacity-50"
                >
                  Pick up
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// My work — things I wrote or picked up
// ─────────────────────────────────────────────────────────────

function MyWorkSection({ stories }: { stories: Story[] }) {
  if (stories.length === 0) return null
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className="text-xs font-medium text-purple-400">My work</h3>
        <span className="text-[10px] text-neutral-600">{stories.length}</span>
      </div>
      <div className="space-y-1.5">
        {stories.map(s => {
          const author = parseAuthor(s)
          const owner = parseOwner(s)
          const statusColor = s.status === 'active' ? 'text-blue-400' : s.status === 'in_progress' ? 'text-purple-400' : s.status === 'blocked' ? 'text-red-400' : 'text-neutral-500'
          return (
            <div
              key={s.id}
              className="flex items-start gap-3 p-3 rounded-lg bg-purple-500/5 border border-purple-500/15 hover:border-purple-500/30 transition-colors"
            >
              {author && <AuthorBadge name={author} />}
              <div className="min-w-0 flex-1">
                <div className="text-sm text-neutral-300">{s.title}</div>
                <div className="text-[11px] text-neutral-500 mt-0.5 flex items-center gap-2">
                  <span className={statusColor}>{s.status.replace('_', ' ')}</span>
                  <span>·</span>
                  <span>{s.priority}</span>
                  {author && owner && author !== owner && (
                    <>
                      <span>·</span>
                      <span className="text-cyan-400">from {author}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function WatchingSection({ tasks }: { tasks: TaskInfo[] }) {
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className="text-xs font-medium text-neutral-500">Watching</h3>
        <span className="text-[10px] text-neutral-600">{tasks.length}</span>
      </div>
      {tasks.length === 0 ? (
        <p className="text-xs text-neutral-700 italic">Loops starting up…</p>
      ) : (
        <div className="space-y-1.5">
          {tasks.map(t => {
            const failed = t.fail_count > 0 && t.fail_count / Math.max(t.run_count, 1) > 0.15
            const result = t.last_result ?? '—'
            const humanName = t.name || t.task_id.replace(/_/g, ' ')
            return (
              <div
                key={t.task_id}
                className="flex items-start gap-3 p-3 rounded-lg bg-neutral-900 border border-neutral-900 hover:border-neutral-800 transition-colors"
              >
                <span className={`w-1.5 h-1.5 mt-1.5 rounded-full shrink-0 ${failed ? 'bg-red-500' : 'bg-neutral-700'}`} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs text-neutral-400">{humanName}</span>
                    <span className="text-[10px] text-neutral-700">{timeAgo(t.last_run)}</span>
                  </div>
                  <div className="text-[11px] text-neutral-500 mt-0.5 truncate">{result}</div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function PendingSection({ actions }: { actions: Action[] }) {
  const qc = useQueryClient()
  const approve = useMutation({
    mutationFn: (id: string) => api.approveAction(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['rn-actions'] })
      qc.invalidateQueries({ queryKey: ['rn-execution'] })
    },
  })
  const reject = useMutation({
    mutationFn: (id: string) => api.rejectAction(id, 'rejected from console'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rn-actions'] }),
  })

  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className="text-xs font-medium text-amber-400">Needs your call</h3>
        <span className="text-[10px] text-neutral-600">{actions.length}</span>
      </div>
      {actions.length === 0 ? (
        <p className="text-xs text-neutral-700 italic">All clear. Nothing waiting on you.</p>
      ) : (
        <div className="space-y-1.5">
          {actions.map(a => (
            <div
              key={a.id}
              className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 hover:border-amber-500/40 transition-colors"
            >
              <div className="flex items-start gap-3">
                <span className="text-amber-400 text-sm mt-0.5 shrink-0">!</span>
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-neutral-300">{a.title}</div>
                  <div className="text-[11px] text-neutral-600 mt-0.5">
                    {a.severity} · {a.source_task} · {timeAgo(a.created_at)}
                  </div>
                </div>
                <div className="flex gap-1 shrink-0">
                  <button
                    onClick={() => approve.mutate(a.id)}
                    className="text-[10px] px-2 py-1 rounded bg-green-500/20 text-green-400 hover:bg-green-500/30 cursor-pointer"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => reject.mutate(a.id)}
                    className="text-[10px] px-2 py-1 rounded text-neutral-500 hover:text-red-400 cursor-pointer"
                  >
                    Reject
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
