import { useEffect, useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  type Agent,
  type AgentRun,
  type AssignmentStatus,
  type GitWorkspaceStatus,
  type GitWriteStatus,
} from '../lib/api'
import { StudioTabs, type StudioView } from './StudioTabs'

type ActivityFilter = 'all' | 'runs' | 'work' | 'git'
type ActivityKind = 'run' | 'assignment' | 'handoff' | 'git'

interface ActivityEvent {
  id: string
  kind: ActivityKind
  status: string
  title: string
  detail: string
  agent: string
  at: string
  meta: string
}

const FILTERS: { id: ActivityFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'runs', label: 'Runs' },
  { id: 'work', label: 'Assignments' },
  { id: 'git', label: 'Git' },
]

export function ActivityOverview({ agents, isLoading, view, onViewChange }: { agents: Agent[]; isLoading: boolean; view: StudioView; onViewChange: (view: StudioView) => void }) {
  const queryClient = useQueryClient()
  const [filter, setFilter] = useState<ActivityFilter>('all')
  const { data: assignmentData } = useQuery({ queryKey: ['assignments'], queryFn: api.assignments, refetchInterval: 5_000 })
  const { data: handoffData } = useQuery({ queryKey: ['handoffs'], queryFn: api.handoffs, refetchInterval: 5_000 })
  const { data: workspaceData } = useQuery({ queryKey: ['git-workspaces'], queryFn: api.gitWorkspaces, refetchInterval: 5_000 })
  const { data: writeData } = useQuery({ queryKey: ['git-writes'], queryFn: () => api.gitWrites(), refetchInterval: 5_000 })

  const assignments = useMemo(() => assignmentData?.assignments ?? [], [assignmentData])
  const handoffs = useMemo(() => handoffData?.handoffs ?? [], [handoffData])
  const workspaces = useMemo(() => workspaceData?.workspaces ?? [], [workspaceData])
  const writes = useMemo(() => writeData?.writes ?? [], [writeData])
  const agentNames = useMemo(() => new Map(agents.map(agent => [agent.id, agent.name])), [agents])

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws`)
    socket.onmessage = message => {
      try {
        const payload = JSON.parse(message.data) as { type?: string }
        if (payload.type !== 'agent_studio_update') return
        for (const key of ['agents', 'assignments', 'handoffs']) {
          queryClient.invalidateQueries({ queryKey: [key] })
        }
      } catch {
        // Polling remains the fallback for malformed or older server events.
      }
    }
    return () => socket.close()
  }, [queryClient])

  const events = useMemo(() => {
    const activity: ActivityEvent[] = []

    for (const agent of agents) {
      for (const [index, run] of agent.runs.entries()) {
        const trigger = runTrigger(run, agent)
        if (trigger === 'assignment') continue
        activity.push({
          id: `run-${agent.id}-${run.at}-${index}`,
          kind: 'run',
          status: run.error ? 'failed' : 'completed',
          title: `${agent.name} ${run.error ? 'failed' : 'completed'} a ${trigger} run`,
          detail: cleanSnippet(run.error || run.result || run.task),
          agent: agent.name,
          at: run.at,
          meta: [trigger, run.model, formatLatency(run.latency_ms)].filter(Boolean).join(' · '),
        })
      }
    }

    for (const assignment of assignments) {
      activity.push({
        id: `assignment-${assignment.id}`,
        kind: 'assignment',
        status: assignment.status,
        title: assignment.title,
        detail: cleanSnippet(assignment.error || assignment.result || assignment.objective),
        agent: agentNames.get(assignment.agent_id) ?? 'Removed agent',
        at: assignment.completed_at || assignment.started_at || assignment.updated_at,
        meta: `${assignment.priority} priority`,
      })
    }

    for (const handoff of handoffs) {
      const from = agentNames.get(handoff.from_agent_id) ?? 'Removed agent'
      const to = agentNames.get(handoff.to_agent_id) ?? 'Removed agent'
      activity.push({
        id: `handoff-${handoff.id}`,
        kind: 'handoff',
        status: handoff.status,
        title: `${from} to ${to}`,
        detail: cleanSnippet(handoff.instructions || handoff.artifact),
        agent: `${from} → ${to}`,
        at: handoff.updated_at,
        meta: `${handoff.artifact_type} handoff`,
      })
    }

    for (const workspace of workspaces) {
      activity.push({
        id: `workspace-${workspace.id}`,
        kind: 'git',
        status: workspace.status,
        title: workspace.title || workspace.branch || 'Git workspace',
        detail: workspace.error || workspace.git_status || `Branch ${workspace.branch}`,
        agent: agentNames.get(workspace.agent_id) ?? 'Removed agent',
        at: workspace.updated_at,
        meta: 'workspace',
      })
    }

    for (const write of writes) {
      const workspace = workspaces.find(item => item.id === write.workspace_id)
      activity.push({
        id: `write-${write.id}`,
        kind: 'git',
        status: write.status,
        title: write.title || formatToken(write.op),
        detail: write.error || formatWriteResult(write.result),
        agent: workspace ? agentNames.get(workspace.agent_id) ?? 'Removed agent' : 'Workspace',
        at: write.updated_at,
        meta: formatToken(write.op),
      })
    }

    return activity.sort((a, b) => timestamp(b.at) - timestamp(a.at))
  }, [agents, agentNames, assignments, handoffs, workspaces, writes])

  const visibleEvents = events.filter(event => {
    if (filter === 'runs') return event.kind === 'run'
    if (filter === 'work') return event.kind === 'assignment' || event.kind === 'handoff'
    if (filter === 'git') return event.kind === 'git'
    return true
  }).slice(0, 80)

  const runsToday = agents.reduce((total, agent) => total + agent.runs.filter(run => isToday(run.at)).length, 0)
  const completedAssignments = assignments.filter(item => item.status === 'completed').length
  const running = agents.filter(agent => agent.status === 'running').length + assignments.filter(item => item.status === 'running').length
  const pendingApprovals = workspaces.filter(item => item.status === 'pending').length + writes.filter(item => item.status === 'pending').length
  const failures = agents.filter(agent => agent.status === 'error').length + assignments.filter(item => item.status === 'failed').length + handoffs.filter(item => item.status === 'failed').length + workspaces.filter(item => item.status === 'failed').length + writes.filter(item => item.status === 'failed').length
  const attention = pendingApprovals + failures
  const scheduled = agents.filter(agent => agent.loop_enabled).length

  return (
    <main className="min-h-0 flex-1 overflow-y-auto bg-neutral-950">
      <header className="flex flex-col gap-5 border-b border-neutral-900 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-cyan-400">Agent Studio</div>
          <h1 className="mt-2 text-2xl font-medium text-white">Operations overview</h1>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
            <span>{agents.length} agents</span>
            <span>{scheduled} scheduled</span>
            <span>{assignments.length} assignments</span>
            <span>{handoffs.length} handoffs</span>
          </div>
        </div>
        <StudioTabs view={view} onChange={onViewChange} />
      </header>

      <section aria-label="Operations summary" className="grid border-b border-neutral-900 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Runs today" value={runsToday} detail={`${events.filter(event => event.kind === 'run').length} direct retained`} />
        <Metric label="In progress" value={running} detail={`${assignments.filter(item => item.status === 'queued').length} queued`} tone={running ? 'active' : 'neutral'} />
        <Metric label="Assignments done" value={completedAssignments} detail={`${handoffs.filter(item => item.status === 'completed').length} handoffs complete`} />
        <Metric label="Needs attention" value={attention} detail={`${pendingApprovals} approvals · ${failures} failures`} tone={attention ? 'warning' : 'good'} />
      </section>

      <div className="grid min-h-[620px] lg:h-[calc(100vh-294px)] lg:min-h-0 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="min-w-0 border-b border-neutral-900 lg:overflow-y-auto lg:border-r lg:border-b-0">
          <div className="sticky top-0 z-10 flex flex-col gap-3 border-b border-neutral-900 bg-neutral-950/95 px-5 py-4 backdrop-blur sm:flex-row sm:items-center sm:justify-between sm:px-7">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-neutral-500">Activity ledger</div>
              <div className="mt-1 text-xs text-neutral-600">Newest first · refreshes every 5 seconds</div>
            </div>
            <div className="flex h-8 max-w-full overflow-x-auto border border-neutral-800 p-0.5" role="tablist" aria-label="Activity filters">
              {FILTERS.map(item => (
                <button key={item.id} role="tab" aria-selected={filter === item.id} onClick={() => setFilter(item.id)} className={`shrink-0 px-3 text-[11px] transition-colors ${filter === item.id ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-200'}`}>
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {!isLoading && visibleEvents.length === 0 ? (
            <div className="px-5 py-16 text-sm text-neutral-600 sm:px-7">No activity in this stream.</div>
          ) : (
            <div className="divide-y divide-neutral-900">
              {visibleEvents.map(event => <ActivityRow key={event.id} event={event} />)}
            </div>
          )}
        </section>

        <aside className="min-h-0 lg:overflow-y-auto">
          <div className="sticky top-0 z-10 border-b border-neutral-900 bg-neutral-950/95 px-5 py-4 backdrop-blur sm:px-6">
            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-neutral-500">Agent load</div>
            <div className="mt-1 text-xs text-neutral-600">Recent execution and assigned work</div>
          </div>
          <div className="divide-y divide-neutral-900">
            {[...agents].sort(compareAgents).map(agent => (
              <AgentLoadRow key={agent.id} agent={agent} assignments={assignments.filter(item => item.agent_id === agent.id)} />
            ))}
          </div>
        </aside>
      </div>
    </main>
  )
}

function Metric({ label, value, detail, tone = 'neutral' }: { label: string; value: number; detail: string; tone?: 'neutral' | 'active' | 'warning' | 'good' }) {
  const color = tone === 'warning' ? 'text-amber-300' : tone === 'active' ? 'text-cyan-300' : tone === 'good' ? 'text-emerald-300' : 'text-white'
  return (
    <div className="min-h-24 border-b border-neutral-900 px-5 py-4 sm:border-r sm:px-7 xl:border-b-0">
      <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-600">{label}</div>
      <div className={`mt-2 text-2xl font-medium ${color}`}>{value}</div>
      <div className="mt-1 truncate text-xs text-neutral-600">{detail}</div>
    </div>
  )
}

function ActivityRow({ event }: { event: ActivityEvent }) {
  return (
    <article className="grid min-h-24 grid-cols-[40px_minmax(0,1fr)] gap-3 px-5 py-4 sm:grid-cols-[44px_minmax(0,1fr)_110px] sm:px-7">
      <div className={`flex h-9 w-9 items-center justify-center border text-[9px] font-semibold ${kindColor(event.kind)}`}>{kindLabel(event.kind)}</div>
      <div className="min-w-0">
        <div className="flex min-w-0 items-center gap-2">
          <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${statusColor(event.status)}`} />
          <h2 className="truncate text-sm font-medium text-neutral-200">{event.title}</h2>
        </div>
        <p className="mt-1 line-clamp-2 whitespace-pre-line text-xs leading-relaxed text-neutral-500">{event.detail || 'No output recorded.'}</p>
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase tracking-[0.12em] text-neutral-600">
          <span>{event.agent}</span>
          <span>{event.meta}</span>
          <span className="sm:hidden">{relativeTime(event.at)}</span>
        </div>
      </div>
      <div className="hidden text-right sm:block">
        <div className="text-[11px] text-neutral-500">{relativeTime(event.at)}</div>
        <div className="mt-1 text-[10px] uppercase tracking-[0.12em] text-neutral-700">{formatToken(event.status)}</div>
      </div>
    </article>
  )
}

function AgentLoadRow({ agent, assignments }: { agent: Agent; assignments: { status: AssignmentStatus }[] }) {
  const successfulRuns = agent.runs.filter(run => !run.error && run.result).length
  const assigned = assignments.filter(item => item.status === 'queued' || item.status === 'running').length
  return (
    <div className="px-5 py-4 sm:px-6">
      <div className="flex min-w-0 items-center gap-2">
        <span className={`h-2 w-2 shrink-0 rounded-full ${statusColor(agent.status)}`} />
        <div className="min-w-0 text-sm font-medium text-neutral-200">{agent.name}</div>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-2 gap-y-1 pl-4 text-[10px] text-neutral-600">
        <span>{agent.repo_id || 'local'} · {agent.loop_enabled ? `${agent.loop_interval_minutes}m schedule` : 'manual'}</span>
        <span>{agent.runs.length} runs · {successfulRuns} complete · {assigned} assigned</span>
      </div>
      <div className="mt-3 flex h-1 overflow-hidden bg-neutral-900">
        {agent.runs.length > 0 && <div className="bg-emerald-500/70" style={{ width: `${Math.max(4, successfulRuns / agent.runs.length * 100)}%` }} />}
        {agent.runs.some(run => run.error) && <div className="flex-1 bg-red-500/70" />}
      </div>
      <div className="mt-2 text-[10px] text-neutral-700">{agent.last_run_at ? `Last activity ${relativeTime(agent.last_run_at)}` : 'No recorded activity'}</div>
    </div>
  )
}

function runTrigger(run: AgentRun, agent: Agent) {
  if (run.trigger) return run.trigger
  if (run.task.startsWith('Assignment:')) return 'assignment'
  return run.task === agent.loop_task ? 'interval' : 'manual'
}

function compareAgents(a: Agent, b: Agent) {
  const statusWeight = { running: 3, error: 2, idle: 1 }
  const statusDelta = statusWeight[b.status] - statusWeight[a.status]
  if (statusDelta) return statusDelta
  return timestamp(b.last_run_at) - timestamp(a.last_run_at)
}

function timestamp(value: string | null | undefined) {
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? 0 : parsed
}

function isToday(value: string) {
  const date = new Date(value)
  const today = new Date()
  return date.getFullYear() === today.getFullYear() && date.getMonth() === today.getMonth() && date.getDate() === today.getDate()
}

function relativeTime(value: string | null) {
  const time = timestamp(value)
  if (!time) return 'never'
  const seconds = Math.max(0, Math.floor((Date.now() - time) / 1000))
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function formatLatency(latency?: number) {
  if (!latency) return ''
  if (latency < 1000) return `${Math.round(latency)}ms`
  return `${(latency / 1000).toFixed(1)}s`
}

function formatToken(value: string) {
  return value.replaceAll('_', ' ')
}

function formatWriteResult(result: Record<string, unknown>) {
  const entries = Object.entries(result)
  if (!entries.length) return 'Awaiting execution details.'
  return entries.slice(0, 3).map(([key, value]) => `${formatToken(key)}: ${String(value)}`).join(' · ')
}

function cleanSnippet(value: string) {
  return value
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\*\*/g, '')
    .replace(/`/g, '')
    .replace(/^\s*[-*]\s+/gm, '')
    .trim()
}

function kindLabel(kind: ActivityKind) {
  return { run: 'RUN', assignment: 'ASG', handoff: 'HOF', git: 'GIT' }[kind]
}

function kindColor(kind: ActivityKind) {
  return {
    run: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300',
    assignment: 'border-purple-500/40 bg-purple-500/10 text-purple-300',
    handoff: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
    git: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  }[kind]
}

function statusColor(status: string | AssignmentStatus | GitWorkspaceStatus | GitWriteStatus) {
  if (status === 'running' || status === 'preparing') return 'bg-cyan-400'
  if (status === 'failed' || status === 'error' || status === 'rejected') return 'bg-red-500'
  if (status === 'pending' || status === 'queued' || status === 'approved') return 'bg-amber-400'
  return 'bg-emerald-500'
}
