import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Check, CirclePause, FileText, GitBranch, Layers3, Play, RefreshCw, Search, X } from 'lucide-react'
import { api, type Agent, type AgentDefinition, type Assignment, type StudioRun } from '../lib/api'
import type { StudioView } from './studioRoute'
import { PageHeader } from './PageHeader'
import { TokenUsage } from './TokenUsage'
import { AgentTokenUsage } from './AgentTokenUsage'
import { runTokens } from './runTokens'
import { DEVELOPMENT_LOOPS } from './developmentLoops'
import { loopState, DOT_COLOR } from './taskStatus'
import { assignmentOrigin, attentionAssignments, attentionSummary, bulkDismissError, reviewLabel } from './assignmentReview'
import { ChannelChip } from './ChannelChip'
import { filterAgents, hasSlackGap, noReviewedOutput, valueGlance, valueSummary } from './agentValue'
import { activeSources, formatRate, reviewMetrics, reviewSummary, type ReviewMetric } from './reviewHealth'
import './switchboard.css'

interface Props {
  title?: string
  agents: Agent[]
  loading: boolean
  agentError: boolean
  onViewChange: (view: StudioView) => void
  onManageAgent: (id: string) => void
  onAssignAgent: (id: string) => void
  onOpenAssignment: (id: string) => void
  onOpenReview: (bucket: ReviewMetric['bucket'] | '') => void
  onTemplate: (draft: AgentDefinition) => void
  initialTaskId?: string
  initialAttention?: boolean
}

export function Switchboard({ title = 'Activity history', agents, loading, agentError, onViewChange, onManageAgent, onAssignAgent, onOpenAssignment, onOpenReview, onTemplate, initialTaskId, initialAttention }: Props) {
  const qc = useQueryClient()
  const inspectorRef = useRef<HTMLElement>(null)
  const ledgerRef = useRef<HTMLElement>(null)
  const attentionRef = useRef<HTMLElement>(null)
  const [showAllAttention, setShowAllAttention] = useState(Boolean(initialAttention))
  useEffect(() => {
    if (initialAttention) {
      attentionRef.current?.focus({ preventScroll: true })
      attentionRef.current?.scrollIntoView({ block: 'start' })
    }
  }, [initialAttention])
  // Bulk dismissal: id -> the review version seen when it was ticked, so a later
  // change to that item refuses the whole batch instead of dismissing unseen work.
  const [selecting, setSelecting] = useState(false)
  const [picked, setPicked] = useState<Record<string, string>>({})
  const [dismissNote, setDismissNote] = useState('')
  const [dismissedCount, setDismissedCount] = useState(0)
  const [agentId, setAgentId] = useState('')
  const [day, setDay] = useState('')
  const [status, setStatus] = useState('')
  const [search, setSearch] = useState('')
  const [onlyQuiet, setOnlyQuiet] = useState(false)
  const [tab, setTab] = useState<'agents' | 'development' | 'ops'>(initialTaskId ? 'ops' : 'agents')
  const [taskId, setTaskId] = useState(initialTaskId ?? '')
  const [runId, setRunId] = useState('')
  useEffect(() => {
    if ((agentId || runId || taskId) && window.matchMedia('(max-width:800px)').matches) {
      inspectorRef.current?.scrollIntoView({ block: 'start' })
    }
  }, [agentId, runId, taskId])
  const activity = useQuery({ queryKey: ['studio-activity', agentId, day, status], queryFn: () => api.studioActivity(agentId, day, status), retry: false, refetchInterval: 5_000 })
  const assignments = useQuery({ queryKey: ['assignments'], queryFn: api.assignments, retry: false, refetchInterval: 5_000 })
  const tasks = useQuery({ queryKey: ['pulse-tasks'], queryFn: api.tasks, refetchInterval: 5_000 })
  const workspaces = useQuery({ queryKey: ['git-workspaces'], queryFn: api.gitWorkspaces, refetchInterval: 5_000 })
  const writes = useQuery({ queryKey: ['git-writes'], queryFn: () => api.gitWrites(), refetchInterval: 5_000 })
  const [reviewWindow, setReviewWindow] = useState(14)
  const review = useQuery({ queryKey: ['review-health', reviewWindow], queryFn: () => api.reviewHealth(reviewWindow), retry: false, refetchInterval: 30_000 })
  const metrics = reviewMetrics(review.data)
  const sources = activeSources(review.data)
  const detail = useQuery({ queryKey: ['studio-run', runId], queryFn: () => api.studioRun(runId), enabled: Boolean(runId) })
  const loop = useMutation({ mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.setAgentLoop(id, enabled), onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }) })
  const data = activity.data
  const agent = agents.find(item => item.id === agentId)
  const selectedTask = tasks.data?.find(item => item.task_id === taskId)
  const selectedWork = assignments.data?.assignments.filter(item => (!agentId || item.agent_id === agentId) && (item.status === 'queued' || item.status === 'running')) ?? []
  const attention = attentionAssignments(assignments.data?.assignments ?? [], agentId)
  const pickedWork = attention.filter(work => picked[work.id])
  const bulkDismiss = useMutation({
    mutationFn: () => api.dismissAssignments(pickedWork.map(work => ({ id: work.id, expected_version: picked[work.id] })), dismissNote.trim()),
    onSuccess: result => {
      setDismissedCount(result.dismissed)
      setPicked({})
      setDismissNote('')
      setSelecting(false)
      void qc.invalidateQueries({ queryKey: ['assignments'] })
      void qc.invalidateQueries({ queryKey: ['review-health'] })
    },
    onError: error => {
      // The item that stopped the batch is unticked, so it is only resubmitted after
      // a person re-checks and ticks it again; the refetch brings current versions.
      const [, blocking] = error.message.split(':')
      if (blocking) setPicked(current => { const next = { ...current }; delete next[blocking]; return next })
      void qc.invalidateQueries({ queryKey: ['assignments'] })
    },
  })
  // A new selection or reason is a new attempt: the last refusal no longer applies.
  function togglePick(work: Assignment) {
    bulkDismiss.reset()
    setPicked(current => {
      const next = { ...current }
      if (next[work.id]) delete next[work.id]
      else next[work.id] = work.review_version ?? ''
      return next
    })
  }
  function pickAllShown() {
    bulkDismiss.reset()
    setPicked(Object.fromEntries(attention.filter(work => work.review_version).map(work => [work.id, work.review_version!])))
  }
  function startSelecting() { setSelecting(true); setDismissedCount(0) }
  function stopSelecting() { setSelecting(false); setPicked({}); bulkDismiss.reset() }
  const summary = attentionSummary(assignments.data?.assignments ?? [], workspaces.data?.workspaces, writes.data?.writes, agentId)
  const today = data?.today ?? new Date().toISOString().slice(0, 10)
  const todayCount = data?.days.find(item => item.day === today)
  const total = data?.days.reduce((sum, item) => sum + item.total, 0) ?? 0
  const failed = data?.days.reduce((sum, item) => sum + item.failed, 0) ?? 0
  const active = agents.filter(item => item.status === 'running')
  const scheduled = agents.filter(item => item.loop_enabled)
  const dailyCap = scheduled.reduce((sum, item) => sum + item.loop_max_runs_per_day, 0)
  const shownAgents = filterAgents(agents, search, onlyQuiet)
    .sort((a, b) => Number(noReviewedOutput(b)) - Number(noReviewedOutput(a)) || Number(b.status === 'running') - Number(a.status === 'running') || Number(b.loop_enabled) - Number(a.loop_enabled) || a.name.localeCompare(b.name))
  const days = Array.from({ length: 91 }, (_, i) => {
    const date = new Date(`${today}T00:00:00Z`)
    date.setUTCDate(date.getUTCDate() - 90 + i)
    return date.toISOString().slice(0, 10)
  })
  const refreshing = activity.isFetching || assignments.isFetching
  function refresh() {
    for (const key of ['agents', 'studio-activity', 'assignments', 'pulse-tasks']) void qc.invalidateQueries({ queryKey: [key] })
  }
  function pickAgent(id: string) { setAgentId(agentId === id ? '' : id); setRunId(''); setTaskId('') }

  return <main className="switchboard">
    {import.meta.env.VITE_STUDIO_PREVIEW === 'true' && <div className="sb-preview" role="status">Isolated preview · execution disabled · copied local data</div>}
    <PageHeader title={title} meta={<>
      <span>{loading ? 'Loading agents' : `${agents.length} agents`}</span>
      <span>{scheduled.length} loops enabled</span>
      <span>Mini execution</span>
    </>} />
    {(activity.isError || agentError || assignments.isError) && <div role="alert" className="sb-alert">Some live data is unavailable. {activity.error?.message || 'Check the Command Center connection.'} <button onClick={refresh}>Retry</button></div>}
    <section className="sb-metrics" aria-label="Switchboard summary">
      <div><span>Runs today / UTC</span><strong>{data ? todayCount?.total ?? 0 : '--'}</strong><small>{agent ? agent.name : 'All agents'} · recorded attempts</small></div>
      <div><span>Mini occupancy</span><strong>{active.length}<em> / 1 slot</em></strong><small>{active.length ? active[0].name : 'No Studio agent running'}</small></div>
      <button onClick={() => onViewChange('assignments')}><span>Work queue</span><strong>{assignments.data ? selectedWork.length : '--'}</strong><small>Open assignments <ArrowRight size={12} /></small></button>
      <div><span>Loop allowance</span><strong>{dailyCap}<em> / day</em></strong><small>Configured ceiling · {scheduled.length} schedules</small></div>
    </section>
    <div className="sb-body">
      <div className="sb-main">
        <section ref={attentionRef} tabIndex={-1} className="sb-attention focus-visible:outline-2 focus-visible:outline-cyan-300" aria-label="Needs attention">
          <div className="sb-section-title"><div><h2>Needs attention {assignments.data ? `(${summary.total})` : ''}</h2><p>{agent?.name || 'All agents'} · {summary.review} to review · {summary.failed} failed · {summary.approvals} approvals</p></div>
            {attention.length > 0 && <button type="button" className="sb-select-toggle" aria-pressed={selecting} onClick={() => selecting ? stopSelecting() : startSelecting()}>{selecting ? 'Cancel' : 'Select to dismiss'}</button>}
          </div>
          {dismissedCount > 0 && !selecting && <p role="status" className="sb-bulk-done">Dismissed {dismissedCount} {dismissedCount === 1 ? 'item' : 'items'}. No verdict was recorded.</p>}
          {selecting && <div className="sb-bulk" role="group" aria-label="Dismiss selected assignments">
            <p>Dismissing stops tracking an item without judging its output: nothing is accepted, and existing verdicts are kept. {agent ? `Showing ${agent.name} only.` : 'Pick an agent below to narrow the list.'}</p>
            <div className="sb-bulk-row">
              <button type="button" className="sb-link" onClick={pickAllShown}>Select all shown ({attention.length})</button>
              {pickedWork.length > 0 && <button type="button" className="sb-link" onClick={() => { bulkDismiss.reset(); setPicked({}) }}>Clear</button>}
              <span>{pickedWork.length} selected</span>
            </div>
            <label className="sb-bulk-note">Reason (required)<textarea value={dismissNote} maxLength={2000} rows={2} onChange={event => { bulkDismiss.reset(); setDismissNote(event.target.value) }} placeholder="Why these no longer need attention" /></label>
            <button type="button" className="sb-command" disabled={!pickedWork.length || !dismissNote.trim() || bulkDismiss.isPending} onClick={() => bulkDismiss.mutate()}>{bulkDismiss.isPending ? 'Dismissing...' : `Dismiss ${pickedWork.length} ${pickedWork.length === 1 ? 'item' : 'items'}`}</button>
            {bulkDismiss.error && <p role="alert">{bulkDismissError(bulkDismiss.error.message, id => attention.find(work => work.id === id)?.title ?? '')}</p>}
          </div>}
          {assignments.isError ? <p role="alert">Assignment status unavailable. {assignments.data ? 'Showing last loaded work.' : 'Retry to load work.'}</p> : assignments.isLoading ? <p>Loading assignments...</p> : null}
          {assignments.data && !assignments.isError && summary.total === 0 && <p>Nothing needs attention in this view.</p>}
          {summary.approvals > 0 && <button className="sb-work" onClick={() => onViewChange('assignments')}><span>{summary.approvals} git {summary.approvals === 1 ? 'approval' : 'approvals'} pending<small>Workspace or write requests waiting for a human decision</small></span><ArrowRight size={14} /></button>}
          {selecting && attention.map(work => <label className="sb-work sb-pick" key={work.id}><input type="checkbox" checked={Boolean(picked[work.id])} disabled={!work.review_version} onChange={() => togglePick(work)} /><span>{work.title}<small>{reviewLabel(work)} · {assignmentOrigin(work)} · {agents.find(item => item.id === work.agent_id)?.name || 'Removed agent'} · {new Date(work.created_at).toLocaleDateString()}</small></span></label>)}
          {!selecting && (showAllAttention ? attention : attention.slice(0, 8)).map(work => <button className="sb-work" key={work.id} onClick={() => onOpenAssignment(work.id)}><span>{work.title}<small>{reviewLabel(work)} · {assignmentOrigin(work)} · {agents.find(item => item.id === work.agent_id)?.name || 'Removed agent'} · {work.priority}</small></span><ArrowRight size={14} /></button>)}
          {!selecting && attention.length > 8 && <button className="sb-command" onClick={() => setShowAllAttention(!showAllAttention)}>{showAllAttention ? 'Show fewer' : `Show all ${attention.length} assignments`}</button>}
        </section>
        <section className="sb-review" aria-label="Review health">
          <div className="sb-section-title">
            <div>
              <h2>Review health</h2>
              <p>{review.isError ? 'Review health unavailable.' : review.isLoading && !review.data ? 'Loading review health...' : reviewSummary(review.data)}</p>
            </div>
            <div className="sb-review-window" role="group" aria-label="Review window">
              {[7, 14, 30].map(days => (
                <button key={days} type="button" aria-pressed={reviewWindow === days} onClick={() => setReviewWindow(days)} className={reviewWindow === days ? 'sb-review-on' : ''}>{days}d</button>
              ))}
            </div>
          </div>
          {review.isError && <p role="alert">Could not load review outcomes. <button className="sb-command" onClick={() => review.refetch()}>Retry</button></p>}
          <div className="sb-review-grid">
            {metrics.map(metric => (
              <button key={metric.bucket} type="button" className="sb-review-metric" title={metric.help} onClick={() => onOpenReview(metric.bucket)}>
                <span className="sb-review-count">{metric.count}</span>
                <span className="sb-review-label">{metric.label}</span>
                <span className="sb-review-rate">{metric.bucket === 'open' ? 'of the window' : `${formatRate(metric.rate)} of reviewed`}</span>
              </button>
            ))}
          </div>
          {sources.length > 0 && <div className="sb-review-sources">
            {sources.map(row => <span key={row.source}>{row.label}: {row.open} open of {row.filed} filed</span>)}
          </div>}
          <button className="sb-command" onClick={() => onOpenReview('')}>Open the review inbox</button>
        </section>
        <details className="sb-usage"><summary>Token usage and agent attribution</summary>
          <TokenUsage />
          <AgentTokenUsage data={data} agents={agents} day={day} status={status} selectedAgentId={agentId} onSelectAgent={pickAgent} />
        </details>
        <section className="sb-contributions" aria-label="Agent activity calendar">
          <div className="sb-section-title"><div><h2>Execution activity</h2><p>{data ? `${total} recorded runs · ${failed} failed · last 13 weeks, UTC` : activity.isLoading ? 'Loading recorded activity...' : 'Recorded activity unavailable'}</p></div><button className="sb-icon" title="Refresh activity" aria-label="Refresh activity" onClick={refresh}><RefreshCw size={16} className={refreshing ? 'sb-spin' : ''} /></button></div>
          <div className="sb-calendar-wrap">
            <div className="sb-calendar" role="group" aria-label="Daily run history">
              {days.map(date => {
                const cell = data?.days.find(item => item.day === date)
                const count = cell?.total ?? 0
                const level = count === 0 ? 0 : count < 3 ? 1 : count < 7 ? 2 : 3
                return <button key={date} className={`sb-day sb-level-${level} ${cell?.failed ? 'sb-has-failure' : ''} ${date === day ? 'sb-selected' : ''}`} aria-pressed={date === day} disabled={!data} aria-label={data ? `${date}: ${count} recorded runs, ${cell?.failed ?? 0} failed` : `${date}: history unavailable`} title={data ? `${date} / ${count} runs / ${cell?.failed ?? 0} failed` : `${date}: history unavailable`} onClick={() => { setDay(day === date ? '' : date); setRunId(''); ledgerRef.current?.scrollIntoView({ block: 'start' }) }} />
              })}
            </div>
          </div>
          <div className="sb-calendar-meta"><span>{days[0]} — {today}</span><span className="sb-legend">No records {[0, 1, 2, 3].map(level => <i key={level} className={`sb-level-${level}`} />)} More <i className="sb-failure-key" /> Failure</span></div>
          <p className="sb-coverage">{data?.earliest ? `History available from ${data.earliest.slice(0, 10)}. ${data.imported} imported records; earlier activity may be incomplete.` : activity.isError ? 'Run history unavailable; recorded counts cannot be confirmed.' : activity.isLoading ? 'Loading run history...' : 'No run history recorded yet.'}</p>
        </section>

        <section className="sb-fleet">
          <div className="sb-fleet-tools"><div className="sb-segments" role="tablist" aria-label="Switchboard panels">{(['agents', 'development', 'ops'] as const).map(value => <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)}>{value === 'agents' ? 'Agent lanes' : value === 'development' ? 'Development loops' : 'System loops'}</button>)}</div>{tab === 'agents' && <><label className="sb-search"><Search size={14} /><input aria-label="Find agent" value={search} onChange={event => setSearch(event.target.value)} placeholder="Find agent or repository" /></label><button type="button" className="sb-select-toggle" aria-pressed={onlyQuiet} onClick={() => setOnlyQuiet(!onlyQuiet)}>No reviewed output in 14 days</button></>}</div>
          {tab === 'agents' && <>
            <div className="sb-lane-head"><span>Agent / job / channel</span><span>Last 14 days</span><span>Schedule / value</span></div>
            {loading && <p className="sb-empty">Loading agent lanes...</p>}
            {!loading && shownAgents.length === 0 && <p className="sb-empty">No agents match this view.</p>}
            {shownAgents.map(item => {
              const queued = assignments.data?.assignments.filter(work => work.agent_id === item.id && work.status === 'queued').length ?? 0
              return <button key={item.id} className={`sb-lane ${agentId === item.id ? 'sb-lane-selected' : ''}`} onClick={() => pickAgent(item.id)} aria-pressed={agentId === item.id}>
                <div className="sb-agent-label"><span className={`sb-status sb-status-${item.status}`} /><div><strong>{item.name}</strong><small>{item.one_liner || item.mission}</small><small><GitBranch size={11} /> {item.repo_id || 'No repository'} · {queued} queued · <ChannelChip channel={item.output_channel} note={item.output_channel_note} /></small></div></div>
                <div className="sb-mini-history" aria-label="Recent recorded runs">{days.slice(-14).map(date => { const cell = data?.agent_days.find(row => row.agent_id === item.id && row.day === date); return <i key={date} title={`${date}: ${cell?.total ?? 0} recorded runs`} className={cell?.failed ? 'sb-mini-failed' : cell?.total ? 'sb-mini-done' : ''} /> })}</div>
                <div className="sb-lane-state"><strong className={noReviewedOutput(item) ? 'sb-value-quiet' : ''}>{valueGlance(item)}</strong><small>{item.loop_enabled ? `Every ${interval(item.loop_interval_minutes)}` : 'Manual'} · {item.status === 'running' ? 'Running' : item.status === 'error' ? 'Last run failed' : item.loop_enabled ? loopDue(item, today) : item.last_run_at ? 'Idle' : 'Never run'}</small></div>
              </button>
            })}
            <p className="sb-coverage">Lane history: 14 UTC days. Green: completed. Coral: one or more failed attempts.</p>
          </>}
          {tab === 'development' && <div className="sb-recipes">{DEVELOPMENT_LOOPS.map(recipe => <article key={recipe.id}><div className="sb-recipe-icon"><FileText size={18} /></div><div><h3>{recipe.name}</h3><p>{recipe.draft.mission}</p><small>{recipe.output} · {recipe.cadence}</small><div className="sb-recipe-meta"><span>{recipe.draft.repo_id}</span><span>Repository read</span><span>Starts paused</span></div></div><button title={`Configure ${recipe.name}`} aria-label={`Configure ${recipe.name}`} className="sb-icon" onClick={() => onTemplate({ ...recipe.draft })}><ArrowRight size={18} /></button></article>)}</div>}
          {tab === 'ops' && <div>{tasks.isError && <p className="sb-alert">System loop status unavailable.</p>}{tasks.isLoading && <p className="sb-empty">Loading system loops...</p>}{tasks.data?.map(task => <button key={task.task_id} className={`sb-system-row ${taskId === task.task_id ? 'sb-lane-selected' : ''}`} onClick={() => { setTaskId(task.task_id); setRunId(''); setAgentId('') }}><i className={`sb-status ${DOT_COLOR[loopState(task)]}`} /><div><strong>{task.name}</strong><small>{task.last_result || 'No result recorded'}</small></div><span>{loopState(task)}<small>{time(task.last_run)}</small></span></button>)}</div>}
        </section>

        <section ref={ledgerRef} className="sb-runs" aria-label="Run ledger">
          <div className="sb-section-title"><div><h2>Run ledger</h2><p>{day || 'All dates'} · {agent?.name || 'All agents'} · {data?.matching ?? '--'} matching</p></div><select aria-label="Run outcome" value={status} onChange={event => setStatus(event.target.value)}><option value="">All outcomes</option><option value="completed">Completed</option><option value="failed">Failed</option></select></div>
          {(day || agentId) && <button className="sb-clear" onClick={() => { setDay(''); setAgentId('') }}><X size={12} /> Clear filters</button>}
          {activity.isLoading && <p className="sb-empty">Loading run history...</p>}
          {data && data.runs.length === 0 && <p className="sb-empty">No recorded attempts match these filters.</p>}
          {data?.runs.map(run => <button key={run.id} onClick={() => { setRunId(run.id); setTaskId('') }} className={`sb-run-row ${runId === run.id ? 'sb-lane-selected' : ''}`}><span className={`sb-outcome sb-outcome-${run.status}`}>{run.status === 'completed' ? <Check size={14} /> : <X size={14} />}</span><div><strong>{run.agent_name}</strong><small>{run.trigger} · {run.model || 'Model unrecorded'}{run.legacy ? ' · imported' : ''}</small><small>{runTokens(run)}</small></div><span>{run.latency_ms > 0 ? `${Math.round(run.latency_ms / 1000)}s` : '--'}<small>{new Date(run.at).toLocaleString()}</small></span><ArrowRight size={14} /></button>)}
          {(data?.matching ?? 0) > 200 && <p className="sb-coverage">Showing the latest 200 matches. Select an agent or date to narrow the history.</p>}
        </section>
      </div>

      <aside ref={inspectorRef} className="sb-inspector" aria-label="Activity inspector">
        {runId ? <><div className="sb-section-title"><h2>Run details</h2><button className="sb-icon" aria-label="Close run details" onClick={() => setRunId('')}><X size={16} /></button></div>{detail.isLoading && <p>Loading run...</p>}{detail.isError && <p role="alert">Could not load this run.</p>}{detail.data && <RunDetail run={detail.data.run} onOpenAssignment={onOpenAssignment} />}</>
        : selectedTask ? <><div className="sb-eyebrow">System loop</div><h2>{selectedTask.name}</h2><p>{selectedTask.description}</p><dl><dt>Current signal</dt><dd>{loopState(selectedTask)}</dd><dt>Last run</dt><dd>{time(selectedTask.last_run)}</dd><dt>Lifetime runs</dt><dd>{selectedTask.run_count}</dd><dt>Lifetime failures</dt><dd>{selectedTask.fail_count}</dd></dl><h3>Latest result</h3><pre>{selectedTask.last_result || 'No result recorded.'}</pre></>
        : agent ? <><div className="sb-eyebrow">Agent controls</div><h2>{agent.name}</h2><p>{agent.one_liner || agent.mission}</p>{hasSlackGap(agent) && <p className="sb-alert">slack not configured</p>}<p className="sb-coverage">{valueSummary(agent)}</p><dl><dt>Output</dt><dd><ChannelChip channel={agent.output_channel} note={agent.output_channel_note} /></dd><dt>State</dt><dd>{agent.status}</dd><dt>Repository</dt><dd>{agent.repo_id || 'None'}</dd><dt>Schedule</dt><dd>{agent.loop_enabled ? 'Enabled' : 'Paused'}</dd><dt>Interval</dt><dd>{interval(agent.loop_interval_minutes)}</dd><dt>Runs / UTC day</dt><dd>{agent.loop_day === today ? agent.loop_runs_today : 0} / {agent.loop_max_runs_per_day}</dd><dt>Last loop check</dt><dd>{agent.loop_checked_at ? time(agent.loop_checked_at) : 'Never'}</dd><dt>Last loop decision</dt><dd>{agent.loop_skip_reason === 'awaiting_review' ? 'Waiting for review' : agent.loop_skip_reason === 'unchanged_repository_input' ? 'Skipped · repository unchanged' : agent.last_run_at ? 'Executed' : 'No decision'}</dd><dt>Failure streak</dt><dd>{agent.loop_consecutive_failures ?? 0}</dd><dt>Backoff until</dt><dd>{agent.loop_backoff_until ? time(agent.loop_backoff_until) : 'None'}</dd><dt>Temperature</dt><dd>{agent.temperature}</dd><dt>Token cap</dt><dd>{agent.max_tokens}</dd></dl>
          {agent.loop_task && <button className="sb-command" disabled={loop.isPending} onClick={() => loop.mutate({ id: agent.id, enabled: !agent.loop_enabled })}>{agent.loop_enabled ? <CirclePause size={15} /> : <Play size={15} />}{agent.loop_enabled ? 'Pause loop' : 'Enable loop'}</button>}
          {loop.isError && <p role="alert" className="sb-alert">{loop.error.message}</p>}
          <button className="sb-command" onClick={() => onManageAgent(agent.id)}><Layers3 size={15} /> Edit configuration</button><button className="sb-command" onClick={() => onAssignAgent(agent.id)}><FileText size={15} /> Assign work</button>
          {agent.loop_task && <><h3>Recurring task</h3><pre>{agent.loop_task}</pre></>}
          <h3>Open assignments</h3>{selectedWork.length ? selectedWork.map(work => <button className="sb-work" key={work.id} onClick={() => onOpenAssignment(work.id)}>{work.title}<ArrowRight size={14} /></button>) : <p>No open assignments.</p>}
        </> : <><div className="sb-eyebrow">Execution node</div><h2>AIIA Mini</h2><div className="sb-node"><span className={`sb-status ${active.length ? 'sb-status-running' : 'sb-status-idle'}`} /><strong>{active.length ? 'Studio run active' : 'Studio idle'}</strong><small>Single execution slot</small></div><dl><dt>Enabled schedules</dt><dd>{scheduled.length}</dd><dt>Maximum loop runs</dt><dd>{dailyCap} / day</dd><dt>Run history</dt><dd>{data?.total ?? '--'} recorded</dd><dt>Storage</dt><dd>Local SQLite</dd></dl><h3>Scheduled agents</h3>{scheduled.map(item => <button className="sb-work" key={item.id} onClick={() => pickAgent(item.id)}><span>{item.name}<small>{interval(item.loop_interval_minutes)} · {item.loop_max_runs_per_day}/day</small></span><ArrowRight size={14} /></button>)}<button className="sb-command" onClick={() => setTab('development')}><Layers3 size={15} /> Configure development work</button><p className="sb-note">Studio serialization does not include every background service on the Mini.</p></>}
      </aside>
    </div>
  </main>
}

function RunDetail({ run, onOpenAssignment }: { run: StudioRun; onOpenAssignment: (id: string) => void }) {
  return <><h2>{run.agent_name}</h2><dl><dt>Outcome</dt><dd>{run.status}</dd><dt>Delivered to</dt><dd>{run.delivered_channel || 'Unrecorded'}{run.delivery_note ? ` · ${run.delivery_note}` : ''}</dd><dt>Trigger</dt><dd>{run.trigger}</dd><dt>Recorded</dt><dd>{new Date(run.at).toLocaleString()}</dd><dt>Model</dt><dd>{run.model || 'Unrecorded'}</dd><dt>Duration</dt><dd>{run.latency_ms ? `${(run.latency_ms / 1000).toFixed(1)}s` : 'Unrecorded'}</dd><dt>Input tokens</dt><dd>{run.input_tokens?.toLocaleString('en-US') ?? 'Unrecorded'}</dd><dt>Output tokens</dt><dd>{run.output_tokens?.toLocaleString('en-US') ?? 'Unrecorded'}</dd><dt>Total</dt><dd>{runTokens(run)}</dd></dl>{run.assignment_id && <button className="sb-command" onClick={() => onOpenAssignment(run.assignment_id)}><FileText size={15} /> Open assignment</button>}<h3>Task</h3><pre>{run.task}</pre><h3>{run.error ? 'Failure' : 'Work product'}</h3><pre>{run.error || run.result || 'No output recorded.'}</pre></>
}

function interval(minutes: number) { return minutes >= 60 && minutes % 60 === 0 ? `${minutes / 60}h` : `${minutes}m` }
function time(value: string | null) { return value ? new Date(value).toLocaleString() : 'Never' }
function loopDue(agent: Agent, today: string) {
  if (agent.loop_skip_reason === 'awaiting_review') return 'Waiting for review'
  if (agent.loop_skip_reason === 'check_incomplete') return 'Check failed · see Needs attention'
  if (agent.loop_skip_reason === 'assignment_capacity_reached') return 'Work registry full · review or dismiss items'
  if (agent.loop_day === today && agent.loop_runs_today >= agent.loop_max_runs_per_day) return 'Daily cap reached'
  if (agent.loop_backoff_until) {
    const backoffMinutes = Math.ceil((Date.parse(agent.loop_backoff_until) - Date.now()) / 60_000)
    if (backoffMinutes > 0) return `Backoff · retry in ${interval(backoffMinutes)}`
  }
  const lastCheck = agent.loop_checked_at || agent.last_run_at
  if (!lastCheck) return 'Due'
  const minutes = Math.ceil((Date.parse(lastCheck) + agent.loop_interval_minutes * 60_000 - Date.now()) / 60_000)
  if (agent.loop_skip_reason === 'unchanged_repository_input' && minutes > 0) return `No changes · check in ${interval(minutes)}`
  return minutes <= 0 ? 'Due' : `Due in ${interval(minutes)}`
}
