import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, CalendarClock, Plus, RefreshCw } from 'lucide-react'
import { api, type Agent, type Assignment } from '../lib/api'
import { attentionAssignments, attentionSummary, reviewLabel } from './assignmentReview'
import { PageHeader } from './PageHeader'
import { jobState, jobTime } from './jobHelpers'

const link = 'inline-flex min-h-11 items-center gap-2 rounded px-2 text-sm text-emerald-300 hover:text-emerald-200 focus-visible:outline-2 focus-visible:outline-emerald-300'

export function Today({ agents, loading, agentError, attention }: { agents: Agent[]; loading: boolean; agentError: boolean; attention?: boolean }) {
  const attentionRef = useRef<HTMLElement>(null)
  const assignments = useQuery({ queryKey: ['assignments'], queryFn: api.assignments, retry: false, refetchInterval: 5_000 })
  const workspaces = useQuery({ queryKey: ['git-workspaces'], queryFn: api.gitWorkspaces, retry: false, refetchInterval: 10_000 })
  const writes = useQuery({ queryKey: ['git-writes'], queryFn: () => api.gitWrites(), retry: false, refetchInterval: 10_000 })
  const work = assignments.data?.assignments ?? []
  const flagged = attentionAssignments(work)
  const summary = attentionSummary(work, workspaces.data?.workspaces, writes.data?.writes)
  const approvals = [
    ...(workspaces.data?.workspaces ?? []).filter(item => item.status === 'pending').map(item => ({ id: `workspace-${item.id}`, assignmentId: item.assignment_id, label: 'Approve workspace' })),
    ...(writes.data?.writes ?? []).filter(item => item.status === 'pending').map(item => ({ id: `write-${item.id}`, assignmentId: item.assignment_id, label: `Review ${item.op.replaceAll('_', ' ')}` })),
  ]
  const active = work.filter(item => item.status === 'running' || item.status === 'queued')
  const scheduled = agents.filter(item => item.loop_enabled)
  const accepted = work.filter(item => item.status === 'completed' && item.review_status === 'accepted')
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at)).slice(0, 3)
  const incomplete = assignments.isError || workspaces.isError || writes.isError
  const checking = assignments.isPending || workspaces.isPending || writes.isPending
  useEffect(() => {
    if (attention) {
      attentionRef.current?.focus({ preventScroll: true })
      attentionRef.current?.scrollIntoView({ block: 'start' })
    }
  }, [attention])

  return <main className="flex h-full min-h-0 flex-col bg-neutral-950">
    <PageHeader title="Today" actions={<div className="flex items-center gap-2"><button type="button" aria-label="Refresh today" title="Refresh today" className="flex h-11 w-11 items-center justify-center rounded text-neutral-400 focus-visible:outline-2 focus-visible:outline-emerald-300" disabled={assignments.isFetching || workspaces.isFetching || writes.isFetching} onClick={() => { void assignments.refetch(); void workspaces.refetch(); void writes.refetch() }}><RefreshCw size={16} aria-hidden="true" /></button><a className={link} href="#/jobs"><Plus size={16} aria-hidden="true" />Set up a job</a></div>} />
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-5xl px-5 pb-8 sm:px-7">
        <section ref={attentionRef} tabIndex={-1} aria-label="Needs attention" className="border-b border-neutral-800 py-6 outline-none focus-visible:ring-2 focus-visible:ring-emerald-400">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-medium text-white">Needs attention {checking ? '' : `(${summary.total})`}</h2>
            <a href="#/history?attention=1" className={link}>Review all<ArrowRight size={16} aria-hidden="true" /></a>
          </div>
          <p className="mt-1 text-sm text-neutral-400">{summary.review} to review · {summary.failed} failures · {summary.approvals} approvals</p>
          {incomplete && <p role="alert" className="mt-3 text-sm text-amber-300">Some review sources are unavailable. These counts may be incomplete.</p>}
          {checking ? <p role="status" className="py-4 text-sm text-neutral-400">Checking work...</p> : <>
            {flagged.slice(0, 5).map(item => <WorkRow key={item.id} item={item} detail={reviewLabel(item)} />)}
            {approvals.slice(0, 3).map(item => <a key={item.id} href={`#/assignments/${encodeURIComponent(item.assignmentId)}`} className="flex min-h-16 items-center justify-between gap-3 border-t border-neutral-900 py-3 text-sm focus-visible:outline-2 focus-visible:outline-emerald-300"><div className="min-w-0"><p className="break-words text-white">{work.find(assignment => assignment.id === item.assignmentId)?.title || 'Work approval'}</p><p className="mt-1 text-amber-300">{item.label}</p></div><ArrowRight size={16} className="shrink-0" aria-hidden="true" /></a>)}
            {summary.approvals > 0 && <a href="#/assignments" className={`${link} mt-2`}>Open {summary.approvals} pending approvals<ArrowRight size={16} aria-hidden="true" /></a>}
            {summary.total === 0 && !incomplete && <p className="py-4 text-sm text-neutral-400">Nothing needs a decision right now.</p>}
          </>}
        </section>

        <section aria-label="Work in progress" className="border-b border-neutral-800 py-6">
          <div className="flex items-center justify-between gap-2"><h2 className="text-lg font-medium text-white">Work in progress</h2><a href="#/assignments" className={link}>Open work<ArrowRight size={16} aria-hidden="true" /></a></div>
          {active.slice(0, 4).map(item => <WorkRow key={item.id} item={item} detail={item.status === 'running' ? 'Running on the Mini' : 'Awaiting manual start'} />)}
          {!active.length && <p className="py-3 text-sm text-neutral-400">{assignments.isError ? 'Work status unavailable.' : assignments.isPending ? 'Checking work...' : 'No active assignments.'}</p>}
        </section>

        <section aria-label="Recurring jobs" className="border-b border-neutral-800 py-6">
          <div className="flex items-center justify-between gap-2"><h2 className="text-lg font-medium text-white">Recurring jobs</h2><a href="#/jobs" className={link}>Manage jobs<ArrowRight size={16} aria-hidden="true" /></a></div>
          {agentError && <p role="alert" className="py-3 text-sm text-amber-300">Job status unavailable.</p>}
          {loading ? <p className="py-3 text-sm text-neutral-400">Checking jobs...</p> : scheduled.slice(0, 4).map(agent => <a key={agent.id} href="#/jobs" className="flex min-h-16 items-center gap-3 border-t border-neutral-900 py-3 focus-visible:outline-2 focus-visible:outline-emerald-300">
            <CalendarClock size={18} className="shrink-0 text-neutral-400" aria-hidden="true" />
            <div className="min-w-0 flex-1"><p className="break-words text-sm text-white">{agent.name}</p><p className="mt-1 text-sm text-neutral-400">Every {agent.loop_interval_minutes} min · up to {agent.loop_max_runs_per_day}/day</p></div>
            <span className="max-w-32 text-right text-sm text-neutral-400" title={jobState(agent).detail}>{jobState(agent).label}</span>
          </a>)}
          {!loading && !agentError && !scheduled.length && <p className="py-3 text-sm text-neutral-400">No recurring agent jobs enabled.</p>}
        </section>

        {!!accepted.length && <section aria-label="Recently accepted" className="border-b border-neutral-800 py-6"><h2 className="mb-3 text-lg font-medium text-white">Recently accepted</h2>{accepted.map(item => <WorkRow key={item.id} item={item} detail="Accepted output" />)}</section>}
        <footer className="flex flex-wrap justify-between gap-2 pt-5"><a href="#/memory?review=all" className={link}>Open the review inbox<ArrowRight size={16} aria-hidden="true" /></a><a href="#/history" className={link}>Activity and usage<ArrowRight size={16} aria-hidden="true" /></a></footer>
      </div>
    </div>
  </main>
}

function WorkRow({ item, detail }: { item: Assignment; detail: string }) {
  return <a href={`#/assignments/${encodeURIComponent(item.id)}`} className="flex min-h-16 items-center justify-between gap-3 border-t border-neutral-900 py-3 focus-visible:outline-2 focus-visible:outline-emerald-300">
    <div className="min-w-0"><p className="break-words text-sm font-medium text-white">{item.title}</p><p className={`mt-1 text-sm ${item.status === 'failed' ? 'text-amber-300' : 'text-neutral-400'}`}>{detail}</p><time dateTime={item.created_at} className="mt-1 block text-xs text-neutral-500">{jobTime(item.created_at)}</time></div>
    <ArrowRight size={16} className="shrink-0 text-neutral-500" aria-hidden="true" />
  </a>
}
