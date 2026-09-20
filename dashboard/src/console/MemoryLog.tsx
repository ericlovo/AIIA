import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, MEMORY_CATEGORIES, MEMORY_PRIORITIES, type Agent, type MemoryCategory, type MemoryIdea, type MemoryIdeaStatus, type MemoryInboxSort, type MemoryPriority } from '../lib/api'
import { StudioTabs, type StudioView } from './StudioTabs'
import { captureText, MEMORY_POST_CHANNEL, memoryPostLabel, priorityLabel, receiptLabel, type PriorityTone, type ReceiptTone } from './memoryText'

type Filter = MemoryIdeaStatus | ''
const FILTERS: { id: Filter; label: string }[] = [
  { id: 'unreviewed', label: 'Unreviewed' },
  { id: 'promoted', label: 'Logged' },
  { id: 'dismissed', label: 'Dismissed' },
  { id: '', label: 'All' },
]
// Two things arrive in this inbox: what a person said in Slack, and what an
// unattended loop proposed. They are reviewed the same way but read differently,
// so the view names which one you are looking at.
type Origin = 'slack' | 'loops' | 'all'
const ORIGINS: { id: Origin; label: string; source: string; project: string; blurb: string }[] = [
  { id: 'slack', label: 'From Slack', source: 'slack', project: 'mindmoor', blurb: 'Captures from the allowed Slack channels.' },
  { id: 'loops', label: 'From loops', source: 'backlog_steward', project: '', blurb: 'Proposals the backlog steward filed. A rerun does not refile the same finding.' },
  { id: 'all', label: 'All', source: '', project: '', blurb: 'Everything waiting for review, whoever raised it.' },
]
const TONE: Record<ReceiptTone, string> = {
  sent: 'text-emerald-300',
  pending: 'text-cyan-200',
  failed: 'text-red-300',
  none: 'text-neutral-600',
}
const PRIORITY_TONE: Record<PriorityTone, string> = {
  urgent: 'border-red-500/60 text-red-200',
  high: 'border-amber-500/60 text-amber-200',
  normal: 'border-neutral-700 text-neutral-400',
  low: 'border-neutral-800 text-neutral-500',
}

export function MemoryLog({ agents, view, onViewChange }: { agents: Agent[]; view: StudioView; onViewChange: (view: StudioView) => void }) {
  const qc = useQueryClient()
  const [filter, setFilter] = useState<Filter>('unreviewed')
  const [origin, setOrigin] = useState<Origin>('slack')
  const scope = ORIGINS.find(item => item.id === origin) ?? ORIGINS[0]
  const [query, setQuery] = useState('')
  const [offset, setOffset] = useState(0)
  const [priority, setPriority] = useState<MemoryPriority | ''>('')
  const [sort, setSort] = useState<MemoryInboxSort>('newest')
  const [notice, setNotice] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null)
  const page = useQuery({
    queryKey: ['memory-inbox', scope.id, filter, query, offset, priority, sort],
    queryFn: () => api.memoryInbox({ project: scope.project, source: scope.source, status: filter, query, offset, priority, sort }),
    retry: false,
    refetchInterval: 15_000,
  })
  const slack = useQuery({ queryKey: ['slack-capture-status'], queryFn: api.slackCaptureStatus, retry: false, refetchInterval: 30_000 })
  const done = (text: string) => {
    setNotice({ tone: 'ok', text })
    qc.invalidateQueries({ queryKey: ['memory-inbox'] })
    qc.invalidateQueries({ queryKey: ['slack-capture-status'] })
    qc.invalidateQueries({ queryKey: ['mind-memories'] })
  }
  const fail = (error: Error) => setNotice({ tone: 'error', text: describe(error.message) })
  const promote = useMutation({
    mutationFn: ({ id, category, priority, postToSlack }: { id: string; category: MemoryCategory; priority: MemoryPriority; postToSlack: boolean }) => api.promoteIdea(id, category, '', { priority, postToSlack }),
    onSuccess: result => done(`Logged to AIIA memory as ${result.idea.memory_category} at ${priorityLabel(result.idea.priority).text.toLowerCase()} priority. ${receiptLabel(result.idea.promotion_status, null, 'Memory').text}.${result.idea.post_requested ? ` ${memoryPostLabel(result.idea.memory_post_status, null, true)?.text}.` : ''}`),
    onError: fail,
  })
  const dismiss = useMutation({ mutationFn: (id: string) => api.dismissIdea(id), onSuccess: () => done('Capture dismissed. It stays in the inbox under Dismissed.'), onError: fail })
  const restore = useMutation({ mutationFn: (id: string) => api.restoreIdea(id), onSuccess: () => done('Capture restored to Unreviewed.'), onError: fail })
  const retry = useMutation({ mutationFn: ({ id, kind }: { id: string; kind: 'capture' | 'promotion' | 'memory_post' }) => api.retryIdeaReceipt(id, kind), onSuccess: (_, { kind }) => done(kind === 'memory_post' ? `Post to ${MEMORY_POST_CHANNEL} queued again.` : 'Receipt queued again.'), onError: fail })
  const assign = useMutation({
    mutationFn: ({ id, agentId }: { id: string; agentId: string }) => api.assignCapture(id, agentId),
    onSuccess: result => {
      qc.invalidateQueries({ queryKey: ['assignments'] })
      done(`Queued for ${agents.find(item => item.id === result.assignment.agent_id)?.name || 'the agent'} as "${result.assignment.title}". It waits in Work until you run it.`)
    },
    onError: fail,
  })
  const canPost = slack.data?.memory_posts_configured === true
  const busy = promote.isPending || dismiss.isPending || restore.isPending || retry.isPending || assign.isPending
  const data = page.data
  const counts = data?.counts
  const ideas = data?.ideas ?? []

  return (
    <main className="h-full min-h-0 flex-1 overflow-y-auto bg-neutral-950">
      <header className="flex flex-col gap-5 border-b border-neutral-900 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-cyan-400">Agent Studio</div>
          <h1 className="mt-2 text-2xl font-medium text-white">Memory log</h1>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
            <span>{agents.length} agents</span>
            <span>{counts ? `${counts.unreviewed} unreviewed` : 'Loading inbox'}</span>
            <span>{counts ? `${counts.promoted} logged to memory` : ''}</span>
            <span>{slackSummary(slack.data, slack.isError)}</span>
          </div>
        </div>
        <StudioTabs view={view} onChange={onViewChange} />
      </header>

      {notice && <div role={notice.tone === 'error' ? 'alert' : 'status'} className={`sticky top-0 z-20 flex items-start justify-between gap-4 border-b border-neutral-800 bg-neutral-950 px-5 py-3 text-sm sm:px-7 ${notice.tone === 'error' ? 'text-red-300' : 'text-cyan-200'}`}>
        <span>{notice.text}</span>
        <button type="button" onClick={() => setNotice(null)} className="shrink-0 text-xs text-neutral-500 hover:text-neutral-200">Dismiss notice</button>
      </div>}

      <section aria-label="Memory log" className="min-w-0">
        <div className="sticky top-0 z-10 flex flex-col gap-3 border-b border-neutral-900 bg-neutral-950/95 px-5 py-4 backdrop-blur sm:flex-row sm:items-center sm:justify-between sm:px-7">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-neutral-500">Mindmoor captures from Slack</div>
            <div className="mt-1 text-xs text-neutral-600">{sort === 'priority' ? 'Highest priority first' : 'Newest first'} · captures stay unreviewed until you log or dismiss them · refreshes every 15 seconds</div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select value={priority} onChange={event => { setPriority(event.target.value as MemoryPriority | ''); setOffset(0) }} aria-label="Filter by priority" className="h-8 border border-neutral-800 bg-neutral-900 px-1 text-xs text-neutral-200">
              <option value="">Any priority</option>
              {MEMORY_PRIORITIES.map(item => <option key={item} value={item}>{priorityLabel(item).text}</option>)}
            </select>
            <select value={sort} onChange={event => { setSort(event.target.value as MemoryInboxSort); setOffset(0) }} aria-label="Sort captures" className="h-8 border border-neutral-800 bg-neutral-900 px-1 text-xs text-neutral-200">
              <option value="newest">Newest first</option>
              <option value="priority">Priority first</option>
            </select>
            <input value={query} onChange={event => { setQuery(event.target.value); setOffset(0) }} placeholder="Search captures" aria-label="Search captures" className="h-8 w-44 border border-neutral-800 bg-neutral-900 px-2 text-xs text-neutral-200 outline-none placeholder:text-neutral-600 focus:border-cyan-500/50" />
            <div className="flex h-8 max-w-full overflow-x-auto border border-neutral-800 p-0.5" role="tablist" aria-label="Capture origin">
              {ORIGINS.map(item => (
                <button key={item.id} role="tab" aria-selected={origin === item.id} onClick={() => { setOrigin(item.id); setOffset(0) }} className={`shrink-0 px-3 text-[11px] transition-colors ${origin === item.id ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-200'}`}>
                  {item.label}
                </button>
              ))}
            </div>
            <div className="flex h-8 max-w-full overflow-x-auto border border-neutral-800 p-0.5" role="tablist" aria-label="Capture filters">
              {FILTERS.map(item => (
                <button key={item.id || 'all'} role="tab" aria-selected={filter === item.id} onClick={() => { setFilter(item.id); setOffset(0) }} className={`shrink-0 px-3 text-[11px] transition-colors ${filter === item.id ? 'bg-neutral-700 text-white' : 'text-neutral-500 hover:text-neutral-200'}`}>
                  {item.label}{counts && item.id ? ` ${counts[item.id]}` : ''}
                </button>
              ))}
            </div>
          </div>
        </div>

        {page.isError && <p role="alert" className="px-5 py-4 text-sm text-red-300 sm:px-7">Memory inbox unavailable. {data ? 'Showing the last loaded captures.' : 'Retry to load captures.'} <button type="button" onClick={() => page.refetch()} className="underline">Retry</button></p>}
        {page.isLoading && <p className="px-5 py-6 text-sm text-neutral-500 sm:px-7">Loading captures...</p>}
        {data && ideas.length === 0 && <p className="px-5 py-6 text-sm text-neutral-500 sm:px-7">{query || priority ? 'No captures match this search.' : filter === 'unreviewed' ? (origin === 'loops' ? 'Nothing proposed. The backlog steward files what it finds here after its daily run.' : 'Nothing waiting. New @AIIA mentions and /aiia-capture ideas from the allowed Slack channels land here.') : 'No captures in this state.'}</p>}

        <ul className="divide-y divide-neutral-900">
          {ideas.map(idea => <IdeaRow key={idea.id} idea={idea} busy={busy} canPost={canPost} agents={agents}
            onPromote={(category, priority, postToSlack) => promote.mutate({ id: idea.id, category, priority, postToSlack })}
            onDismiss={() => dismiss.mutate(idea.id)}
            onRestore={() => restore.mutate(idea.id)}
            onAssign={agentId => assign.mutate({ id: idea.id, agentId })}
            onRetry={kind => retry.mutate({ id: idea.id, kind })} />)}
        </ul>

        {data && data.total > 50 && <div className="flex items-center justify-between px-5 py-4 text-xs text-neutral-500 sm:px-7">
          <span>Showing {offset + 1}-{Math.min(offset + 50, data.total)} of {data.total}</span>
          <div className="flex gap-2">
            <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))} className="border border-neutral-800 px-3 py-1 disabled:opacity-40">Newer</button>
            <button type="button" disabled={offset + 50 >= data.total} onClick={() => setOffset(offset + 50)} className="border border-neutral-800 px-3 py-1 disabled:opacity-40">Older</button>
          </div>
        </div>}

        <p className="px-5 py-4 text-[11px] leading-relaxed text-neutral-600 sm:px-7">
          {scope.blurb} "Queue as work" turns a capture into a queued assignment for one agent, with the capture text carried as untrusted input and its origin recorded. Nothing runs until you start it in Work. Logging stores the capture as a Brain fact with Slack provenance (capture ID, channel, author, time) and, when the capture came from a thread, queues one fixed receipt back to that thread through the AIIA Slack app. Receipts never carry the captured text. Only when you check "Post to {MEMORY_POST_CHANNEL}" is the capture text, with its priority and category, posted to that one channel. Dismissing keeps the record locally and sends nothing.
        </p>
      </section>
    </main>
  )
}

function IdeaRow({ idea, busy, canPost, agents, onPromote, onDismiss, onRestore, onAssign, onRetry }: { idea: MemoryIdea; busy: boolean; canPost: boolean; agents: Agent[]; onPromote: (category: MemoryCategory, priority: MemoryPriority, postToSlack: boolean) => void; onDismiss: () => void; onRestore: () => void; onAssign: (agentId: string) => void; onRetry: (kind: 'capture' | 'promotion' | 'memory_post') => void }) {
  const [category, setCategory] = useState<MemoryCategory>('project')
  const [priority, setPriority] = useState<MemoryPriority>('normal')
  const [postToSlack, setPostToSlack] = useState(false)
  const [owner, setOwner] = useState('')
  const posted = memoryPostLabel(idea.memory_post_status, idea.memory_post_error, idea.post_requested === 1)
  const badge = priorityLabel(idea.priority)
  const text = captureText(idea.text) || '(mention only, no text)'
  const save = receiptLabel(idea.acknowledgement_status, idea.acknowledgement_error, 'Save')
  const memory = receiptLabel(idea.promotion_status, idea.promotion_error, 'Memory')
  return (
    <li className="px-5 py-4 sm:px-7" data-idea-status={idea.status} data-idea-priority={badge.tone}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 flex-1">
          <p className="whitespace-pre-wrap break-words text-sm text-neutral-100">{text}</p>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-neutral-500">
            <span className={`uppercase tracking-wider ${idea.status === 'promoted' ? 'text-emerald-300' : idea.status === 'dismissed' ? 'text-neutral-500' : 'text-amber-300'}`}>{idea.status === 'promoted' ? 'Logged to memory' : idea.status === 'dismissed' ? 'Dismissed' : 'Unreviewed'}</span>
            {idea.status === 'promoted' && <span className={`border px-1.5 uppercase tracking-wider ${PRIORITY_TONE[badge.tone]}`} aria-label={`Priority ${badge.text}`}>{badge.text}</span>}
            <span>{new Date(idea.created_at).toLocaleString()}</span>
            <span>{idea.source === 'slack' ? `slack · channel ${idea.channel_id} · author ${idea.author_id}` : `${idea.source.replace(/_/g, ' ')} · proposed${idea.project ? ` · ${idea.project}` : ''}`}</span>
            <span>capture {idea.id.slice(0, 8)}</span>
            {idea.status === 'promoted' && <span>memory {idea.memory_category} · {idea.memory_id}</span>}
            {idea.review_note && <span>note: {idea.review_note}</span>}
            {idea.assignment_id && <span className="text-cyan-200">queued as work {idea.assignment_id.slice(0, 8)}</span>}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
            <span className={TONE[save.tone]}>{save.text}</span>
            {save.tone === 'failed' && <button type="button" disabled={busy} onClick={() => onRetry('capture')} className="underline text-neutral-400">Retry save receipt</button>}
            {idea.status === 'promoted' && <span className={TONE[memory.tone]}>{memory.text}</span>}
            {idea.status === 'promoted' && memory.tone === 'failed' && <button type="button" disabled={busy} onClick={() => onRetry('promotion')} className="underline text-neutral-400">Retry memory receipt</button>}
            {posted && <span className={TONE[posted.tone]}>{posted.text}</span>}
            {posted?.tone === 'failed' && <button type="button" disabled={busy} onClick={() => onRetry('memory_post')} className="underline text-neutral-400">Retry memory post</button>}
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {idea.status === 'unreviewed' && <>
            <label className="text-[11px] text-neutral-500">Category
              <select value={category} onChange={event => setCategory(event.target.value as MemoryCategory)} className="ml-1 h-8 border border-neutral-800 bg-neutral-900 px-1 text-xs text-neutral-200" aria-label={`Memory category for capture ${idea.id.slice(0, 8)}`}>
                {MEMORY_CATEGORIES.map(item => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label className="text-[11px] text-neutral-500">Priority
              <select value={priority} onChange={event => setPriority(event.target.value as MemoryPriority)} className="ml-1 h-8 border border-neutral-800 bg-neutral-900 px-1 text-xs text-neutral-200" aria-label={`Priority for capture ${idea.id.slice(0, 8)}`}>
                {MEMORY_PRIORITIES.map(item => <option key={item} value={item}>{priorityLabel(item).text}</option>)}
              </select>
            </label>
            {canPost && <label className="flex h-8 items-center gap-1.5 text-[11px] text-neutral-300">
              <input type="checkbox" checked={postToSlack} onChange={event => setPostToSlack(event.target.checked)} className="accent-cyan-400" aria-label={`Post capture ${idea.id.slice(0, 8)} to ${MEMORY_POST_CHANNEL}`} />
              Post to {MEMORY_POST_CHANNEL}
            </label>}
            <button type="button" disabled={busy} onClick={() => onPromote(category, priority, postToSlack)} className="h-8 border border-cyan-500/60 bg-cyan-500/10 px-3 text-xs text-cyan-100 hover:bg-cyan-500/20 disabled:opacity-40">Log to memory</button>
            <button type="button" disabled={busy} onClick={onDismiss} className="h-8 border border-neutral-800 px-3 text-xs text-neutral-300 hover:text-white disabled:opacity-40">Dismiss</button>
          </>}
          {idea.status !== 'dismissed' && !idea.assignment_id && agents.length > 0 && <>
            <label className="text-[11px] text-neutral-500">Agent
              <select value={owner} onChange={event => setOwner(event.target.value)} className="ml-1 h-8 border border-neutral-800 bg-neutral-900 px-1 text-xs text-neutral-200" aria-label={`Agent for capture ${idea.id.slice(0, 8)}`}>
                <option value="">Choose agent</option>
                {agents.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </label>
            <button type="button" disabled={busy || !owner} onClick={() => onAssign(owner)} className="h-8 border border-neutral-800 px-3 text-xs text-neutral-300 hover:text-white disabled:opacity-40">Queue as work</button>
          </>}
          {idea.status === 'dismissed' && <button type="button" disabled={busy} onClick={onRestore} className="h-8 border border-neutral-800 px-3 text-xs text-neutral-300 hover:text-white disabled:opacity-40">Restore</button>}
        </div>
      </div>
    </li>
  )
}

function slackSummary(status: { configured: boolean; acknowledgements_configured: boolean; channel_ids: string[]; memory_posts_configured?: boolean } | undefined, failed: boolean): string {
  if (failed) return 'Slack status unavailable'
  if (!status) return 'Checking Slack capture'
  if (!status.configured) return 'Slack capture not configured'
  return `Slack capture on ${status.channel_ids.length} channel${status.channel_ids.length === 1 ? '' : 's'} · receipts ${status.acknowledgements_configured ? 'on' : 'off'} · posts to ${MEMORY_POST_CHANNEL} ${status.memory_posts_configured ? 'on' : 'off'}`
}

function describe(code: string): string {
  const messages: Record<string, string> = {
    memory_quality_rejected: 'The Brain memory gate rejected this capture as too short or vague. Nothing was logged.',
    brain_unavailable: 'The local Brain did not answer. Nothing was logged; try again once it is back.',
    memory_saved_inbox_update_failed: 'The Brain stored the fact but the inbox row did not update. Do not log it again; refresh and check.',
    idea_already_promoted: 'This capture is already logged to memory.',
    idea_not_dismissable: 'Only unreviewed captures can be dismissed.',
    idea_not_restorable: 'Only dismissed captures can be restored.',
    memory_posting_disabled: `Posting to ${MEMORY_POST_CHANNEL} is not enabled on the Mini. Nothing was logged; untick the post option or ask the owner to enable it.`,
    invalid_priority: 'Choose a priority of urgent, high, normal, or low.',
    idea_has_no_content: 'This capture is only a mention with no text to log.',
    memory_inbox_unavailable: 'The memory inbox storage is unavailable.',
    slack_receipts_not_configured: 'Slack receipts are not configured on the Mini.',
    no_failed_receipt: 'No failed receipt to retry.',
    idea_already_assigned: 'This capture already has work queued for it. Open it in Work.',
    idea_not_assignable: 'A dismissed capture cannot be queued as work. Restore it first.',
    assigned_agent_not_found: 'That agent no longer exists. Pick another one.',
    assignment_persistence_failed: 'The assignment was not saved. Nothing was queued; try again.',
  }
  return messages[code] ?? code
}
