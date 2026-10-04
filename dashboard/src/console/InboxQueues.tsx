import { useQuery } from '@tanstack/react-query'
import { ArrowRight, ClipboardCheck, Inbox, MessageSquare } from 'lucide-react'
import { api } from '../lib/api'
import { attentionSummary } from './assignmentReview'
import type { InboxSource } from './studioRoute'
import { navigate } from './useStudioRoute'

export function InboxQueues({ source }: { source?: InboxSource }) {
  const slack = useQuery({ queryKey: ['memory-inbox', 'counts', 'slack'], queryFn: () => api.memoryInbox({ source: 'slack', status: 'unreviewed' }), retry: false, refetchInterval: 15_000 })
  const proposals = useQuery({ queryKey: ['memory-inbox', 'counts', 'loops', 'open'], queryFn: () => api.memoryInbox({ source: 'local_proposals', status: 'unreviewed', outcome: 'open' }), retry: false, refetchInterval: 15_000 })
  const assignments = useQuery({ queryKey: ['assignments'], queryFn: api.assignments, retry: false, refetchInterval: 5_000 })
  const workspaces = useQuery({ queryKey: ['git-workspaces'], queryFn: api.gitWorkspaces, retry: false, refetchInterval: 10_000 })
  const writes = useQuery({ queryKey: ['git-writes'], queryFn: () => api.gitWrites(), retry: false, refetchInterval: 10_000 })
  const workError = assignments.isError || workspaces.isError || writes.isError
  const workPending = assignments.isPending || workspaces.isPending || writes.isPending
  const summary = attentionSummary(assignments.data?.assignments ?? [], workspaces.data?.workspaces, writes.data?.writes)
  const queues = [
    { id: 'slack', label: 'Slack', detail: 'Unreviewed captures', icon: MessageSquare, href: '#/inbox?source=slack', count: slack.isError ? 'Unavailable' : slack.data?.counts?.unreviewed ?? 'Loading' },
    { id: 'loops', label: 'Proposals', detail: 'Pending findings', icon: Inbox, href: '#/inbox?source=loops', count: proposals.isError ? 'Unavailable' : proposals.data?.counts?.unreviewed ?? 'Loading' },
    { id: 'work', label: 'Work review', detail: 'Reports, failures, approvals', icon: ClipboardCheck, href: '#/history?attention=1', count: workError ? 'Unavailable' : workPending ? 'Loading' : summary.total },
  ]
  return <nav aria-label="Inbox queues" className="grid grid-cols-3 divide-x divide-neutral-800 border-b border-neutral-800">
    {queues.map(({ id, label, detail, icon: Icon, href, count }) => <a key={id} href={href} aria-label={`${label}: ${count}. ${detail}`} aria-current={source === id ? 'page' : undefined} onClick={event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
      event.preventDefault()
      navigate(id === 'work' ? { view: 'history', attention: true } : { view: 'inbox', source: id as 'slack' | 'loops' })
    }} className={`flex min-w-0 items-center gap-3 px-3 py-4 focus-visible:outline-2 focus-visible:outline-emerald-300 sm:px-7 ${source === id ? 'bg-neutral-900' : 'hover:bg-neutral-900/60'}`}>
      <Icon size={18} aria-hidden="true" className="hidden shrink-0 text-neutral-400 md:block" />
      <div className="min-w-0 flex-1"><div className="flex flex-col gap-x-3 gap-y-1 text-sm text-white sm:flex-row sm:flex-wrap sm:items-baseline"><span>{label}</span><span className={`font-medium tabular-nums text-amber-200 ${typeof count === 'string' ? 'text-xs' : ''}`}>{count}</span></div><p className="mt-1 hidden text-xs text-neutral-400 sm:block">{detail}</p></div>
      <ArrowRight size={16} aria-hidden="true" className="hidden shrink-0 text-neutral-500 sm:block" />
    </a>)}
  </nav>
}
