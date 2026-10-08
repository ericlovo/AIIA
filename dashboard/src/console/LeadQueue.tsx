import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, RefreshCw } from 'lucide-react'
import { api, type Agent, type LeadDecision, type LeadSignal } from '../lib/api'
import { LeadQualification } from './LeadQualification'
import { AgentPicker } from './AgentPicker'
import { activeAgents } from './agentRoster'
import { formatRoute } from './studioRoute'

const LABELS: Record<LeadDecision, string> = { unreviewed: 'Not reviewed', research: 'Needs research', watch: 'Watch', qualified: 'Qualified for follow-up', rejected: 'Not a fit' }

export function LeadQueue() {
  const [status, setStatus] = useState<LeadDecision | 'all'>('all')
  const [search, setSearch] = useState('')
  const [company, setCompany] = useState('')
  const [offset, setOffset] = useState(0)
  const queue = useQuery({ queryKey: ['lead-queue', status, company, offset], queryFn: () => api.leadQueue(status, company, offset), retry: false, refetchOnWindowFocus: false })
  const agents = useQuery({ queryKey: ['agents'], queryFn: api.agents })
  const groups = new Map<string, { name: string; leads: LeadSignal[] }>()
  for (const lead of queue.data?.leads ?? []) {
    const key = lead.company.trim().toLowerCase() || `unknown:${lead.id}`
    const group = groups.get(key) ?? { name: lead.company || 'Company not identified', leads: [] }
    group.leads.push(lead)
    groups.set(key, group)
  }
  return <section aria-label="Lead queue" className="border-t border-neutral-800">
    <div className="space-y-4 px-5 py-5 sm:px-7">
      <div className="flex items-center justify-between"><h2 className="text-lg text-white">Lead queue</h2><button type="button" aria-label="Refresh lead queue" title="Refresh lead queue" className="flex h-11 w-11 items-center justify-center" onClick={() => void queue.refetch()}><RefreshCw size={18} /></button></div>
      <div className="flex flex-wrap items-end gap-4">
        <label className="text-sm">Qualification<select aria-label="Qualification filter" className="mt-1 block min-h-11 max-w-full border border-neutral-700 bg-neutral-900 px-3" value={status} onChange={event => { setStatus(event.target.value as typeof status); setOffset(0) }}><option value="all">All decisions</option>{Object.entries(LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <form className="flex min-w-0 flex-wrap items-end gap-2" onSubmit={event => { event.preventDefault(); setCompany(search.trim()); setOffset(0) }}><label className="text-sm">Company<input className="mt-1 block min-h-11 w-full border border-neutral-700 bg-neutral-900 px-3" maxLength={200} value={search} onChange={event => setSearch(event.target.value)} /></label><button className="min-h-11 border border-neutral-700 px-3">Search</button></form>
      </div>
      {queue.isPending && <p role="status">Loading leads...</p>}
      {agents.isError && <p role="alert" className="text-red-300">Research agents unavailable. <button className="underline" onClick={() => void agents.refetch()}>Retry agents</button></p>}
      {agents.data && activeAgents(agents.data.agents).length === 0 && <p className="text-sm text-neutral-400">No research agents available.</p>}
      {queue.isError && <p role="alert" className="text-red-300">Lead queue unavailable. Refresh to retry.</p>}
      {queue.data && !queue.isError && <p className="text-xs text-neutral-400">{queue.data.total} matching signals · {queue.data.leads.length ? offset + 1 : 0}-{offset + queue.data.leads.length} shown · grouped by company name on this page</p>}
    </div>
    {!queue.isError && queue.data && <>
      {queue.data.leads.length === 0 && <p className="px-7 py-5 text-sm text-neutral-400">No signals match this view.</p>}
      {[...groups.entries()].map(([key, group]) => <section key={key} aria-label={group.name} className="border-t border-neutral-800 px-5 py-5 sm:px-7"><h3 className="break-words text-base font-medium text-white">{group.name}</h3><ul className="divide-y divide-neutral-800">{group.leads.map(lead => <LeadRow key={lead.id} lead={lead} agents={agents.data?.agents ?? []} />)}</ul></section>)}
      <div className="flex items-center gap-4 border-t border-neutral-800 px-7 py-4"><button aria-label="Previous lead page" title="Previous lead page" className="flex h-11 w-11 items-center justify-center disabled:opacity-30" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}><ArrowLeft size={18} /></button><button aria-label="Next lead page" title="Next lead page" className="flex h-11 w-11 items-center justify-center disabled:opacity-30" disabled={offset + queue.data.limit >= queue.data.total} onClick={() => setOffset(offset + 25)}><ArrowRight size={18} /></button></div>
    </>}
  </section>
}

function LeadRow({ lead, agents }: { lead: LeadSignal; agents: Agent[] }) {
  const [agentId, setAgentId] = useState('')
  const client = useQueryClient()
  const assign = useMutation({ mutationFn: () => api.assignCapture(lead.id, agentId), onSuccess: () => {
    void client.invalidateQueries({ queryKey: ['lead-queue'] })
    void client.invalidateQueries({ queryKey: ['assignments'] })
    void client.invalidateQueries({ queryKey: ['memory-inbox'] })
  } })
  const assignmentId = assign.data?.assignment.id || lead.assignment_id
  return <li className="min-w-0 py-4 text-sm">
    <div className="flex flex-wrap gap-3 text-xs"><span className={lead.decision === 'qualified' ? 'text-emerald-300' : 'text-amber-200'}>{LABELS[lead.decision]}</span><span className="text-neutral-400">{new Date(lead.created_at).toLocaleDateString()} · Inbox: {lead.inbox_status === 'unreviewed' ? 'open' : lead.inbox_status}</span></div>
    <p className="mt-3 max-w-3xl whitespace-pre-wrap break-words text-neutral-300">{lead.text}</p>
    {lead.review && <dl className="mt-3 max-w-3xl space-y-2 break-words text-neutral-400"><div><dt className="text-xs">Account fit</dt><dd>{lead.review.account_fit || 'Not recorded'}</dd></div><div><dt className="text-xs">Observed change</dt><dd>{lead.review.observed_change || 'Not recorded'}</dd></div><div><dt className="text-xs">Evidence URL</dt><dd>{/^https:\/\//i.test(lead.review.evidence_url) ? <a href={lead.review.evidence_url} target="_blank" rel="noopener noreferrer" className="text-cyan-200 underline">{lead.review.evidence_url}</a> : 'Not recorded'}</dd></div></dl>}
    <LeadQualification ideaId={lead.id} />
    {assignmentId ? <a href={formatRoute({ view: 'assignments', assignmentId })} className="inline-flex min-h-11 items-center text-cyan-200 underline">Open research assignment</a> : lead.inbox_status !== 'dismissed' && <div className="mt-3 flex flex-wrap items-end gap-3"><label>Research agent<AgentPicker agents={agents} value={agentId} onChange={setAgentId} placeholder="Choose agent" className="mt-1 block min-h-11 max-w-full border border-neutral-700 bg-neutral-900 px-3" aria-label={`Research agent for ${lead.id}`} /></label><button disabled={!agentId || assign.isPending} onClick={() => assign.mutate()} className="min-h-11 border border-cyan-600 px-3 text-cyan-200 disabled:opacity-40">{assign.isPending ? 'Queuing...' : 'Queue research'}</button></div>}
    {assign.isError && <p role="alert" className="mt-3 text-red-300">Research was not confirmed queued. Refresh the queue before retrying.</p>}
    {assign.isSuccess && <p role="status" className="mt-2 text-emerald-300">Research queued, not started.</p>}
  </li>
}
