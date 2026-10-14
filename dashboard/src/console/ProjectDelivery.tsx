import { useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { formatRoute } from './studioRoute'
import { deliverySignals, hasObservedWorkflowFailure, parsePRNumbers, type DeliveryFeature, type ProjectDeliveryData } from './deliveryEvidence'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { cache: 'no-store', ...init })
  const payload = await response.json().catch(() => null)
  if (!response.ok) throw new Error(typeof payload?.detail === 'string' ? payload.detail : `Request failed (${response.status})`)
  return payload as T
}
const linkClass = 'inline-flex min-h-11 max-w-full items-center break-all text-sm text-cyan-300 hover:underline'
const buttonClass = 'min-h-11 rounded border border-neutral-700 px-3 text-sm hover:border-cyan-300 disabled:opacity-40'
function Timestamp({ value }: { value?: string }) {
  return value ? <time dateTime={value}>{new Date(value).toLocaleString()}</time> : <>Unavailable</>
}
function Feature({ feature, data, repoId, saved }: { feature: DeliveryFeature; data: ProjectDeliveryData; repoId: string; saved: () => Promise<unknown> }) {
  const [editing, setEditing] = useState(false)
  const [ids, setIds] = useState(feature.assignment_ids)
  const [numbers, setNumbers] = useState(feature.pr_numbers.join(', '))
  const [error, setError] = useState('')
  const [expectedUpdatedAt, setExpectedUpdatedAt] = useState(feature.updated_at)
  const [saving, setSaving] = useState(false)
  const candidates = data.assignment_candidates ?? []
  const save = async () => {
    setError(''); setSaving(true)
    try {
      const pr_numbers = parsePRNumbers(numbers)
      await request(`/api/projects/${encodeURIComponent(repoId)}/features/${encodeURIComponent(feature.id)}/links`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ assignment_ids: ids, pr_numbers, expected_updated_at: expectedUpdatedAt }),
      })
      setEditing(false)
      await saved()
    } catch (reason) { setError(reason instanceof Error && reason.message === 'delivery_links_changed_refresh_required' ? 'This feature changed. Cancel, refresh delivery, and reopen the editor before saving.' : reason instanceof Error ? reason.message : 'Save failed') }
    finally { setSaving(false) }
  }
  return <article aria-label={feature.title} className="min-w-0 rounded-lg border border-neutral-800 bg-neutral-900/40 p-4 sm:p-5">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0"><h3 className="break-words text-base font-medium text-white">{feature.title}</h3><p className="mt-2 text-xs text-neutral-400">Roadmap: {feature.status} / {feature.priority}</p></div>
      <button type="button" className={buttonClass} onClick={() => { setExpectedUpdatedAt(feature.updated_at); setIds(feature.assignment_ids); setNumbers(feature.pr_numbers.join(', ')); setError(''); setEditing(!editing) }}>Edit links</button>
    </div>
    <p className="mt-3 text-sm text-amber-200">{deliverySignals(feature).join(' / ')}</p>
    {hasObservedWorkflowFailure(feature) && <p className="mt-2 text-xs text-amber-300">Failed workflow observed at a linked PR head. Required-check readiness is unverified.</p>}
    {feature.description && <p className="mt-3 whitespace-pre-wrap break-words text-sm text-neutral-400">{feature.description}</p>}
    {editing && <form aria-label={`Links for ${feature.title}`} className="mt-4 space-y-3 rounded border border-neutral-700 p-3" onSubmit={event => { event.preventDefault(); void save() }}>
      <p className="text-xs text-neutral-400">Choose existing assignments explicitly. These choices span repositories; confirm they belong to this feature. Saving replaces this feature’s links.</p>
      <fieldset disabled={saving} className="min-w-0"><legend className="text-sm">Assignments</legend>
        {candidates.map(a => <label key={a.id} className="flex min-h-11 items-center gap-3 break-words text-sm"><input type="checkbox" checked={ids.includes(a.id)} onChange={event => setIds(event.target.checked ? [...ids, a.id] : ids.filter(id => id !== a.id))} /><span className="min-w-0 break-all">{a.title || a.id} / {a.status} <span className="text-neutral-500">({a.id})</span></span></label>)}
        {!candidates.length && <p className="py-3 text-xs text-neutral-400">No assignment choices available.</p>}
        {ids.filter(id => !candidates.some(a => a.id === id)).map(id => <label key={id} className="flex min-h-11 items-center gap-3 break-all text-sm"><input type="checkbox" checked onChange={() => setIds(ids.filter(value => value !== id))} />Unavailable assignment: {id}</label>)}
        <label className="mt-3 block text-sm">PR numbers (comma separated)<input value={numbers} onChange={event => setNumbers(event.target.value)} className="mt-2 h-11 w-full rounded border border-neutral-600 bg-neutral-950 px-3" placeholder="12, 34" /></label>
      </fieldset>
      {error && <p role="alert" className="break-words text-sm text-red-300">Links were not saved: {error}</p>}
      <div className="flex gap-3"><button className={buttonClass} disabled={saving} type="submit">{saving ? 'Saving...' : 'Save links'}</button><button type="button" className={buttonClass} disabled={saving} onClick={() => setEditing(false)}>Cancel</button></div>
    </form>}
    <div className="mt-4 grid min-w-0 gap-5 lg:grid-cols-2">
      <div className="min-w-0"><h4 className="text-xs font-medium uppercase tracking-wide text-neutral-500">Agent work</h4>
        {!feature.assignments.length && <p className="mt-3 text-sm text-neutral-400">No assignment linked.</p>}
        {feature.assignments.map(a => <div key={a.id} className="mt-3 min-w-0 border-t border-neutral-800 pt-2">
          <a className={linkClass} href={formatRoute({ view: 'assignments', assignmentId: a.id })}>{a.title || a.id}</a>
          <p className="text-sm">Run: {a.status} / Review: {a.review_status || 'unrecorded'}</p>
          {a.agent_id && <a className={linkClass} href={formatRoute({ view: 'agents', agentId: a.agent_id })}>{a.agent_name || a.agent_id}{a.agent_status ? ` / ${a.agent_status}` : ''}</a>}
        </div>)}
      </div>
      <div className="min-w-0"><h4 className="text-xs font-medium uppercase tracking-wide text-neutral-500">PR and check evidence</h4>
        {!feature.pull_requests.length && <p className="mt-3 text-sm text-neutral-400">No PR linked.</p>}
        {feature.pull_requests.map(pr => <div key={pr.number} className="mt-3 min-w-0 border-t border-neutral-800 pt-2">
          {pr.html_url ? <a className={linkClass} href={pr.html_url} target="_blank" rel="noopener noreferrer">#{pr.number} {pr.title}</a> : <p className="py-3 text-sm">PR #{pr.number} / unavailable</p>}
          {pr.status === 'available' && <p className="text-sm">{pr.merged ? 'Merged' : pr.draft ? 'Draft' : pr.state || 'State unavailable'}</p>}
          {pr.mergeable === false && <p className="mt-2 text-sm text-amber-300">Merge conflict reported</p>}
          {pr.head_sha && <p className="mt-2 break-all font-mono text-xs text-neutral-400">PR head: {pr.head_sha}</p>}
          <p className="mt-2 text-xs text-neutral-500">PR fetched <Timestamp value={pr.fetched_at} /></p>
          {pr.errors.map((e, i) => <p key={i} role="alert" className="mt-2 text-sm text-amber-300">{e.message}</p>)}
          {pr.ci && <><p className="mt-3 text-xs text-neutral-400">GitHub Actions workflows for this PR head / {pr.ci.status}</p><p className="mt-1 text-xs text-neutral-500">Workflows fetched <Timestamp value={pr.ci.fetched_at} /></p>
            {pr.ci.errors.map((e, i) => <p key={i} role="alert" className="mt-2 text-sm text-amber-300">{e.message}</p>)}
            {pr.ci.runs.filter(run => run.head_sha === pr.head_sha).map(run => <div key={run.id} className="mt-2 text-xs"><a className={linkClass} href={run.html_url} target="_blank" rel="noopener noreferrer">{run.name || 'Workflow'} #{run.id}</a><p>{run.status} / {run.conclusion || 'No conclusion'}</p><p className="mt-1 text-neutral-500">Updated <Timestamp value={run.updated_at} /></p></div>)}
            {pr.ci.status === 'available' && !pr.ci.runs.some(run => run.head_sha === pr.head_sha) && <p className="mt-2 text-xs text-neutral-400">No workflow evidence for this PR head.</p>}
            {pr.ci.has_more && <p className="mt-2 text-xs text-neutral-400">More checks may exist on GitHub.</p>}
          </>}
        </div>)}
      </div>
    </div>
    <p className="mt-5 border-t border-neutral-800 pt-3 text-xs text-neutral-500">Feature release: Unverified / no feature-to-deployment mapping.</p>
  </article>
}
export function ProjectDelivery({ repoId }: { repoId: string }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 60_000); return () => clearInterval(timer) }, [])
  const queryClient = useQueryClient()
  const queryKey = ['project-delivery', repoId]
  const delivery = useQuery({ queryKey, queryFn: ({ signal }) => request<ProjectDeliveryData>(`/api/projects/${encodeURIComponent(repoId)}/delivery`, { signal }), retry: false, staleTime: 30_000, refetchOnWindowFocus: false })
  const [filter, setFilter] = useState('All')
  const data = delivery.data
  const features = data?.features ?? []
  return <section aria-label="Feature delivery" className="border-b border-neutral-800 px-5 py-5 sm:px-7">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-base font-medium text-white">Feature delivery</h2><button type="button" disabled={delivery.isFetching} onClick={() => void delivery.refetch()} className={buttonClass}>Refresh delivery</button></div>
    <p className="mt-2 text-sm text-neutral-400">Roadmap features with explicitly linked agent work and PR evidence. Completed runs still need review; merged PRs do not establish deployment.</p>
    {delivery.isPending && <p role="status" className="mt-4 text-sm">Loading feature delivery...</p>}
    {delivery.isError && <p role="alert" className="mt-4 text-sm text-red-300">Delivery unavailable: {delivery.error.message}. Refresh to retry.</p>}
    {!delivery.isError && data && <>
      <p className="mt-3 text-xs text-neutral-500">Sources checked <Timestamp value={data.checked_at} /> / {data.status}{now - Date.parse(data.checked_at) > 300_000 ? ' / Snapshot older than 5 minutes; refresh for current evidence.' : ''}</p>
      {Object.entries(data.sources ?? {}).map(([name, source]) => <p key={name} className="mt-1 text-xs text-neutral-500">{name}: {source.status} / <Timestamp value={source.fetched_at} /></p>)}
      {data.deployments && <div aria-label="Deployment evidence" className="mt-4 rounded border border-neutral-800 p-3">
        <h3 className="text-sm font-medium">Last reported deployments</h3>
        <p className="mt-2 text-xs text-neutral-400">Provider reports, not a live health probe. Checked <Timestamp value={data.deployments.checked_at} /> / {data.deployments.status}</p>
        {data.deployments.notice && <p className="mt-2 text-xs text-neutral-400">{data.deployments.notice}</p>}
        {data.deployments.environments.map(environment => <div key={environment.environment} className="mt-3 text-sm"><p>{environment.environment} / {environment.state}</p><a className={linkClass} href={environment.source_url} target="_blank" rel="noopener noreferrer">Deployment source</a><p className="break-all font-mono text-xs">{environment.sha}</p><p className="mt-1 text-xs text-neutral-500">Updated <Timestamp value={environment.updated_at} /></p></div>)}
        {!data.deployments.environments.length && <p className="mt-2 text-xs text-neutral-400">No deployment evidence available.</p>}
        {data.deployments.errors.map((error, i) => <p key={i} role="alert" className="mt-2 text-sm text-amber-300">{error.message}</p>)}
      </div>}
      {data.errors.map((error, i) => <p key={i} role="alert" className="mt-3 text-sm text-amber-300">{error.message}</p>)}
      <div className="mt-4 flex flex-wrap gap-2" aria-label="Delivery filters">{['All', 'Moving', 'Awaiting start', 'Blocked', 'Needs review', 'Unlinked', 'Broken link'].map(label => <button type="button" key={label} aria-pressed={filter === label} className={`${buttonClass} ${filter === label ? 'border-cyan-500 text-cyan-200' : ''}`} onClick={() => setFilter(label)}>{label} ({features.filter(feature => label === 'All' || deliverySignals(feature).includes(label)).length})</button>)}</div>
      {!features.length && <p className="py-5 text-sm text-neutral-400">No roadmap features available for this repository.</p>}
      <div className="mt-4 space-y-4">{features.filter(feature => filter === 'All' || deliverySignals(feature).includes(filter)).map(feature => <Feature key={feature.id} feature={feature} data={data} repoId={repoId} saved={() => queryClient.invalidateQueries({ queryKey })} />)}</div>
    </>}
  </section>
}
