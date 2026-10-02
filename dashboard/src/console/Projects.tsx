import { useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ExternalLink, FolderGit2, GitBranch, Plus, RefreshCw } from 'lucide-react'
import type { Agent } from '../lib/api'
import { PageHeader } from './PageHeader'
import { formatRoute } from './studioRoute'

interface EvidenceError {
  code: string
  message: string
}

interface Project {
  id: string
  name: string
  path: string
  mounted: boolean
  checkout_status: 'available' | 'unavailable'
  branch: string | null
  head_sha: string | null
  detached: boolean | null
  github_repo: string | null
  html_url: string | null
  checked_at: string
  errors: EvidenceError[]
  deployment: { status: 'unknown'; connection: 'not_connected' }
}

interface WorkflowRun {
  id: number
  name: string | null
  html_url: string
  head_sha: string
  head_branch: string | null
  status: string
  conclusion: string | null
  created_at: string
  updated_at: string
  fetched_at: string
}

interface ProjectCI {
  project: Project
  status: 'available' | 'unavailable'
  provider: 'github_actions'
  scope: 'repository_recent_runs'
  source_url: string | null
  api_url: string | null
  attempted_at: string
  fetched_at: string | null
  limit: number
  total_count: number | null
  has_more: boolean | null
  runs: WorkflowRun[]
  errors: EvidenceError[]
}

async function get<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, cache: 'no-store' })
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null)
    const detail = payload && typeof payload === 'object' && 'detail' in payload && typeof payload.detail === 'string'
      ? payload.detail : `Request failed (${response.status})`
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

function date(value: string) {
  return new Date(value).toLocaleString()
}

function stateColor(run: WorkflowRun) {
  if (run.status !== 'completed') return 'text-amber-300'
  if (run.conclusion === 'success') return 'text-emerald-300'
  if (['failure', 'timed_out', 'action_required', 'startup_failure'].includes(run.conclusion ?? '')) return 'text-red-300'
  return 'text-neutral-300'
}

function SourceLink({ href, children }: { href: string; children: ReactNode }) {
  return <a href={href} target="_blank" rel="noopener noreferrer" className="inline-flex min-h-11 max-w-full items-center gap-2 text-sm text-cyan-300 hover:underline">
    <span className="min-w-0 break-all">{children}</span><ExternalLink size={14} className="shrink-0" aria-hidden="true" />
  </a>
}

export function Projects() {
  const [selectedId, setSelectedId] = useState('')
  const [agentId, setAgentId] = useState('')
  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: ({ signal }) => get<{ projects: Project[] }>('/api/projects', signal),
    staleTime: 30_000, retry: false, refetchOnWindowFocus: false, refetchInterval: false,
  })
  const available = projects.data?.projects ?? []
  const selected = available.find(project => project.id === selectedId)
    ?? available.find(project => project.mounted) ?? available[0]
  const ci = useQuery({
    queryKey: ['project-ci', selected?.id],
    queryFn: ({ signal }) => get<ProjectCI>(`/api/projects/${encodeURIComponent(selected!.id)}/ci`, signal),
    enabled: !!selected,
    staleTime: 30_000, retry: false, refetchOnWindowFocus: false, refetchInterval: false,
  })
  const agents = useQuery({
    queryKey: ['agents'],
    queryFn: ({ signal }) => get<{ agents: Agent[] }>('/api/agents', signal),
    enabled: !!selected?.mounted,
    staleTime: 30_000, retry: false,
  })
  const scopedAgents = (agents.data?.agents ?? []).filter(agent => agent.repo_id === selected?.id)
  const selectedAgent = scopedAgents.find(agent => agent.id === agentId) ?? scopedAgents[0]
  const project = selected && ci.data?.project.id === selected.id && ci.data.project.checked_at > selected.checked_at
    ? ci.data.project : selected
  const busy = projects.isFetching || ci.isFetching
  const refresh = () => {
    void projects.refetch()
    if (selected) void ci.refetch()
  }

  return <main className="h-full min-h-0 overflow-y-auto bg-neutral-950 text-neutral-200">
    <PageHeader title="Projects" meta={<span>Mounted checkouts and GitHub Actions</span>} actions={
      <button type="button" onClick={refresh} disabled={busy} title="Refresh projects and CI" aria-label="Refresh projects and CI" className="flex h-11 w-11 shrink-0 items-center justify-center rounded border border-neutral-700 hover:border-neutral-400 disabled:opacity-40">
        <RefreshCw size={17} className={busy ? 'animate-spin' : ''} />
      </button>
    } />
    {projects.isPending && <p role="status" className="px-7 py-6">Loading projects...</p>}
    {projects.isError && <p role="alert" className="px-7 py-4 text-sm text-red-300">Projects unavailable: {projects.error.message}. Check Command Center connectivity, then refresh.</p>}
    {projects.isSuccess && available.length === 0 && <p className="px-7 py-6 text-sm text-neutral-400">No repositories configured.</p>}
    {project && <>
      <section aria-label="Project checkout" className="border-b border-neutral-800 px-5 py-5 sm:px-7">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <label className="flex min-w-0 max-w-full flex-col gap-2 text-xs text-neutral-400">
            Repository
            <select aria-label="Repository" value={selected?.id} onChange={event => { setSelectedId(event.target.value); setAgentId('') }} className="h-11 w-72 max-w-full rounded border border-neutral-700 bg-neutral-900 px-3 text-sm text-white">
              {available.map(item => <option key={item.id} value={item.id}>{item.name}{item.mounted ? '' : ' (not mounted)'}</option>)}
            </select>
          </label>
          {project.html_url && <SourceLink href={project.html_url}>{project.github_repo}</SourceLink>}
        </div>
        <div className="mt-5 flex min-w-0 items-start gap-2">
          <FolderGit2 size={17} className="mt-0.5 shrink-0 text-neutral-400" aria-hidden="true" />
          <span className="break-all font-mono text-xs text-neutral-400">{project.path}</span>
        </div>
        <dl className="mt-5 grid min-w-0 grid-cols-1 gap-5 text-sm md:grid-cols-2 xl:grid-cols-3">
          <div className="min-w-0"><dt className="text-neutral-500">Checkout branch</dt><dd className="mt-1 flex items-start gap-2"><GitBranch size={16} className="mt-0.5 shrink-0" aria-hidden="true" /><span className="break-all">{project.detached ? 'Detached HEAD' : project.branch ?? 'Unavailable'}</span></dd></div>
          <div className="min-w-0"><dt className="text-neutral-500">Checkout commit</dt><dd className="mt-1 break-all font-mono text-xs">
            {project.head_sha && project.html_url ? <SourceLink href={`${project.html_url}/commit/${project.head_sha}`}>{project.head_sha}</SourceLink> : project.head_sha ?? 'Unavailable'}
          </dd></div>
          <div><dt className="text-neutral-500">Deployment</dt><dd className="mt-1 text-neutral-400">Unknown / not connected</dd></div>
        </dl>
        <p className="mt-4 text-xs text-neutral-500">Checkout checked <time dateTime={project.checked_at}>{date(project.checked_at)}</time></p>
        {project.errors.map(error => <p key={error.code} role="alert" className="mt-3 text-sm text-amber-300">{error.message}</p>)}
      </section>

      <section aria-label="GitHub Actions evidence" className="border-b border-neutral-800 px-5 py-5 sm:px-7">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-medium text-white">Recent workflow runs</h2>
          {ci.data?.source_url && <SourceLink href={ci.data.source_url}>GitHub Actions</SourceLink>}
        </div>
        <p className="mt-2 text-xs text-neutral-400">All repository branches / latest {ci.data?.limit ?? 20} runs</p>
        {ci.isPending && <p role="status" className="py-5 text-sm">Loading CI evidence...</p>}
        {ci.isFetching && !ci.isPending && <p role="status" className="mt-3 text-xs text-neutral-400">Refreshing evidence...</p>}
        {ci.isError && <p role="alert" className="py-5 text-sm text-red-300">CI evidence unavailable: {ci.error.message}. Check Command Center connectivity, then refresh.</p>}
        {!ci.isError && ci.data?.status === 'unavailable' && <div role="alert" className="py-5 text-sm text-amber-300">
          <p className="font-medium">CI evidence unavailable</p>
          {ci.data.errors.map(error => <p key={error.code} className="mt-2">{error.message}</p>)}
          <p className="mt-3 text-xs text-neutral-500">Last attempt <time dateTime={ci.data.attempted_at}>{date(ci.data.attempted_at)}</time></p>
        </div>}
        {!ci.isError && ci.data?.status === 'available' && <>
          <p className="mt-3 text-xs text-neutral-500">Fetched <time dateTime={ci.data.fetched_at!}>{date(ci.data.fetched_at!)}</time></p>
          {ci.data.runs.length === 0 ? <p className="py-6 text-sm text-neutral-400">No workflow runs reported by GitHub.</p> :
            <ul className="mt-4 divide-y divide-neutral-800">
              {ci.data.runs.map(run => <li key={run.id} className="grid min-w-0 grid-cols-1 gap-3 py-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)]">
                <div className="min-w-0">
                  <SourceLink href={run.html_url}>{run.name ?? 'Workflow'} #{run.id}</SourceLink>
                  <p className={`mt-1 break-words text-sm ${stateColor(run)}`}>{run.status.replaceAll('_', ' ')}{run.status === 'completed' ? ` / ${run.conclusion?.replaceAll('_', ' ') ?? 'conclusion unavailable'}` : ''}</p>
                </div>
                <div className="min-w-0 text-xs">
                  <p className="break-all text-neutral-300">Source branch: {run.head_branch ?? 'Unavailable'}</p>
                  <p className="mt-2 break-all font-mono text-neutral-400">{run.head_sha}</p>
                  <p className="mt-2 text-neutral-500">{project.head_sha ? run.head_sha === project.head_sha ? 'Matches checkout commit' : 'Different from checkout commit' : 'Checkout comparison unavailable'}</p>
                </div>
                <div className="text-xs text-neutral-400">
                  <p>Created <time dateTime={run.created_at}>{date(run.created_at)}</time></p>
                  <p className="mt-2">Updated <time dateTime={run.updated_at}>{date(run.updated_at)}</time></p>
                </div>
              </li>)}
            </ul>}
          {ci.data.has_more && <p className="mt-3 text-xs text-neutral-400">Showing {ci.data.runs.length} of {ci.data.total_count?.toLocaleString()} runs. Older evidence is on GitHub.</p>}
        </>}
      </section>

      {project.mounted && <section aria-label="Project work" className="px-5 py-5 sm:px-7">
        <h2 className="text-base font-medium text-white">Project work</h2>
        {agents.isPending && <p role="status" className="mt-3 text-sm text-neutral-400">Loading repository agents...</p>}
        {agents.isError && <div role="alert" className="mt-3 flex items-center gap-3 text-sm text-red-300">Repository agents unavailable.
          <button type="button" onClick={() => void agents.refetch()} aria-label="Refresh repository agents" title="Refresh repository agents" className="flex h-11 w-11 items-center justify-center"><RefreshCw size={16} /></button>
        </div>}
        {!agents.isError && selectedAgent && <div className="mt-4 flex flex-wrap items-end gap-4">
          <label className="flex min-w-0 max-w-full flex-col gap-2 text-xs text-neutral-400">Repository agent
            <select aria-label="Repository agent" value={selectedAgent.id} onChange={event => setAgentId(event.target.value)} className="h-11 w-64 max-w-full rounded border border-neutral-700 bg-neutral-900 px-3 text-sm text-white">
              {scopedAgents.map(agent => <option key={agent.id} value={agent.id}>{agent.name}</option>)}
            </select>
          </label>
          <a href={formatRoute({ view: 'assignments', agentId: selectedAgent.id })} className="inline-flex min-h-11 items-center gap-2 rounded border border-neutral-600 px-3 text-sm text-white hover:border-cyan-300"><Plus size={16} aria-hidden="true" />Create work</a>
        </div>}
        {agents.isSuccess && scopedAgents.length === 0 && <p className="mt-3 text-sm text-neutral-400">No agent connected to this repository. <a href={formatRoute({ view: 'agents' })} className="inline-flex min-h-11 items-center text-cyan-300 hover:underline">Open agents</a></p>}
      </section>}
    </>}
  </main>
}
