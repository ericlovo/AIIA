import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, ArrowRight, Check, Clock3, FlaskConical, Loader2, Pause, Play, Plus, RefreshCw, X } from 'lucide-react'
import { api, type Agent, type Assignment } from '../lib/api'
import { navigate } from './useStudioRoute'
import { buildJob, canResumeJob, JOB_RECIPES, jobError, jobState, jobTestDefinition, jobTestPassed, jobTime, latestJobTest, nextJobCheck, runsToday, type JobDraft } from './jobHelpers'

export interface JobsProps {
  agents: Agent[]
  loading: boolean
  agentError: boolean
}

const control = 'inline-flex min-h-11 items-center justify-center gap-2 rounded border border-neutral-700 px-3 py-2 text-sm text-neutral-200 hover:border-neutral-500 hover:bg-neutral-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-400 disabled:cursor-not-allowed disabled:opacity-50'
const primary = `${control} border-emerald-600 bg-emerald-950 text-emerald-200 hover:border-emerald-400`
const input = 'min-h-11 w-full min-w-0 rounded border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 focus:outline-2 focus:outline-emerald-400'
const initialDraft: JobDraft = { recipeId: JOB_RECIPES[0].id, name: JOB_RECIPES[0].name, repoId: '', interval: '240', cap: '3' }

export function Jobs({ agents, loading, agentError }: JobsProps) {
  const qc = useQueryClient()
  const resources = useQuery({ queryKey: ['agent-resources'], queryFn: api.agentResources, retry: false })
  const tasks = useQuery({ queryKey: ['tasks'], queryFn: api.tasks, refetchInterval: 30_000, retry: false })
  const work = useQuery({ queryKey: ['assignments'], queryFn: api.assignments, refetchInterval: 10_000, retry: false })
  const [context, setContext] = useState<string | null>(null)
  const [draft, setDraft] = useState<JobDraft>(initialDraft)
  const [confirmed, setConfirmed] = useState<Agent | null>(null)
  const [notice, setNotice] = useState('')
  const [inspectedId, setInspectedId] = useState('')
  const heading = useRef<HTMLHeadingElement>(null)
  useEffect(() => { if (context) heading.current?.focus() }, [context])

  const selectedFromProps = agents.find(agent => agent.id === context)
  const selected = confirmed?.id === context && (!selectedFromProps || confirmed.updated_at >= selectedFromProps.updated_at) ? confirmed : selectedFromProps
  const jobs = agents.filter(agent => agent.loop_task.trim())
  const repos = resources.data?.repos ?? []
  const selectedRepo = repos.find(repo => repo.id === draft.repoId)
  const assignment = selected ? latestJobTest(selected, work.data?.assignments ?? []) : undefined
  const testPassed = jobTestPassed(assignment)

  function saveAgent(agent: Agent) {
    setConfirmed(agent)
    qc.setQueryData<{ agents: Agent[] }>(['agents'], old => ({ agents: old?.agents.some(item => item.id === agent.id) ? old.agents.map(item => item.id === agent.id ? agent : item) : [...(old?.agents ?? agents), agent] }))
  }
  function saveAssignment(item: Assignment) {
    qc.setQueryData<{ assignments: Assignment[] }>(['assignments'], old => ({ assignments: [item, ...(old?.assignments ?? []).filter(existing => existing.id !== item.id)] }))
  }
  function refresh() {
    void qc.invalidateQueries({ queryKey: ['agents'] })
    void qc.invalidateQueries({ queryKey: ['assignments'] })
    void qc.invalidateQueries({ queryKey: ['studio-activity'] })
  }

  const create = useMutation({
    mutationFn: () => api.createAgent(buildJob(draft, repos)),
    onSuccess: ({ agent }) => { saveAgent(agent); setContext(agent.id); setNotice('Job saved paused.'); refresh() },
  })
  const toggle = useMutation({
    mutationFn: ({ agent, enabled }: { agent: Agent; enabled: boolean }) => api.setAgentLoop(agent.id, enabled),
    onSuccess: ({ agent }) => { saveAgent(agent); setNotice(agent.loop_enabled ? `${agent.name}: interval checks enabled.` : `${agent.name}: future checks paused. Any current run continues.`) },
    onSettled: refresh,
  })
  const test = useMutation({
    mutationFn: async ({ agent, existing }: { agent: Agent; existing?: Assignment }) => {
      const item = existing && ['queued', 'failed'].includes(existing.status) ? existing : (await api.createAssignment(jobTestDefinition(agent))).assignment
      saveAssignment(item)
      const response = await api.runAssignment(item.id)
      saveAssignment(response.assignment)
      saveAgent(response.agent)
      return response
    },
    onSuccess: ({ assignment: item }) => setNotice(item.status === 'completed' ? 'Test saved in Work. Review the evidence before enabling.' : 'Test did not complete. Open its saved work.'),
    onSettled: refresh,
  })
  const busy = create.isPending || toggle.isPending || test.isPending

  function openContext(next: string | null) {
    if (busy) return
    setContext(next)
    setInspectedId('')
    create.reset(); toggle.reset(); test.reset()
    setNotice('')
    if (next === 'new') setDraft(initialDraft)
  }

  return <main className="mx-auto h-full min-h-0 w-full max-w-6xl space-y-8 overflow-y-auto p-4 text-sm leading-6 text-neutral-300 sm:p-6">
    <header className="flex flex-wrap items-center justify-between gap-3">
      <h1 className="text-2xl font-semibold text-white">Jobs</h1>
      <div className="flex gap-2">
        <button type="button" className={control} aria-label="Refresh jobs" title="Refresh jobs" disabled={busy} onClick={() => { refresh(); void tasks.refetch(); void resources.refetch() }}><RefreshCw size={18} /></button>
        {!context && <button type="button" className={primary} onClick={() => openContext('new')}><Plus size={18} />New job</button>}
      </div>
    </header>

    {notice && <p role="status" className="flex items-start gap-2 text-emerald-300"><Check className="mt-1 shrink-0" size={16} />{notice}</p>}
    {toggle.error && <ErrorMessage message={`Schedule change not confirmed: ${jobError(toggle.error)}`} />}

    {context === 'new' ? <section aria-labelledby="job-context-heading" className="border-y border-neutral-800 py-6">
      <ContextHeading title="New job" busy={busy} headingRef={heading} onClose={() => openContext(null)} />
      <form className="mt-5 max-w-2xl space-y-5" onSubmit={event => { event.preventDefault(); if (!busy) create.mutate() }}>
        <label className="block space-y-2"><span>Recipe</span><select autoFocus className={input} value={draft.recipeId} onChange={event => { const recipe = JOB_RECIPES.find(item => item.id === event.target.value)!; setDraft({ ...draft, recipeId: recipe.id, name: recipe.name }) }} disabled={busy}>{JOB_RECIPES.map(recipe => <option key={recipe.id} value={recipe.id}>{recipe.name}</option>)}</select></label>
        <label className="block space-y-2"><span>Job name</span><input className={input} required maxLength={80} value={draft.name} onChange={event => setDraft({ ...draft, name: event.target.value })} disabled={busy} /></label>
        <label className="block space-y-2"><span>Project / repository</span><select className={input} required value={draft.repoId} onChange={event => setDraft({ ...draft, repoId: event.target.value })} disabled={busy || resources.isPending || resources.isError}><option value="">Choose a repository</option>{repos.map(repo => <option key={repo.id} value={repo.id}>{repo.name} / {repo.branch || 'branch unknown'}</option>)}</select></label>
        {resources.isPending && <p role="status">Loading repositories...</p>}
        {resources.isError && <ErrorMessage message={`Repositories unavailable: ${jobError(resources.error)}`} />}
        {!resources.isPending && !resources.isError && repos.length === 0 && <p className="text-amber-300">No repository resources available.</p>}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <label className="block space-y-2"><span>Interval (minutes)</span><input type="number" className={input} required min={15} max={1440} step={1} value={draft.interval} onChange={event => setDraft({ ...draft, interval: event.target.value })} disabled={busy} /></label>
          <label className="block space-y-2"><span>Daily cap (UTC)</span><input type="number" className={input} required min={1} max={48} step={1} value={draft.cap} onChange={event => setDraft({ ...draft, cap: event.target.value })} disabled={busy} /></label>
        </div>
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 border-y border-neutral-800 py-4">
          <dt className="text-neutral-400">Reads</dt><dd className="break-words">{selectedRepo ? `${selectedRepo.name}, mounted ${selectedRepo.branch || 'unknown branch'}${selectedRepo.dirty ? ' (modified)' : ''}` : 'Selected repository snapshot'}</dd>
          <dt className="text-neutral-400">Produces</dt><dd>{JOB_RECIPES.find(recipe => recipe.id === draft.recipeId)?.output}</dd>
          <dt className="text-neutral-400">Execution</dt><dd>Mini / Ollama</dd>
          <dt className="text-neutral-400">Delivery</dt><dd>Work, for human review</dd>
          <dt className="text-neutral-400">Writes</dt><dd>Saved work only. No repository changes or publication.</dd>
          <dt className="text-neutral-400">Starts</dt><dd>Paused. No scheduled checks until enabled.</dd>
          <dt className="text-neutral-400">Limits</dt><dd>Daily cap, pending-review limit, unavailable inputs, and failure backoff.</dd>
        </dl>
        {create.error && <ErrorMessage message={`Job creation not confirmed: ${jobError(create.error)}`} />}
        <button type="submit" className={primary} disabled={busy || resources.isPending || resources.isError || !repos.length}>{create.isPending ? <Loader2 size={18} className="animate-spin" /> : <Plus size={18} />}Create paused job</button>
      </form>
    </section> : context && selected ? <section aria-labelledby="job-context-heading" className="border-y border-neutral-800 py-6">
      <ContextHeading title={selected.name} busy={busy} headingRef={heading} onClose={() => openContext(null)} />
      <p className="mt-3 text-neutral-400">{jobState(selected).detail} Every {selected.loop_interval_minutes} minutes, up to {selected.loop_max_runs_per_day} runs per UTC day.</p>
      <p className="mt-2 break-words">{selected.mission}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        <button type="button" className={control} disabled={busy} onClick={() => navigate({ view: 'agents', agentId: selected.id })}>Agent<ArrowRight size={16} /></button>
        <button type="button" className={control} disabled={busy} onClick={() => navigate({ view: 'assignments', agentId: selected.id })}>Create work<ArrowRight size={16} /></button>
      </div>
      <h3 className="mt-6 text-base font-medium text-white">Test evidence</h3>
      {work.isError ? <ErrorMessage message={`Saved test status unavailable: ${jobError(work.error)}`} /> : work.isPending ? <p role="status">Loading saved tests...</p> : assignment ? <div className="mt-3 space-y-3">
        <p className="text-neutral-400">{assignment.status === 'queued' ? 'Awaiting manual start' : assignment.status === 'completed' ? 'Result ready for review' : assignment.status === 'running' ? 'Running on Mini' : 'Failed'} / {jobTime(assignment.updated_at)}</p>
        {assignment.error && <ErrorMessage message={assignment.error} />}
        {(assignment.review_status === 'rejected' || assignment.dismissed_at) && <p className="text-amber-300">This test was rejected or closed. A new test is required before enabling.</p>}
        {assignment.result && <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words border-l-2 border-neutral-700 pl-4 font-sans text-sm leading-6">{assignment.result}</pre>}
        <button type="button" className={control} disabled={busy} onClick={() => navigate({ view: 'assignments', assignmentId: assignment.id })}>Open saved test<ArrowRight size={16} /></button>
      </div> : <p className="mt-2 text-neutral-400">No saved test for this configuration.</p>}
      {test.error && <div className="mt-3"><ErrorMessage message={`Test request failed: ${jobError(test.error)}. Check saved work before retrying.`} /></div>}
      {testPassed && assignment && !selected.loop_enabled && <label className="mt-4 flex min-h-11 items-center gap-3"><input type="checkbox" className="h-5 w-5 accent-emerald-500" checked={inspectedId === assignment.id} disabled={busy} onChange={event => setInspectedId(event.target.checked ? assignment.id : '')} />I reviewed the test evidence.</label>}
      <div className="mt-4 flex flex-wrap gap-3">
        <button type="button" className={control} disabled={busy || selected.loop_enabled || selected.status === 'running' || work.isPending || work.isError || assignment?.status === 'running' || assignment?.recovery_pending} onClick={() => { setInspectedId(''); test.mutate({ agent: selected, existing: assignment }) }}>{test.isPending ? <Loader2 size={18} className="animate-spin" /> : <FlaskConical size={18} />}{assignment?.status === 'queued' ? 'Start saved test' : assignment?.status === 'failed' ? 'Retry saved test' : assignment?.status === 'completed' ? 'Test again' : 'Test once on Mini'}</button>
        <button type="button" className={primary} disabled={busy || agentError || (!selected.loop_enabled && (!testPassed || inspectedId !== assignment?.id || work.isError))} onClick={() => toggle.mutate({ agent: selected, enabled: !selected.loop_enabled })}>{selected.loop_enabled ? <Pause size={18} /> : <Play size={18} />}{selected.loop_enabled ? 'Pause job' : 'Enable job'}</button>
      </div>
      {selected.loop_enabled && <p className="mt-2 text-neutral-400">Pause future checks before running a manual test.</p>}
      {!selected.loop_enabled && testPassed && <p className="mt-2 text-neutral-400">After enabling, next eligible check: {nextJobCheck({ ...selected, loop_enabled: true })}. Execution depends on scheduler and Mini availability.</p>}
    </section> : null}

    {!context && <section aria-labelledby="recurring-jobs-heading">
      <h2 id="recurring-jobs-heading" className="text-base font-medium text-white">Recurring jobs <span className="ml-2 text-neutral-400">{jobs.length}</span></h2>
      {agentError && <ErrorMessage message="Jobs could not be refreshed. Displayed records may be stale." />}
      {loading ? <p role="status" className="py-6">Loading jobs...</p> : jobs.length === 0 ? <p className="py-6 text-neutral-400">{agentError ? 'Jobs are unavailable.' : 'No recurring jobs configured.'}</p> : <ul className="mt-3 divide-y divide-neutral-800 border-y border-neutral-800">
        {jobs.map(original => {
          const agent = confirmed?.id === original.id && confirmed.updated_at >= original.updated_at ? confirmed : original
          const state = jobState(agent)
          const repo = repos.find(item => item.id === agent.repo_id)
          const latestWork = (work.data?.assignments ?? []).filter(item => item.agent_id === agent.id).sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0]
          return <li key={agent.id} className="py-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1"><h3 className="break-words text-base font-medium text-white">{agent.name}</h3><p className="mt-1 break-words text-neutral-400">{repo?.name || agent.repo_id || 'No project'} / Agent interval</p></div>
              <span className={`shrink-0 ${state.tone === 'warning' ? 'text-amber-300' : state.tone === 'good' ? 'text-emerald-300' : 'text-neutral-300'}`}>{state.label}</span>
            </div>
            <p className="mt-2 break-words">{state.detail}</p>
            {agent.status === 'error' && !agent.loop_enabled && <p className="mt-2 break-words text-amber-300">Last run failed: {agent.last_error || 'No error detail reported.'}</p>}
            <dl className="mt-3 grid gap-3 text-neutral-400 sm:grid-cols-2 lg:grid-cols-4">
              <div><dt>Interval / daily cap</dt><dd className="text-neutral-200">{agent.loop_interval_minutes} min / {runsToday(agent)} of {agent.loop_max_runs_per_day} runs (UTC)</dd></div>
              <div><dt>Last check</dt><dd className="text-neutral-200">{agent.loop_checked_at ? jobTime(agent.loop_checked_at) : 'No scheduled check yet'}</dd></div>
              <div><dt>Next eligible check</dt><dd className="text-neutral-200">{nextJobCheck(agent)}</dd></div>
              <div><dt>Last execution</dt><dd className="text-neutral-200">{agent.last_run_at ? jobTime(agent.last_run_at) : 'Not run yet'}</dd></div>
            </dl>
            <div className="mt-4 flex flex-wrap gap-2">
              {agent.loop_enabled ? <button type="button" className={control} disabled={busy || agentError} onClick={() => toggle.mutate({ agent, enabled: false })}><Pause size={16} />Pause</button> : canResumeJob(agent, work.data?.assignments ?? []) ? <button type="button" className={control} disabled={busy || agentError || work.isPending || work.isError} onClick={() => toggle.mutate({ agent, enabled: true })}><Play size={16} />Resume</button> : null}
              <button type="button" className={control} disabled={busy} onClick={() => openContext(agent.id)}><FlaskConical size={16} />Test / result</button>
              <button type="button" className={control} disabled={busy || work.isPending || work.isError} onClick={() => navigate(latestWork ? { view: 'assignments', assignmentId: latestWork.id } : { view: 'assignments', agentId: agent.id })}>{latestWork ? 'Latest work' : 'Create work'}<ArrowRight size={16} /></button>
              <button type="button" className={control} disabled={busy} onClick={() => navigate({ view: 'agents', agentId: agent.id })}>Agent<ArrowRight size={16} /></button>
            </div>
          </li>
        })}
      </ul>}
      <p className="mt-3 text-neutral-400">Intervals are eligibility windows, not reserved run times. Pausing preserves history and does not stop a current run.</p>
    </section>}

    {!context && <>
      <section className="border-t border-neutral-800 pt-6" aria-label="System tasks">
        <details>
          <summary className="min-h-11 cursor-pointer py-2 text-base font-medium text-white focus-visible:outline-2 focus-visible:outline-emerald-400">System tasks <span className="ml-2 text-sm font-normal text-neutral-400">Read only / Built-in task runner</span></summary>
          {tasks.isPending && <p role="status">Loading system tasks...</p>}
          {tasks.isError && <ErrorMessage message={`System tasks unavailable: ${jobError(tasks.error)}`} />}
          {tasks.data?.length === 0 && <p className="py-4 text-neutral-400">No system tasks reported.</p>}
          <ul className="divide-y divide-neutral-800">{tasks.data?.map(task => <li key={task.task_id} className="py-4">
            <div className="flex flex-wrap justify-between gap-2"><h3 className="break-words font-medium text-neutral-100">{task.name}</h3><span className="text-neutral-400">{task.enabled ? 'Enabled' : 'Disabled'} / {task.last_status || 'Not run yet'}</span></div>
            <p className="mt-1 break-words text-neutral-400">{task.description}</p>
            <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1"><span>Last run: {jobTime(task.last_run)}</span><span className="inline-flex items-center gap-2"><Clock3 size={16} />Next run: {task.enabled ? jobTime(task.next_run) : 'Disabled'}</span></div>
          </li>)}</ul>
        </details>
      </section>
      <section className="flex flex-wrap items-center justify-between gap-3 border-t border-neutral-800 pt-6" aria-label="Public signals">
        <div><h2 className="text-base font-medium text-white">Public signals</h2><p className="mt-1 text-neutral-400">Separate discovery runner / 12-hour intervals</p></div>
        <button type="button" className={control} onClick={() => navigate({ view: 'signals' })}>Open public signals<ArrowRight size={16} /></button>
      </section>
    </>}
  </main>
}

function ErrorMessage({ message }: { message: string }) {
  return <p role="alert" className="flex items-start gap-2 break-words py-2 text-sm text-amber-300"><AlertCircle className="mt-1 shrink-0" size={16} /><span className="min-w-0">{message}</span></p>
}

function ContextHeading({ title, busy, headingRef, onClose }: { title: string; busy: boolean; headingRef: React.RefObject<HTMLHeadingElement | null>; onClose: () => void }) {
  return <div className="flex items-start justify-between gap-3"><h2 id="job-context-heading" ref={headingRef} tabIndex={-1} className="min-w-0 break-words text-xl font-medium text-white outline-none">{title}</h2><button type="button" className={control} aria-label="Close job" title="Close job" disabled={busy} onClick={onClose}><X size={18} /></button></div>
}
