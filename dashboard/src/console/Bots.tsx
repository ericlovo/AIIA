import { useEffect, useRef, useState } from 'react'
import { useIsMutating, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type Agent } from '../lib/api'
import { PageHeader } from './PageHeader'
import { formatRoute } from './studioRoute'

function purpose(agent: Agent) { return agent.one_liner?.trim() || agent.use_when?.trim() || agent.mission }
function status(agent: Agent, pending = false) { return pending || agent.status === 'running' ? 'Working' : agent.status === 'error' ? 'Needs attention' : agent.retired ? 'Retired' : 'Ready' }
function initials(name: string) { return name.trim().split(/\s+/).slice(0, 2).map(word => word[0]).join('').toUpperCase() }
const focus = 'focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan-300'

function TaskPanel({ agent }: { agent: Agent }) {
  const qc = useQueryClient()
  const draftKey = ['bot-task', agent.id]
  const { data: draft } = useQuery({ queryKey: draftKey, queryFn: () => ({ task: '', error: '' }), initialData: { task: '', error: '' }, staleTime: Infinity, gcTime: Infinity, refetchOnWindowFocus: false })
  const task = draft.task
  const setTask = (text: string) => qc.setQueryData(draftKey, { ...draft, task: text })
  const lock = useRef(false)
  const heading = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    heading.current?.focus({ preventScroll: true })
    heading.current?.scrollIntoView({ block: 'start' })
  }, [agent.id])
  const mutationKey = ['bot-run', agent.id]
  const pending = useIsMutating({ mutationKey }) > 0
  const run = useMutation({
    mutationKey,
    mutationFn: ({ text }: { text: string; context: string }) => api.runAgent(agent.id, text),
    onMutate: () => { qc.setQueryData<{ task: string; error: string }>(draftKey, old => ({ task: old?.task ?? '', error: '' })) },
    onSuccess: ({ agent: updated }, submitted) => {
      qc.setQueryData<{ agents: Agent[] }>(['agents'], old => old ? { ...old, agents: old.agents.map(item => item.id === updated.id ? updated : item) } : { agents: [updated] })
      qc.setQueryData<{ task: string; error: string }>(draftKey, old => ({ task: old?.task.trim() === submitted.context ? '' : old?.task ?? '', error: '' }))
    },
    onError: (error) => { qc.setQueryData<{ task: string; error: string }>(draftKey, old => ({ task: old?.task ?? '', error: error.message })) },
    onSettled: () => {
      lock.current = false
      void qc.invalidateQueries({ queryKey: ['agents'] })
      void qc.invalidateQueries({ queryKey: ['bot-history', agent.id] })
    },
  })
  const busy = pending || agent.status === 'running'
  const taskText = (useCaseTask = '') => useCaseTask ? `${useCaseTask}${task.trim() ? `\n\nAdditional context:\n${task.trim()}` : ''}` : task.trim()
  const submit = (useCaseTask = '') => {
    const text = taskText(useCaseTask)
    if (!text || text.length > 8000 || busy || lock.current || agent.retired) return
    lock.current = true
    run.mutate({ text, context: task.trim() })
  }
  const useCases = agent.use_cases?.slice(0, 3) ?? []
  return <section aria-label={`${agent.name} interaction`} className="min-w-0 rounded-xl border border-neutral-700 bg-neutral-900/50 p-5 sm:p-7">
    <div className="flex items-start justify-between gap-4"><div className="min-w-0"><h2 ref={heading} tabIndex={-1} className="scroll-mt-6 break-words text-xl font-medium text-white">{agent.name}</h2><p className="mt-2 text-sm leading-relaxed text-neutral-400">{purpose(agent)}</p></div><span className="shrink-0 text-xs text-cyan-200">{status(agent, pending)}</span></div>
    {useCases.length > 0 && <section aria-label="Use cases" className="mt-6">
      <h3 className="text-sm font-medium text-neutral-200">Use cases</h3>
      <p className="mt-1 text-xs text-neutral-400">Add optional context below, then choose a task to run it.</p>
      <div className="mt-3 flex flex-wrap gap-2">{useCases.map((item, index) => <button key={index} type="button" disabled={busy || agent.retired || taskText(item.task).length > 8000} onClick={() => submit(item.task)} className={`min-h-11 max-w-full rounded-lg border border-cyan-800 bg-cyan-950/30 px-3 py-2 text-left text-sm text-cyan-100 break-words disabled:cursor-not-allowed disabled:opacity-40 ${focus}`}>{item.label}</button>)}</div>
    </section>}
    <form className="mt-6" onSubmit={event => { event.preventDefault(); submit() }}>
      <h3 className="text-sm font-medium text-neutral-200">Give task</h3>
      <label className="mt-2 block text-sm text-white">Additional context <span className="text-neutral-400">(optional for use cases)</span><textarea aria-label="Additional context" value={task} onChange={event => setTask(event.target.value)} disabled={busy || agent.retired} rows={4} maxLength={8000} placeholder={`Add context for a use case, or write a free-form task for ${agent.name}.`} className={`mt-2 w-full resize-y rounded-lg border border-neutral-700 bg-neutral-950 p-3 text-sm text-white placeholder:text-neutral-400 disabled:opacity-60 ${focus}`} /></label>
      <button disabled={!task.trim() || busy || agent.retired} type="submit" className={`mt-3 min-h-11 rounded-lg bg-cyan-300 px-5 text-sm font-medium text-neutral-950 disabled:cursor-not-allowed disabled:opacity-40 ${focus}`}>{busy ? 'Working...' : 'Send task'}</button>
      {busy && <p role="status" className="mt-3 text-sm text-neutral-400">{agent.name} is working. You can open another bot while this finishes.</p>}
      {agent.retired && <p className="mt-3 text-sm text-neutral-400">This bot is retired. Its previous output is still available.</p>}
      {draft.error && <p role="alert" className="mt-3 break-words text-sm text-red-300">Task request failed: {draft.error}. Your task is still here; check the bot status before retrying.</p>}
    </form>
    <section aria-label="Latest result" className="mt-6 border-t border-neutral-800 pt-5">
      <h3 className="text-sm font-medium text-neutral-200">Latest result</h3>
      {agent.last_run_at && <p className="mt-1 text-xs text-neutral-400"><time dateTime={agent.last_run_at}>{new Date(agent.last_run_at).toLocaleString()}</time></p>}
      {agent.last_error ? <p role="alert" className="mt-3 whitespace-pre-wrap break-words text-sm text-red-300">Last run failed: {agent.last_error}</p> : <p className="mt-3 max-h-96 overflow-y-auto whitespace-pre-wrap break-words text-sm leading-relaxed text-neutral-300">{agent.last_result?.trim() || 'No result yet. Give this bot its first task.'}</p>}
    </section>
    <BotHistory key={agent.id} agent={agent} />
    <a className={`mt-5 inline-flex min-h-11 items-center text-sm text-neutral-400 hover:text-cyan-200 ${focus}`} href={formatRoute({ view: 'agents', agentId: agent.id })}>Bot settings</a>
  </section>
}

function BotHistory({ agent }: { agent: Agent }) {
  const [offset, setOffset] = useState(0)
  const history = useQuery({ queryKey: ['bot-history', agent.id, offset], queryFn: () => api.botHistory(agent.id, offset), refetchInterval: 10_000, retry: false })
  return <section aria-label="History" className="mt-6 border-t border-neutral-800 pt-5">
    <h3 className="text-sm font-medium text-neutral-200">History</h3>
    {history.isPending && <p role="status" className="mt-3 text-sm text-neutral-400">Loading history...</p>}
    {history.isError && <p role="alert" className="mt-3 text-sm text-red-300">History unavailable. <button className={`min-h-11 px-2 underline ${focus}`} onClick={() => void history.refetch()}>Retry history</button></p>}
    {history.data && <>
      <p className="mt-2 text-xs text-neutral-400">{history.data.total ? `${offset + 1}–${offset + history.data.runs.length} of ${history.data.total} runs · Newest first` : 'No runs yet. Run a use case or send a task to start.'}</p>
      <ol className="mt-3 divide-y divide-neutral-800">{history.data.runs.map(run => <li key={run.id} className="py-3">
        <details>
          <summary className={`min-h-11 cursor-pointer break-words text-sm ${focus}`}><span className={run.status === 'failed' ? 'text-red-300' : 'text-emerald-300'}>{run.status === 'failed' ? 'Failed' : 'Completed'}</span> <time className="text-xs text-neutral-400" dateTime={run.at}>{new Date(run.at).toLocaleString()}</time><span className="mt-1 block truncate text-neutral-300">{run.task || 'Task unrecorded'}</span></summary>
          <div className="mt-3 space-y-3 text-sm"><div><h4 className="text-xs text-neutral-400">Task input</h4><p className="mt-1 whitespace-pre-wrap break-words">{run.task || 'Task unrecorded'}</p></div><div><h4 className="text-xs text-neutral-400">Result</h4><p className={`mt-1 whitespace-pre-wrap break-words ${run.status === 'failed' ? 'text-red-300' : 'text-neutral-300'}`}>{run.error || run.result || 'No result recorded.'}</p></div></div>
        </details>
      </li>)}</ol>
      {(offset > 0 || offset + history.data.runs.length < history.data.total) && <div className="mt-3 flex gap-3"><button className={`min-h-11 px-3 text-sm text-cyan-200 disabled:opacity-40 ${focus}`} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - history.data.limit))}>Newer runs</button><button className={`min-h-11 px-3 text-sm text-cyan-200 disabled:opacity-40 ${focus}`} disabled={offset + history.data.runs.length >= history.data.total} onClick={() => setOffset(offset + history.data.limit)}>Older runs</button></div>}
    </>}
  </section>
}

function BotTile({ agent, selected }: { agent: Agent; selected: boolean }) {
  const pending = useIsMutating({ mutationKey: ['bot-run', agent.id] }) > 0
  const latest = (agent.last_error ? `Failed: ${agent.last_error}` : agent.last_result?.trim() || 'No result recorded.').replace(/\s+/g, ' ')
  return <a href={formatRoute({ view: 'bots', agentId: agent.id })} aria-label={`Open ${agent.name}`} aria-current={selected ? 'page' : undefined} className={`block min-w-0 rounded-xl border p-5 transition-colors ${focus} ${selected ? 'border-cyan-400 bg-cyan-950/30' : 'border-neutral-800 bg-neutral-900/40 hover:border-neutral-600'}`}>
    <div className="flex items-center gap-3"><span aria-hidden="true" className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-neutral-600 bg-neutral-800 text-sm font-medium text-cyan-100">{initials(agent.name)}</span><div className="min-w-0"><h2 className="break-words text-base font-medium text-white">{agent.name}</h2><p className={`mt-1 text-xs ${pending || agent.status === 'running' ? 'text-amber-200' : 'text-neutral-400'}`}>{status(agent, pending)}</p></div></div>
    <p className="mt-4 line-clamp-2 break-words text-sm leading-relaxed text-neutral-300">{purpose(agent)}</p>
    <div className="mt-3 text-xs leading-relaxed text-neutral-400"><p>{agent.last_run_at ? <>Last activity: <time dateTime={agent.last_run_at}>{new Date(agent.last_run_at).toLocaleString()}</time></> : 'No activity yet.'}</p><p className="mt-1 truncate" title={latest}>{latest}</p></div>
  </a>
}

export function Bots({ agents, loading, error, agentId }: { agents: Agent[]; loading: boolean; error: boolean; agentId?: string }) {
  const qc = useQueryClient()
  const selected = agents.find(agent => agent.id === agentId)
  const visible = agents.filter(agent => !agent.retired || agent.id === agentId)
  return <main className="h-full min-h-0 overflow-y-auto bg-neutral-950 text-neutral-200">
    <PageHeader title="Bots" meta={<span>Pick a bot. Give it a task.</span>} />
    {loading && <p role="status" className="px-5 py-6 sm:px-7">Loading bots...</p>}
    {error && <div role="alert" className="px-5 py-6 text-sm text-red-300 sm:px-7">Bots unavailable. <button className={`min-h-11 px-3 underline ${focus}`} onClick={() => void qc.invalidateQueries({ queryKey: ['agents'] })}>Retry</button></div>}
    {!loading && !error && <div className={`grid min-w-0 gap-6 px-5 pb-8 pt-5 sm:px-7 ${selected ? 'xl:grid-cols-[minmax(0,1fr)_minmax(360px,1fr)]' : ''}`}>
      <section aria-label="Choose a bot" className="min-w-0">
        {!visible.length && <p className="text-sm text-neutral-400">No active bots available. <a className={`inline-flex min-h-11 items-center text-cyan-300 ${focus}`} href={formatRoute({ view: 'agents' })}>Open agent settings</a></p>}
        <div className={`grid min-w-0 gap-4 ${selected ? 'sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2' : 'sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'}`}>{visible.map(agent => <BotTile key={agent.id} agent={agent} selected={agent.id === agentId} />)}</div>
      </section>
      {selected && <div className="min-w-0 max-xl:row-start-1"><TaskPanel key={selected.id} agent={selected} /></div>}
      {agentId && !selected && <p role="alert" className="text-sm text-amber-200">This bot is unavailable. Choose another bot.</p>}
    </div>}
  </main>
}
