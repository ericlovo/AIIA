import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type Agent, type StudioRun } from '../lib/api'
import { formatRoute } from './studioRoute'

export function AgentTimeline({ agent }: { agent: Agent }) {
  const [status, setStatus] = useState('')
  const activity = useQuery({ queryKey: ['studio-activity', agent.id, '', status], queryFn: () => api.studioActivity(agent.id, '', status), refetchInterval: 10_000, retry: false })
  return <section aria-label={`${agent.name} activity`} className="px-5 py-5 text-sm text-neutral-300">
    <h2 className="mb-3 break-words text-lg text-white">{agent.name}</h2>
    <p className="break-words">{agent.mission}</p>
    <div className="mt-3 flex flex-wrap gap-2 text-xs text-neutral-400"><span>{agent.status}</span><span>{agent.loop_enabled ? `Every ${agent.loop_interval_minutes} minutes` : 'Manual'}</span><span>{agent.repo_id || 'No repository'}</span></div>
    <a className="mt-4 inline-flex min-h-11 items-center text-cyan-200 underline" href={formatRoute({ view: 'assignments', agentId: agent.id })}>Assign work</a>
    <label className="mt-4 block text-xs text-neutral-400">Run status<select className="mt-2 min-h-11 w-full border border-neutral-700 bg-neutral-900 px-3 text-sm text-white" value={status} onChange={event => setStatus(event.target.value)}><option value="">All recorded runs</option><option value="failed">Failed</option><option value="completed">Completed</option></select></label>
    {activity.isPending && <p role="status" className="mt-4">Loading activity...</p>}
    {activity.isError && <p role="alert" className="mt-4 text-red-300">Activity unavailable. <button className="underline" onClick={() => void activity.refetch()}>Retry</button></p>}
    {activity.data && <p className="my-4 text-xs text-neutral-500">{activity.data.runs.length} recent recorded runs</p>}
    {activity.data?.runs.length === 0 && <p className="py-4 text-neutral-500">No recorded runs in this view.</p>}
    <ol className="divide-y divide-neutral-800">{activity.data?.runs.map(run => <TimelineRun key={run.id} run={run} />)}</ol>
  </section>
}

function TimelineRun({ run }: { run: StudioRun }) {
  const [open, setOpen] = useState(false)
  const detail = useQuery({ queryKey: ['studio-run', run.id], queryFn: () => api.studioRun(run.id), enabled: open, retry: false })
  const tokens = run.input_tokens != null && run.output_tokens != null ? (run.input_tokens + run.output_tokens).toLocaleString() : 'Unrecorded'
  return <li className="py-4">
    <details onToggle={event => setOpen(event.currentTarget.open)}>
      <summary className="min-h-11 cursor-pointer break-words"><span className={run.status === 'failed' ? 'text-red-300' : 'text-emerald-300'}>{run.status === 'failed' ? 'Failed' : 'Completed'}</span><span className="ml-2 text-xs text-neutral-400">{new Date(run.at).toLocaleString()}</span><span className="mt-1 block text-xs text-neutral-400">{run.trigger} · {run.model || 'Model unrecorded'} · {tokens} tokens</span></summary>
      {open && detail.isPending && <p role="status">Loading run...</p>}
      {open && detail.isError && <p role="alert" className="text-red-300">Run unavailable. <button className="underline" onClick={() => void detail.refetch()}>Retry</button></p>}
      {open && detail.data && <div className="space-y-3 py-3">
        <div><h3 className="text-xs text-neutral-500">Task</h3><p className="mt-1 whitespace-pre-wrap break-words">{detail.data.run.task || 'Task unrecorded'}</p></div>
        <div><h3 className="text-xs text-neutral-500">{detail.data.run.error ? 'Failure evidence' : 'Output'}</h3><p className="mt-1 whitespace-pre-wrap break-words">{detail.data.run.error || detail.data.run.result || 'No output recorded'}</p></div>
        {run.assignment_id && <a href={formatRoute({ view: 'assignments', assignmentId: run.assignment_id })} className="inline-flex min-h-11 items-center text-cyan-200 underline">Open assignment</a>}
      </div>}
    </details>
  </li>
}
