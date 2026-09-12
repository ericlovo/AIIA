import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

export function AssignmentHistory({ assignmentId }: { assignmentId: string }) {
  const [offset, setOffset] = useState(0)
  const [selectedId, setSelectedId] = useState('')
  const history = useQuery({
    queryKey: ['assignment-history', assignmentId, offset],
    queryFn: () => api.assignmentHistory(assignmentId, offset),
    retry: false,
    refetchInterval: 5_000,
  })
  const detail = useQuery({
    queryKey: ['studio-run', selectedId],
    queryFn: () => api.studioRun(selectedId),
    enabled: Boolean(selectedId),
    retry: false,
  })
  const data = history.data
  return <section aria-label="Assignment attempt history" className="space-y-3 border-t border-neutral-800 pt-4">
    <div className="flex items-center justify-between gap-2">
      <h3 className="text-sm font-medium text-white">Attempt history</h3>
      <button type="button" aria-label="Refresh attempt history" title="Refresh attempt history" disabled={history.isFetching} onClick={() => { void history.refetch(); if (selectedId) void detail.refetch() }} className="p-2 text-neutral-400 disabled:opacity-40"><RefreshCw size={14} /></button>
    </div>
    {history.isPending && <p role="status" className="text-xs text-neutral-400">Loading saved attempts…</p>}
    {history.isError ? <p role="alert" className="text-xs text-red-300">Attempt history unavailable. Refresh to try again.</p> : data && <>
      <p className="text-xs text-neutral-400">{data.total} saved {data.total === 1 ? 'attempt' : 'attempts'}</p>
      {data.attempt_id && !data.current_output_saved && <p className="text-xs text-amber-300">Latest attempt has no saved output.</p>}
      {!data.attempt_id && <p className="text-xs text-neutral-500">Current attempt identity was not recorded.</p>}
      {data.total === 0 && <p className="text-xs text-neutral-500">No saved attempts for this assignment.</p>}
      <ol className="divide-y divide-neutral-800">
        {data.runs.map(run => <li key={run.id}>
          <button type="button" aria-expanded={selectedId === run.id} onClick={() => setSelectedId(selectedId === run.id ? '' : run.id)} className="w-full space-y-1 py-3 text-left text-xs">
            <span className="flex flex-wrap justify-between gap-2"><span className={run.status === 'failed' ? 'text-red-300' : 'text-emerald-300'}>{run.status === 'failed' ? 'Failed attempt' : 'Output saved'}</span><time className="text-neutral-400" dateTime={run.at}>{new Date(run.at).toLocaleString()}</time></span>
            <span className="block break-words text-neutral-300">{run.model || 'Model unrecorded'} · {run.latency_ms ? `${(run.latency_ms / 1000).toFixed(1)}s` : 'Duration unrecorded'}{run.legacy ? ' · Imported' : ''}</span>
            {run.id === data.completed_run_id && <span className="block text-cyan-300">Applied to assignment</span>}
          </button>
          {selectedId === run.id && <div className="space-y-2 pb-4 text-xs">
            {detail.isPending && <p role="status">Loading attempt…</p>}
            {detail.isError && <p role="alert" className="text-red-300">Could not load this attempt.</p>}
            {detail.data && <>
              <div className="text-neutral-500">Recorded task</div><pre className="max-h-52 overflow-y-auto whitespace-pre-wrap break-words text-neutral-300">{detail.data.run.task || 'Task unrecorded'}</pre>
              {detail.data.run.error && <p className="break-words text-red-300">{detail.data.run.error}</p>}
              <div className="text-neutral-500">Saved output</div><pre className="max-h-72 overflow-y-auto whitespace-pre-wrap break-words text-neutral-300">{detail.data.run.result || 'No output recorded.'}</pre>
            </>}
          </div>}
        </li>)}
      </ol>
      {data.total > data.limit && <nav aria-label="Attempt history pages" className="flex items-center justify-between text-xs text-neutral-400">
        <button type="button" aria-label="Newer attempts" title="Newer attempts" disabled={offset === 0} onClick={() => { setOffset(Math.max(0, offset - data.limit)); setSelectedId('') }} className="p-2 disabled:opacity-30"><ArrowLeft size={16} /></button>
        <span>{data.offset + 1}–{Math.min(data.offset + data.limit, data.total)} of {data.total}</span>
        <button type="button" aria-label="Older attempts" title="Older attempts" disabled={offset + data.limit >= data.total} onClick={() => { setOffset(offset + data.limit); setSelectedId('') }} className="p-2 disabled:opacity-30"><ArrowRight size={16} /></button>
      </nav>}
    </>}
  </section>
}
