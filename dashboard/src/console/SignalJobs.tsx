import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Play, Radar, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'
import { PageHeader } from './PageHeader'
import { navigate } from './useStudioRoute'
import { LeadQueue } from './LeadQueue'

const STATES: Record<string, string> = {
  running: 'Screening', failed: 'Run failed', interrupted: 'Interrupted',
  no_change: 'No new evidence', no_signal: 'No qualifying signals',
  review_ready: 'Ready for review', review_backlog: 'Review queue full',
}

export function SignalJobs() {
  const qc = useQueryClient()
  const query = useQuery({ queryKey: ['signal-jobs'], queryFn: api.signalJobs, refetchInterval: 10_000, retry: false })
  const refresh = () => { void qc.invalidateQueries({ queryKey: ['signal-jobs'] }); void qc.invalidateQueries({ queryKey: ['memory-inbox'] }) }
  const run = useMutation({ mutationFn: api.runSignalJob, onSettled: refresh })
  const configure = useMutation({ mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.configureSignalJob(id, enabled), onSettled: refresh })
  const error = run.error ?? configure.error
  const busy = run.isPending || query.data?.jobs.some(job => job.last_run?.status === 'running')

  return <main className="h-full min-h-0 overflow-y-auto bg-neutral-950 text-neutral-200">
    <PageHeader title="Signals" meta={<span>Performance Labs · Wisconsin / Minnesota / Iowa</span>} />
    <div className="border-b border-neutral-800 px-5 py-4 sm:px-7">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm"><Radar size={18} className="text-emerald-300" /><strong>Jev</strong><span className="text-neutral-400">Public evidence screening</span></div>
        <button onClick={() => navigate({ view: 'memory', source: 'signals' })} className="inline-flex min-h-11 items-center gap-2 text-sm text-cyan-300">Review inbox <ArrowRight size={16} /></button>
      </div>
      {query.data && <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-neutral-400">
        <span>Retrieval {query.data.retrieval_enabled ? 'enabled' : 'disabled'}</span>
        <span>Jev {query.data.configured ? query.data.screening_enabled ? 'enabled' : 'disabled' : 'credential missing'}</span>
        <span>3 items / run · 6 pending maximum</span><span>No outreach</span>
      </div>}
    </div>
    {query.isPending && <p role="status" className="p-7">Loading jobs...</p>}
    {query.isError && <div role="alert" className="p-7 text-red-300">Signal jobs unavailable. <button aria-label="Retry signal jobs" title="Retry signal jobs" onClick={() => void query.refetch()}><RefreshCw size={16} /></button></div>}
    {error && <p role="alert" className="px-7 py-3 text-sm text-red-300">{error.message}</p>}
    <LeadQueue />
    <details className="border-t border-neutral-800">
      <summary className="cursor-pointer px-5 py-5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-cyan-300 sm:px-7">Discovery automation{query.data ? ` (${query.data.jobs.filter(job => job.enabled).length}/${query.data.jobs.length} scheduled)` : ''}</summary>
      {query.data && !query.data.ready && <p role="status" className="px-5 pb-4 text-sm text-amber-200 sm:px-7">Discovery is paused.{!query.data.retrieval_enabled && ' Public retrieval is disabled.'}{!query.data.screening_enabled && ' Jev screening is disabled.'}{!query.data.configured && ' A Jev credential is required.'}</p>}
    <div className="divide-y divide-neutral-800">
      {query.data?.jobs.map(job => {
        const last = job.last_run
        const usage = last?.result.usage
        const cooldown = last ? last.started * 1000 + job.interval_hours * 3600_000 > query.dataUpdatedAt : false
        return <section key={job.id} className="px-5 py-6 sm:px-7">
          <div className="flex flex-wrap items-start justify-between gap-5">
            <div className="min-w-0">
              <div className="flex items-center gap-2"><Radar aria-hidden="true" size={18} className="shrink-0 text-emerald-300" /><h2 className="text-base font-medium text-white">{job.name}</h2></div>
              <p className="mt-1 text-sm text-neutral-400">{job.specialty}</p>
            </div>
            <div className="flex flex-wrap items-center gap-5">
              <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={job.enabled} disabled={configure.isPending || (!job.enabled && !query.data?.ready)} onChange={event => configure.mutate({ id: job.id, enabled: event.target.checked })} />Every 12 hours</label>
              <button aria-label={`Run ${job.name}`} title={cooldown ? 'Available 12 hours after the last attempt' : `Run ${job.name}`} disabled={!query.data?.ready || busy || cooldown} onClick={() => run.mutate(job.id)} className="inline-flex h-9 items-center gap-2 border border-neutral-600 px-3 text-sm hover:border-emerald-300 disabled:opacity-40"><Play size={14} />Run</button>
            </div>
          </div>
          <dl className="mt-5 grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
            <div><dt className="text-neutral-500">Status</dt><dd className="mt-1">{last ? STATES[last.status] ?? last.status : 'Not run'}</dd></div>
            <div><dt className="text-neutral-500">Last attempt</dt><dd className="mt-1">{last ? new Date(last.started * 1000).toLocaleString() : 'None'}</dd></div>
            <div><dt className="text-neutral-500">Review items</dt><dd className="mt-1">{last?.result.created ?? '—'}</dd></div>
            <div><dt className="text-neutral-500">Last run tokens</dt><dd className="mt-1">{usage ? (usage.input_tokens + usage.output_tokens).toLocaleString() : 'Not reported'}</dd></div>
          </dl>
          {last?.result.error && <p className="mt-4 text-sm text-red-300">{last.result.error}</p>}
        </section>
      })}
    </div>
    </details>
  </main>
}
