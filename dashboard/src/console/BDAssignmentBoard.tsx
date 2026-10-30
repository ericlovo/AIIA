import { formatSeconds, jobLabel, medianLatencySeconds, modelWorkingJobs, unavailableCount, type BDSnapshot } from './bdModel'

export function BDAssignmentBoard({ data, now, stopped }: { data: BDSnapshot; now: number; stopped?: boolean }) {
  const qwenJobs = modelWorkingJobs(data, 'qwen', now)
  const jevJobs = modelWorkingJobs(data, 'jev', now)
  const extractWait = data.jobs.filter(job => job.state === 'queued' || job.state === 'leased' || job.stage === 'queued')
  const judgeWait = data.jobs.filter(job => job.stage === 'extract' && job.state === 'running')
  const qwenNow = qwenJobs[0]
  const jevNow = jevJobs[0]
  const off = stopped || data.kill_switch || !data.schedule_enabled
  const qwenLabel = off ? (data.kill_switch ? 'Stopped' : 'Paused') : qwenNow ? `Extracting ${jobLabel(qwenNow, data.packets)}` : 'Idle, nothing waiting'
  const jevLabel = off ? (data.kill_switch ? 'Stopped' : 'Paused') : jevNow ? `Judging ${jobLabel(jevNow, data.packets)}` : 'Idle, nothing waiting'
  const mix = data.assignments?.jev.verdict_mix ?? data.jev.verdict_mix
  const mixTotal = mix ? (mix.Choice ?? 0) + (mix.Score ?? 0) + (mix.Noul ?? 0) : 0
  const qwenLatency = medianLatencySeconds(data.events, ['qwen_completed', 'qwen_replayed'])
  const jevLatency = medianLatencySeconds(data.events, ['jev_completed', 'jev_replayed'])
  const qwenLive = data.events.filter(event => event.kind === 'qwen_completed' && event.mode === 'live' && Date.parse(event.created_at) > now - 60_000).length
  const throughput = off
    ? `0 jobs a minute (${data.kill_switch ? 'stopped' : 'paused'})`
    : data.mode === 'replay'
      ? 'Recorded replay · local, no live cost'
      : data.health.qwen.status === 'data_unavailable' && qwenLive === 0
        ? 'data_unavailable'
        : `About ${Math.max(qwenLive, data.assignments?.qwen.calls ?? 0)} jobs recently · local, no cost`
  return <section className="bd-panel" aria-labelledby="h-workers">
    <h2 id="h-workers">Jev and Qwen, right now</h2>
    <div className="bd-workers">
      <article className={`bd-worker bd-model-qwen ${qwenJobs.length ? 'bd-model-working is-working' : ''}`} aria-label="Qwen assignment">
        <h3>Qwen</h3>
        <p className="bd-role">Local · reads public records, drafts</p>
        <dl className="bd-kv">
          <div><dt>Doing now</dt><dd className="bd-doing">{qwenLabel}</dd></div>
          <div><dt>Waiting to extract</dt><dd>{extractWait.length} jobs</dd></div>
          <div><dt>Throughput</dt><dd>{throughput}</dd></div>
          <div><dt>Median time per job</dt><dd>{formatSeconds(qwenLatency ?? (data.assignments?.qwen.latency_ms == null ? null : data.assignments.qwen.latency_ms / 1000))}</dd></div>
        </dl>
      </article>
      <article className={`bd-worker bd-model-jev ${jevJobs.length ? 'bd-model-working is-working' : ''}`} aria-label="Jev assignment">
        <h3>Jev</h3>
        <p className="bd-role">Makes narrow judgments</p>
        <dl className="bd-kv">
          <div><dt>Doing now</dt><dd className="bd-doing">{jevLabel}</dd></div>
          <div><dt>Waiting to judge</dt><dd>{judgeWait.length} jobs</dd></div>
          <div>
            <dt><span className="bd-tip" title="Jev's typed verdicts" tabIndex={0}>Verdicts today</span></dt>
            <dd>
              {mix == null ? 'data_unavailable' : <div className="bd-bars" aria-label="Jev verdict mix">
                {(['Choice', 'Score', 'Noul'] as const).map(kind => <div className="bd-bar" key={kind}><span>{kind}</span><i><b style={{ width: `${mixTotal ? Math.round((mix[kind] ?? 0) / mixTotal * 100) : 0}%` }} /></i><span>{unavailableCount(mix[kind])}</span></div>)}
              </div>}
            </dd>
          </div>
          <div><dt>Median time per job</dt><dd>{formatSeconds(jevLatency ?? (data.jev.latency_ms == null ? null : data.jev.latency_ms / 1000))}</dd></div>
        </dl>
      </article>
    </div>
  </section>
}
