import { ageLabel, latestPackets, modelWorkingJobs, queueJobs, type BDPacket, type BDSnapshot } from './bdModel'

const count = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString()

function ModelWork({ data, provider, now, lane }: { data: BDSnapshot; provider: 'qwen' | 'jev'; now: number; lane: string }) {
  const activity = data.model_activity?.[provider]
  const working = modelWorkingJobs(data, provider, now).filter(job => lane === 'all' || job.bot_id === lane)
  const recent = data.events.find(event => ['called', 'completed', 'error', 'replayed'].some(phase => event.kind === `${provider}_${phase}`) && (lane === 'all' || event.bot_id === lane))
  const replay = data.mode === 'replay'
  const unavailable = data.health[provider]?.status === 'data_unavailable'
  const status = working.length ? 'Working now' : replay ? 'Recorded replay' : unavailable ? 'Unavailable' : 'Idle'
  return <article className={`bd-model-card bd-model-${provider} ${working.length ? 'bd-model-working' : ''}`} aria-label={`${provider === 'qwen' ? 'Qwen' : 'Jev'} activity`}>
    <div className="bd-model-heading"><div><span className="bd-machine-label">{provider === 'qwen' ? 'LOCAL EXTRACTION' : 'TYPED JUDGMENT'}</span><h3>{provider === 'qwen' ? 'Qwen' : 'Jev'}</h3></div><span className={`bd-work-state ${working.length ? 'is-working' : ''}`}><i aria-hidden="true" />{status}</span></div>
    <p className="bd-model-task">{working.length ? working.map(job => data.bots.find(bot => bot.id === job.bot_id)?.name ?? job.bot_id).join(' · ') : provider === 'qwen' ? 'Public evidence → structured research' : 'Research → fit, priority & judgment'}</p>
    <dl className="bd-model-counts"><div><dt>{replay ? 'Recorded results' : 'Attempts'}</dt><dd>{count(replay ? activity?.replayed : activity?.attempts ?? activity?.calls)}</dd></div><div><dt>{replay ? 'Live calls' : 'Completed'}</dt><dd>{count(replay ? activity?.calls : activity?.completed)}</dd></div><div><dt>Errors</dt><dd>{count(activity?.errors)}</dd></div></dl>
    <div className="bd-model-last"><span>{recent ? recent.kind.replaceAll('_', ' ') : 'No phase event in this view'}</span>{recent && <time dateTime={recent.created_at}>{ageLabel(recent.created_at, now)} ago</time>}</div>
    <p className="bd-model-scope">{activity ? 'Counters: latest 200 events, all lanes.' : 'Activity counters unavailable.'}{replay ? ' No inference is running.' : ''}</p>
  </article>
}

export function BDActivityBoard({ data, now, lane, onReview }: { data: BDSnapshot; now: number; lane: string; onReview: (packet: BDPacket) => void }) {
  const jobs = data.jobs.filter(job => lane === 'all' || job.bot_id === lane)
  const queue = queueJobs(jobs)
  const totals = lane === 'all' ? data.job_counts : undefined
  const queueTotal = totals ? totals.queued + totals.leased : queue.length
  const active = jobs.filter(job => job.state === 'running')
  const fresh = active.some(job => modelWorkingJobs(data, 'qwen', now).some(work => work.id === job.id) || modelWorkingJobs(data, 'jev', now).some(work => work.id === job.id))
  const completed = jobs.filter(job => job.state === 'completed')
  const packets = latestPackets(data.packets.filter(packet => lane === 'all' || packet.bot_id === lane))
  const pending = packets.filter(packet => packet.review_status === 'human_review_required')
  const botName = (id: string) => data.bots.find(bot => bot.id === id)?.name ?? id
  return <section className="bd-activity-board" aria-label="Research activity">
    <div className="bd-flow-summary">
      <div className="bd-flow-state"><span className={`bd-work-state ${fresh ? 'is-working' : ''}`}><i aria-hidden="true" />{data.kill_switch ? 'Conductor stopped' : active.length ? data.mode === 'replay' ? 'Replay in progress' : 'Research in progress' : queueTotal ? 'Work queued' : 'Waiting for work'}</span><p>{data.mode === 'replay' ? 'Synthetic research · replay only' : 'Public-source research · human review'}</p></div>
      <dl className="bd-flow-counts"><div><dt>Queued / leased</dt><dd>{queueTotal}</dd></div><div><dt>Running</dt><dd>{totals?.running ?? active.length}</dd></div><div><dt>Jobs complete</dt><dd>{totals?.completed ?? completed.length}</dd></div><div><dt>To review</dt><dd>{pending.length}</dd></div></dl>
    </div>
    <div className="bd-action-grid">
      <section className="bd-queue-panel" aria-label="Work queue"><div className="bd-panel-title"><h2>Up next <span>{queueTotal}</span></h2><span>Oldest first</span></div>
        {queue.length ? <ol className="bd-queue-list">{queue.slice(0, 4).map((job, index) => <li key={job.id}><span className="bd-queue-position">{String(index + 1).padStart(2, '0')}</span><div><strong>{botName(job.bot_id)}</strong><p>{job.segment}</p><small>{job.state === 'leased' ? 'Leased · awaiting execution' : job.step && job.step !== 'queued' ? job.step.replaceAll('_', ' ') : 'Queued'} · {ageLabel(job.created_at, now)}</small></div></li>)}</ol> : <div className="bd-queue-empty"><strong>{queueTotal ? 'Queue outside the visible window.' : 'The queue is clear.'}</strong><p>{queueTotal ? `${queueTotal} jobs are queued or leased; active jobs fill the current snapshot.` : data.kill_switch ? 'Release the kill switch before starting work.' : data.schedule_enabled ? 'Waiting for the next scheduled cycle.' : data.mode === 'replay' ? 'The schedule is off. Start a shadow batch to see the research flow.' : 'No jobs are queued. The schedule is disabled.'}</p></div>}
        {queue.length > 4 && <details className="bd-queue-more"><summary>{queue.length - 4} more queued jobs</summary><ul>{queue.slice(4).map(job => <li key={job.id}>{botName(job.bot_id)} · {job.state} · {ageLabel(job.created_at, now)}</li>)}</ul></details>}
        <p className="bd-board-note">{totals ? 'All recorded jobs' : 'Visible job window'} · {totals?.failed ?? jobs.filter(job => job.state === 'failed').length} failed · {totals?.cancelled ?? jobs.filter(job => job.state === 'cancelled').length} cancelled</p>{queueTotal > 0 && <p className="bd-board-note">{queue.length} of {queueTotal} queued / leased jobs in this snapshot.</p>}
      </section>
      <div className="bd-model-stack"><ModelWork data={data} provider="qwen" now={now} lane={lane} /><ModelWork data={data} provider="jev" now={now} lane={lane} /></div>
      <section className="bd-delivered-panel" aria-label="Completed research"><div className="bd-panel-title"><h2>Research delivered <span>{packets.length}</span></h2><span>Newest first</span></div><p className="bd-delivered-note">Packets delivered here for review. No emails sent.</p>
        {packets.length ? <ol className="bd-delivered-list">{packets.slice(0, 3).map(packet => <li key={packet.id}><div className="bd-delivered-stamp"><span>{packet.mode === 'replay' ? 'REPLAY PACKET' : 'RESEARCH COMPLETE'}</span><time dateTime={packet.created_at}>{ageLabel(packet.created_at, now)} ago</time></div><h3>{packet.organization}</h3><p>{botName(packet.bot_id)}</p><div className="bd-delivered-facts"><span>{packet.signals.length} {packet.signals.length === 1 ? 'signal' : 'signals'}</span><span>{packet.emails.length} {packet.emails.length === 1 ? 'email' : 'emails'} found</span><span>{packet.phones.length} {packet.phones.length === 1 ? 'phone' : 'phones'} found</span></div><button onClick={() => onReview(packet)}>Review packet <span aria-hidden="true">→</span></button></li>)}</ol> : <div className="bd-queue-empty"><strong>No research delivered yet.</strong><p>Completed work will land here with its sources and contact restrictions.</p></div>}
      </section>
    </div>
  </section>
}
