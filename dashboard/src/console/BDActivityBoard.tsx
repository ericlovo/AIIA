import { packetOutcome, scheduleIdleText, scheduleQueueText, ageLabel, latestPackets, modelWorkingJobs, queueJobs, type BDPacket, type BDSnapshot } from './bdModel'
import { BDAssignmentBoard } from './BDAssignmentBoard'

export function BDActivityBoard({ data, now, lane, onReview }: { data: BDSnapshot; now: number; lane: string; onReview: (packet: BDPacket) => void }) {
  const jobs = data.jobs.filter(job => lane === 'all' || job.bot_id === lane)
  const queue = queueJobs(jobs)
  const totals = lane === 'all' ? data.job_counts : undefined
  const queueTotal = totals ? totals.queued + totals.leased : queue.length
  const active = jobs.filter(job => job.state === 'running')
  const fresh = active.some(job => modelWorkingJobs(data, 'qwen', now).some(work => work.id === job.id) || modelWorkingJobs(data, 'jev', now).some(work => work.id === job.id))
  const completed = jobs.filter(job => job.state === 'completed')
  const packets = latestPackets(data.packets.filter(packet => (lane === 'all' || packet.bot_id === lane) && packetOutcome(packet) === 'supported' && packet.review_status !== 'dismissed'))
  const pending = packets.filter(packet => packet.review_status === 'human_review_required')
  const botName = (id: string) => data.bots.find(bot => bot.id === id)?.name ?? id
  return <section className="bd-activity-board" aria-label="Research activity">
    <div className="bd-flow-summary">
      <div className="bd-flow-state"><span className={`bd-work-state ${fresh ? 'is-working' : ''}`}><i aria-hidden="true" />{data.kill_switch || !active.length && !queueTotal ? scheduleIdleText(data, now) : active.length ? data.mode === 'replay' ? 'Replay in progress' : 'Research in progress' : 'Work queued'}</span><p>{data.mode === 'replay' ? 'Synthetic research · replay only' : 'Public-source research · human review'}</p></div>
      <dl className="bd-flow-counts"><div><dt>Queued / leased</dt><dd>{queueTotal}</dd></div><div><dt>Running</dt><dd>{totals?.running ?? active.length}</dd></div><div><dt>Jobs complete</dt><dd>{totals?.completed ?? completed.length}</dd></div><div><dt>Candidates to review</dt><dd>{pending.length}</dd></div></dl>
    </div>
    <div className="bd-action-grid">
      <section className="bd-queue-panel" aria-label="Work queue"><div className="bd-panel-title"><h2>Up next <span>{queueTotal}</span></h2><span>Oldest first</span></div>
        {queue.length ? <ol className="bd-queue-list">{queue.slice(0, 4).map((job, index) => <li key={job.id}><span className="bd-queue-position">{String(index + 1).padStart(2, '0')}</span><div><strong>{botName(job.bot_id)}</strong><p>{job.segment}</p><small>{job.state === 'leased' ? 'Leased · awaiting execution' : job.step && job.step !== 'queued' ? job.step.replaceAll('_', ' ') : 'Queued'} · {ageLabel(job.created_at, now)}</small></div></li>)}</ol> : <div className="bd-queue-empty"><strong>{queueTotal ? 'Queue outside the visible window.' : 'The queue is clear.'}</strong><p>{queueTotal ? `${queueTotal} jobs are queued or leased; active jobs fill the current snapshot.` : scheduleQueueText(data, now)}</p></div>}
        {queue.length > 4 && <details className="bd-queue-more"><summary>{queue.length - 4} more queued jobs</summary><ul>{queue.slice(4).map(job => <li key={job.id}>{botName(job.bot_id)} · {job.state} · {ageLabel(job.created_at, now)}</li>)}</ul></details>}
        <p className="bd-board-note">{totals ? 'All recorded jobs' : 'Visible job window'} · {totals?.failed ?? jobs.filter(job => job.state === 'failed').length} failed · {totals?.cancelled ?? jobs.filter(job => job.state === 'cancelled').length} cancelled</p>{queueTotal > 0 && <p className="bd-board-note">{queue.length} of {queueTotal} queued / leased jobs in this snapshot.</p>}
      </section>
      <BDAssignmentBoard data={data} now={now} />
      <section className="bd-delivered-panel" aria-label="Completed research"><div className="bd-panel-title"><h2>Supported candidates <span>{packets.length}</span></h2><span>Newest first</span></div><p className="bd-delivered-note">Evidence supports further research, not qualification or outreach. Counts reflect this snapshot.</p>
        {packets.length ? <ol className="bd-delivered-list">{packets.slice(0, 3).map(packet => <li key={packet.id}><div className="bd-delivered-stamp"><span>{packet.mode === 'replay' ? 'REPLAY CANDIDATE' : 'SUPPORTED CANDIDATE'}</span><time dateTime={packet.created_at}>{ageLabel(packet.created_at, now)} ago</time></div><h3>{packet.organization}</h3><p>{botName(packet.bot_id)}</p><div className="bd-delivered-facts"><span>{packet.signals.length} {packet.signals.length === 1 ? 'signal' : 'signals'}</span><span>{packet.emails.length} {packet.emails.length === 1 ? 'email' : 'emails'} found</span><span>{packet.phones.length} {packet.phones.length === 1 ? 'phone' : 'phones'} found</span></div><button onClick={() => onReview(packet)}>Review packet <span aria-hidden="true">→</span></button></li>)}</ol> : <div className="bd-queue-empty"><strong>No supported candidates in this view.</strong><p>Rejected and inconclusive results remain in Research history below.</p></div>}
      </section>
    </div>
  </section>
}
