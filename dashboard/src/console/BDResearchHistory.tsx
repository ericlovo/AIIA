import { ageLabel, latestPackets, outcomeLabel, packetOutcome, researchReason, type BDPacket, type BDSnapshot } from './bdModel'

export function BDResearchHistory({ data, now, lane, onReview }: { data: BDSnapshot; now: number; lane: string; onReview: (packet: BDPacket) => void }) {
  const packets = latestPackets(data.packets.filter(packet => (lane === 'all' || packet.bot_id === lane) && packetOutcome(packet) !== 'supported'))
  const jobs = data.jobs.filter(job => job.state === 'completed' && (lane === 'all' || job.bot_id === lane) && !data.packets.some(packet => packet.job_id === job.id) && job.research_outcome !== 'supported')
    .sort((a, b) => Date.parse(b.finished_at ?? b.created_at) - Date.parse(a.finished_at ?? a.created_at))
  const total = packets.length + jobs.length
  return <section className="bd-research-history" aria-label="Research history">
    <div className="bd-section-heading"><h2>Research history <span>{total}</span></h2><span className="bd-muted">Rejected / inconclusive</span></div>
    <p className="bd-muted">Retained results and completed runs that did not produce a supported candidate. Counts reflect this snapshot.</p>
    {total === 0 ? <p className="bd-empty">No rejected or inconclusive results in this snapshot.</p> : <details><summary>Inspect research outcomes ({total})</summary>
      <ul className="bd-history-list">
        {packets.map(packet => <li key={packet.id}><div><strong>{packet.organization || 'Unnamed research result'}</strong><p>{outcomeLabel(packetOutcome(packet))} · {packet.segment} · {packet.mode}</p><p>{researchReason(packet.candidate_reason)}</p>{packet.next_query && <p>Next query: {packet.next_query}</p>}<small>{ageLabel(packet.created_at, now)} ago · {packet.review_status.replaceAll('_', ' ')}</small></div><button onClick={() => onReview(packet)}>Inspect evidence</button></li>)}
        {jobs.map(job => <li key={job.id}><div><strong>{data.bots.find(bot => bot.id === job.bot_id)?.name ?? job.bot_id}</strong><p>{outcomeLabel(job.research_outcome ?? 'inconclusive')} · {job.segment} · {job.mode ?? data.mode} · no candidate packet</p><p>{researchReason(job.result_reason)}</p>{job.next_query && <p>Next query: {job.next_query}</p>}<small>{ageLabel(job.finished_at ?? job.created_at, now)} ago · {job.id}</small></div></li>)}
      </ul>
    </details>}
  </section>
}
