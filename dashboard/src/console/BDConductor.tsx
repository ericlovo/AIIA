import { studioFetch, studioJSON } from '../lib/connection'
import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { packetOutcome, outcomeLabel, researchReason, plannedWaiting, providerStatus, scheduleQueueText, ageLabel, heartbeatFresh, publicSource, statusBanner, type BDSnapshot, type Budget } from './bdModel'
import { BDScheduleStrip } from './BDScheduleStrip'
import { BDResearchHistory } from './BDResearchHistory'
import { BDActivityBoard } from './BDActivityBoard'
import './BDConductor.css'

async function request(path: string, body?: unknown): Promise<BDSnapshot> {
  const response = await studioFetch(`/api/bd/${path}`, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const result = await studioJSON<BDSnapshot>(response)
  if (path === 'snapshot' && (!Array.isArray(result?.bots) || !Array.isArray(result?.jobs) || !Array.isArray(result?.packets) || !Array.isArray(result?.events) || !result?.health || !result?.jev || !result?.budget)) {
    throw new Error('Conductor returned an invalid snapshot.')
  }
  return result
}
const metric = (value: number | null | undefined) => value == null ? 'Unavailable' : value.toLocaleString()
const healthNames = { qwen: 'Qwen / Ollama', jev: 'Jev router', managed_leads: 'Managed Leads' }

function BudgetReadout({ budget }: { budget: Budget }) {
  return <dl className="bd-budget">{(['tokens', 'calls', 'seconds'] as const).map(key => <div key={key}>
    <dt>{key === 'seconds' ? 'Wall time (s)' : key}</dt>
    <dd>{metric(budget[key]?.used)} <span>/ {metric(budget[key]?.limit)}</span></dd>
  </div>)}</dl>
}

export function BDConductor() {
  const qc = useQueryClient()
  const [stream, setStream] = useState('Connecting')
  const [now, setNow] = useState(() => Date.now())
  const [lane, setLane] = useState('all')
  const [outcomeFilter, setOutcomeFilter] = useState('supported')
  const [reviewFilter, setReviewFilter] = useState('active')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [reviewTarget, setReviewTarget] = useState('')
  const [pausedFeed, setPausedFeed] = useState(false)
  const [frozenEvents, setFrozenEvents] = useState<BDSnapshot['events']>([])
  const query = useQuery({ queryKey: ['bd-snapshot'], queryFn: () => request('snapshot'), refetchInterval: 15_000, retry: 1 })
  const data = query.data
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    const source = new EventSource('/api/bd/events')
    source.onopen = () => setStream('Live')
    source.onerror = () => setStream('Reconnecting · polling every 15s')
    source.addEventListener('snapshot', event => {
      try {
        const snapshot = JSON.parse((event as MessageEvent).data) as BDSnapshot
        if (!Array.isArray(snapshot?.bots) || !Array.isArray(snapshot?.jobs) || !Array.isArray(snapshot?.packets) || !Array.isArray(snapshot?.events) || !snapshot?.health || !snapshot?.jev || !snapshot?.budget) throw new Error('Invalid snapshot')
        qc.setQueryData(['bd-snapshot'], snapshot)
        setStream('Live')
      } catch { setStream('Invalid stream data · polling every 15s') }
    })
    return () => { source.close(); window.clearInterval(timer) }
  }, [qc])
  useEffect(() => {
    if (!reviewTarget) return
    const card = document.getElementById(`bd-packet-${reviewTarget}`)
    card?.scrollIntoView({ block: 'start', behavior: 'instant' })
    card?.focus({ preventScroll: true })
  }, [reviewTarget, reviewFilter, outcomeFilter, lane])
  const action = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => request(path, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['bd-snapshot'] }),
  })
  const botName = (id: string) => data?.bots.find(bot => bot.id === id)?.name ?? id
  const active = data?.jobs.filter(job => ['running', 'leased'].includes(job.state)) ?? []
  const lanePackets = data?.packets.filter(packet => lane === 'all' || packet.bot_id === lane) ?? []
  const pendingCount = lanePackets.filter(packet => packetOutcome(packet) === 'supported' && packet.review_status === 'human_review_required').length
  const packets = lanePackets.filter(packet => outcomeFilter === 'all' || (outcomeFilter === 'supported' ? packetOutcome(packet) === 'supported' : packetOutcome(packet) !== 'supported')).filter(packet => reviewFilter === 'active' ? packet.review_status !== 'dismissed' : packet.review_status === reviewFilter)
  const visibleActive = active.filter(job => lane === 'all' || job.bot_id === lane)
  const events = (pausedFeed ? frozenEvents : data?.events ?? []).filter(event => lane === 'all' || event.bot_id === lane)
  const reviewPacket = (packet: BDSnapshot['packets'][number]) => {
    setExpanded(current => ({ ...current, [packet.id]: true }))
    setLane('all')
    setOutcomeFilter(packetOutcome(packet) === 'supported' ? 'supported' : 'history')
    setReviewFilter(packet.review_status === 'dismissed' ? 'dismissed' : 'active')
    setReviewTarget(packet.id)
    if (reviewTarget === packet.id) {
      const card = document.getElementById(`bd-packet-${packet.id}`)
      card?.scrollIntoView({ block: 'start', behavior: 'instant' })
      card?.focus({ preventScroll: true })
    }
  }
  return <main className="bd-page">
    <div className="bd-wrap">
      <header className="bd-heading">
        <div><a className="bd-back" href="#/note">Morning note</a><p className="bd-eyebrow">AIIA / BUSINESS DEVELOPMENT</p><h1>The research desk.</h1><p className="bd-intro">Review completed research below, inspect the queue, and follow the planned schedule.</p></div>
        <div className="bd-actions">
          <span className={`bd-stream ${stream === 'Live' ? 'is-live' : ''}`} role="status"><i aria-hidden="true" />{stream}</span>
          <button className="bd-activate" disabled={!data || action.isPending || query.isError} onClick={() => action.mutate({ path: 'schedule', body: { enabled: !data?.schedule_enabled } })}>{data?.schedule_enabled ? 'Pause daily schedule' : 'Activate daily research'}</button>
          <button className="bd-primary" disabled={!data || data.mode !== 'replay' || data.kill_switch || action.isPending || query.isError} onClick={() => action.mutate({ path: 'shadow', body: {} })}>{action.isPending && action.variables?.path === 'shadow' ? 'Starting replay…' : 'Run shadow batch'}</button>
          <button className="bd-stop" disabled={!data || action.isPending || query.isError} onClick={() => action.mutate({ path: 'kill-switch', body: { enabled: !data?.kill_switch } })}>{data?.kill_switch ? 'Release kill switch' : 'Stop conductor'}</button>
        </div>
      </header>
      {action.isError && <p className="bd-error" role="alert">Action failed: {action.error.message}</p>}
      {action.isSuccess && <p className="bd-notice" role="status">{action.variables.path === 'shadow' ? 'Shadow batch accepted. Follow its progress below.' : action.variables.path === 'kill-switch' ? 'Kill-switch change accepted.' : action.variables.path === 'schedule' ? (action.variables.body && typeof action.variables.body === 'object' && 'enabled' in action.variables.body && action.variables.body.enabled ? 'Daily research schedule is on. The worker will rotate enabled bots during Chicago working hours.' : 'Daily research schedule is paused. Existing jobs stay queued.') : 'Local research triage saved.'}</p>}
      {query.isError && <div className="bd-error" role="alert">data_unavailable: {query.error.message} {data && 'Showing the last received snapshot; controls are disabled.'}<button onClick={() => query.refetch()}>Retry connection</button></div>}
      {!data && query.isLoading && <p className="bd-empty" role="status">Connecting to the conductor…</p>}
      {data && <>
        <section className={`bd-status-banner is-${statusBanner(data).state}`} role="status" aria-label="Conductor status">
          <strong>{statusBanner(data).label}</strong>
          <span>{statusBanner(data).state === 'running' ? 'Standing Chicago cadence is armed. Quiet time still means the worker is on.' : statusBanner(data).state === 'paused_kill_switch' ? 'No jobs will lease until the kill switch is released.' : 'One labelled action turns the standing schedule on. Nothing runs until you do.'}</span>
        </section>
        <section className="bd-mode" aria-label="Operating mode">
          <div><strong>{data.mode === 'replay' ? 'Shadow · replay only' : data.mode}</strong><span>{data.mode === 'replay' ? 'Recorded judgments and synthetic candidates. No outreach.' : 'Public-source research. Human review before outreach.'}</span></div>
          <div><strong>Schedule {data.schedule_enabled ? 'enabled' : 'disabled'}</strong><span>{plannedWaiting(data.schedule, now) ? 'Waiting for planned start' : data.kill_switch ? 'Kill switch engaged · work paused' : 'Kill switch released'}</span></div>
          <div><strong>Research → proposal → human</strong><span>Packets are evidence for review, not qualified leads.</span></div>
        </section>
        {data.daily_summary && <section className="bd-daily-summary" aria-label="Daily research summary">
          <div><span className="bd-machine-label">TODAY · {data.daily_summary.timezone}</span><strong>{data.daily_summary.day}</strong></div>
          <dl>
            <div><dt>Candidates proposed</dt><dd>{metric(data.daily_summary.candidates_proposed)}</dd></div>
            <div><dt>Inconclusive</dt><dd>{metric(data.daily_summary.inconclusive)}</dd></div>
            <div><dt>Failed</dt><dd>{metric(data.daily_summary.failed)}</dd></div>
            <div><dt>Awaiting human review</dt><dd>{metric(data.daily_summary.awaiting_human_review)}</dd></div>
          </dl>
        </section>}
        <BDScheduleStrip data={data} now={now} />
        <div className="bd-view-controls"><span>Research flow</span><label>Show <select value={lane} onChange={event => setLane(event.target.value)}><option value="all">All segments & clients</option>{data.bots.map(bot => <option key={bot.id} value={bot.id}>{bot.name}</option>)}</select></label></div>
        <BDActivityBoard data={data} now={now} lane={lane} onReview={reviewPacket} />
        <BDResearchHistory data={data} now={now} lane={lane} onReview={reviewPacket} />
        <details className="bd-connection-details"><summary>Connections & integration health</summary>
        <section className="bd-health" aria-label="Connection health">
          {Object.entries(healthNames).map(([key, name]) => {
            const item = data.health[key as keyof typeof healthNames]
            return <div key={key}><span className={`bd-dot ${item?.status === 'available' ? 'is-live' : ''}`} aria-hidden="true" /><div><strong>{name}</strong><span>{providerStatus(item)}{item?.detail ? ` · ${item.detail}` : ''}</span></div></div>
          })}
        </section>
        </details>
        <div className="bd-section-heading"><h2>Research lanes</h2><span className="bd-muted">Bot budgets & completed jobs</span></div>
        <section className="bd-lanes" aria-label="Research lanes">
          {data.bots.map(bot => <article className={`bd-lane ${lane === bot.id ? 'selected' : ''}`} key={bot.id}>
            <p className="bd-eyebrow">{bot.tenant_id ? `Tenant · ${bot.tenant_id}` : 'Industry research'}</p><h3>{bot.name}</h3><p className="bd-muted">{bot.segment}</p>
            <dl className="bd-counts"><div><dt>Queued</dt><dd>{metric(bot.counts.queued)}</dd></div><div><dt>Running</dt><dd>{metric(bot.counts.running)}</dd></div><div><dt>Done</dt><dd>{metric(bot.counts.completed)}</dd></div></dl>
            {bot.counts.failed > 0 && <p className="bd-failure">{bot.counts.failed} failed · inspect live feed</p>}
            <details><summary>Bot budget</summary><BudgetReadout budget={bot.budgets} /></details>
          </article>)}
        </section>
        <div className="bd-columns">
          <div className="bd-main-column">
            <section className="bd-section" aria-label="Now running">
              <div className="bd-section-heading"><h2>Now running <span>{visibleActive.length}</span></h2><span className="bd-muted">Elapsed / heartbeat</span></div>
              {visibleActive.length === 0 ? <p className="bd-empty">No active jobs in this snapshot. {scheduleQueueText(data, now)}</p> : <ul className="bd-jobs">{visibleActive.map(job => <li key={job.id}><span className={`bd-dot ${heartbeatFresh(job.heartbeat_at, now) ? 'is-live bd-pulse' : ''}`} aria-hidden="true" /><div><strong>{botName(job.bot_id)}</strong><p>{job.step || job.state}</p><small>{job.tenant_id} · {job.segment} · {job.id}</small></div><div className="bd-job-time"><strong>{ageLabel(job.started_at, now)}</strong><small>Heartbeat {ageLabel(job.heartbeat_at, now)} ago</small></div></li>)}</ul>}
            </section>
            <section className="bd-section" aria-label="Waiting on a human">
              <div className="bd-section-heading"><h2>Research packets <span>{packets.length}</span></h2><label>Outcome <select aria-label="Research outcome" value={outcomeFilter} onChange={event => setOutcomeFilter(event.target.value)}><option value="supported">Supported candidates</option><option value="history">Rejected / inconclusive</option><option value="all">All research</option></select></label><label>Review <select aria-label="Packet review status" value={reviewFilter} onChange={event => setReviewFilter(event.target.value)}><option value="active">Active packets</option><option value="human_review_required">Waiting on a human</option><option value="research_next">Research next</option><option value="watch">Watch</option><option value="dismissed">Dismissed</option></select></label></div>
              <p className="bd-muted">{pendingCount} supported candidates waiting on a human. Local triage only; no outreach or CRM changes.</p>
              {packets.length === 0 && <p className="bd-empty">No packets in this view yet. Completed research will appear here.</p>}
              <div className="bd-packets">{packets.map(packet => <article key={packet.id} id={`bd-packet-${packet.id}`} tabIndex={-1} className="bd-packet">
                <div className="bd-packet-meta"><span>{packet.tenant_id} / {packet.segment}</span><span>{ageLabel(packet.created_at, now)} ago</span></div>
                <h3>{packet.organization || 'Unnamed research result'}</h3><p className="bd-outcome-label">{outcomeLabel(packetOutcome(packet))}</p><p>{researchReason(packet.candidate_reason ?? (packetOutcome(packet) === 'supported' ? 'source_supported_candidate' : undefined))}</p>{packet.next_query && packetOutcome(packet) === 'inconclusive' && <p className="bd-next-query">Next query: {packet.next_query}</p>}<p>{packet.summary}</p>
                <div className="bd-tags"><span>{packet.mode}</span><span>{packet.review_status.replaceAll('_', ' ')}</span><span>{botName(packet.bot_id)}</span></div>
                <details open={!!expanded[packet.id]} onToggle={event => { const open = event.currentTarget.open; setExpanded(current => current[packet.id] === open ? current : { ...current, [packet.id]: open }) }}><summary>Review evidence & contacts</summary><div className="bd-evidence">
                  {publicSource(packet.source_url) ? <a href={publicSource(packet.source_url)!} target="_blank" rel="noopener noreferrer">Open source evidence ↗</a> : <p>Source evidence unavailable</p>}
                  {packet.source_quote && <blockquote>{packet.source_quote}</blockquote>}<p className="bd-contact-hold">Contact hold: {packet.contact_policy?.replaceAll('_', ' ') || 'pending tenant restriction check'}</p>{packet.disposition && <p>Disposition: {packet.disposition.replaceAll('_', ' ')}</p>}<h4>Public contacts · unverified</h4><p>Email: {packet.emails.length ? packet.emails.join(', ') : 'Not found in source'}</p><p>Phone: {packet.phones.length ? packet.phones.join(', ') : 'Not found in source'}</p>
                  <h4>Market signals</h4>{packet.signals.length ? <ul>{packet.signals.map((signal, i) => <li key={i}>{signal}</li>)}</ul> : <p>No signals recorded</p>}
                  <h4>Unverified hypotheses</h4>{packet.insights.length ? <ul>{packet.insights.map((insight, i) => <li key={i}>{insight}</li>)}</ul> : <p>No insights recorded</p>}
                </div></details>
                <div className="bd-triage" aria-label={`Triage ${packet.organization}`}>
                  {(packet.review_status === 'dismissed' ? [['human_review_required', 'Restore']] : [['research_next', 'Research next'], ['watch', 'Watch'], ['dismissed', 'Dismiss']]).map(([status, label]) => <button key={status} disabled={action.isPending || query.isError || packet.review_status === status} onClick={() => action.mutate({ path: `packets/${encodeURIComponent(packet.job_id)}/${packet.index}/review`, body: { status } })}>{label}</button>)}
                </div>
              </article>)}</div>
            </section>
          </div>
          <aside className="bd-sidebar">
            <section className="bd-section" aria-label="Live feed"><div className="bd-section-heading"><h2>Live feed</h2><button aria-pressed={pausedFeed} onClick={() => { if (!pausedFeed) setFrozenEvents(data.events); setPausedFeed(!pausedFeed) }}>{pausedFeed ? 'Resume feed' : 'Pause feed'}</button></div>
              <p className="bd-muted">Newest first{pausedFeed ? ' · display paused' : ''}</p>
              {events.length === 0 && <p className="bd-empty">Waiting for conductor events.</p>}
              <ol className="bd-events">{events.slice(0, 40).map(event => <li key={event.id} className={event.kind.includes('fail') ? 'bd-event-failed' : ''}><div><strong>{event.kind.replaceAll('_', ' ')}</strong><time dateTime={event.created_at}>{ageLabel(event.created_at, now)} ago</time></div><p>{event.message}</p><small>{botName(event.bot_id)}</small></li>)}</ol>
            </section>
            <section className="bd-section" aria-label="Jev router"><div className="bd-section-heading"><h2>Jev router{data.mode === 'replay' ? ' · replay' : ''}</h2><span className="bd-muted">{providerStatus(data.health.jev)}</span></div><dl className="bd-router"><div><dt>{data.mode === 'replay' ? 'Replayed calls / min' : 'Calls / minute'}</dt><dd>{metric(data.jev.calls_per_minute)}</dd></div><div><dt>Latency</dt><dd>{metric(data.jev.latency_ms)}{data.jev.latency_ms !== null ? ' ms' : ''}</dd></div><div><dt>Errors</dt><dd>{metric(data.jev.errors)}</dd></div></dl><p className="bd-muted">Verdict mix</p><div className="bd-tags">{(['Choice', 'Score', 'Noul'] as const).map(type => <span key={type}>{type} {metric(data.jev.verdict_mix?.[type])}</span>)}</div></section>
            <section className="bd-section" aria-label="Daily budget"><div className="bd-section-heading"><h2>Daily budget</h2></div><BudgetReadout budget={data.budget} /><p className="bd-muted">Reserved use / configured limit · UTC day</p></section>
          </aside>
        </div>
      </>}
    </div>
  </main>
}
