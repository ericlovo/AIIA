import { studioFetch, studioJSON } from '../lib/connection'
import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ageLabel, cadencePhrase, jobLabel, jobStage, laneProfile, leadBanner, leadContact, operatorTargets,
  orderedBots, packetOutcome, providerStatus, publicSource, researchReason, weekdayInterval,
  type BDPacket, type BDSnapshot,
} from './bdModel'
import { BDAssignmentBoard } from './BDAssignmentBoard'
import { BDSteeringControls, type SteerBody } from './BDSteeringControls'
import { BDScheduleStrip } from './BDScheduleStrip'
import { BDResearchHistory } from './BDResearchHistory'
import './BDConductor.css'

const MAX_CARDS = 5
const healthNames = { qwen: 'Qwen / Ollama', jev: 'Jev router', managed_leads: 'Managed Leads' }

async function request(path: string, body?: unknown): Promise<BDSnapshot | { job_ids?: string[]; mode?: string }> {
  const response = await studioFetch(`/api/bd/${path}`, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const result = await studioJSON<BDSnapshot>(response)
  if (path === 'snapshot' && (!Array.isArray(result?.bots) || !Array.isArray(result?.jobs) || !Array.isArray(result?.packets) || !Array.isArray(result?.events) || !result?.health || !result?.jev || !result?.budget)) {
    throw new Error('Conductor returned an invalid snapshot.')
  }
  return result
}

type Toast = { message: string; undo?: () => void }
type ChangeLine = { id: string; at: string; text: string }
type FixState = { contact: string; segment: string }

export function BDConductor() {
  const qc = useQueryClient()
  const details = useRef<HTMLDialogElement>(null)
  const toastTimer = useRef<number>(0)
  const [now, setNow] = useState(() => Date.now())
  const [sheetOpen, setSheetOpen] = useState(false)
  const [toast, setToast] = useState<Toast | null>(null)
  const [log, setLog] = useState<ChangeLine[]>([])
  const [leaving, setLeaving] = useState<Record<string, boolean>>({})
  const [fixing, setFixing] = useState<Record<string, FixState>>({})
  const [fixes, setFixes] = useState<Record<string, FixState>>({})
  const query = useQuery({ queryKey: ['bd-snapshot'], queryFn: () => request('snapshot') as Promise<BDSnapshot>, refetchInterval: 15_000, retry: 1 })
  const data = query.data
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    const source = new EventSource('/api/bd/events')
    source.addEventListener('snapshot', event => {
      try {
        const snapshot = JSON.parse((event as MessageEvent).data) as BDSnapshot
        if (!Array.isArray(snapshot?.bots) || !Array.isArray(snapshot?.jobs) || !Array.isArray(snapshot?.packets) || !Array.isArray(snapshot?.events) || !snapshot?.health || !snapshot?.jev || !snapshot?.budget) throw new Error('Invalid snapshot')
        qc.setQueryData(['bd-snapshot'], snapshot)
      } catch { /* keep last good snapshot; polling continues */ }
    })
    return () => { source.close(); window.clearInterval(timer) }
  }, [qc])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setSheetOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const announce = (message: string) => {
    window.clearTimeout(toastTimer.current)
    setToast({ message })
    toastTimer.current = window.setTimeout(() => setToast(null), 6000)
  }
  const showToast = (message: string, undo?: () => void) => {
    window.clearTimeout(toastTimer.current)
    setToast({ message, undo })
    toastTimer.current = window.setTimeout(() => setToast(null), 6000)
  }
  const addLog = (text: string) => setLog(current => [{ id: `${Date.now()}-${text}`, at: new Date().toISOString(), text }, ...current].slice(0, 20))
  const action = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => request(path, body),
    onSuccess: result => {
      if (result && 'bots' in result) qc.setQueryData(['bd-snapshot'], result)
      else qc.invalidateQueries({ queryKey: ['bd-snapshot'] })
    },
  })
  const run = (path: string, body: unknown, after?: (result: BDSnapshot) => void) => {
    action.mutate({ path, body }, {
      onSuccess: result => {
        if (result && 'bots' in result) {
          qc.setQueryData(['bd-snapshot'], result)
          after?.(result)
        } else {
          qc.invalidateQueries({ queryKey: ['bd-snapshot'] })
          after?.(result as BDSnapshot)
        }
      },
    })
  }
  const steer = (body: SteerBody, undo: SteerBody | null, line: string, toastText?: string) => {
    run('steer', body, result => {
      addLog(line)
      const queued = body.action === 'queue' ? result.jobs.find(job => job.bot_id === body.bot_id && (job.operator_query === body.query || job.operator_target === body.target) && job.state === 'queued') : undefined
      const undoBody = body.action === 'queue' && queued ? { action: 'cancel', job_id: queued.id } : undo
      showToast(toastText || line, undoBody ? () => {
        run('steer', undoBody)
        addLog(`Undid: ${line.replace(/^You /, '')}`)
        announce('Undone')
      } : undefined)
    })
  }
  const review = (packet: BDPacket, status: 'research_next' | 'dismissed' | 'human_review_required', line: string, toastText?: string) => {
    run(`packets/${encodeURIComponent(packet.job_id)}/${packet.index}/review`, { status }, () => {
      addLog(line)
      showToast(toastText || line, status === 'human_review_required' ? undefined : () => {
        run(`packets/${encodeURIComponent(packet.job_id)}/${packet.index}/review`, { status: 'human_review_required' })
        addLog(`Undid: ${line.replace(/^You /, '')}`)
        announce('Undone')
      })
    })
  }
  const disabled = action.isPending || query.isError
  const banner = data ? leadBanner(data, now) : null
  const bots = data ? orderedBots(data.bots) : []
  const leads = data ? data.packets.filter(packet => packetOutcome(packet) === 'supported' && packet.review_status === 'human_review_required' && !leaving[packet.id]) : []
  const shown = leads.slice(0, MAX_CARDS)
  const extra = Math.max(0, leads.length - shown.length)
  const lede = !data || banner?.mode !== 'stopped'
    ? 'Five bots look for new business around the clock. Qwen pulls facts from public records, Jev judges each one, and the good ones wait here for you.'
    : 'Every bot is stopped. Nothing is being checked or judged until you resume.'
  const pauseAll = () => {
    if (!data) return
    if (data.kill_switch) {
      run('kill-switch', { enabled: false }, () => run('steer', { action: 'resume' }, () => {
        addLog('You turned everything back on.')
        showToast('You turned everything back on.', () => { run('kill-switch', { enabled: true }); addLog('Undid: turned everything back on.'); announce('Undone') })
      }))
      return
    }
    if (data.schedule_enabled) steer({ action: 'pause' }, { action: 'resume' }, 'You paused every bot.')
    else steer({ action: 'resume' }, { action: 'pause' }, 'You resumed every bot.')
  }
  const stopAll = () => {
    if (!data) return
    run('kill-switch', { enabled: true }, () => {
      addLog('You stopped every bot.')
      showToast('Everything is stopped. Use Resume everything at the top to turn it back on.', () => {
        run('kill-switch', { enabled: false })
        addLog('Undid: stopped every bot.')
        announce('Undone')
      })
    })
  }
  return <main className="bd-page">
    <div className="bd-wrap">
      <div className="bd-top">
        <a className="bd-back" href="#/note">← Morning note</a>
        <button type="button" className="bd-linkish" onClick={() => details.current?.showModal()}>Details</button>
      </div>
      <header>
        <h1>Lead finding</h1>
        <p className="bd-lede">{lede}</p>
        <p className="bd-rule-note">Bots only propose. Nothing is sent or saved to a CRM without you.</p>
      </header>
      {action.isError && <p className="bd-error" role="alert">Action failed: {action.error.message}</p>}
      {query.isError && <div className="bd-error" role="alert">data_unavailable: {query.error.message} {data && 'Showing the last received snapshot; controls are disabled.'}<button type="button" className="bd-btn" onClick={() => query.refetch()}>Retry connection</button></div>}
      {!data && query.isLoading && <p className="bd-empty" role="status">Connecting to the conductor…</p>}
      {data && banner && <>
        <section className={`bd-status ${banner.mode}`} role="status" aria-label="Conductor status">
          <span className="bd-status-text"><span className="bd-dot" aria-hidden="true" /><span>{banner.label}</span><small>{banner.sub}</small></span>
          <button type="button" className={`bd-btn${banner.mode === 'running' ? '' : ' bd-btn-primary'}`} disabled={disabled} onClick={pauseAll}>
            {banner.mode === 'running' ? 'Pause all' : banner.mode === 'stopped' ? 'Resume everything' : 'Resume all'}
          </button>
        </section>
        <div className="bd-grid">
          <div className="bd-main">
            <section aria-labelledby="h-lanes">
              <h2 id="h-lanes">Bots, live</h2>
              <div className="bd-lanes">
                {bots.map(bot => <Lane key={bot.id} data={data} bot={bot} disabled={disabled} onSteer={steer} />)}
              </div>
            </section>
            <section aria-labelledby="h-review">
              <h2 id="h-review">Leads for you {leads.length > 0 && <span className="bd-count">{leads.length}</span>}</h2>
              {shown.length === 0 && <p className="bd-empty" tabIndex={-1}>Nothing to review right now.</p>}
              <ol className="bd-cards">
                {shown.map(packet => {
                  const bot = data.bots.find(item => item.id === packet.bot_id)
                  const overlay = fixes[packet.id]
                  const contact = overlay?.contact ?? leadContact(packet)
                  const segment = overlay?.segment ?? (bot ? laneProfile(bot).title : packet.segment)
                  const draft = fixing[packet.id]
                  return <li key={packet.id} id={`bd-packet-${packet.id}`} tabIndex={-1} className={`bd-card bd-packet${leaving[packet.id] ? ' leaving' : ''}`} aria-labelledby={`t-${packet.id}`}>
                    <p className="bd-card-product">{bot ? laneProfile(bot).title : packet.segment}</p>
                    <h3 className="bd-card-title" id={`t-${packet.id}`}>{packet.organization || 'Unnamed research result'}</h3>
                    <p className="bd-card-why">Jev: {packet.summary || researchReason(packet.candidate_reason)}</p>
                    <p className="bd-lead-meta">
                      {packet.evidence_score != null && <span>Evidence <span className="bd-score">{packet.evidence_score.toFixed(1)} / 2</span></span>}
                      <span>Verdict: {packet.verdict || packet.disposition || 'typed'}</span>
                      <span>Contact: {contact}</span>
                      <span>Segment: {segment}</span>
                    </p>
                    {draft && <form className="bd-fix" onSubmit={event => {
                      event.preventDefault()
                      const previous = { contact, segment }
                      const next = { contact: draft.contact.trim() || contact, segment: draft.segment }
                      setFixes(current => ({ ...current, [packet.id]: next }))
                      setFixing(current => { const copy = { ...current }; delete copy[packet.id]; return copy })
                      addLog(`You fixed the displayed contact on ${packet.organization}.`)
                      showToast('Saved your fix. Still waiting for your approval. Nothing is sent.', () => {
                        setFixes(current => ({ ...current, [packet.id]: previous }))
                        announce('Undone')
                      })
                    }}>
                      <label>Contact<input className="bd-field" value={draft.contact} onChange={event => setFixing(current => ({ ...current, [packet.id]: { ...draft, contact: event.target.value } }))} /></label>
                      <label>Segment<select className="bd-field" value={draft.segment} onChange={event => setFixing(current => ({ ...current, [packet.id]: { ...draft, segment: event.target.value } }))}>{bots.map(item => <option key={item.id}>{laneProfile(item).title}</option>)}</select></label>
                      <div className="bd-row">
                        <button type="submit" className="bd-btn bd-btn-primary">Save</button>
                        <button type="button" className="bd-btn" onClick={() => setFixing(current => { const copy = { ...current }; delete copy[packet.id]; return copy })}>Cancel</button>
                      </div>
                    </form>}
                    <div className="bd-card-actions">
                      <button type="button" className="bd-btn bd-btn-primary" disabled={disabled} onClick={() => {
                        setLeaving(current => ({ ...current, [packet.id]: true }))
                        window.setTimeout(() => {
                          review(packet, 'research_next', `You approved ${packet.organization}.`, `Approved · ${packet.organization}. Nothing is sent; it's just marked for you.`)
                          setLeaving(current => { const copy = { ...current }; delete copy[packet.id]; return copy })
                        }, 280)
                      }}>Approve</button>
                      <button type="button" className="bd-btn" disabled={disabled} onClick={() => {
                        setLeaving(current => ({ ...current, [packet.id]: true }))
                        window.setTimeout(() => {
                          review(packet, 'dismissed', `You rejected ${packet.organization}.`, `Rejected · ${packet.organization}. Nothing is sent; it's just marked for you.`)
                          setLeaving(current => { const copy = { ...current }; delete copy[packet.id]; return copy })
                        }, 280)
                      }}>Reject</button>
                      <button type="button" className="bd-btn bd-btn-quiet" disabled={disabled} onClick={() => setFixing(current => ({ ...current, [packet.id]: { contact, segment } }))}>Fix</button>
                    </div>
                  </li>
                })}
              </ol>
              {extra > 0 && <p className="bd-more-line">Plus {extra} more waiting</p>}
            </section>
          </div>
          <aside className="bd-side">
            <BDAssignmentBoard data={data} now={now} stopped={Boolean(data.kill_switch) || !data.schedule_enabled} />
            {sheetOpen && <button type="button" className="bd-scrim" aria-label="Close steer" onClick={() => setSheetOpen(false)} />}
            <BDSteeringControls
              data={data}
              disabled={disabled}
              sheetOpen={sheetOpen}
              localLog={log}
              onClose={() => setSheetOpen(false)}
              onSteer={steer}
              onPauseAll={pauseAll}
              onKill={stopAll}
              onBudget={(body, undo, line) => steer(body, undo, line)}
            />
          </aside>
        </div>
      </>}
    </div>
    <button type="button" className="bd-btn bd-btn-primary bd-steer-open" aria-controls="steer" aria-expanded={sheetOpen} onClick={() => setSheetOpen(true)}>Steer</button>
    <dialog className="bd-details" ref={details} aria-labelledby="d-title" onClick={event => { if (event.target === details.current) details.current.close() }}>
      {data && <div className="bd-details-inner">
        <div className="bd-details-head"><h2 id="d-title">Details</h2><button type="button" className="bd-btn" onClick={() => details.current?.close()}>Close</button></div>
        <p className="bd-details-sub">What's under the hood. Live snapshot values, not the prototype placeholders.</p>
        <h3>Models</h3>
        <ul>
          <li><span className="bd-r-name">Qwen · local</span><span className="bd-r-last">{qwenModel(data)} · loopback Ollama · local, no per-job dollar cost</span></li>
          <li><span className="bd-r-name">Jev · typed judgments</span><span className="bd-r-last">Returns Choice / Score / Noul per job via typesafe.signals</span></li>
        </ul>
        <h3>Prompts (summaries)</h3>
        <ul>
          <li><span className="bd-r-name">Extract</span><span className="bd-r-last">From this public record, pull the named organization, event, and supporting quote. Draft a one-line summary.</span></li>
          <li><span className="bd-r-name">Judge</span><span className="bd-r-last">Given the extracted fields and the bot's targets, return a typed verdict with one sentence why. Contacts stay held.</span></li>
        </ul>
        <h3>Recent job traces</h3>
        <ul>
          {data.events.filter(event => ['qwen_completed', 'qwen_replayed', 'jev_completed', 'jev_replayed', 'jev_verdict'].includes(event.kind)).slice(0, 8).map(event => <li key={event.id}><span className="bd-r-name">{event.kind.replaceAll('_', ' ')} · {data.bots.find(bot => bot.id === event.bot_id)?.name ?? event.bot_id}</span><span className="bd-r-last">{event.model || event.provider || 'snapshot'} · {event.latency_ms != null ? `${Math.round(event.latency_ms)} ms` : 'latency data_unavailable'} · {event.primitive || event.message} · {ageLabel(event.created_at, now)} ago</span></li>)}
          {data.events.length === 0 && <li><span className="bd-r-last">No traces in this snapshot.</span></li>}
        </ul>
        <h3>Permit fields</h3>
        <PermitDetails data={data} />
        <h3>Spend today</h3>
        <ul><li><span className="bd-r-name">{metric(data.budget.tokens.used)} of {metric(data.budget.tokens.limit)} tokens</span><span className="bd-r-last">{metric(data.budget.calls.used)} of {metric(data.budget.calls.limit)} calls · {metric(data.budget.seconds.used)} of {metric(data.budget.seconds.limit)} s reserved. Not a dollar invoice.</span></li></ul>
        <h3>Connections</h3>
        <div className="bd-health">
          {Object.entries(healthNames).map(([key, name]) => {
            const item = data.health[key as keyof typeof healthNames]
            return <div key={key}><strong>{name}</strong><span className="bd-r-last">{providerStatus(item)}{item?.detail ? ` · ${item.detail}` : ''}</span></div>
          })}
        </div>
        <h3>Operating mode</h3>
        <ul>
          <li><span className="bd-r-name">{data.mode === 'replay' ? 'Shadow · replay only' : data.mode}</span><span className="bd-r-last">{data.mode === 'replay' ? 'Recorded judgments and synthetic candidates. No outreach.' : 'Public-source research. Human review before outreach.'}</span></li>
          <li><span className="bd-r-name">Cadence</span><span className="bd-r-last">{cadencePhrase(weekdayInterval(data))} · {data.cadence?.timezone || 'America/Chicago'}</span></li>
        </ul>
        {data.mode === 'replay' && <button type="button" className="bd-btn bd-btn-primary" disabled={disabled || data.kill_switch} onClick={() => run('shadow', {}, () => showToast('Shadow batch accepted. Follow its progress on the lanes.'))}>{action.isPending && action.variables?.path === 'shadow' ? 'Starting replay…' : 'Run shadow batch'}</button>}
        <BDScheduleStrip data={data} now={now} />
        <BDResearchHistory data={data} now={now} lane="all" onReview={packet => { details.current?.close(); document.getElementById(`bd-packet-${packet.id}`)?.scrollIntoView({ block: 'start' }) }} />
      </div>}
    </dialog>
    <div className={`bd-toast${toast ? ' show' : ''}`} role="status">
      {toast && <span>{toast.message}</span>}
      {toast?.undo && <button type="button" className="bd-toast-undo" onClick={() => toast.undo?.()}>Undo</button>}
    </div>
    <div className="bd-sr" aria-live="polite">{toast?.message}</div>
  </main>
}

function Lane({ data, bot, disabled, onSteer }: { data: BDSnapshot; bot: BDSnapshot['bots'][number]; disabled: boolean; onSteer: (body: SteerBody, undo: SteerBody | null, log: string, toast?: string) => void }) {
  const profile = laneProfile(bot)
  const focused = Boolean(bot.focused)
  const dim = Boolean(data.steering?.focus) && !focused
  const jobs = data.jobs.filter(job => job.bot_id === bot.id)
  const extract = jobs.filter(job => ['queued', 'leased'].includes(job.state) || jobStage(job) === 'extract' || jobStage(job) === 'queued')
  const judge = jobs.filter(job => jobStage(job) === 'judge')
  const awaiting = data.packets.filter(packet => packet.bot_id === bot.id && packetOutcome(packet) === 'supported' && packet.review_status === 'human_review_required')
  const workingExtract = extract.find(job => job.state === 'running')
  const workingJudge = judge.find(job => job.state === 'running')
  const bumped = jobs.some(job => (job.priority ?? 0) > 0 && job.state === 'queued')
  const interval = weekdayInterval(data)
  const mine = operatorTargets(data, bot.id)
  const chip = (job: typeof jobs[number], stage: 'extract' | 'judge') => <button type="button" key={job.id} className={`bd-job${(job.priority ?? 0) > 0 ? ' prio' : ''}${job === workingExtract || job === workingJudge ? ' working' : ''}`} title={stage === 'extract' || stage === 'judge' ? 'Tap to move this one to the front' : 'Waiting for you below'} onClick={() => {
    if ((job.priority ?? 0) > 0 || job.state === 'completed') return
    onSteer({ action: 'bump', job_id: job.id }, null, `You moved "${jobLabel(job, data.packets)}" to the front.`)
  }}>{(job.priority ?? 0) > 0 ? '↑ ' : ''}{jobLabel(job, data.packets)}</button>
  return <article className={`bd-lane${dim ? ' dim' : ''}${focused ? ' focus' : ''}${bumped ? ' prio' : ''}`} aria-labelledby={`ln-${bot.id}`}>
    <div className="bd-lane-head">
      <div>
        <h3 className="bd-lane-name" id={`ln-${bot.id}`}>{profile.title}{profile.isNew && <span className="bd-tag bd-tag-new">New</span>}
          {bot.paused && <span className="bd-tag bd-tag-paused">Paused</span>}
          {focused && <span className="bd-tag bd-tag-focus">Focused</span>}
          {bumped && <span className="bd-tag bd-tag-prio">Bumped</span>}
        </h3>
        <p className="bd-lane-sub">{profile.kind} · checks {cadencePhrase(interval)}</p>
        <div className="bd-targets">
          {profile.targets.map(target => <span className="bd-target" key={target}>{target}</span>)}
          {mine.map(target => <span className="bd-target mine" key={target}>+ {target}</span>)}
        </div>
      </div>
      <div className="bd-lane-ctl">
        <button type="button" className="bd-btn" aria-pressed={bot.paused} disabled={disabled} aria-label={`${bot.paused ? 'Resume' : 'Pause'} ${profile.title}`} onClick={() => onSteer({ action: bot.paused ? 'resume_bot' : 'pause_bot', bot_id: bot.id }, { action: bot.paused ? 'pause_bot' : 'resume_bot', bot_id: bot.id }, `You ${bot.paused ? 'resumed' : 'paused'} ${profile.title}.`)}>{bot.paused ? 'Resume' : 'Pause'}</button>
        <button type="button" className="bd-btn" aria-pressed={focused} disabled={disabled} aria-label={`${focused ? 'Unfocus' : 'Focus'} ${profile.title}`} onClick={() => onSteer(focused ? { action: 'clear_focus' } : { action: 'focus', bot_id: bot.id }, focused ? { action: 'focus', bot_id: bot.id } : { action: 'clear_focus' }, focused ? `You cleared focus on ${profile.title}.` : `You focused on ${profile.title}. Others slow down.`)}>{focused ? 'Unfocus' : 'Focus'}</button>
        <button type="button" className="bd-btn" disabled={disabled || !extract[0]} aria-label={`Bump ${profile.title}`} onClick={() => extract[0] && onSteer({ action: 'bump', job_id: extract[0].id }, null, `You bumped ${profile.title} to the front.`)}>Bump</button>
      </div>
    </div>
    <div className="bd-stages">
      <div className="bd-stage"><p className="bd-stage-h"><span>Extract <em>· Qwen</em></span><b>{extract.length}</b></p><div className="bd-chips">{extract.slice(0, 4).map(job => chip(job, 'extract'))}{extract.length > 4 && <span className="bd-more-chip">+{extract.length - 4} more</span>}</div></div>
      <div className="bd-stage"><p className="bd-stage-h"><span>Judge <em>· Jev</em></span><b>{judge.length}</b></p><div className="bd-chips">{judge.slice(0, 4).map(job => chip(job, 'judge'))}{judge.length > 4 && <span className="bd-more-chip">+{judge.length - 4} more</span>}</div></div>
      <div className="bd-stage"><p className="bd-stage-h"><span>Awaiting you</span><b>{awaiting.length}</b></p><div className="bd-chips">{awaiting.slice(0, 4).map(packet => <button type="button" key={packet.id} className="bd-job" title="Waiting for you below" onClick={() => document.getElementById('h-review')?.scrollIntoView({ behavior: 'smooth' })}>{packet.organization}</button>)}{awaiting.length > 4 && <span className="bd-more-chip">+{awaiting.length - 4} more</span>}</div></div>
    </div>
  </article>
}

function PermitDetails({ data }: { data: BDSnapshot }) {
  const packet = data.packets.find(item => item.permit && item.permit.status !== 'data_unavailable')
  if (!packet?.permit) return <ul><li><span className="bd-r-last">No permit fields in this snapshot.</span></li></ul>
  return <ul>
    <li><span className="bd-r-name">Permit</span><span className="bd-r-last">{packet.permit.number || 'data_unavailable'}</span></li>
    <li><span className="bd-r-name">Project</span><span className="bd-r-last">{packet.permit.project_title || 'data_unavailable'}</span></li>
    <li><span className="bd-r-name">Address</span><span className="bd-r-last">{packet.permit.address || 'data_unavailable'}</span></li>
    <li><span className="bd-r-name">Status</span><span className="bd-r-last">{packet.permit.permit_status || packet.permit.status || 'data_unavailable'}</span></li>
    <li><span className="bd-r-name">Source</span><span className="bd-r-last">{publicSource(packet.source_url) ? <a href={publicSource(packet.source_url)!} target="_blank" rel="noopener noreferrer">Open source evidence</a> : 'Source evidence unavailable'}</span></li>
  </ul>
}

function qwenModel(data: BDSnapshot) {
  const named = data.events.map(event => event.model).find(model => model && /qwen/i.test(model))
  return named || 'qwen3:8b'
}

function metric(value: number | null | undefined) {
  return value == null ? 'data_unavailable' : value.toLocaleString()
}
