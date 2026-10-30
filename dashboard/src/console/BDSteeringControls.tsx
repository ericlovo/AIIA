import { useState } from 'react'
import { CADENCE_OPTIONS, auditLine, cadencePhrase, laneProfile, orderedBots, selectedCadence, weekdayInterval, type BDSnapshot } from './bdModel'

export type SteerBody = {
  action: string; bot_id?: string; job_id?: string; segment?: string; query?: string; target?: string
  enabled?: boolean; overnight_enabled?: boolean; around_the_clock?: boolean
  weekday_interval_seconds?: number; overnight_interval_seconds?: number
  tokens?: number; calls?: number; seconds?: number
}

const SUGGESTIONS = [
  { bot: 'comcast', text: 'Uptown restaurant build-outs' },
  { bot: 'comcast', text: 'Permit type commercial tenant improvement' },
  { bot: 'healthcare', text: 'Rural clinics adding telehealth' },
]

export function BDSteeringControls({
  data, disabled, sheetOpen, localLog, onClose, onSteer, onPauseAll, onKill, onBudget,
}: {
  data: BDSnapshot
  disabled: boolean
  sheetOpen: boolean
  localLog: Array<{ id: string; at: string; text: string }>
  onClose: () => void
  onSteer: (body: SteerBody, undo: SteerBody | null, log: string, toast?: string) => void
  onPauseAll: () => void
  onKill: () => void
  onBudget: (body: SteerBody, undo: SteerBody, log: string) => void
}) {
  const bots = orderedBots(data.bots)
  const [botId, setBotId] = useState(bots[0]?.id ?? '')
  const [query, setQuery] = useState('')
  const [confirmKill, setConfirmKill] = useState(false)
  const selected = bots.find(bot => bot.id === botId) ?? bots[0]
  const interval = weekdayInterval(data)
  const cadenceId = selectedCadence(interval)
  const overnight = Boolean(data.steering?.overnight_enabled ?? data.cadence?.overnight?.enabled)
  const tokenLimit = data.budget.tokens.limit
  const callLimit = data.budget.calls.limit
  const tokenOptions = [10000, 40000, 50000, 200000, tokenLimit].filter((value, index, list): value is number => value != null && list.indexOf(value) === index)
  const callOptions = [50, 100, 200, callLimit].filter((value, index, list): value is number => value != null && list.indexOf(value) === index)
  const audit = data.steering?.audit ?? []
  const lines = localLog.length ? localLog : audit.slice(0, 12).map(item => ({ id: String(item.id), at: item.created_at, text: auditLine(item) }))
  const addTarget = (id: string, text: string) => {
    const trimmed = text.trim()
    if (!trimmed || !id) return
    const bot = bots.find(item => item.id === id)
    onSteer(
      { action: 'queue', bot_id: id, query: trimmed },
      null,
      `You added a target to ${laneProfile(bot ?? { id, name: id, kind: 'segment' }).title}: "${trimmed}".`,
      `Added "${trimmed}" to ${laneProfile(bot ?? { id, name: id, kind: 'segment' }).title}. New finds will look for it.`,
    )
    setQuery('')
  }
  return <section className={`bd-panel bd-steer${sheetOpen ? ' open' : ''}`} id="steer" aria-labelledby="h-steer">
    <div className="bd-sheet-head"><h2 id="h-steer-m" style={{ margin: 0 }}>Steer</h2><button type="button" className="bd-btn" onClick={onClose}>Done</button></div>
    <h2 id="h-steer" className="bd-steer-desk">Steer</h2>
    <div className="bd-steer-group">
      <span className="bd-steer-label">Everything</span>
      <div className="bd-row">
        <button type="button" className="bd-btn" disabled={disabled} onClick={onPauseAll}>{data.kill_switch || !data.schedule_enabled ? 'Resume all' : 'Pause all'}</button>
        {data.steering?.focus && <button type="button" className="bd-btn" disabled={disabled} onClick={() => onSteer({ action: 'clear_focus' }, { action: 'focus', bot_id: data.steering?.focus?.kind === 'bot' ? data.steering.focus.id : undefined, segment: data.steering?.focus?.kind === 'segment' ? data.steering.focus.id : undefined }, 'You cleared focus.')}>Clear focus</button>}
      </div>
      <p className="bd-steer-note">Pause, focus or bump a single bot from its own row.</p>
    </div>
    <div className="bd-steer-group">
      <label className="bd-steer-label" htmlFor="bd-target">Add a target</label>
      <form className="bd-row" onSubmit={event => { event.preventDefault(); addTarget(selected?.id ?? '', query) }}>
        <select className="bd-field" id="bd-target-bot" aria-label="Which bot" value={selected?.id ?? ''} onChange={event => setBotId(event.target.value)}>
          {bots.map(bot => <option key={bot.id} value={bot.id}>{laneProfile(bot).title}</option>)}
        </select>
        <input className="bd-field" id="bd-target" aria-label="Steer query" value={query} maxLength={200} placeholder="e.g. Uptown restaurant build-outs" onChange={event => setQuery(event.target.value)} />
        <button className="bd-btn bd-btn-primary" type="submit" disabled={disabled || !selected || !query.trim()}>Add</button>
      </form>
      <div className="bd-suggest">
        {SUGGESTIONS.filter(item => bots.some(bot => bot.id === item.bot)).map(item => <button type="button" className="bd-sugg" key={item.text} disabled={disabled} onClick={() => { setBotId(item.bot); addTarget(item.bot, item.text) }}>{item.text}</button>)}
      </div>
    </div>
    <div className="bd-steer-group">
      <span className="bd-steer-label" id="cad-l">How often to check</span>
      <div className="bd-seg" role="group" aria-labelledby="cad-l">
        {CADENCE_OPTIONS.map(option => <button type="button" key={option.id} aria-pressed={cadenceId === option.id} disabled={disabled} onClick={() => {
          if (option.seconds === interval) return
          onSteer(
            { action: 'set_cadence', weekday_interval_seconds: option.seconds },
            interval == null ? null : { action: 'set_cadence', weekday_interval_seconds: interval },
            `You set checks to ${option.phrase}.`,
          )
        }}>{option.label}</button>)}
      </div>
      <p className="bd-steer-note">Each bot checks {cadencePhrase(interval)}.</p>
      <div className="bd-row" style={{ marginTop: 8 }}>
        <button type="button" className="bd-btn" aria-pressed={overnight} disabled={disabled || !selected} onClick={() => {
          const next = !overnight
          onSteer(
            { action: 'set_cadence', overnight_enabled: next, around_the_clock: next, bot_id: selected?.id },
            { action: 'set_cadence', overnight_enabled: overnight, around_the_clock: overnight, bot_id: selected?.id },
            next ? `You turned overnight 24/7 on for ${laneProfile(selected ?? { id: '', name: 'bot', kind: 'segment' }).title}.` : 'You turned overnight 24/7 off.',
          )
        }}>{overnight ? 'Overnight 24/7 on' : 'Overnight 24/7'}</button>
      </div>
      <p className="bd-steer-note">Overnight is a slower hourly window. It stays off until you press this; only the selected bot is marked around the clock.</p>
    </div>
    <div className="bd-steer-group">
      <span className="bd-steer-label">Daily budget</span>
      <div className="bd-row">
        <select className="bd-field" aria-label="Daily token cap" value={tokenLimit ?? ''} disabled={disabled} onChange={event => {
          const tokens = Number(event.target.value)
          if (!tokens || tokens === tokenLimit) return
          onBudget({ action: 'set_budget', tokens }, { action: 'set_budget', tokens: tokenLimit ?? tokens }, `You set the daily token ceiling to ${tokens.toLocaleString()}.`)
        }}>
          {tokenOptions.map(value => <option key={value} value={value}>{value.toLocaleString()} tokens</option>)}
        </select>
        <select className="bd-field" aria-label="Daily call cap" value={callLimit ?? ''} disabled={disabled} onChange={event => {
          const calls = Number(event.target.value)
          if (!calls || calls === callLimit) return
          onBudget({ action: 'set_budget', calls }, { action: 'set_budget', calls: callLimit ?? calls }, `You capped the day at ${calls} calls.`)
        }}>
          {callOptions.map(value => <option key={value} value={value}>{value} jobs</option>)}
        </select>
      </div>
      <p className="bd-steer-note">Used {metric(data.budget.tokens.used)} of {metric(data.budget.tokens.limit)} tokens · {metric(data.budget.calls.used)} of {metric(data.budget.calls.limit)} jobs today. Reserved capacity, not a dollar invoice.</p>
    </div>
    <div className="bd-steer-group">
      <span className="bd-steer-label">Stop everything</span>
      <button type="button" className="bd-btn bd-btn-danger" disabled={disabled || data.kill_switch} onClick={() => setConfirmKill(true)}>{data.kill_switch ? 'Everything is stopped' : 'Stop all bots'}</button>
      {confirmKill && <div className="bd-confirm">
        <p>Stop every bot now? Nothing new is checked or judged until you resume. Leads already waiting stay here.</p>
        <div className="bd-row">
          <button type="button" className="bd-btn bd-btn-danger-solid" onClick={() => { setConfirmKill(false); onKill() }}>Yes, stop everything</button>
          <button type="button" className="bd-btn" onClick={() => setConfirmKill(false)}>Cancel</button>
        </div>
      </div>}
    </div>
    <div className="bd-steer-group">
      <span className="bd-steer-label">What changed</span>
      {lines.length ? <ul className="bd-log">{lines.map(item => <li key={item.id}><time dateTime={item.at}>{clock(item.at)}</time>{item.text}</li>)}</ul> : <p className="bd-log-empty">Nothing yet. Your changes show up here.</p>}
    </div>
  </section>
}

function metric(value: number | null | undefined) {
  return value == null ? 'data_unavailable' : value.toLocaleString()
}
function clock(value: string) {
  const at = Date.parse(value)
  if (!Number.isFinite(at)) return ''
  return new Date(at).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}
