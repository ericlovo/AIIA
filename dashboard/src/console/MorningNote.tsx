import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type Agent } from '../lib/api'
import {
  assignmentTitle,
  countdown,
  formatClock,
  formatLongDate,
  greeting,
  morningLede,
  pickAgentForAsk,
  STATE_LABEL,
  type MorningDecision,
  type MorningLine,
  type MorningNotePayload,
} from './morningNote'
import './MorningNote.css'

const EMPTY_NOTE: MorningNotePayload = {
  date: '',
  products: [],
  customers: [],
  decisions: [],
  needs_you_total: 0,
  failure: null,
}

export function MorningNote({ agents, onDetails }: { agents: Agent[]; onDetails: () => void }) {
  const qc = useQueryClient()
  const note = useQuery({ queryKey: ['morning-note'], queryFn: api.morningNote, refetchInterval: 60_000 })
  const data = note.data ?? EMPTY_NOTE
  const [now, setNow] = useState(() => new Date())
  const [openId, setOpenId] = useState<string | null>(null)
  const [ask, setAsk] = useState('')
  const [taken, setTaken] = useState<Array<{ id: string; agent: string; text: string; why: string; at: string }>>([])
  const [askError, setAskError] = useState('')
  const [gone, setGone] = useState<Set<string>>(new Set())

  useEffect(() => {
    const tick = window.setInterval(() => setNow(new Date()), 30_000)
    return () => window.clearInterval(tick)
  }, [])

  const remaining = useMemo(
    () => data.decisions.filter(item => !gone.has(`${item.kind}:${item.id}`)),
    [data.decisions, gone],
  )
  const shown = remaining.slice(0, 5)
  const more = Math.max(data.needs_you_total - gone.size - shown.length, remaining.length - shown.length)
  const needCount = Math.max(data.needs_you_total - gone.size, remaining.length)

  const approve = useMutation({
    mutationFn: (item: MorningDecision) => item.kind === 'review'
      ? api.reviewAssignment(item.id, 'accepted', item.expected_version || '', '')
      : api.promoteIdea(item.id, 'decisions', '', { postToSlack: false }),
    onSuccess: (_result, item) => {
      setGone(prev => new Set(prev).add(`${item.kind}:${item.id}`))
      void qc.invalidateQueries({ queryKey: ['morning-note'] })
      void qc.invalidateQueries({ queryKey: ['assignments'] })
      void qc.invalidateQueries({ queryKey: ['memory-inbox'] })
    },
  })
  const later = useMutation({
    mutationFn: (item: MorningDecision) => item.kind === 'review'
      ? api.dismissAssignment(item.id, true, item.expected_version || '', 'Not now')
      : api.dismissIdea(item.id, 'Not now'),
    onSuccess: (_result, item) => {
      setGone(prev => new Set(prev).add(`${item.kind}:${item.id}`))
      void qc.invalidateQueries({ queryKey: ['morning-note'] })
      void qc.invalidateQueries({ queryKey: ['assignments'] })
      void qc.invalidateQueries({ queryKey: ['memory-inbox'] })
    },
  })
  const create = useMutation({
    mutationFn: async (text: string) => {
      const picked = pickAgentForAsk(text, agents)
      if (!picked) throw new Error('No one is available to take this yet.')
      const assignment = await api.createAssignment({
        title: assignmentTitle(text),
        objective: text,
        agent_id: picked.agent.id,
        priority: 'normal',
        context: '',
        success_criteria: '',
      })
      return { assignment: assignment.assignment, picked }
    },
    onSuccess: ({ picked }, text) => {
      setTaken(prev => [{ id: `${Date.now()}`, agent: picked.agent.name, text, why: picked.why, at: formatClock(new Date()) }, ...prev])
      setAsk('')
      setAskError('')
      void qc.invalidateQueries({ queryKey: ['assignments'] })
    },
    onError: (error: Error) => setAskError(error.message),
  })

  const busy = approve.isPending || later.isPending

  return (
    <main className="morning-note" aria-labelledby="greet">
      <div className="wrap">
        <div className="top">
          <span className="mark" aria-label="AIIA">AIIA</span>
          <button type="button" className="linkish" onClick={onDetails}>Details</button>
        </div>

        <header>
          <h1 id="greet">{greeting(now)}, Eric.</h1>
          <p className="date">{formatLongDate(now)}</p>
          <p className="lede">{note.isPending ? 'Reading where things stand…' : morningLede(data.products, data.customers, needCount, now)}</p>
        </header>

        {note.isError && <p className="err" role="alert">Could not load today’s note. Details still works.</p>}

        <section aria-labelledby="h-status">
          <h2 id="h-status">Where things stand</h2>
          <StatusList items={data.products} now={now} />
          <div className="sub">
            <h3 className="sub-h">Customers</h3>
            <StatusList items={data.customers} now={now} />
          </div>
        </section>

        <section aria-labelledby="h-decide">
          <h2 id="h-decide">Needs you {needCount > 0 && <span className="count">{needCount}</span>}</h2>
          {shown.length === 0 && !note.isPending && <p className="empty">Nothing needs you right now. Enjoy the quiet.</p>}
          <ol className="cards">
            {shown.map(item => {
              const key = `${item.kind}:${item.id}`
              const opened = openId === key
              return (
                <li key={key} className="card">
                  {item.product && <p className="card-product">{item.product}</p>}
                  <h3 className="card-title">{item.title}</h3>
                  {item.why && <p className="card-why">{item.why}</p>}
                  {opened && item.more && <div className="card-more"><p>{item.more}</p></div>}
                  <div className="card-actions">
                    <button type="button" className="btn btn-primary" disabled={busy} onClick={() => approve.mutate(item)}>Approve</button>
                    <button type="button" className="btn" disabled={busy} onClick={() => later.mutate(item)}>Not now</button>
                    <button type="button" className="btn btn-quiet" aria-expanded={opened} onClick={() => setOpenId(opened ? null : key)}>{opened ? 'Close' : 'Open'}</button>
                  </div>
                </li>
              )
            })}
          </ol>
          {more > 0 && <p className="more-line">{more === 1 ? 'Plus 1 more waiting' : `Plus ${more} more waiting`}</p>}
          {(approve.error || later.error) && <p className="err" role="alert">{(approve.error ?? later.error)?.message}</p>}
        </section>

        <section aria-labelledby="h-ask">
          <h2 id="h-ask">Ask for anything</h2>
          <form className="ask-form" onSubmit={event => { event.preventDefault(); const text = ask.trim(); if (text) create.mutate(text) }}>
            <label htmlFor="morning-ask" className="sr-only">What do you need done?</label>
            <textarea id="morning-ask" className="ask-input" rows={1} value={ask} placeholder="e.g. have someone look at why Sanction CI is red" onChange={event => setAsk(event.target.value)} onInput={event => { const el = event.currentTarget; el.style.height = 'auto'; el.style.height = `${Math.min(el.scrollHeight + 2, 200)}px` }} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); const text = ask.trim(); if (text) create.mutate(text) } }} />
            <button className="btn btn-primary" type="submit" disabled={create.isPending || !ask.trim()}>Ask</button>
          </form>
          <p className="ask-hint">Write it how you’d say it. The right person picks it up, and you’ll see who.</p>
          {askError && <p className="err" role="alert">{askError}</p>}
          <ul className="taken-list" aria-label="Asks you’ve handed off">
            {taken.map(item => (
              <li key={item.id} className="taken">
                <p className="taken-who"><span className="dot dot-working" aria-hidden="true" />Taken by <strong>{item.agent}</strong> <span className="taken-time">· {item.at}</span></p>
                <p className="taken-ask">“{item.text}”</p>
                <p className="taken-why">Picked because {item.why}.</p>
              </li>
            ))}
          </ul>
        </section>

        <footer>
          <span>AIIA</span>
          <button type="button" className="linkish" onClick={onDetails}>Details: who’s working on what</button>
        </footer>
      </div>
    </main>
  )
}

function StatusList({ items, now }: { items: MorningLine[]; now: Date }) {
  if (!items.length) return <p className="empty">Nothing to show yet.</p>
  return (
    <ul className="lines">
      {items.map(item => {
        const ticking = Boolean(item.target)
        const state = ticking ? countdown(item.target!, now).text : (STATE_LABEL[item.state] || item.state)
        return (
          <li key={item.id} className="line">
            <span className="line-name"><span className={`dot dot-${item.state}`} aria-hidden="true" />{item.name}</span>
            <span className={`line-state w-${item.state}${ticking ? ' countdown' : ''}`}>{state}</span>
            <span className="line-note">{item.note}</span>
          </li>
        )
      })}
    </ul>
  )
}
