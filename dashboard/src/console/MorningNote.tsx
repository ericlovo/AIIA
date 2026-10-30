import { useEffect, useState } from 'react'
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
  type MorningLine,
  type MorningNotePayload,
} from './morningNoteModel'
import './MorningNote.css'
import { useDecisionQueue } from './useDecisionQueue'
import { assignmentNextAction } from './assignmentReview'
import { formatRoute } from './studioRoute'

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
  const data = { ...EMPTY_NOTE, ...(note.data ?? {}) }
  const [now, setNow] = useState(() => new Date())
  const [ask, setAsk] = useState('')
  const [taken, setTaken] = useState<Array<{ id: string; agent: string; text: string; why: string; at: string }>>([])
  const [askError, setAskError] = useState('')
  const queue = useDecisionQueue()

  useEffect(() => {
    const tick = window.setInterval(() => setNow(new Date()), 30_000)
    return () => window.clearInterval(tick)
  }, [])

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


  return (
    <main className="morning-note" aria-labelledby="greet">
      <div className="wrap">
        <div className="top">
          <span className="mark" aria-label="AIIA">AIIA</span>
          <button type="button" className="linkish" onClick={onDetails}>Review work</button>
        </div>

        <header>
          <h1 id="greet">{greeting(now)}.</h1>
          <p className="date">{formatLongDate(now)}</p>
          <p className="lede">{note.isPending || queue.loading ? 'Reading where things stand…' : queue.error ? 'The decision queue is unavailable. Refresh before deciding.' : morningLede(data.products, data.customers, queue.total, now)}</p>
        </header>

        {note.isError && <p className="err" role="alert">Could not load today’s note. You can still open Review work.</p>}

        <section aria-labelledby="h-status">
          <h2 id="h-status">Where things stand</h2>
          <StatusList items={data.products} now={now} />
          <div className="sub">
            <h3 className="sub-h">Customers</h3>
            <StatusList items={data.customers} now={now} />
          </div>
        </section>

        <section aria-labelledby="h-decide">
          <h2 id="h-decide">Needs you <span className="count">{queue.error ? 'Unavailable' : queue.loading ? 'Loading' : queue.total}</span></h2>
          <p className="ask-hint">The same decision queue as Work. Open the evidence before deciding; reviewing output never grants Git permissions.</p>
          {queue.error && <p className="err" role="alert">Could not refresh all decisions. Counts may be stale; no all-clear is available.</p>}
          {!queue.loading && !queue.error && queue.total === 0 && <p className="empty">Nothing needs you right now.</p>}
          <ol className="cards">
            {queue.groups.slice(0, 5).map(group => {
              const item = group[0]
              const approval = group.some(record => queue.pendingApprovalIds.has(record.id))
              return <li key={item.id} className="card">
                <h3 className="card-title">{item.title}</h3>
                <p className="card-why">{approval ? 'Permission approval needed. Review each request separately.' : assignmentNextAction(item)}</p>
                {group.length > 1 && <p>{group.length} matching records; each original is preserved.</p>}
                <a className="btn btn-primary" href={formatRoute({ view: 'assignments', assignmentId: approval ? group.find(record => queue.pendingApprovalIds.has(record.id))!.id : item.id })}>Review decision and evidence</a>
              </li>
            })}
          </ol>
          {queue.savedIdeas > 0 && <p><a className="btn" href={formatRoute({ view: 'inbox', source: 'all' })}>Review saved ideas ({queue.savedIdeas})</a> Decide what to keep as memory. This does not approve execution.</p>}
          {queue.missingApprovalIds.length > 0 && <p className="err" role="alert">{queue.missingApprovalIds.length} permission requests reference unavailable work. These approvals remain pending; inspect Work before proceeding.</p>}
          <button type="button" className="btn" onClick={onDetails}>Open full decision queue</button>
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
          <button type="button" className="linkish" onClick={onDetails}>Review work and decisions</button>
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
