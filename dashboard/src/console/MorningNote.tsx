import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, MEMORY_CATEGORIES, type Agent, type Assignment, type MemoryCategory, type MemoryIdea } from '../lib/api'
import {
  assignmentTitle,
  countdown,
  formatClock,
  formatLongDate,
  greeting,
  pickAgentForAsk,
  STATE_LABEL,
  type MorningLine,
  type MorningNotePayload,
} from './morningNoteModel'
import './MorningNote.css'
import { formatRoute } from './studioRoute'
import { GradientButton } from '../components/kokonut/GradientButton'
import { WorkBoard } from './WorkBoard'
import { navigate } from './useStudioRoute'
import { ArrowUpRight, Layers3, Orbit, Hexagon, AudioLines, Aperture, Building2 } from 'lucide-react'
import { useDecisionQueue } from './useDecisionQueue'
import { reviewLabel } from './assignmentReview'

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
        <header className="hero-glass">
          <p className="date">{formatLongDate(now)}</p>
          <h1 id="greet">{greeting(now)}.</h1>
          <GradientButton className="mt-4" onClick={() => document.getElementById('h-decide')?.scrollIntoView({ block: 'start' })}>
            {queue.loading ? 'Loading decisions' : queue.error ? 'Decision queue unavailable' : `${queue.total} pending decision${queue.total === 1 ? '' : 's'}`} <ArrowUpRight size={17} />
          </GradientButton>
        </header>
        {note.isError && <p className="err" role="alert">Could not load today’s note. You can still open Review work.</p>}

        <section aria-labelledby="h-status">
          <h2 id="h-status">Where things stand</h2>
          <StatusList items={data.products} now={now} agents={agents} queue={queue} />
          <div className="sub">
            <h3 className="sub-h">Customers</h3>
            <StatusList items={data.customers} now={now} agents={agents} queue={queue} />
          </div>
        </section>

        <section aria-labelledby="h-decide">
          <h2 id="h-decide">Needs you <span className="count">{queue.error ? 'Unavailable' : queue.loading ? 'Loading' : queue.total}</span></h2>
          {queue.error && <p className="err" role="alert">Could not refresh all decisions. Counts may be stale; no all-clear is available.</p>}
          {!queue.loading && !queue.error && queue.total === 0 && <p className="empty">Nothing needs you right now.</p>}
          {queue.executing.length > 0 && <section className="mini-queue" aria-label="Mini execution queue">
            <h3>Mini execution queue</h3>
            <ol>{queue.executing.map(item => (
              <li key={item.id}>
                <a href={formatRoute({ view: 'assignments', assignmentId: item.id })}>{item.title}</a>
                <span className={item.run_ticket ? 'waiting' : 'active'}>{item.run_ticket ? 'Queued to run' : 'Running'}</span>
              </li>
            ))}</ol>
          </section>}
          <ol className="cards">
            {queue.groups.flat().map(item => <li key={item.id}><WorkDecision item={item} agents={agents} approval={queue.pendingApprovalIds.has(item.id)} /></li>)}
          </ol>
          {queue.savedIdeas > 0 && <SavedIdeas />}
          {queue.missingApprovalIds.length > 0 && <p className="err" role="alert">{queue.missingApprovalIds.length} permission requests reference unavailable work. These approvals remain pending; inspect Work before proceeding.</p>}
          <p className="queue-meta">Needs verification: {queue.error ? 'Unavailable' : queue.loading ? 'Loading' : queue.verification} · Activity: {queue.error ? 'Unavailable' : queue.loading ? 'Loading' : queue.activity}</p>
          <button type="button" className="linkish" onClick={onDetails}>Open Work <ArrowUpRight size={15} /></button>
        </section>

        <section aria-labelledby="h-ask">
          <h2 id="h-ask">Ask for anything</h2>
          <form className="ask-form" onSubmit={event => { event.preventDefault(); const text = ask.trim(); if (text) create.mutate(text) }}>
            <label htmlFor="morning-ask" className="sr-only">What do you need done?</label>
            <textarea id="morning-ask" className="ask-input" rows={1} value={ask} placeholder="e.g. have someone look at why Sanction CI is red" onChange={event => setAsk(event.target.value)} onInput={event => { const el = event.currentTarget; el.style.height = 'auto'; el.style.height = `${Math.min(el.scrollHeight + 2, 200)}px` }} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); const text = ask.trim(); if (text) create.mutate(text) } }} />
            <button className="btn btn-primary" type="submit" disabled={create.isPending || !ask.trim()}>Ask</button>
          </form>
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

      </div>
    </main>
  )
}

function WorkDecision({ item, agents, approval }: { item: Assignment; agents: Agent[]; approval: boolean }) {
  const [open, setOpen] = useState(false)
  return <details className="work-decision" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><span>{item.title}</span><small>{approval ? 'Permission request' : reviewLabel(item) || (item.status === 'queued' ? 'Awaiting start' : item.status)}</small></summary>
    {open && <WorkBoard embedded agents={agents} view="assignments" initialAssignmentId={item.id}
      onRouteHandoff={(from, to) => navigate({ view: 'handoffs', from, to })} />}
  </details>
}

function SavedIdeas() {
  const [offset, setOffset] = useState(0)
  const page = useQuery({ queryKey: ['memory-inbox', 'home', offset], queryFn: () => api.memoryInbox({ status: 'unreviewed', attention: 'needs_you', offset }) })
  return <div>
    {page.isError && <p role="alert">Could not load saved ideas.</p>}
    {page.isPending && <p>Loading saved ideas…</p>}
    {page.data?.ideas.map(idea => <IdeaDecision key={idea.id} idea={idea} />)}
    {offset > 0 && <button onClick={() => setOffset(Math.max(0, offset - 50))}>Previous ideas</button>}
    {offset + 50 < (page.data?.total ?? 0) && <button onClick={() => setOffset(offset + 50)}>More ideas</button>}
  </div>
}

function IdeaDecision({ idea }: { idea: MemoryIdea }) {
  const qc = useQueryClient()
  const [category, setCategory] = useState<MemoryCategory>('project')
  const save = useMutation({
    mutationFn: (keep: boolean) => keep ? api.promoteIdea(idea.id, category, '', { priority: idea.priority }) : api.dismissIdea(idea.id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['memory-inbox'] }),
  })
  return <details className="work-decision"><summary>{idea.text.split('\n')[0]}<small>Saved idea</small></summary>
    <p className="whitespace-pre-wrap">{idea.text}</p>
    <label>Memory category <select value={category} onChange={e => setCategory(e.target.value as MemoryCategory)}>{MEMORY_CATEGORIES.map(name => <option key={name}>{name}</option>)}</select></label>
    <div className="card-actions"><button className="btn" disabled={save.isPending} onClick={() => save.mutate(true)}>Save to memory</button>
    <button className="btn" disabled={save.isPending} onClick={() => save.mutate(false)}>Dismiss idea</button></div>
    {save.error && <p role="alert">{save.error.message}</p>}
  </details>
}

const productIcons = { aiia: Orbit, mindmoor: Layers3, sanction: Hexagon, mia: AudioLines, morrow: Aperture }
function StatusList({ items, now, agents, queue }: { items: MorningLine[]; now: Date; agents: Agent[]; queue: ReturnType<typeof useDecisionQueue> }) {
  if (!items.length) return <p className="empty">Nothing to show yet.</p>
  return <ul className="lines">{items.map(item => {
    const work = queue.needsYou.filter(record => item.agent_ids?.includes(record.agent_id))
    const state = work.length && item.state !== 'blocked' ? 'waiting' : item.state
    const Icon = productIcons[item.id as keyof typeof productIcons] ?? Building2
    return <li className="project-tile" key={item.id}>
      <div className="tile-heading"><Icon aria-hidden="true" size={20} /><strong>{item.name}</strong></div>
      <p className={`tile-state w-${state}`}><span className={`dot dot-${state}`} />{work.length ? `${work.length} decision${work.length === 1 ? '' : 's'}` : item.target && state === 'countdown' ? countdown(item.target, now).text : item.incomplete && state === 'clean' ? 'Checks unavailable' : STATE_LABEL[state] || state}</p>
      {item.note && item.note !== 'nothing blocked' && <p className="tile-note">{item.note}</p>}
      {work.length > 0 && <details className="tile-decisions"><summary>View decisions</summary>{work.map(record => <WorkDecision key={record.id} item={record} agents={agents} approval={queue.pendingApprovalIds.has(record.id)} />)}</details>}
      {item.kind === 'customer' && <a className="repo-link mr-3" href={formatRoute({ view: 'meetings', clientId: item.id })}>Meeting notes <ArrowUpRight size={14} /></a>}
      {item.github && <a className="repo-link" href={`https://github.com/${item.github}`} target="_blank" rel="noreferrer">Repository <ArrowUpRight size={14}/></a>}
    </li>
  })}</ul>
}
