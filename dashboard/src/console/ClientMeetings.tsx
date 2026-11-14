import { useState, type ChangeEvent, type FormEvent } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, ArrowUpRight, CalendarDays, FileText, Plus } from 'lucide-react'
import { api, type AssignmentDefinition, type ClientMeeting, type MeetingSource } from '../lib/api'
import { GradientButton } from '../components/kokonut/GradientButton'
import { PageHeader } from './PageHeader'
import { formatRoute } from './studioRoute'
import { navigate } from './useStudioRoute'
import './ClientMeetings.css'

const sources = { zoom: 'Zoom', morrow: 'Morrow', manual: 'Manual notes' }
const dateLabel = (time: string) => new Date(time).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
const clientQuery = { queryKey: ['meeting-clients'], queryFn: api.clients }
const callsQuery = (client: string) => ({
  queryKey: ['meetings', client], queryFn: () => api.meetings(client), enabled: Boolean(client),
})

export function MeetingScope({ draft, onChange }: { draft: AssignmentDefinition; onChange: (draft: AssignmentDefinition) => void }) {
  const clients = useQuery(clientQuery)
  const calls = useQuery(callsQuery(draft.client_id || ''))
  return <fieldset className="space-y-2"><legend>Shared meeting context</legend>
    <label className="block">Client<select aria-label="Client" className="block w-full" value={draft.client_id || ''} onChange={e => onChange({ ...draft, client_id: e.target.value, meeting_id: '' })}>
      <option value="">No client attached</option>
      {clients.data?.clients?.map(client => <option key={client.id} value={client.id}>{client.name}</option>)}
    </select></label>
    {draft.client_id && <label className="block">Call<select aria-label="Call" className="block w-full" value={draft.meeting_id || ''} onChange={e => onChange({ ...draft, meeting_id: e.target.value })}>
      <option value="">Recent client meetings</option>
      {calls.data?.meetings?.map(call => <option key={call.id} value={call.id}>{call.title} · {dateLabel(call.starts_at)}</option>)}
    </select></label>}
    {(clients.error || calls.error) && <p role="alert">Meeting context could not be loaded.</p>}
  </fieldset>
}

export function ClientMeetings({ clientId = '', meetingId = '' }: { clientId?: string; meetingId?: string }) {
  const clients = useQuery(clientQuery)
  const client = clients.data?.clients?.find(item => item.id === clientId)
  return <main className="client-meetings h-full min-h-0 overflow-y-auto">
    <PageHeader title="Client meetings" />
    <div className="meeting-layout">
      <nav aria-label="Meeting clients" className="meeting-clients">
        <h2>Clients</h2>
        {clients.isPending && <p role="status">Loading clients…</p>}
        {clients.error && <p role="alert">{clients.error.message}</p>}
        {clients.data?.clients?.map(item => <a key={item.id} aria-current={clientId === item.id ? 'page' : undefined} href={formatRoute({ view: 'meetings', clientId: item.id })}>{item.name}</a>)}
      </nav>
      {client ? <div key={client.id} className="meeting-content">
        {meetingId ? <CallDetail key={meetingId} client={client.id} name={client.name} id={meetingId} /> :
          <CallList client={client.id} name={client.name} />}
      </div> : clients.isSuccess && <section className="meeting-panel"><h2>{clientId ? 'Client unavailable' : 'Choose a client'}</h2><p>Open a client to find a call or add meeting notes.</p></section>}
    </div>
  </main>
}

function CallList({ client, name }: { client: string; name: string }) {
  const calls = useQuery({ ...callsQuery(client), refetchInterval: 5000 })
  const qc = useQueryClient()
  const [creating, setCreating] = useState(false)
  const [title, setTitle] = useState('')
  const [time, setTime] = useState('')
  const save = useMutation({
    mutationFn: () => api.createMeeting(client, title.trim(), new Date(time).toISOString()),
    onSuccess: call => {
      void qc.invalidateQueries({ queryKey: ['meetings', client] })
      navigate({ view: 'meetings', clientId: client, meetingId: call.id })
    },
  })
  return <section className="meeting-panel">
    <div className="meeting-heading"><h2>{name}</h2><button type="button" className="meeting-action" onClick={() => setCreating(!creating)}>{creating ? 'Cancel' : <><Plus size={16} /> New call</>}</button></div>
    {creating && <form onSubmit={e => { e.preventDefault(); if (!save.isPending) save.mutate() }}>
      <label>Call title<input required maxLength={160} value={title} onChange={e => setTitle(e.target.value)} /></label>
      <label>Start time ({Intl.DateTimeFormat().resolvedOptions().timeZone})<input required type="datetime-local" value={time} onChange={e => setTime(e.target.value)} /></label>
      {save.error && <p role="alert">{save.error.message}</p>}
      <GradientButton type="submit" disabled={save.isPending || !title.trim() || !time}>Save call</GradientButton>
    </form>}
    {calls.error && <p role="alert">{calls.error.message}</p>}
    {calls.isPending && <p role="status">Loading calls…</p>}
    {calls.data?.meetings.length === 0 && <p>No calls recorded yet.</p>}
    <ul className="meeting-list">{calls.data?.meetings.map(call => <li key={call.id}><a href={formatRoute({ view: 'meetings', clientId: client, meetingId: call.id })}>
      <CalendarDays aria-hidden="true" size={19} /><span><strong>{call.title}</strong><small>{dateLabel(call.starts_at)} · {call.note_count} {call.note_count === 1 ? 'note' : 'notes'}</small></span><ArrowUpRight size={16} />
    </a></li>)}</ul>
    <a className="meeting-link" href={formatRoute({ view: 'assignments', clientId: client })}>Create work with recent client notes <ArrowUpRight size={15} /></a>
  </section>
}

function CallDetail({ client, name, id }: { client: string; name: string; id: string }) {
  const key = ['meetings', client, id]
  const call = useQuery({ queryKey: key, queryFn: () => api.meeting(client, id), refetchInterval: 5000 })
  return <>
    <section className="meeting-panel">
      <a className="meeting-link" href={formatRoute({ view: 'meetings', clientId: client })}><ArrowLeft size={15} /> {name}</a>
      {call.error && <p role="alert">{call.error.message}</p>}
      {call.isPending && <p role="status">Loading call…</p>}
      {call.data && <>
        <h2>{call.data.title}</h2><p><time dateTime={call.data.starts_at}>{dateLabel(call.data.starts_at)}</time></p>
        <a className="meeting-link" href={formatRoute({ view: 'assignments', clientId: client, meetingId: id })}>Create work with these notes <ArrowUpRight size={15} /></a>
        <h3>Shared notes</h3>
        {call.data.notes.length === 0 && <p>Add the first contribution to this call.</p>}
        {call.data.notes.map(note => <article key={note.id} className="meeting-note">
          <header><FileText size={16} aria-hidden="true" /><strong>{note.author}</strong><span>{sources[note.source]}</span></header>
          <small>Added {dateLabel(note.created_at)}{note.source_ref && ` · ${note.source_ref}`}</small>
          <p>{note.text}</p>
        </article>)}
      </>}
    </section>
    {call.data && <AddNote meeting={call.data} />}
  </>
}

function AddNote({ meeting }: { meeting: ClientMeeting }) {
  const qc = useQueryClient()
  const [author, setAuthor] = useState('')
  const [source, setSource] = useState<MeetingSource>('manual')
  const [sourceRef, setSourceRef] = useState('')
  const [text, setText] = useState('')
  const [fileError, setFileError] = useState('')
  const [reading, setReading] = useState(false)
  const save = useMutation({
    mutationFn: () => api.addMeetingNote(meeting.client_id, meeting.id, { author, source, text, source_ref: sourceRef }),
    onSuccess: () => {
      setText(''); setSourceRef('')
      void qc.invalidateQueries({ queryKey: ['meetings', meeting.client_id] })
    },
  })
  async function loadFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setFileError(''); setReading(true)
    try {
      if (file.size > 400000) throw new Error('Use an excerpt of up to 100,000 characters.')
      const content = await file.text()
      if (content.length > 100000) throw new Error('Use an excerpt of up to 100,000 characters.')
      setText(content)
    } catch (error) { setFileError(error instanceof Error ? error.message : 'File could not be read.') }
    finally { setReading(false) }
  }
  function submit(e: FormEvent) { e.preventDefault(); if (!save.isPending && !reading) save.mutate() }
  return <section className="meeting-panel"><h2>Add a note</h2>
    <form onSubmit={submit}><fieldset disabled={save.isPending || reading}>
      <div className="meeting-fields">
        <label>Contributor<input autoComplete="name" required maxLength={100} value={author} onChange={e => setAuthor(e.target.value)} /></label>
        <label>Source<select aria-label="Source" value={source} onChange={e => setSource(e.target.value as MeetingSource)}>{Object.entries(sources).map(([value,label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      </div>
      <label>Notes<textarea required rows={5} maxLength={100000} value={text} onChange={e => setText(e.target.value)} /></label>
      <label>Load a text file<input type="file" accept=".md,.txt,.vtt,text/plain,text/markdown,text/vtt" onChange={loadFile} /></label>
      <label>Source reference (optional)<input maxLength={500} value={sourceRef} onChange={e => setSourceRef(e.target.value)} placeholder="Capture ID or recording reference" /></label>
      {fileError && <p role="alert">{fileError}</p>}
      {save.error && <p role="alert">{save.error.message}</p>}
      <GradientButton type="submit" disabled={!author.trim() || !text.trim()}>Save note</GradientButton>
      {save.isSuccess && !text && <p role="status">Saved to this call.</p>}
    </fieldset></form>
  </section>
}
