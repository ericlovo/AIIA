import { useEffect, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowUpRight, Check, RefreshCw } from 'lucide-react'
import { studioFetch, studioJSON } from '../lib/connection'
import { packetOutcome, publicSource, type BDPacket } from './bdModel'
import type { BDReceipt } from './bdRecordModel'
import { formatRoute } from './studioRoute'
import './Desk.css'

const decisions = {
  research_next: 'Research next',
  watch: 'Watch',
  dismissed: 'Dismiss',
} as const
type Decision = keyof typeof decisions
type PacketQueue = { packets: BDPacket[]; has_more: boolean }
type Submission = { packet: BDPacket; status: Decision }

async function loadQueue(signal: AbortSignal) {
  const response = await studioFetch('/api/bd/desk', { signal, cache: 'no-store' })
  const data = await studioJSON<PacketQueue>(response)
  if (!Array.isArray(data?.packets)) throw new Error('The packet queue could not be read.')
  if (!data.packets.every(packet =>
    packet?.bot_id && packet.job_id && packet.tenant_id && Number.isInteger(packet.index) &&
    packet.index >= 0 && typeof packet.organization === 'string' &&
    packet.review_status === 'human_review_required' && packetOutcome(packet) === 'supported'
  )) throw new Error('The queue contains a packet that cannot be reviewed.')
  return data
}

async function decide({ packet, status }: Submission) {
  const path = '/api/bd/desk/' + [packet.bot_id, packet.job_id, String(packet.index)].map(encodeURIComponent).join('/') + '/review'
  const response = await studioFetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status, expected_tenant_id: packet.tenant_id }),
  })
  if (response.status === 409) throw new Error('This packet has changed since it was opened.')
  if (response.status === 404) throw new Error('This packet is no longer available.')
  const { receipt } = await studioJSON<{ receipt: BDReceipt }>(response)
  if (!receipt || !Number.isSafeInteger(receipt.id) || receipt.id <= 0 ||
    receipt.tenant_id !== packet.tenant_id || receipt.bot_id !== packet.bot_id ||
    receipt.job_id !== packet.job_id || receipt.packet_index !== packet.index || receipt.status !== status
  ) throw new Error('The server did not confirm a matching receipt.')
  return receipt
}

function Packet({ packet, busy, disabled, onDecide }: {
  packet: BDPacket; busy: boolean; disabled: boolean; onDecide: (status: Decision) => void
}) {
  const source = publicSource(packet.source_url)
  const time = Date.parse(packet.created_at)
  const mode = packet.mode === 'replay' ? 'Synthetic replay' : packet.mode === 'live' ? 'Live research' : 'Mode unavailable'
  return <section className="desk-packet-layout" aria-label="BD Research packet">
    <article className="desk-packet">
      <p className="desk-kicker">Awaiting review</p>
      <h2>{packet.organization}</h2>
      {packet.summary && <p className="desk-summary">{packet.summary}</p>}
      <div className="desk-source">
        <h3>Source evidence</h3>
        {packet.source_quote ? <blockquote>{packet.source_quote}</blockquote> : <p>No source passage recorded.</p>}
        {source && <a href={source} target="_blank" rel="noopener noreferrer">
          Open source <ArrowUpRight size={15} aria-hidden="true" />
        </a>}
      </div>
      <div className="desk-actions" role="group" aria-label="Packet decision" aria-busy={busy}>
        {(Object.keys(decisions) as Decision[]).map(status =>
          <button type="button" key={status} className={status === 'research_next' ? 'desk-primary' : ''}
            disabled={disabled} onClick={() => onDecide(status)}>{decisions[status]}</button>
        )}
      </div>
      {busy && <p role="status">Saving your decision…</p>}
    </article>
    <aside className="desk-run" aria-label="Run details">
      <h3>Run details</h3>
      <dl>
        <div><dt>Mode</dt><dd>{mode}</dd></div>
        <div><dt>Workspace</dt><dd>{packet.tenant_id}</dd></div>
        <div><dt>Bot</dt><dd>{packet.bot_id}</dd></div>
        <div><dt>Run</dt><dd className="desk-run-id">{packet.job_id}</dd></div>
        <div><dt>Recorded</dt><dd>{Number.isFinite(time) ? <time dateTime={packet.created_at}>{new Date(time).toLocaleString()}</time> : 'Time unavailable'}</dd></div>
      </dl>
      <details><summary>Research findings</summary>
        <h4>Extracted signals</h4>
        {packet.signals?.length ? <ul>{packet.signals.map((signal, i) => <li key={i}>{signal}</li>)}</ul> : <p>No signals recorded.</p>}
        <h4>Recorded judgment</h4><p>{packet.disposition?.replaceAll('_', ' ') || 'Unavailable'}</p>
      </details>
    </aside>
  </section>
}

export function Desk() {
  const client = useQueryClient()
  const query = useQuery({
    queryKey: ['bd-desk'], queryFn: ({ signal }) => loadQueue(signal), retry: false,
    refetchOnWindowFocus: false, refetchOnReconnect: false,
  })
  const lock = useRef(false)
  const confirmation = useRef<HTMLHeadingElement>(null)
  const review = useMutation({
    mutationFn: decide, retry: false,
    onSuccess: (_receipt, { packet }) => {
      client.setQueryData<PacketQueue>(['bd-desk'], old => old && ({
        ...old, packets: old.packets.filter(item => !(item.job_id === packet.job_id && item.index === packet.index && item.tenant_id === packet.tenant_id && item.bot_id === packet.bot_id)),
      }))
      for (const key of ['bd-desk', 'bd-record', 'bd-snapshot']) void client.invalidateQueries({ queryKey: [key] })
    },
    onSettled: () => { lock.current = false },
  })
  const receipt = review.data
  useEffect(() => { if (receipt) confirmation.current?.focus() }, [receipt])
  const packet = review.isPending || review.isError ? review.variables?.packet : query.data?.packets[0]
  const refresh = async () => {
    const result = await query.refetch()
    if (result.isSuccess) review.reset()
  }
  const submit = (status: Decision) => {
    if (!packet || lock.current || query.isError || query.isFetching || review.isError || receipt) return
    lock.current = true
    review.mutate({ packet, status })
  }
  return <main className="desk-page">
    <div className="desk-sheet">
      <header className="desk-header"><div><p className="desk-kicker">BD Research</p><h1>Desk</h1></div>
        <button type="button" className="desk-refresh" disabled={query.isFetching || review.isPending} onClick={() => void refresh()}>
          <RefreshCw size={15} aria-hidden="true" /> Refresh packets
        </button>
      </header>
      {receipt && <section className="desk-confirm" aria-label="Saved decision">
        <Check size={22} aria-hidden="true" /><div>
          <h2 ref={confirmation} tabIndex={-1}>Decision recorded</h2>
          <p>{review.variables.packet.organization} · {decisions[review.variables.status]}</p>
          <a href={formatRoute({ view: 'record', receiptId: receipt.id })}>View receipt in Record <ArrowUpRight size={16} aria-hidden="true" /></a>
          {!!query.data?.packets.length && <button type="button" disabled={query.isFetching} onClick={() => review.reset()}>Next packet</button>}
        </div>
      </section>}
      {query.isError && <p role="alert" className="desk-error">Packet queue unavailable. {query.error.message}</p>}
      {review.isError && <p role="alert" className="desk-error">{review.error.message} Refresh packets to check the saved state before trying again.</p>}
      {!receipt && packet && <Packet packet={packet} busy={review.isPending}
        disabled={review.isPending || review.isError || query.isError || query.isFetching} onDecide={submit} />}
      {!receipt && !packet && !query.isError && (query.isPending ? <p role="status">Loading research packets…</p> :
        <section className="desk-empty"><h2>No packets awaiting review</h2><p>Supported candidates from BD Research appear here when they need a decision.</p>
          <a href={formatRoute({ view: 'record' })}>Open Record <ArrowUpRight size={16} aria-hidden="true" /></a></section>)}
    </div>
  </main>
}
