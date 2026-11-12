import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { studioFetch, studioJSON } from '../lib/connection'
import { publicSource } from './bdModel'
import { bdDecisionLabel, type BDReceipt, type BDReceiptDetail } from './bdRecordModel'
import { formatRoute } from './studioRoute'
import './BDRecord.css'

async function get<T>(path: string, signal: AbortSignal) {
  return studioJSON<T>(await studioFetch(path, { signal, cache: 'no-store' }))
}

function RecordedAt({ value }: { value: string }) {
  return <time dateTime={value}>{new Date(value).toLocaleString()}</time>
}

function StatusMessage({ error, retry }: { error: Error; retry: () => void }) {
  return <div role="alert" className="receipt-notice"><p>Record unavailable. {error.message}</p><button type="button" onClick={retry}>Retry</button></div>
}

function ReceiptList() {
  const [search, setSearch] = useState('')
  const query = useQuery({
    queryKey: ['bd-record'],
    queryFn: ({ signal }) => get<{ receipts: BDReceipt[]; has_more: boolean }>('/api/bd/record', signal),
    retry: false,
  })
  const filter = search.trim().toLowerCase()
  const rows = query.data?.receipts.filter(r => [r.organization, r.job_id, r.tenant_id, r.bot_id, bdDecisionLabel(r.status)]
    .some(value => value.toLowerCase().includes(filter))) ?? []
  return <>
    <h1>Record</h1><p className="receipt-intro">Decisions made on BD Research packets.</p>
    {query.isError ? <StatusMessage error={query.error} retry={() => void query.refetch()} /> : query.isPending ? <p role="status">Loading receipts…</p> : <>
      <label className="receipt-search">Find a receipt<input type="search" placeholder="Organization, run or decision" value={search} onChange={e => setSearch(e.target.value)} /></label>
      {query.data.has_more && <p>Showing the latest {query.data.receipts.length} decisions.</p>}
      {!rows.length && <p>{filter ? 'No matching receipts in this list.' : 'No BD Research decisions recorded yet.'}</p>}
      <ol className="receipt-list">{rows.map(r => <li key={r.id}><a href={formatRoute({ view: 'record', receiptId: r.id })}>
        <span><strong>{r.organization}</strong><small>{r.tenant_id} · {r.job_id}</small></span>
        <span>{bdDecisionLabel(r.status)}<small><RecordedAt value={r.created_at} /></small></span>
      </a></li>)}</ol>
    </>}
  </>
}

function ReceiptDetail({ id, showRun }: { id: number; showRun: boolean }) {
  const query = useQuery({
    queryKey: ['bd-record', id],
    queryFn: async ({ signal }) => {
      const data = await get<BDReceiptDetail>(`/api/bd/record/${id}`, signal)
      const { receipt, packet } = data
      if (receipt.id !== id || packet.job_id !== receipt.job_id || packet.index !== receipt.packet_index
          || packet.bot_id !== receipt.bot_id || packet.tenant_id !== receipt.tenant_id || !Array.isArray(data.stages)
          || data.stages.some(event => event.job_id !== receipt.job_id || event.bot_id !== receipt.bot_id)) {
        throw new Error('The receipt and run evidence could not be matched.')
      }
      return data
    }, retry: false,
  })
  const data = query.data
  const source = data && publicSource(data.packet.source_url)
  return <>
    <a className="receipt-back" href={showRun ? formatRoute({ view: 'record', receiptId: id }) : formatRoute({ view: 'record' })}>{showRun ? '← Back to receipt' : '← All receipts'}</a><h1>{showRun ? 'Run record' : 'Decision receipt'}</h1>
    {query.isError ? <StatusMessage error={query.error} retry={() => void query.refetch()} /> : !data ? <p role="status">Loading receipt…</p> : <>
      {showRun ? <p className="receipt-intro">{data.receipt.organization} · {data.receipt.tenant_id} · {data.receipt.mode === 'replay' ? 'Synthetic replay' : data.receipt.mode}</p> : <header className="receipt-card"><small>BD Research · Receipt {data.receipt.id}</small>
        <h2>{data.receipt.organization}</h2><p><strong>{bdDecisionLabel(data.receipt.status)}</strong> · <RecordedAt value={data.receipt.created_at} /></p>
        <p>{data.receipt.tenant_id} · {data.receipt.mode === 'replay' ? 'Synthetic replay' : data.receipt.mode}</p>
        <a className="receipt-back" href={formatRoute({ view: 'record', receiptId: id, showRun: true })}>Inspect run {data.receipt.job_id}</a>
      </header>}
      <section className="receipt-evidence" aria-label="Run evidence"><h2>Run evidence</h2>
        <p>Run <code>{data.receipt.job_id}</code> · Packet index {data.receipt.packet_index} · {data.receipt.bot_id}</p>
        {data.packet.source_quote && <blockquote>{data.packet.source_quote}</blockquote>}
        {source && <a href={source} target="_blank" rel="noopener noreferrer">Open source evidence</a>}
        <details key={String(showRun)} open={showRun}><summary>Events through this decision ({data.stages.length})</summary>
          {data.stages.length === 0 ? <p>No events recorded for this run.</p> : <ol>{data.stages.map(event => <li key={event.id}>
            <strong>{event.kind.replaceAll('_', ' ')}</strong> · <RecordedAt value={event.created_at} />
            {event.message && event.message !== event.kind.replaceAll('_', ' ') && <p>{event.message}</p>}
          </li>)}</ol>}
        </details>
      </section>
    </>}
  </>
}

export function BDRecord({ receiptId, showRun = false }: { receiptId?: number; showRun?: boolean }) {
  return <main className="bd-record"><div className="receipt-wrap">
    {receiptId === undefined && <a className="receipt-back" href={formatRoute({ view: 'desk' })}>← Desk</a>}
    {receiptId === undefined ? <ReceiptList /> : <ReceiptDetail key={receiptId} id={receiptId} showRun={showRun} />}
  </div></main>
}
