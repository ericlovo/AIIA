import { useQuery } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

const number = new Intl.NumberFormat('en-US')

export function TokenUsage() {
  const today = useQuery({ queryKey: ['tokens-today'], queryFn: api.tokensToday, refetchInterval: 15_000 })
  const history = useQuery({ queryKey: ['tokens-recent'], queryFn: api.tokensRecent, refetchInterval: 15_000 })
  const data = today.data
  const providers = Object.entries(data?.by_provider ?? {})
  const input = providers.reduce((sum, [, value]) => sum + value.input_tokens, 0)
  const output = providers.reduce((sum, [, value]) => sum + value.output_tokens, 0)
  const purposes = Object.entries(data?.by_purpose ?? {}).sort((a, b) => b[1].tokens - a[1].tokens)
  const days = [...(history.data?.days ?? [])].reverse()
  const maximum = Math.max(1, ...days.map(day => day.total_tokens))
  const stale = today.isError || history.isError

  return <section className="sb-tokens" aria-label="Platform token usage">
    <div className="sb-section-title">
      <div><h2>Token usage</h2><p>Platform-wide / {data?.date ?? 'today'} UTC {stale ? '/ last loaded data' : ''}</p></div>
      <button className="sb-icon" aria-label="Refresh token usage" title="Refresh token usage" onClick={() => { void today.refetch(); void history.refetch() }}>
        <RefreshCw size={16} className={today.isFetching || history.isFetching ? 'sb-spin' : ''} />
      </button>
    </div>
    {stale && <p role="alert">Usage refresh failed. Totals may be incomplete; use refresh to retry.</p>}
    <dl className="sb-token-totals">
      <div><dt>Reported tokens</dt><dd>{data ? number.format(data.total_tokens) : today.isPending ? 'Loading' : 'Unavailable'}</dd></div>
      <div><dt>Input</dt><dd>{data ? number.format(input) : '--'}</dd></div>
      <div><dt>Output</dt><dd>{data ? number.format(output) : '--'}</dd></div>
      <div><dt>Requests</dt><dd>{data ? number.format(data.total_requests) : '--'}</dd></div>
    </dl>
    <div className="sb-token-providers">{providers.map(([provider, value]) => <span key={provider}>{provider === 'local' ? 'Mini / local' : provider}: <strong>{number.format(value.tokens)}</strong></span>)}</div>
    <details>
      <summary>Usage breakdown and 14-day history</summary>
      <div className="sb-token-history" aria-label="Daily reported token usage">
        {days.map(day => <div key={day.date} title={`${day.date}: ${number.format(day.total_tokens)} tokens, ${day.total_requests} requests`}>
          <div className="sb-token-bar"><i style={{ height: `${day.total_tokens / maximum * 100}%` }} /></div>
          <span>{day.date.slice(5)}</span>
          <span className="sr-only">{number.format(day.total_tokens)} tokens, {day.total_requests} requests</span>
        </div>)}
      </div>
      {!history.data && <p>{history.isPending ? 'Loading history...' : 'History unavailable.'}</p>}
      <div className="sb-token-table"><table>
        <caption>Today by purpose</caption>
        <thead><tr><th scope="col">Purpose / last reported model</th><th scope="col">Tokens</th><th scope="col">Requests</th></tr></thead>
        <tbody>{purposes.map(([purpose, value]) => <tr key={purpose}><th scope="row">{purpose}<small>{value.model || 'Model unrecorded'} / {value.providers.join(', ')}</small></th><td>{number.format(value.tokens)}</td><td>{number.format(value.requests)}</td></tr>)}</tbody>
      </table></div>
      {data && purposes.length === 0 && <p>No attributed usage reported today.</p>}
      <p className="sb-coverage">Measured model usage, not token caps. Reporting is best-effort; missing reports are not reconstructed. Studio purposes aggregate multiple agents. Local inference has no per-token API charge.</p>
    </details>
  </section>
}
