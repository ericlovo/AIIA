import type { DailySummary, YieldCounts, YieldReport } from './bdModel'

function inconclusiveLabel(counts: Record<string, number> | undefined): string {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  if (!entries.length) return 'none'
  return entries.map(([reason, count]) => `${reason.replaceAll('_', ' ')} ${count}`).join('; ')
}

function Row({ item, showQuery }: { item: YieldCounts; showQuery?: boolean }) {
  const queryId = item.query_id || item.id || 'unknown'
  return <tr>
    <th scope="row">{showQuery ? queryId : item.segment}</th>
    <td>{item.jobs}</td>
    <td>{item.supported}</td>
    <td>{item.rejected ?? 0}</td>
    <td>{item.failed}</td>
    <td>{inconclusiveLabel(item.inconclusive)}</td>
  </tr>
}

export function BDYieldReport({ report, today }: { report?: YieldReport; today?: DailySummary['yield'] }) {
  const segments = report?.segments ?? []
  const queries = segments.flatMap(segment => (segment.queries || []).map(query => ({ ...query, segment: query.segment || segment.segment })))
  if (!segments.length && !today?.segments?.length) return null
  return <section className="bd-yield" aria-label="Query yield">
    <div className="bd-section-heading">
      <h2>Query yield</h2>
      <span className="bd-muted">{report?.scope === 'lifetime_stored_runs' ? 'Lifetime stored runs' : 'Stored research jobs'}</span>
    </div>
    <p className="bd-muted">Supported candidates still require human review. Low-yield queries decay after four jobs with no supported result.</p>
    {today?.segments?.length ? <div className="bd-yield-today">
      <span className="bd-machine-label">Today by segment</span>
      <ul>{today.segments.map(segment => <li key={segment.segment}><strong>{segment.segment}</strong> {segment.supported}/{segment.jobs} supported</li>)}</ul>
    </div> : null}
    {segments.length ? <table>
      <caption>Per segment</caption>
      <thead><tr><th scope="col">Segment</th><th scope="col">Jobs</th><th scope="col">Supported</th><th scope="col">Rejected</th><th scope="col">Failed</th><th scope="col">Inconclusive</th></tr></thead>
      <tbody>{segments.map(segment => <Row key={segment.segment} item={segment} />)}</tbody>
    </table> : null}
    {queries.length ? <table>
      <caption>Per query</caption>
      <thead><tr><th scope="col">Query</th><th scope="col">Jobs</th><th scope="col">Supported</th><th scope="col">Rejected</th><th scope="col">Failed</th><th scope="col">Inconclusive</th></tr></thead>
      <tbody>{queries.map(query => <Row key={`${query.segment}:${query.query_id || query.id}`} item={query} showQuery />)}</tbody>
    </table> : null}
  </section>
}
