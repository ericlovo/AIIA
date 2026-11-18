import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowDown, ArrowUpRight, Building2, Check, ChevronRight, Database, ExternalLink, Globe2, Layers3, RefreshCw, Search, Users, X } from 'lucide-react'
import { consumerInsightsApi, type BusinessInsights, type DemographicBand, type Demographics, type InsightProspect } from '../lib/consumerInsightsApi'
import { formatRoute, type StudioRoute } from './studioRoute'
import { navigate } from './useStudioRoute'
import './ConsumerInsights.css'

type InsightRoute = Extract<StudioRoute, { view: 'insights' }>
const count = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('en-US').format(value)
const compact = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
const money = (value: number | null) => value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)
const date = (value: string | null, full = false) => value && Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric', ...(full ? { year: 'numeric' } : {}), timeZone: 'UTC' }) : 'Not recorded'
const reviewLabel = (status: string) => ({ human_review_required: 'Needs review', accepted: 'Reviewed', approved: 'Approved', dismissed: 'Dismissed', rejected: 'Rejected' }[status] || status.replaceAll('_', ' '))
const sourceUrl = (value: string) => { try { const url = new URL(value); return ['https:', 'http:'].includes(url.protocol) ? url.href : undefined } catch { return undefined } }
const sourceHost = (value: string) => { try { return new URL(value).hostname.replace(/^www\./, '') } catch { return 'Source unavailable' } }
const colors = ['#77eed0', '#a698ff', '#f4bc80', '#6985aa']

function showProspects() {
  const heading = document.getElementById('ci-prospects-title')
  const surface = heading?.closest('.studio-surface')
  if (!heading || !surface) return
  heading.focus({ preventScroll: true })
  surface.scrollTo({
    top: surface.scrollTop + heading.getBoundingClientRect().top - surface.getBoundingClientRect().top - 20,
    behavior: 'instant',
  })
}

function SourceLink({ url, children }: { url: string; children: ReactNode }) {
  const href = sourceUrl(url)
  return href ? <a href={href} target="_blank" rel="noreferrer">{children}<ArrowUpRight size={14} aria-hidden="true" /></a> : <span>{children}</span>
}

function Notice({ title, message, retry }: { title: string; message?: string; retry?: () => void }) {
  return <div className="ci-notice" role="status"><div><strong>{title}</strong>{message && <p>{message}</p>}</div>{retry && <button onClick={retry}><RefreshCw size={14} aria-hidden="true" /> Retry</button>}</div>
}

function EvidenceChart({ data }: { data: BusinessInsights }) {
  const items = [
    { label: 'Supported', value: data.job_outcomes.supported },
    { label: 'Inconclusive', value: data.job_outcomes.inconclusive },
    { label: 'Rejected', value: data.job_outcomes.rejected },
    { label: 'Unclassified', value: data.job_outcomes.unknown },
  ]
  const total = items.reduce((sum, item) => sum + item.value, 0)
  let cursor = 0
  const gradient = items.map((item, index) => { const start = cursor; cursor += total ? item.value / total * 100 : 0; return `${colors[index]} ${start}% ${cursor}%` }).join(',')
  return <section className="ci-panel ci-evidence" aria-labelledby="ci-evidence-title">
    <div className="ci-panel-heading"><div><span className="ci-eyebrow">All research runs</span><h3 id="ci-evidence-title">Research outcomes</h3></div><Database size={19} aria-hidden="true" /></div>
    <div className="ci-evidence-body">
      <div className="ci-donut" style={{ background: total ? `conic-gradient(from 225deg, ${gradient})` : '#29384c' }} role="img" aria-label={items.map(item => `${item.label}: ${item.value} research runs`).join(', ')}><div><strong>{count(total)}</strong><span>research runs</span></div></div>
      <ul className="ci-legend">{items.map((item, index) => <li key={item.label}><i style={{ background: colors[index] }} aria-hidden="true" /><span>{item.label}</span><strong>{count(item.value)}</strong></li>)}</ul>
    </div>
    <div className="ci-panel-foot"><span>{count(data.totals.packets)} retained packets</span><span>{count(data.totals.failed_jobs)} failed runs</span></div>
  </section>
}

function ActivityChart({ data }: { data: BusinessInsights }) {
  const days = data.trend.slice(-21)
  const max = Math.max(1, ...days.map(day => day.packets))
  return <section className="ci-panel ci-activity" aria-labelledby="ci-activity-title">
    <div className="ci-panel-heading"><div><span className="ci-eyebrow">Research cadence</span><h3 id="ci-activity-title">Research activity</h3></div><span className="ci-subtle">{days.length ? `${date(days[0].date)} – ${date(days[days.length - 1].date)}` : 'No activity yet'}</span></div>
    {days.length ? <>
      <div className="ci-chart-total"><strong>{count(days.reduce((sum, day) => sum + day.packets, 0))}</strong><span>packets over {days.length} recorded days</span></div>
      <div className="ci-activity-bars" role="img" aria-label={days.map(day => `${day.date}: ${day.packets} research packets, ${day.prospects} prospect records`).join('; ')}>
        {days.map(day => <div className="ci-day" key={day.date} title={`${date(day.date)} · ${count(day.packets)} packets`}><div style={{ height: `${day.packets / max * 100}%`, minHeight: day.packets ? '3px' : 0 }} /></div>)}
      </div>
      <div className="ci-chart-axis"><span>{date(days[0].date)}</span><span>{date(days[days.length - 1].date)}</span></div>
    </> : <p className="ci-empty">Completed research will build this timeline.</p>}
  </section>
}

function BandChart({ title, bands, unit }: { title: string; bands: DemographicBand[]; unit: string }) {
  return <div className="ci-band-chart"><h3>{title}<span>% of {unit}</span></h3><div className="ci-band-list">{bands.map((band, index) => <div className="ci-band" key={band.label}>
    <div><span>{band.label}</span><strong>{band.share == null ? '—' : `${band.share.toFixed(1)}%`}</strong></div>
    <div className="ci-band-track"><i style={{ width: `${Math.min(100, Math.max(0, band.share ?? 0))}%`, '--band-color': index % 2 ? '#a99bff' : '#77eed0' } as CSSProperties} /></div>
  </div>)}</div></div>
}

function Population({ route, data, loading, error, retry }: { route: InsightRoute; data?: Demographics; loading: boolean; error: Error | null; retry: () => void }) {
  const unavailable = !data || data.status === 'unavailable'
  return <section className="ci-population ci-panel" aria-labelledby="ci-population-title">
    <div className="ci-panel-heading"><div><span className="ci-eyebrow"><Globe2 size={13} aria-hidden="true" /> Population context</span><h2 id="ci-population-title">Demographics</h2></div>
      <label className="ci-geography"><span>Census geography</span><select aria-label="Census geography" value={route.state || 'us'} onChange={event => navigate({ ...route, state: event.target.value === 'us' ? undefined : event.target.value })}>
        {data ? data.geographies.map(item => <option key={item.id} value={item.id}>{item.name}</option>) : <option value={route.state || 'us'}>{route.state ? `State ${route.state}` : 'United States'}</option>}
      </select></label>
    </div>
    <div className="ci-scope-line"><span>{data?.geography.name || 'Population estimates'}</span><span>{data ? `${data.period} ACS estimates` : 'U.S. Census Bureau'}</span><span>Independent of research segments</span></div>
    {loading ? <div className="ci-loading" role="status">Loading Census estimates…</div> : unavailable || error ? <Notice title="Census estimates unavailable" message={error?.message || data?.error || 'The source could not be reached.'} retry={retry} /> : null}
    {data && !unavailable && <>
      {data.status === 'stale' && <Notice title="Showing the last Census snapshot" message={data.error || 'The source could not be refreshed.'} retry={retry} />}
      <div className="ci-population-stats">
        <div><span>Population</span><strong title={count(data.metrics.population)}>{compact(data.metrics.population)}</strong><small>residents</small></div>
        <div><span>Households</span><strong title={count(data.metrics.households)}>{compact(data.metrics.households)}</strong><small>households</small></div>
        <div><span>Median income</span><strong>{money(data.metrics.median_household_income)}</strong><small>per household · {data.year} USD</small></div>
        <div><span>Median age</span><strong>{data.metrics.median_age ?? '—'}<em> yrs</em></strong><small>population midpoint</small></div>
      </div>
      <div className="ci-demographic-charts"><BandChart title="Age distribution" bands={data.age_bands} unit="people" /><BandChart title="Household income" bands={data.income_bands} unit="households" /></div>
    </>}
    <div className="ci-panel-foot"><SourceLink url={data?.source.api_url || 'https://www.census.gov/programs-surveys/acs'}>{data?.source.name || 'U.S. Census Bureau'}</SourceLink><span>{data?.fetched_at ? `Retrieved ${date(data.fetched_at, true)}` : 'Public population data'}</span></div>
  </section>
}

function ProspectDetail({ prospect, pending, error, close, segmentLabel }: { prospect?: InsightProspect; pending: boolean; error: Error | null; close: () => void; segmentLabel: string }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { const element = dialog.current; element?.showModal(); return () => { element?.close() } }, [])
  return <dialog ref={dialog} className="ci-detail" aria-labelledby="ci-detail-title" onCancel={event => { event.preventDefault(); close() }} onClick={event => { if (event.target === event.currentTarget) close() }}>
    <div className="ci-detail-inner">
      <div className="ci-detail-top"><span className="ci-eyebrow">Prospect evidence</span><button className="ci-icon-button" aria-label="Close prospect" onClick={close} autoFocus><X size={20} /></button></div>
      <h2 id="ci-detail-title">{prospect?.organization || (pending ? 'Loading prospect…' : 'Prospect unavailable')}</h2>
      {error && <Notice title="Research could not be loaded" message={error.message} />}
      {!prospect && !pending && !error && <p className="ci-empty">This address does not match a current research record. Close this view to browse the available prospects.</p>}
      {prospect && <>
        <div className="ci-detail-meta"><span>{segmentLabel}</span><span className="ci-review-badge">{reviewLabel(prospect.review_status)}</span><span>{count(prospect.packet_count)} evidence {prospect.packet_count === 1 ? 'packet' : 'packets'}</span><span>Scope: {prospect.tenant_id}</span></div>
        <p className="ci-detail-intro">Public-source research. Review the evidence before qualification or outreach.</p>
        <div className="ci-evidence-records">{prospect.sources.map((source, index) => <article key={source.id} className="ci-evidence-record">
          <div className="ci-source-heading"><span className="ci-source-index">{String(index + 1).padStart(2, '0')}</span><div><SourceLink url={source.source_url}>{sourceHost(source.source_url)}</SourceLink><span>{source.provenance.captured_at ? 'Captured' : 'Recorded'} {date(source.provenance.captured_at || source.created_at, true)} · {reviewLabel(source.review_status)}</span></div></div>
          {source.source_quote ? <blockquote>{source.source_quote}</blockquote> : <p className="ci-subtle">No source quote recorded.</p>}
          {source.signals.length > 0 && <div className="ci-record-section"><h3>Observed signals</h3><ul>{source.signals.map((signal, item) => <li key={item}>{signal}</li>)}</ul></div>}
          {source.insights.length > 0 && <div className="ci-record-section ci-hypotheses"><h3>Potential implications <span>Unverified</span></h3><ul>{source.insights.map((insight, item) => <li key={item}>{insight}</li>)}</ul></div>}
          <div className="ci-source-provenance"><span>{source.provenance.publisher || source.bot_id}</span><span>{source.provenance.published_at ? `Published ${date(source.provenance.published_at, true)}` : 'Publication date not recorded'}</span></div>
        </article>)}</div>
      </>}
    </div>
  </dialog>
}

function ProspectList({ prospects, route, segments }: { prospects: InsightProspect[]; route: InsightRoute; segments: BusinessInsights['available_segments'] }) {
  const [limit, setLimit] = useState(12)
  return <>
    <div className="ci-prospect-grid">{prospects.slice(0, limit).map((prospect, index) => <article key={prospect.id} className="ci-prospect-card" style={{ '--card-order': Math.min(index, 8) } as CSSProperties}>
      <div className="ci-prospect-top"><span className="ci-company-avatar" aria-hidden="true">{prospect.organization.slice(0, 1).toUpperCase()}</span><span className={`ci-review-badge ${prospect.pending_packet_count > 0 ? 'needs-review' : ''}`}>{prospect.pending_packet_count > 0 ? `${prospect.pending_packet_count} to review` : reviewLabel(prospect.review_status)}</span></div>
      <a className="ci-prospect-name" href={formatRoute({ ...route, segment: prospect.segment, prospect: prospect.id })}><h3>{prospect.organization}</h3><ArrowUpRight size={19} aria-hidden="true" /></a>
      <p className="ci-prospect-segment">{segments.find(item => item.id === prospect.segment)?.label || prospect.segment}</p>
      <p className="ci-prospect-quote">{prospect.source_quote || prospect.signals[0] || 'Open the evidence record for source details.'}</p>
      <div className="ci-prospect-bottom"><SourceLink url={prospect.source_url}>{sourceHost(prospect.source_url)}</SourceLink><span>{count(prospect.packet_count)} {prospect.packet_count === 1 ? 'packet' : 'packets'}</span></div>
    </article>)}</div>
    {prospects.length > limit && <button className="ci-show-more" onClick={() => setLimit(value => value + 12)}>Show {Math.min(12, prospects.length - limit)} more prospects <ArrowDown size={15} /></button>}
  </>
}

export function ConsumerInsights({ route }: { route: InsightRoute }) {
  const segment = route.segment || 'all'
  const all = useQuery({ queryKey: ['consumer-insights', 'research', 'all'], queryFn: () => consumerInsightsApi.research(), refetchInterval: 60_000, retry: 1 })
  const research = useQuery({ queryKey: ['consumer-insights', 'research', segment], queryFn: () => consumerInsightsApi.research(segment), refetchInterval: 60_000, retry: 1 })
  const population = useQuery({ queryKey: ['consumer-insights', 'demographics', route.state || 'us'], queryFn: () => consumerInsightsApi.demographics(route.state), staleTime: 3_600_000, refetchInterval: false, retry: 1 })
  const data = research.data
  const segmentLabel = all.data?.available_segments.find(item => item.id === segment)?.label || (segment === 'all' ? 'All research segments' : segment)
  const normalizedQuery = (route.query || '').toLocaleLowerCase()
  const prospects = data?.prospects.filter(prospect => (!route.review || route.review === 'all' || (route.review === 'human_review_required' ? prospect.pending_packet_count > 0 : prospect.review_status === route.review)) && (!normalizedQuery || [prospect.organization, prospect.source_quote, ...prospect.signals].join(' ').toLocaleLowerCase().includes(normalizedQuery))) || []
  const selected = all.data?.prospects.find(prospect => prospect.id === route.prospect)
  const refresh = () => { void all.refetch(); if (segment !== 'all') void research.refetch(); void population.refetch() }
  const update = (values: Partial<InsightRoute>) => navigate({ ...route, ...values, prospect: undefined })
  const chartSegments = all.data?.segments || []
  const largestSegment = Math.max(1, ...chartSegments.map(item => item.prospects))
  const reviewOptions = Array.from(new Set(['human_review_required', ...(data?.review.map(item => item.status) || []), ...(data?.prospects.map(item => item.review_status) || []), ...(route.review ? [route.review] : [])]))
  return <main className="consumer-insights">
    <div className="ci-wrap">
      <header className="ci-hero">
        <div><div className="ci-eyebrow"><i aria-hidden="true" /> AIIA intelligence</div><h1>Consumer <span>Insights</span></h1></div>
        <div className="ci-hero-actions"><span>{data ? `Research updated ${date(data.as_of)}` : 'Research & population context'}</span><button onClick={refresh} disabled={research.isFetching || population.isFetching}><RefreshCw size={15} className={research.isFetching || population.isFetching ? 'ci-spinning' : ''} aria-hidden="true" /> Refresh data</button></div>
      </header>

      <section aria-labelledby="ci-research-title">
        <div className="ci-section-heading"><div><span className="ci-eyebrow">Recorded research</span><h2 id="ci-research-title">Business intelligence</h2></div><label className="ci-segment-select"><span>Research segment</span><select aria-label="Research segment" value={segment} onChange={event => update({ segment: event.target.value === 'all' ? undefined : event.target.value })}><option value="all">All research segments</option>{all.data?.available_segments.filter(item => item.id !== 'all').map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div>
        {research.isPending && <div className="ci-loading" role="status">Loading recorded research…</div>}
        {research.error && <Notice title="Research unavailable" message={research.error.message} retry={() => { void research.refetch() }} />}
        {data && <>
          <div className="ci-metrics" aria-label={`${segmentLabel} research summary`}>
            <a href={formatRoute({ ...route, review: undefined, query: undefined, prospect: undefined })} onClick={event => { event.preventDefault(); update({ review: undefined, query: undefined }); showProspects() }} className="ci-metric ci-metric-primary"><div><Building2 size={18} aria-hidden="true" /><ArrowDown size={15} aria-hidden="true" /></div><strong>{count(data.totals.prospects)}</strong><span>Source-backed prospects</span><small>Organizations in {segment === 'all' ? 'all segments' : segmentLabel}</small></a>
            <div className="ci-metric"><div><Layers3 size={18} aria-hidden="true" /></div><strong>{count(data.totals.insights)}</strong><span>Research hypotheses</span><small>Potential implications · unverified</small></div>
            <div className="ci-metric"><div><ExternalLink size={18} aria-hidden="true" /></div><strong>{count(data.totals.sources)}</strong><span>Source articles</span><small>{count(data.source_coverage.domains)} link domains</small></div>
            <a href={formatRoute({ ...route, review: 'human_review_required', query: undefined, prospect: undefined })} onClick={event => { event.preventDefault(); update({ review: 'human_review_required', query: undefined }); showProspects() }} className="ci-metric ci-metric-review"><div><Users size={18} aria-hidden="true" /><ArrowDown size={15} aria-hidden="true" /></div><strong>{count(data.totals.awaiting_review)}</strong><span>Reports awaiting review</span><small>Open the supporting prospects</small></a>
          </div>
          <div className="ci-research-grid"><section className="ci-panel ci-segments" aria-labelledby="ci-segments-title"><div className="ci-panel-heading"><div><span className="ci-eyebrow">Coverage</span><h3 id="ci-segments-title">Market coverage</h3></div><span className="ci-subtle">Source-backed prospects</span></div>
            <div className="ci-segment-bars">{chartSegments.map((item, index) => <button key={item.id} className={segment === item.id ? 'selected' : ''} aria-pressed={segment === item.id} onClick={() => update({ segment: segment === item.id ? undefined : item.id })}>
              <div><span>{item.label}</span><strong>{count(item.prospects)}{segment === item.id ? <Check size={13} aria-hidden="true" /> : <ChevronRight size={13} aria-hidden="true" />}</strong></div><span className="ci-segment-track"><i style={{ width: `${item.prospects / largestSegment * 100}%`, '--segment-color': colors[index % colors.length] } as CSSProperties} /></span>
            </button>)}</div>
            {!chartSegments.length && <p className="ci-empty">No research segments have recorded prospects yet.</p>}
          </section><EvidenceChart data={data} /><ActivityChart data={data} /></div>
        </>}
      </section>

      <Population route={route} data={population.data} loading={population.isPending} error={population.error} retry={() => { void population.refetch() }} />

      <section className="ci-prospects" aria-labelledby="ci-prospects-title">
        <div className="ci-section-heading"><div><span className="ci-eyebrow">Explore the evidence</span><h2 id="ci-prospects-title" tabIndex={-1}>Prospects</h2></div><span className="ci-result-count">{count(prospects.length)} prospects<span>{segmentLabel}</span></span></div>
        <div className="ci-prospect-filters"><form key={route.query || ''} onSubmit={event => { event.preventDefault(); const values = new FormData(event.currentTarget); update({ query: String(values.get('q') || '').trim() || undefined }) }} className="ci-search"><Search size={18} aria-hidden="true" /><input name="q" aria-label="Search prospects" type="search" placeholder="Search organizations or signals…" defaultValue={route.query || ''} /><button type="submit">Search</button></form><label><span className="sr-only">Prospect review status</span><select aria-label="Prospect review status" value={route.review || 'all'} onChange={event => update({ review: event.target.value === 'all' ? undefined : event.target.value })}><option value="all">All review states</option>{reviewOptions.map(status => <option key={status} value={status}>{reviewLabel(status)}</option>)}</select></label>{(route.query || route.review) && <button className="ci-clear" onClick={() => update({ query: undefined, review: undefined })}>Clear filters <X size={14} /></button>}</div>
        {research.isPending ? <div className="ci-loading" role="status">Finding prospects…</div> : research.error ? <Notice title="Prospects could not be loaded" message={research.error.message} retry={() => { void research.refetch() }} /> : prospects.length ? <ProspectList key={`${segment}:${route.query || ''}:${route.review || ''}`} prospects={prospects} route={route} segments={all.data?.available_segments || []} /> : <div className="ci-empty ci-empty-prospects"><Search size={26} aria-hidden="true" /><h3>{route.query || route.review ? 'No prospects match these filters.' : 'No supported prospects in this segment yet.'}</h3><p>{route.query || route.review ? 'Try another organization, signal, or review state.' : 'Completed public-source research will appear here with its evidence.'}</p></div>}
      </section>
      <footer className="ci-footer"><span>AIIA · Consumer Insights</span><span>Recorded research & public Census estimates · {all.data ? `${count(all.data.excluded_replay_jobs)} replay runs excluded` : 'Source-linked data'}</span></footer>
    </div>
    {route.prospect && <ProspectDetail key={route.prospect} prospect={selected} pending={all.isPending} error={all.error} close={() => navigate({ ...route, prospect: undefined })} segmentLabel={all.data?.available_segments.find(item => item.id === selected?.segment)?.label || selected?.segment || ''} />}
  </main>
}
