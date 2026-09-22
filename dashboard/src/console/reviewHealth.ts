import type { ReviewHealth, ReviewBucket } from '../lib/api'

/** A metric is a doorway: every number here opens the inbox filtered to itself. */
export interface ReviewMetric {
  bucket: ReviewBucket
  label: string
  count: number
  /** Share of reviewed proposals, or null when nothing has been reviewed yet. */
  rate: number | null
  help: string
}

const METRICS: { bucket: ReviewBucket; label: string; help: string }[] = [
  { bucket: 'open', label: 'Open', help: 'Filed by a loop and still waiting for you.' },
  { bucket: 'needs_work', label: 'Accepted as work', help: 'Queued for an agent. It still waits in Work until you start it.' },
  { bucket: 'already_fixed', label: 'Already fixed', help: 'Real, but resolved before you got to it.' },
  { bucket: 'declined', label: 'Declined', help: 'Not worth doing. A high rate means the loop is asking the wrong question.' },
  { bucket: 'external_failure', label: 'External / tooling', help: 'The finding was about something outside the code, such as a vendor outage.' },
  { bucket: 'unclassified', label: 'Unclassified', help: 'Closed without a verdict, usually before outcomes were recorded. Not counted as declined.' },
]

/**
 * Rates are shares of *reviewed* proposals, so an untouched backlog cannot
 * flatter a loop. Open is a count only: it has no verdict to be a share of.
 */
export function reviewMetrics(health: ReviewHealth | undefined): ReviewMetric[] {
  const totals = health?.totals
  const reviewed = health?.reviewed ?? 0
  return METRICS.map(metric => {
    const count = totals?.[metric.bucket] ?? 0
    const rated = metric.bucket !== 'open' && reviewed > 0
    return { ...metric, count, rate: rated ? count / reviewed : null }
  })
}

export function formatRate(rate: number | null): string {
  if (rate === null) return '—'
  return `${Math.round(rate * 100)}%`
}

/** One line a person can act on, rather than six numbers they have to weigh. */
export function reviewSummary(health: ReviewHealth | undefined): string {
  if (!health) return 'Review health unavailable.'
  if (health.filed === 0) return `No local proposals filed in the last ${health.window_days} days.`
  const open = health.totals.open
  if (health.reviewed === 0) {
    return `${health.filed} filed in ${health.window_days} days, none reviewed yet.`
  }
  const accepted = formatRate(health.totals.needs_work / health.reviewed)
  return `${health.filed} filed in ${health.window_days} days · ${open} open · ${accepted} of reviewed became work.`
}

/** Sources worth naming: the ones that actually filed something in the window. */
export function activeSources(health: ReviewHealth | undefined): { source: string; label: string; filed: number; open: number }[] {
  return (health?.by_source ?? []).map(row => ({
    source: row.source,
    label: row.source.replace(/_/g, ' '),
    filed: row.open + row.needs_work + row.already_fixed + row.declined + row.external_failure + row.unclassified,
    open: row.open,
  }))
}
