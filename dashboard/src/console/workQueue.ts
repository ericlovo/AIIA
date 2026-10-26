import type { Assignment } from '../lib/api'

export type WorkLane = 'needs-you' | 'working' | 'history'

// Only explicit exercise labels qualify; ordinary work mentioning tests stays visible.
export function isTestFixture(item: Assignment): boolean {
  return /^(?:Eval [A-Z]+-\d+\b|Weekend Test —|Tune-up smoke\b)/i.test(item.title.replace(/^(?:Continue: )+/, ''))
}

export function workLane(item: Assignment, approvalPending = false): WorkLane {
  if (approvalPending) return 'needs-you'
  if (isTestFixture(item) || item.dismissed_at) return 'history'
  if (item.status === 'running') return 'working'
  if (item.recovery_pending || item.status === 'queued' || item.status === 'failed' || (item.status === 'completed' && !item.result.trim())) return 'needs-you'
  return !item.review_status || item.review_status === 'unreviewed' || item.review_status === 'rejected' ? 'needs-you' : 'history'
}

export function groupWork(items: Assignment[]): Assignment[][] {
  const groups = new Map<string, Assignment[]>()
  for (const item of items) {
    // Manual source references identify individual runs, not distinct task scope.
    // This is display grouping only; every record still requires its own decision.
    const key = JSON.stringify([item.source_kind === 'manual' ? '' : item.agent_id, item.priority, item.schedule_key, item.source_kind, item.source_kind === 'manual' ? '' : item.source_ref, item.source_handoff_id, item.title, item.objective, item.context, item.success_criteria, item.status, item.review_status, Boolean(item.dismissed_at), item.revision_of, Boolean(item.recovery_pending)])
    const group = groups.get(key) ?? []
    group.push(item)
    groups.set(key, group)
  }
  return [...groups.values()].map(group => group.sort((a, b) => b.updated_at.localeCompare(a.updated_at)))
    .sort((a, b) => b[0].updated_at.localeCompare(a[0].updated_at))
}
