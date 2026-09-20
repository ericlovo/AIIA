import type { Assignment } from '../lib/api'

/** The output verdict, which survives dismissal. */
export function reviewLabel(assignment: Assignment): string {
  if (assignment.status === 'failed') return assignment.error === 'interrupted_by_restart' ? 'Interrupted run' : 'Failed run'
  if (assignment.status !== 'completed') return ''
  if (!assignment.result.trim()) return 'Missing output'
  if (assignment.review_status === 'accepted') return 'Accepted output'
  if (assignment.review_status === 'rejected') return 'Rejected output'
  return 'Awaiting review'
}

/** Verdict plus tracking state, which are independent decisions. */
export function assignmentLabel(assignment: Assignment): string {
  const verdict = reviewLabel(assignment)
  if (!assignment.dismissed_at) return verdict
  return verdict ? `${verdict} · Dismissed` : 'Dismissed'
}

export function assignmentOrigin(assignment: Assignment): string {
  if (assignment.source_kind === 'memory_capture') return 'From Slack capture'
  if (assignment.trigger === 'interval') return 'Scheduled loop'
  if (assignment.trigger === 'handoff') return 'Handoff'
  if (assignment.trigger === 'revision') return 'Revision'
  return 'Manual'
}

/** Work a human still has to look at. A dismissed record keeps its verdict but stops nagging. */
export function attentionAssignments(assignments: Assignment[], agentId = ''): Assignment[] {
  const rank = (item: Assignment) => item.status === 'failed' ? 0 : item.review_status === 'rejected' ? 1 : 2
  const priority = { urgent: 0, high: 1, normal: 2, low: 3 }
  return assignments.filter(item => (!agentId || item.agent_id === agentId)
    && !item.dismissed_at
    && (item.status === 'failed' || (item.status === 'completed' && (!item.result.trim() || item.review_status !== 'accepted'))))
    .sort((a, b) => rank(a) - rank(b) || priority[a.priority] - priority[b.priority]
      || a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))
}
