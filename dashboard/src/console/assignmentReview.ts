import type { Assignment } from '../lib/api'

export function reviewLabel(assignment: Assignment): string {
  if (assignment.status === 'failed') return assignment.error === 'interrupted_by_restart' ? 'Interrupted run' : 'Failed run'
  if (assignment.status !== 'completed') return ''
  if (!assignment.result.trim()) return 'Missing output'
  if (assignment.review_status === 'accepted') return 'Accepted output'
  if (assignment.review_status === 'rejected') return 'Rejected output'
  return 'Awaiting review'
}

export function attentionAssignments(assignments: Assignment[], agentId = ''): Assignment[] {
  const rank = (item: Assignment) => item.status === 'failed' ? 0 : item.review_status === 'rejected' ? 1 : 2
  const priority = { urgent: 0, high: 1, normal: 2, low: 3 }
  return assignments.filter(item => (!agentId || item.agent_id === agentId)
    && (item.status === 'failed' || (item.status === 'completed' && (!item.result.trim() || item.review_status !== 'accepted'))))
    .sort((a, b) => rank(a) - rank(b) || priority[a.priority] - priority[b.priority]
      || a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))
}
