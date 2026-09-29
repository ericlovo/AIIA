import type { Assignment, GitWorkspace, GitWrite } from '../lib/api'

/** The output verdict, which survives dismissal. */
export function reviewLabel(assignment: Assignment): string {
  // A scheduled check that could not read its inputs: no model ran, nothing was verified.
  if (assignment.source_kind === 'loop_check') return 'Check incomplete'
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
  if (assignment.source_kind === 'loop_check') return 'Scheduled check'
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

export interface AttentionSummary {
  review: number
  failed: number
  approvals: number
  total: number
}

/**
 * The single "needs attention" count. Today and Overview both render it, so
 * they can never disagree. Everything counted here can be cleared by a human:
 * review or dismiss the assignment, or decide the approval.
 */
export function attentionSummary(
  assignments: Assignment[],
  workspaces: Pick<GitWorkspace, 'id' | 'agent_id' | 'status'>[] = [],
  writes: Pick<GitWrite, 'workspace_id' | 'status'>[] = [],
  agentId = '',
): AttentionSummary {
  const flagged = attentionAssignments(assignments, agentId)
  const failed = flagged.filter(item => item.status === 'failed').length
  const scoped = agentId ? workspaces.filter(item => item.agent_id === agentId) : workspaces
  const scopedIds = new Set(scoped.map(item => item.id))
  const approvals = scoped.filter(item => item.status === 'pending').length
    + writes.filter(item => item.status === 'pending' && (!agentId || scopedIds.has(item.workspace_id))).length
  return { review: flagged.length - failed, failed, approvals, total: flagged.length + approvals }
}

/** A refused bulk dismissal names the item that stopped it ("code:assignment_id"). Nothing was dismissed. */
export function bulkDismissError(message: string, titleOf: (id: string) => string): string {
  const [code, id] = message.split(':')
  const item = id ? `"${titleOf(id) || id}"` : 'An item'
  const reasons: Record<string, string> = {
    review_changed_refresh_required: `${item} changed after you selected it. Re-check it, then dismiss again.`,
    assignment_already_dismissed: `${item} was already dismissed. Clear it from the selection and try again.`,
    assignment_not_settled: `${item} is still queued or running.`,
    assignment_not_found: `${item} no longer exists. Refresh and select again.`,
    dismiss_note_required: 'Add a reason before dismissing.',
    too_many_assignments: 'Select at most 250 items at a time.',
    review_persistence_failed: 'Storage is unavailable.',
  }
  return `Nothing was dismissed. ${reasons[code] ?? message}`
}
