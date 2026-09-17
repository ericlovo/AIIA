import type { Assignment, Handoff } from '../lib/api'

export const HANDOFF_INSTRUCTIONS_MAX = 8_000

const HANDOFF_ERRORS: Record<string, string> = {
  source_assignment_not_found: 'The source assignment no longer exists.',
  source_assignment_not_completed: 'Only completed work with a result can be handed off.',
  handoff_requires_different_agent: 'Hand the work to a different agent.',
  target_agent_not_found: 'The target agent no longer exists.',
  invalid_artifact_type: 'That artifact type is not supported.',
  assignment_result_too_long: 'The source result is too long to hand off.',
  handoff_capacity_reached: 'The handoff ledger is full. Remove an old handoff first.',
  handoff_not_found: 'This handoff was already removed.',
  handoff_running: 'The handoff is running. Wait for it to finish before removing it.',
}

export function handoffErrorText(detail: string) {
  return HANDOFF_ERRORS[detail] ?? (detail || 'The handoff request failed.')
}

export function defaultHandoffInstructions(source: Pick<Assignment, 'title'>) {
  return `Continue from “${source.title}”. Use the brief to take the next step and report what changed.`
}

export function handoffInstructionsError(instructions: string) {
  if (!instructions.trim()) return 'Add instructions for the receiving agent.'
  if (instructions.length > HANDOFF_INSTRUCTIONS_MAX) return `Keep instructions under ${HANDOFF_INSTRUCTIONS_MAX.toLocaleString('en-US')} characters.`
  return ''
}

export function formatHandoffTime(iso: string) {
  const date = new Date(iso)
  if (!iso || Number.isNaN(date.getTime())) return 'Unknown time'
  return `${date.toISOString().slice(0, 16).replace('T', ' ')} UTC`
}

// Put a freshly created handoff and its target assignment into the cached lists so
// the edge draws before the refetch lands.
export function withCreatedHandoff(handoffs: Handoff[], handoff: Handoff) {
  return [handoff, ...handoffs.filter(item => item.id !== handoff.id)]
}

export function withCreatedAssignment(assignments: Assignment[], assignment: Assignment) {
  return [...assignments.filter(item => item.id !== assignment.id), assignment]
}

export function withoutHandoff(handoffs: Handoff[], handoffId: string) {
  return handoffs.filter(item => item.id !== handoffId)
}
