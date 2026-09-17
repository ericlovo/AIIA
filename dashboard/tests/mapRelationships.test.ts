import assert from 'node:assert/strict'
import test from 'node:test'
import {
  defaultHandoffInstructions,
  formatHandoffTime,
  handoffErrorText,
  handoffInstructionsError,
  withCreatedAssignment,
  withCreatedHandoff,
  withoutHandoff,
} from '../src/console/mapRelationships.ts'

const handoff = (id: string) => ({
  id, source_assignment_id: 'a1', target_assignment_id: `t-${id}`, from_agent_id: 'x', to_agent_id: 'y',
  artifact_type: 'brief' as const, artifact: '', instructions: 'go', status: 'queued' as const,
  created_at: '2026-09-16T12:00:00Z', updated_at: '2026-09-16T12:00:00Z',
})

test('server handoff details become readable text and unknown details pass through', () => {
  assert.equal(handoffErrorText('handoff_requires_different_agent'), 'Hand the work to a different agent.')
  assert.match(handoffErrorText('handoff_running'), /running/)
  assert.equal(handoffErrorText('something_new'), 'something_new')
  assert.equal(handoffErrorText(''), 'The handoff request failed.')
})

test('instructions must be present and within the server limit', () => {
  assert.match(handoffInstructionsError('   '), /Add instructions/)
  assert.match(handoffInstructionsError('x'.repeat(8_001)), /8,000/)
  assert.equal(handoffInstructionsError('x'.repeat(8_000)), '')
  assert.equal(handoffInstructionsError(defaultHandoffInstructions({ title: 'Scan repo' })), '')
  assert.ok(defaultHandoffInstructions({ title: 'Scan repo' }).includes('Scan repo'))
})

test('created time is formatted deterministically in UTC', () => {
  assert.equal(formatHandoffTime('2026-09-16T12:05:30Z'), '2026-09-16 12:05 UTC')
  assert.equal(formatHandoffTime('not a date'), 'Unknown time')
  assert.equal(formatHandoffTime(''), 'Unknown time')
})

test('a created handoff is placed first once, and removal drops only that edge', () => {
  const list = [handoff('h1'), handoff('h2')]
  const created = withCreatedHandoff(list, handoff('h3'))
  assert.deepEqual(created.map(item => item.id), ['h3', 'h1', 'h2'])
  assert.deepEqual(withCreatedHandoff(created, handoff('h3')).map(item => item.id), ['h3', 'h1', 'h2'])
  assert.deepEqual(withoutHandoff(created, 'h1').map(item => item.id), ['h3', 'h2'])
  assert.equal(list.length, 2)
})

test('the handoff target assignment is appended without duplicates', () => {
  const assignments = [{ id: 'a1' }, { id: 'a2' }] as never[]
  const next = withCreatedAssignment(assignments, { id: 'a3' } as never)
  assert.deepEqual(next.map((item: { id: string }) => item.id), ['a1', 'a2', 'a3'])
  assert.equal(withCreatedAssignment(next, { id: 'a3' } as never).length, 3)
})
