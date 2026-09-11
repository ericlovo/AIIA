import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Assignment } from '../src/lib/api.ts'
import { attentionAssignments, reviewLabel } from '../src/console/assignmentReview.ts'

const work = (id: string, overrides: Partial<Assignment> = {}): Assignment => ({
  id, title: id, objective: 'Assess evidence', agent_id: 'agent', priority: 'normal',
  context: '', success_criteria: '', source_handoff_id: '', status: 'completed', result: 'Evidence', error: '',
  created_at: '2026-09-10', updated_at: '2026-09-10', started_at: null, completed_at: null, ...overrides,
})

test('completed does not mean accepted; legacy and rejected output need attention', () => {
  const records = [work('accepted', { review_status: 'accepted' }), work('legacy'), work('rejected', { review_status: 'rejected' }), work('running', { status: 'running' }), work('failed', { status: 'failed', error: 'interrupted_by_restart' })]
  assert.deepEqual(attentionAssignments(records).map(item => item.id), ['failed', 'rejected', 'legacy'])
  assert.equal(reviewLabel(records[4]), 'Interrupted run')
  assert.equal(reviewLabel(records[0]), 'Accepted output')
  assert.equal(reviewLabel(records[1]), 'Awaiting review')
  assert.equal(records[0].id, 'accepted') // Sorting never mutates the query cache.
})

test('attention keeps missing artifacts visible and scopes only to selected agent', () => {
  assert.equal(reviewLabel(work('blank', { result: ' ' })), 'Missing output')
  const records = [work('blank', { result: '', review_status: 'accepted' }), work('other', { agent_id: 'other' }), work('urgent', { priority: 'urgent' })]
  assert.deepEqual(attentionAssignments(records, 'agent').map(item => item.id), ['urgent', 'blank'])
})
