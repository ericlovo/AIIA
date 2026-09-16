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

test('dismissed work leaves the attention list without changing its outcome', () => {
  const records = [
    work('failed-open', { status: 'failed', result: '', error: 'empty_agent_result' }),
    work('failed-dismissed', { status: 'failed', result: '', error: 'empty_agent_result', review_status: 'dismissed' }),
    work('rejected-dismissed', { review_status: 'dismissed' }),
    work('rejected-open', { review_status: 'rejected' }),
  ]
  assert.deepEqual(attentionAssignments(records).map(item => item.id), ['failed-open', 'rejected-open'])
  // The label reports the human decision, while the run's own outcome is untouched.
  assert.equal(reviewLabel(records[1]), 'Dismissed')
  assert.equal(records[1].status, 'failed')
  assert.equal(records[1].error, 'empty_agent_result')
  assert.equal(reviewLabel(records[2]), 'Dismissed')
  assert.equal(reviewLabel(records[0]), 'Failed run')
})

test('dismissing every flagged record empties attention', () => {
  const records = [
    work('a', { status: 'failed', result: '', review_status: 'dismissed' }),
    work('b', { review_status: 'dismissed' }),
    work('c', { result: '', review_status: 'dismissed' }),
  ]
  assert.deepEqual(attentionAssignments(records), [])
})
