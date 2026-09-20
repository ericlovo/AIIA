import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Assignment } from '../src/lib/api.ts'
import { assignmentLabel, assignmentOrigin, attentionAssignments, reviewLabel } from '../src/console/assignmentReview.ts'

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

test('dismissal is independent of the verdict, which survives it', () => {
  const records = [
    work('failed-open', { status: 'failed', result: '', error: 'empty_agent_result' }),
    work('failed-dismissed', { status: 'failed', result: '', error: 'empty_agent_result', dismissed_at: '2026-09-16T19:00:00Z' }),
    work('rejected-dismissed', { review_status: 'rejected', dismissed_at: '2026-09-16T19:00:00Z' }),
    work('rejected-open', { review_status: 'rejected' }),
  ]
  assert.deepEqual(attentionAssignments(records).map(item => item.id), ['failed-open', 'rejected-open'])
  // The verdict is not overwritten by dismissing, and both read back.
  assert.equal(records[2].review_status, 'rejected')
  assert.equal(reviewLabel(records[2]), 'Rejected output')
  assert.equal(assignmentLabel(records[2]), 'Rejected output · Dismissed')
  // A failed run has no verdict to keep, so it reads as dismissed alone.
  assert.equal(assignmentLabel(records[1]), 'Failed run · Dismissed')
  assert.equal(records[1].error, 'empty_agent_result')
  assert.equal(assignmentLabel(records[0]), 'Failed run')
})

test('restoring a dismissed record returns it to attention with its verdict', () => {
  const dismissed = work('a', { review_status: 'rejected', dismissed_at: '2026-09-16T19:00:00Z' })
  assert.deepEqual(attentionAssignments([dismissed]), [])
  const restored = { ...dismissed, dismissed_at: null }
  assert.deepEqual(attentionAssignments([restored]).map(item => item.id), ['a'])
  assert.equal(reviewLabel(restored), 'Rejected output')
})

test('dismissing every flagged record empties attention', () => {
  const records = [
    work('a', { status: 'failed', result: '', dismissed_at: '2026-09-16T19:00:00Z' }),
    work('b', { review_status: 'rejected', dismissed_at: '2026-09-16T19:00:00Z' }),
    work('c', { result: '', dismissed_at: '2026-09-16T19:00:00Z' }),
  ]
  assert.deepEqual(attentionAssignments(records), [])
})

test('assignment origin distinguishes scheduled work from operator work', () => {
  assert.equal(assignmentOrigin(work('manual')), 'Manual')
  assert.equal(assignmentOrigin(work('loop', { trigger: 'interval' })), 'Scheduled loop')
  assert.equal(assignmentOrigin(work('handoff', { trigger: 'handoff' })), 'Handoff')
  assert.equal(assignmentOrigin(work('revision', { trigger: 'revision' })), 'Revision')
})
