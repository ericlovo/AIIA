import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Assignment, GitWorkspace, GitWrite } from '../src/lib/api.ts'
import { decisionQueue } from '../src/console/decisionQueue.ts'
import { assignmentNextAction } from '../src/console/assignmentReview.ts'
const row = (id: string, fields: Partial<Assignment> = {}): Assignment => ({ id, title: id, agent_id: 'a', objective: 'Inspect', context: '', success_criteria: '', source_handoff_id: '', priority: 'normal', status: 'completed', result: 'Evidence', error: '', created_at: '2026-10-30', updated_at: '2026-10-30', started_at: null, completed_at: null, ...fields })
test('complete shared queue preserves approval-protected history and counts saved ideas', () => {
  const rows = Array.from({ length: 300 }, (_, i) => row(String(i)))
  rows.push(row('archived', { dismissed_at: 'today' }), row('seen', { attention_seen: true }), row('fixture', { title: 'Eval A-1', dismissed_at: 'today' }))
  const before = JSON.stringify(rows)
  const result = decisionQueue(rows, [{ id: 'ws', assignment_id: 'fixture', status: 'pending' } as GitWorkspace], [], 17)
  assert.equal(result.total, 318)
  assert.equal(result.needsYou.length, 301)
  assert.equal(result.needsYou.some(r => r.id === 'archived' || r.id === 'seen'), false)
  assert.equal(result.needsYou.some(r => r.id === 'fixture'), true)
  assert.equal(JSON.stringify(rows), before)
})
test('pending writes and missing work never disappear into an all-clear', () => {
  const result = decisionQueue([row('accepted', { review_status: 'accepted' })], [], [
    { assignment_id: 'accepted', status: 'pending' } as GitWrite,
    { assignment_id: 'missing', status: 'pending' } as GitWrite,
  ])
  assert.equal(result.total, 2)
  assert.deepEqual(result.missingApprovalIds, ['missing'])
})
test('summary evidence and duplicate counts agree without merging original records', () => {
  const rows = [row('a', { title: 'Same', result: '', summary: true, has_result: true }), row('b', { title: 'Same', result: '', summary: true, has_result: true })]
  const result = decisionQueue(rows, [], [])
  assert.equal(result.groups.length, 1)
  assert.equal(result.total, 2)
  assert.match(assignmentNextAction(rows[0]), /accept or reject/)
})
