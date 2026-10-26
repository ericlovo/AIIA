import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Assignment } from '../src/lib/api.ts'
import { groupWork, isTestFixture, workLane } from '../src/console/workQueue.ts'
const work = (id: string, overrides: Partial<Assignment> = {}): Assignment => ({
  id, title: 'Inspect source', objective: 'Assess evidence', agent_id: 'agent', priority: 'normal',
  context: '', success_criteria: '', source_handoff_id: '', status: 'completed', result: id, error: '',
  created_at: '2026-10-26', updated_at: '2026-10-26', started_at: null, completed_at: null, ...overrides,
})
test('needs decisions by default; running and resolved records have their own lanes', () => {
  for (const record of [work('legacy'), work('queued', { status: 'queued' }), work('failed', { status: 'failed' }), work('rejected', { review_status: 'rejected' })]) assert.equal(workLane(record), 'needs-you')
  assert.equal(workLane(work('missing', { review_status: 'accepted', result: ' ' })), 'needs-you')
  assert.equal(workLane(work('running', { status: 'running' })), 'working')
  assert.equal(workLane(work('accepted', { review_status: 'accepted' })), 'history')
  assert.equal(workLane(work('dismissed', { dismissed_at: '2026-10-26' })), 'history')
})
test('explicit exercises stay in history but real test work does not', () => {
  for (const title of ['Eval A-1 output', 'Continue: Eval B-2 output', 'Weekend Test — routing', 'Tune-up smoke task']) assert.equal(workLane(work('fixture', { title })), 'history')
  for (const title of ['Fix test runner', 'Write regression tests', 'Evaluate production evidence']) assert.equal(isTestFixture(work('real', { title })), false)
  assert.equal(workLane(work('fixture', { title: 'Eval A-1', dismissed_at: '2026-10-26' }), true), 'needs-you')
  assert.equal(workLane(work('accepted', { review_status: 'accepted' }), true), 'needs-you')
})
test('matching tasks group without merging evidence or changing input records', () => {
  const records = [work('old', { updated_at: '2026-10-25' }), work('new')]
  const before = JSON.stringify(records)
  assert.deepEqual(groupWork(records).map(g => g.map(a => a.id)), [['new', 'old']])
  assert.deepEqual(groupWork(records)[0].map(a => a.result), ['new', 'old'])
  assert.equal(JSON.stringify(records), before)
})
test('different owners, scope, sources and decisions remain separate', () => {
  for (const difference of [{ agent_id: 'other' }, { objective: 'Different task' }, { context: 'Different scope' }, { source_ref: 'other' }, { source_handoff_id: 'other' }, { priority: 'urgent' }, { review_status: 'accepted' }, { revision_of: 'original' }]) {
    assert.equal(groupWork([work('one'), work('two', difference as Partial<Assignment>)]).length, 2)
  }
})
