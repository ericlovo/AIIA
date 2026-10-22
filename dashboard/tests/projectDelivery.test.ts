import assert from 'node:assert/strict'
import { test } from 'node:test'
import { deliverySignals, parsePRNumbers, type DeliveryFeature } from '../src/console/deliveryEvidence.ts'
const base: DeliveryFeature = { id: 'one', title: 'Feature', status: 'shipped', priority: 'normal', assignment_ids: [], pr_numbers: [], assignments: [], pull_requests: [] }
test('roadmap shipped alone establishes no completed or live evidence', () => assert.deepEqual(deliverySignals(base), ['Unlinked']))
test('completed requires explicit accepted review; statuses can coexist', () => {
  const feature = { ...base, assignment_ids: ['one', 'two'], assignments: [{ id: 'one', status: 'running' }, { id: 'two', status: 'completed' }] }
  assert.deepEqual(deliverySignals(feature), ['Moving', 'Needs review'])
  assert.deepEqual(deliverySignals({ ...feature, assignments: [{ id: 'one', status: 'completed', review_status: 'accepted' }] }), ['Linked'])
  assert.deepEqual(deliverySignals({ ...feature, assignments: [{ id: 'one', status: 'failed' }] }), ['Blocked'])
})
test('PR parser rejects ambiguous, unsafe or nonpositive values', () => {
  assert.deepEqual(parsePRNumbers('12, 34,12'), [12, 34])
  assert.deepEqual(parsePRNumbers(' '), [])
  for (const input of ['0', '-1', '1.2', '12,', '#12', '9007199254740992']) assert.throws(() => parsePRNumbers(input))
})

test('queued work waits for start; missing assignments have broken links', () => {
  assert.deepEqual(deliverySignals({ ...base, assignment_ids: ['one'], assignments: [{ id: 'one', status: 'queued' }] }), ['Awaiting start'])
  assert.deepEqual(deliverySignals({ ...base, assignment_ids: ['one'], assignments: [{ id: 'one', status: 'missing' }] }), ['Broken link'])
})
test('only latest observed workflow per exact PR head can block', () => {
  const run = { id: 1, workflow_id: 7, name: 'CI', html_url: '', head_sha: 'head', status: 'completed', conclusion: 'failure', updated_at: '2026-10-01' }
  const pr = { number: 1, status: 'available', head_sha: 'head', errors: [], ci: { status: 'available', scope: 'pull_request_head', errors: [], runs: [run] } }
  const feature = { ...base, pr_numbers: [1], pull_requests: [pr] }
  assert.deepEqual(deliverySignals(feature), ['Blocked'])
  pr.ci.runs.push({ ...run, id: 2, conclusion: 'success', updated_at: '2026-10-02' })
  assert.deepEqual(deliverySignals(feature), ['Linked'])
  pr.ci.runs = [{ ...run, head_sha: 'old-head' }]
  assert.deepEqual(deliverySignals(feature), ['Linked'])
  assert.deepEqual(deliverySignals({ ...feature, pull_requests: [{ ...pr, mergeable: false }] }), ['Blocked'])
})
