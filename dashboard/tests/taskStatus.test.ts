import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loopState } from '../src/console/taskStatus.ts'

const recent = new Date().toISOString()
test('recovered task uses latest outcome, not lifetime failure ratio', () => {
  assert.equal(loopState({ status: 'idle', last_run: recent, run_history: [{ status: 'done' }] }), 'healthy')
})
test('a failed first attempt is a failure even with zero successful runs', () => {
  assert.equal(loopState({ status: 'failed', last_run: recent }), 'failing')
})
test('running and stale are distinct from idle', () => {
  assert.equal(loopState({ status: 'running', last_run: null }), 'running')
  assert.equal(loopState({ last_status: 'done', last_run: '2020-01-01T00:00:00Z', interval_seconds: 60 }), 'stale')
  assert.equal(loopState({ last_run: null }), 'idle')
})
