import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ageLabel, publicSource, heartbeatFresh } from '../src/console/bdModel.ts'

test('missing and malformed telemetry stays unavailable; future times do not turn negative', () => {
  const now = Date.parse('2026-10-23T12:00:00Z')
  assert.equal(ageLabel(null, now), 'unavailable')
  assert.equal(ageLabel('broken', now), 'unavailable')
  assert.equal(ageLabel('2026-10-23T11:58:25Z', now), '1m 35s')
  assert.equal(ageLabel('2026-10-23T12:01:00Z', now), '0s')
  assert.equal(heartbeatFresh('2026-10-23T11:59:59Z', now), true)
  assert.equal(heartbeatFresh('2026-10-23T11:59:29Z', now), false)
  assert.equal(heartbeatFresh('2026-10-23T12:01:00Z', now), false)
  assert.equal(heartbeatFresh(null, now), false)
})
test('untrusted evidence links cannot execute scripts or open local resources', () => {
  for (const value of ['javascript:alert(1)', 'file:///etc/passwd', 'data:text/html,test', '/relative', '']) assert.equal(publicSource(value), null)
  assert.equal(publicSource('https://example.com/evidence'), 'https://example.com/evidence')
})

test('provider activity requires a live running job with fresh heartbeat; replay and stopped jobs never animate', async () => {
  const { modelWorkingJobs } = await import('../src/console/bdModel.ts')
  const now = Date.parse('2026-10-23T12:00:00Z')
  const job = { id: 'job', state: 'running', step: 'qwen_called', heartbeat_at: '2026-10-23T11:59:59Z' }
  const snapshot = { mode: 'live', kill_switch: false, jobs: [job] } as import('../src/console/bdModel.ts').BDSnapshot
  assert.equal(modelWorkingJobs(snapshot, 'qwen', now).length, 1)
  assert.equal(modelWorkingJobs(snapshot, 'jev', now).length, 0)
  assert.equal(modelWorkingJobs({ ...snapshot, mode: 'replay' }, 'qwen', now).length, 0)
  assert.equal(modelWorkingJobs({ ...snapshot, kill_switch: true }, 'qwen', now).length, 0)
  job.state = 'completed'
  assert.equal(modelWorkingJobs(snapshot, 'qwen', now).length, 0)
  job.state = 'running'
  job.heartbeat_at = '2026-10-23T11:59:00Z'
  assert.equal(modelWorkingJobs(snapshot, 'qwen', now).length, 0)
})

test('queue excludes terminal jobs and orders oldest first without changing input', async () => {
  const { queueJobs } = await import('../src/console/bdModel.ts')
  const jobs = [
    { id: 'new', state: 'queued', created_at: '2026-10-23T12:01:00Z' },
    { id: 'done', state: 'completed', created_at: '2026-10-23T11:00:00Z' },
    { id: 'old', state: 'leased', created_at: '2026-10-23T12:00:00Z' },
  ] as import('../src/console/bdModel.ts').BDJob[]
  assert.deepEqual(queueJobs(jobs).map(job => job.id), ['old', 'new'])
  assert.equal(jobs[0].id, 'new')
})
