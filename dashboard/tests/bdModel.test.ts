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

test('planned waiting requires an explicit future start; manual kill switches stay paused', async () => {
  const { plannedWaiting, scheduleIdleText, scheduleQueueText } = await import('../src/console/bdModel.ts')
  const now = Date.parse('2026-10-23T17:00:00Z')
  const schedule = { state: 'waiting', start_at: '2026-10-23T20:00:00Z', next_run_at: '2026-10-23T20:00:00Z', stop_at: '2026-10-26T01:00:00Z', timezone: 'America/Chicago' } as import('../src/console/bdModel.ts').BDSchedule
  const data = { schedule, kill_switch: true, mode: 'live' } as import('../src/console/bdModel.ts').BDSnapshot
  assert.equal(plannedWaiting(schedule, now), true)
  assert.equal(scheduleIdleText(data, now), 'Waiting for planned start')
  assert.match(scheduleQueueText(data, now), /Currently paused/)
  assert.equal(plannedWaiting({ ...schedule, start_at: null }, now), false)
  assert.equal(plannedWaiting({ ...schedule, start_at: 'broken' }, now), false)
  assert.equal(plannedWaiting(schedule, Date.parse('2026-10-24T00:00:00Z')), false)
  assert.equal(scheduleIdleText({ ...data, schedule: undefined }, now), 'Conductor paused')
  assert.equal(scheduleIdleText({ ...data, schedule: { ...schedule, state: 'paused' } }, now), 'Conductor paused')
  assert.equal(scheduleIdleText({ ...data, schedule: { ...schedule, state: 'expired' } }, now), 'Scheduled window ended')
})

test('schedule times use the configured timezone and DST, independent of browser timezone', async () => {
  const { scheduleTime } = await import('../src/console/bdModel.ts')
  assert.match(scheduleTime('2026-10-23T20:00:00Z', 'America/Chicago'), /3:00 PM CDT/)
  assert.match(scheduleTime('2026-11-02T20:00:00Z', 'America/Chicago'), /2:00 PM CST/)
  assert.equal(scheduleTime(null), 'Not scheduled')
  assert.equal(scheduleTime('invalid'), 'Not scheduled')
  assert.equal(scheduleTime('2026-10-23T20:00:00Z', 'Invalid/Zone'), 'Time unavailable')
})

test('stale success is unverified, a real failed observation remains a failure', async () => {
  const { providerStatus } = await import('../src/console/bdModel.ts')
  assert.equal(providerStatus({ status: 'stale', detail: 'old success', observed_at: '2026-10-23T12:00:00Z' }), 'Unverified now')
  assert.equal(providerStatus({ status: 'data_unavailable', detail: 'provider failed', observed_at: '2026-10-23T12:00:00Z' }), 'Last call failed')
  assert.equal(providerStatus({ status: 'data_unavailable', detail: 'not observed' }), 'Unverified')
  assert.equal(providerStatus(undefined), 'Unverified')
})

test('candidate display requires current projection; old dispositions and manual triage never promote history', async () => {
  const { packetOutcome } = await import('../src/console/bdModel.ts')
  const packet = { organization: 'Synthetic operator', source_url: 'https://example.com/evidence', disposition: 'research' } as import('../src/console/bdModel.ts').BDPacket
  for (const disposition of ['research', 'reject', 'watch', 'uncertain', '']) {
    for (const review_status of ['human_review_required', 'research_next', 'watch']) {
      assert.equal(packetOutcome({ ...packet, disposition, review_status }), 'inconclusive')
    }
  }
  assert.equal(packetOutcome({ ...packet, candidate_status: 'supported' }), 'supported')
  assert.equal(packetOutcome({ ...packet, candidate_status: 'rejected' }), 'rejected')
  assert.equal(packetOutcome({ ...packet, candidate_status: 'supported', organization: '  ' }), 'inconclusive')
  assert.equal(packetOutcome({ ...packet, candidate_status: 'supported', source_url: 'javascript:alert(1)' }), 'inconclusive')
  assert.equal(packet.candidate_status, undefined, 'classification does not rewrite history')
})
test('status banner never treats a quiet system as working', async () => {
  const { statusBanner } = await import('../src/console/bdModel.ts')
  assert.deepEqual(statusBanner({ kill_switch: true, schedule_enabled: true }), { state: 'paused_kill_switch', label: 'Paused: kill switch on' })
  assert.deepEqual(statusBanner({ kill_switch: false, schedule_enabled: false }), { state: 'paused_schedule_off', label: 'Paused: schedule off' })
  assert.deepEqual(statusBanner({ kill_switch: false, schedule_enabled: true }), { state: 'waiting', label: 'Waiting for next run' })
  assert.equal(statusBanner({ kill_switch: false, schedule_enabled: false, status_banner: { state: 'running', label: 'Running' } }).label, 'Paused: schedule off')
})

test('withheld outcomes explain absent evidence instead of implying provider failure', async () => {
  const { researchReason, outcomeLabel } = await import('../src/console/bdModel.ts')
  assert.equal(researchReason('no_new_evidence'), 'No new evidence was found in this run.')
  assert.match(researchReason('segment_entity_or_event_unconfirmed'), /not confirmed/)
  assert.match(researchReason(undefined), /not established/)
  assert.equal(outcomeLabel('rejected'), 'Rejected research')
  assert.equal(outcomeLabel('inconclusive'), 'Inconclusive research')
})
