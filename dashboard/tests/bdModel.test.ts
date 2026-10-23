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
