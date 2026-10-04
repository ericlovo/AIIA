import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Agent } from '../src/lib/api.ts'
import { channelLabel, filterAgents, hasSlackGap, noReviewedOutput, sortAgents, valueGlance, valueSummary } from '../src/console/agentValue.ts'

const agent = (id: string, overrides: Partial<Agent> = {}): Agent => ({
  id, name: id, mission: 'Do the job.', persona: '', skills: [], tools: [], repo_id: '',
  temperature: 0.35, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60,
  loop_task: '', loop_max_runs_per_day: 4, loop_runs_today: 0, loop_day: '',
  status: 'idle', last_run_at: null, last_result: '', last_error: '', runs: [],
  created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z',
  ...overrides,
})

test('channel labels and slack-not-configured notes', () => {
  assert.equal(channelLabel('studio_inbox'), 'Studio inbox')
  assert.equal(channelLabel('slack'), 'Slack')
  assert.equal(hasSlackGap(agent('a', { output_channel: 'slack', output_channel_note: 'slack not configured' })), true)
  assert.equal(hasSlackGap(agent('a', { output_channel: 'slack', output_channel_note: '' })), false)
  assert.equal(hasSlackGap(agent('a', { output_channel: 'studio_inbox' })), false)
})

test('zero reviewed outputs in 14 days is the quiet signal', () => {
  const quiet = agent('quiet', { value: { window_days: 14, runs: 3, last_run_at: '2026-10-03T00:00:00Z', reviewed: 0, unreviewed: 2 } })
  const earning = agent('earn', { value: { window_days: 14, runs: 3, last_run_at: '2026-10-03T00:00:00Z', reviewed: 2, unreviewed: 0 } })
  assert.equal(noReviewedOutput(quiet), true)
  assert.equal(noReviewedOutput(earning), false)
  assert.equal(valueGlance(quiet), '0 reviewed / 14d')
  assert.match(valueSummary(quiet), /3 runs \/ 14d · 0 reviewed · 2 waiting/)
  assert.deepEqual(filterAgents([quiet, earning], '', true).map(item => item.id), ['quiet'])
})

test('least-reviewed sort puts idle agents first', () => {
  const agents = [
    agent('b', { value: { window_days: 14, runs: 1, last_run_at: '2026-10-02T00:00:00Z', reviewed: 4, unreviewed: 0 } }),
    agent('a', { value: { window_days: 14, runs: 8, last_run_at: '2026-10-04T00:00:00Z', reviewed: 0, unreviewed: 3 } }),
  ]
  assert.deepEqual(sortAgents(agents, 'reviewed').map(item => item.id), ['a', 'b'])
  assert.deepEqual(sortAgents(agents, 'runs').map(item => item.id), ['a', 'b'])
})
