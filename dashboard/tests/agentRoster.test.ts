import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Agent } from '../src/lib/api.ts'
import {
  activeAgents,
  gitStanceLabel,
  groupAgentsByKind,
  pausedReviewLabel,
  pickerGroups,
  pickerOptionLabel,
  productRepoLabel,
  resolveKind,
  resolveUseWhen,
} from '../src/console/agentRoster.ts'

const agent = (id: string, overrides: Partial<Agent> = {}): Agent => ({
  id, name: id, mission: 'Do the job.', persona: '', skills: [], tools: [], repo_id: '',
  temperature: 0.35, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60,
  loop_task: '', loop_max_runs_per_day: 4, loop_runs_today: 0, loop_day: '',
  status: 'idle', last_run_at: null, last_result: '', last_error: '', runs: [],
  created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z',
  ...overrides,
})

test('kind is stored when set and otherwise derived from repo or coding tools', () => {
  assert.equal(resolveKind(agent('a', { kind: 'product', repo_id: 'aiia' })), 'product')
  assert.equal(resolveKind(agent('a', { tools: ['Git workspace'] })), 'coding')
  assert.equal(resolveKind(agent('a', { tools: ['Repository read'] })), 'coding')
  assert.equal(resolveKind(agent('a', { repo_id: 'mindmoor' })), 'coding')
  assert.equal(resolveKind(agent('a', { tools: ['Local memory'] })), '')
})

test('use_when falls back to the one-liner', () => {
  assert.equal(resolveUseWhen(agent('a', { use_when: 'Ship the brief.', one_liner: 'Other.' })), 'Ship the brief.')
  assert.equal(resolveUseWhen(agent('a', { one_liner: 'Review CI.' })), 'Review CI.')
  assert.equal(resolveUseWhen(agent('a')), '')
})

test('cards name the product/repo, git stance, and paused review state', () => {
  assert.equal(productRepoLabel(agent('a', { suite: 'mindmoor', repo_id: 'mindmoor' })), 'mindmoor')
  assert.equal(productRepoLabel(agent('a', { suite: 'ops', repo_id: 'aiia' })), 'ops · aiia')
  assert.equal(productRepoLabel(agent('a')), 'No repo')
  assert.equal(gitStanceLabel(agent('a', { tools: ['Git workspace'] })), 'Proposes git (approval)')
  assert.equal(gitStanceLabel(agent('a', { tools: ['Repository read'] })), 'Read-only')
  assert.equal(pausedReviewLabel(agent('a', { loop_skip_reason: 'awaiting_review', value: { window_days: 14, runs: 2, last_run_at: null, reviewed: 0, unreviewed: 3 } })), 'Paused: 3 awaiting review')
  assert.equal(pausedReviewLabel(agent('a', { loop_skip_reason: 'unchanged_repository_input' })), null)
})

test('grouping follows Coding / Product / Ops / Unsorted and hides retired from pickers', () => {
  const coding = agent('CI Fixer', { kind: 'coding', use_when: 'Unblock red checks.' })
  const product = agent('Delivery Lead', { kind: 'product', use_when: 'Sequence the next ship.' })
  const ops = agent('Inbox Clerk', { kind: 'ops', use_when: 'Triage inbound mail.' })
  const unsorted = agent('Scratch', { tools: ['Local memory'] })
  const retired = agent('Old Diplomat', { kind: 'coding', retired: true, use_when: 'Do not pick.' })
  const groups = groupAgentsByKind([unsorted, ops, retired, product, coding])
  assert.deepEqual(groups.map(group => group.label), ['Coding', 'Product', 'Ops', 'Unsorted'])
  assert.deepEqual(groups[0].agents.map(item => item.id), ['Old Diplomat', 'CI Fixer'])
  assert.deepEqual(activeAgents([coding, retired]).map(item => item.id), ['CI Fixer'])

  const pickers = pickerGroups([retired, ops, coding, product, unsorted], [coding.id])
  assert.deepEqual(pickers.map(group => group.label), ['Product', 'Ops', 'Unsorted'])
  assert.ok(!pickers.some(group => group.agents.some(item => item.retired)))
  assert.ok(!pickers.some(group => group.agents.some(item => item.id === coding.id)))
  assert.equal(pickerOptionLabel(product), 'Delivery Lead — Sequence the next ship.')
})
