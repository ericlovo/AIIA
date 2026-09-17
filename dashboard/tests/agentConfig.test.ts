import { test } from 'node:test'
import assert from 'node:assert/strict'
import { listSummary, loopSummary, modelSummary, truncateText } from '../src/console/agentConfig.ts'

test('long results are collapsed and cut with an ellipsis at the limit', () => {
  assert.equal(truncateText('  short\n\nanswer  '), 'short answer')
  const cut = truncateText('x'.repeat(500), 20)
  assert.equal(cut.length, 20)
  assert.ok(cut.endsWith('…'))
  assert.equal(truncateText(undefined), '')
})

test('an empty model names the task default instead of pretending one is pinned', () => {
  assert.equal(modelSummary('llama3.1:8b', 'qwen3:8b'), 'llama3.1:8b')
  assert.equal(modelSummary('', 'qwen3:8b'), 'Task default: qwen3:8b')
  assert.equal(modelSummary(undefined, undefined), 'Task default')
})

test('loop state carries the interval and runs today against the daily maximum', () => {
  const loop = { loop_enabled: true, loop_interval_minutes: 90, loop_runs_today: 2, loop_max_runs_per_day: 6 }
  assert.equal(loopSummary(loop), 'On · every 90m · 2 of 6 runs today')
  assert.equal(loopSummary({ ...loop, loop_interval_minutes: 120 }), 'On · every 2h · 2 of 6 runs today')
  assert.equal(loopSummary({ ...loop, loop_enabled: false }), 'Off')
})

test('empty lists read as None', () => {
  assert.equal(listSummary([]), 'None')
  assert.equal(listSummary(undefined), 'None')
  assert.equal(listSummary(['Research', 'Coding']), 'Research, Coding')
})
