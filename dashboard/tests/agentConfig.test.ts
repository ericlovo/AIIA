import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Agent } from '../src/lib/api.ts'
import {
  NUMBER_LIMITS, listSummary, loopSummary, modelChoices, modelSummary, parseBoundedNumber, patchFailureMessage,
  previousFields, readableError, settlePending, truncateText, withAgentFields, withAgentRecord,
} from '../src/console/agentConfig.ts'

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

test('number fields enforce the AgentCreateRequest limits before a request is sent', () => {
  assert.deepEqual(parseBoundedNumber('0.5', NUMBER_LIMITS.temperature), { value: 0.5 })
  assert.deepEqual(parseBoundedNumber('1.2', NUMBER_LIMITS.temperature), { error: 'must be between 0 and 1' })
  assert.deepEqual(parseBoundedNumber('127', NUMBER_LIMITS.max_tokens), { error: 'must be between 128 and 2000' })
  assert.deepEqual(parseBoundedNumber('300.5', NUMBER_LIMITS.max_tokens), { error: 'must be a whole number' })
  assert.deepEqual(parseBoundedNumber(' ', NUMBER_LIMITS.loop_interval_minutes), { error: 'must be a number' })
  assert.deepEqual(parseBoundedNumber('48', NUMBER_LIMITS.loop_max_runs_per_day), { value: 48 })
})

test('contract error codes read as sentences naming the field that was not saved', () => {
  assert.match(patchFailureMessage({ loop_enabled: true }, 'loop_task_required'), /^Loop not saved: the loop needs a loop task/)
  assert.equal(patchFailureMessage({ model: 'ghost:1b' }, 'unknown_model'), 'Model not saved: that model is not installed on the Mini.')
  assert.match(patchFailureMessage({ model: 'qwen3:8b' }, 'models_unavailable'), /^Model not saved: Ollama is not reachable/)
  assert.equal(readableError('[object Object]'), 'the Mini rejected that value.')
  assert.equal(readableError('Synthetic outage'), 'Synthetic outage')
})

const agent = (id: string, overrides: Partial<Agent> = {}) => ({ id, name: id, temperature: 0.35, model: '', suite: '', ...overrides }) as Agent

test('optimistic fields touch one agent, and rollback restores only the patched field', () => {
  const data = { agents: [agent('a', { temperature: 0.2, suite: 'release' }), agent('b')] }
  const optimistic = withAgentFields(data, 'a', { temperature: 0.9 })!
  assert.equal(optimistic.agents[0].temperature, 0.9)
  assert.equal(optimistic.agents[0].suite, 'release')
  assert.equal(optimistic.agents[1], data.agents[1])
  assert.equal(data.agents[0].temperature, 0.2) // The cache snapshot is never mutated.
  const previous = previousFields(data.agents[0], { temperature: 0.9 })
  assert.deepEqual(previous, { temperature: 0.2 })
  const concurrent = withAgentFields(optimistic, 'a', { suite: 'ops' })
  assert.deepEqual(withAgentFields(concurrent, 'a', previous)!.agents[0], { ...data.agents[0], suite: 'ops' })
  assert.equal(withAgentFields(undefined, 'a', {}), undefined)
  const record = agent('b', { model: 'qwen3:8b' })
  assert.equal(withAgentRecord(data, record)!.agents[1], record)
})

test('a settled edit leaves a newer pending edit to the same field in place', () => {
  assert.deepEqual(settlePending({ temperature: 0.9, suite: 'ops' }, { temperature: 0.9 }), { suite: 'ops' })
  assert.deepEqual(settlePending({ temperature: 0.7 }, { temperature: 0.9 }), { temperature: 0.7 })
})

test('model picker offers the task default first and keeps an uninstalled pinned model visible', () => {
  const catalog = { default: 'qwen3:8b', models: [{ id: 'qwen3:8b', label: 'qwen3:8b', family: 'qwen3', parameter_size: '8.2B', size_gb: 5.2, default: true }] }
  assert.deepEqual(modelChoices('', catalog), [
    { id: '', label: 'Task default: qwen3:8b' },
    { id: 'qwen3:8b', label: 'qwen3:8b · 8.2B' },
  ])
  assert.deepEqual(modelChoices('gone:3b', catalog).at(-1), { id: 'gone:3b', label: 'gone:3b (not installed)' })
  assert.deepEqual(modelChoices('gone:3b', undefined), [{ id: '', label: 'Task default' }, { id: 'gone:3b', label: 'gone:3b' }])
})
