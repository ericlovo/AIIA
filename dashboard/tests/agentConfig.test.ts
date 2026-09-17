import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Agent } from '../src/lib/api.ts'
import {
  NUMBER_LIMITS, listSummary, loopSummary, modelChoices, modelSummary, parseBoundedNumber, patchFailureMessage,
  beginPatch, isNewestEdit, pendingValues, readableError, runFailureMessage, runSuccessMessage, settlePatch, truncateText, withAgentFields, withAgentRecord,
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
  const concurrent = withAgentFields(optimistic, 'a', { suite: 'ops' })
  assert.deepEqual(withAgentFields(concurrent, 'a', { temperature: 0.2 })!.agents[0], { ...data.agents[0], suite: 'ops' })
  assert.equal(withAgentFields(undefined, 'a', {}), undefined)
  const record = agent('b', { model: 'qwen3:8b' })
  assert.equal(withAgentRecord(data, record)!.agents[1], record)
})

test('only the newest edit of a field decides what it shows, whatever order responses arrive in', () => {
  const server = agent('a', { model: 'llama3.1:8b' })
  // Older success lands after the newer success: nothing is written back over the newer value.
  let ledger = beginPatch({}, server, { model: 'qwen3:8b' }, 1)
  ledger = beginPatch(ledger, server, { model: '' }, 2)
  assert.deepEqual(pendingValues(ledger), { model: '' })
  assert.equal(isNewestEdit(ledger, { model: 'qwen3:8b' }, 1), false)
  let result = settlePatch(ledger, { model: '' }, 2, { ...server, model: '' })
  assert.deepEqual(result.settled, { model: '' })
  assert.deepEqual(pendingValues(result.ledger), {})
  result = settlePatch(result.ledger, { model: 'qwen3:8b' }, 1, { ...server, model: 'qwen3:8b' })
  assert.deepEqual(result.settled, {})

  // Older success lands first, newer edit then fails: restore what the server confirmed, not the original.
  ledger = beginPatch(beginPatch({}, server, { model: 'qwen3:8b' }, 3), server, { model: 'ghost:1b' }, 4)
  result = settlePatch(ledger, { model: 'qwen3:8b' }, 3, { ...server, model: 'qwen3:8b' })
  assert.deepEqual(result.settled, {})
  assert.deepEqual(pendingValues(result.ledger), { model: 'ghost:1b' })
  assert.deepEqual(settlePatch(result.ledger, { model: 'ghost:1b' }, 4, null).settled, { model: 'qwen3:8b' })

  // Both fail: the original value comes back, never the first optimistic one.
  ledger = beginPatch(beginPatch({}, server, { model: 'qwen3:8b' }, 5), server, { model: 'ghost:1b' }, 6)
  result = settlePatch(ledger, { model: 'qwen3:8b' }, 5, null)
  assert.deepEqual(result.settled, {})
  assert.deepEqual(settlePatch(result.ledger, { model: 'ghost:1b' }, 6, null).settled, { model: 'llama3.1:8b' })

  // Edits to different fields settle independently.
  ledger = beginPatch(beginPatch({}, server, { temperature: 0.9 }, 7), server, { suite: 'ops' }, 8)
  result = settlePatch(ledger, { temperature: 0.9 }, 7, null)
  assert.deepEqual(result.settled, { temperature: 0.35 })
  assert.deepEqual(pendingValues(result.ledger), { suite: 'ops' })
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

test('a busy Mini keeps its own wording, other run failures never read as a result', () => {
  assert.equal(runFailureMessage('Mini busy — wait for the active run to finish.'), 'Mini busy — wait for the active run to finish.')
  assert.equal(runFailureMessage('local_model_unavailable'), 'Run failed: the local model is unavailable.')
  assert.equal(runSuccessMessage('qwen3:8b', 2340), 'Run finished on qwen3:8b in 2.3s.')
  assert.equal(runSuccessMessage('', undefined), 'Run finished.')
})
