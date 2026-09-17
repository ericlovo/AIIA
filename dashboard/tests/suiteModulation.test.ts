import assert from 'node:assert/strict'
import test from 'node:test'
import {
  EMPTY_SUITE_FORM,
  buildSuitePatch,
  describeSuiteSettings,
  suiteDetailText,
  withUpdatedAgents,
} from '../src/console/suiteModulation.ts'

test('an untouched form sends nothing and asks for a change', () => {
  const { patch, errors } = buildSuitePatch(EMPTY_SUITE_FORM)
  assert.deepEqual(patch, {})
  assert.deepEqual(errors, ['Choose at least one setting to change.'])
})

test('only the chosen fields are sent, never identity or membership', () => {
  const { patch, errors } = buildSuitePatch({
    ...EMPTY_SUITE_FORM, model: 'custom', modelId: ' qwen3:8b ', temperature: '0.6', maxTokens: '1600',
    loop: 'on', loopInterval: '30', loopMaxRuns: '6',
  })
  assert.deepEqual(errors, [])
  assert.deepEqual(patch, {
    model: 'qwen3:8b', temperature: 0.6, max_tokens: 1600, loop_enabled: true,
    loop_interval_minutes: 30, loop_max_runs_per_day: 6,
  })
  for (const field of ['name', 'mission', 'suite']) assert.equal(field in patch, false)
})

test('task default sends an empty model and loop off sends false', () => {
  assert.deepEqual(buildSuitePatch({ ...EMPTY_SUITE_FORM, model: 'default' }).patch, { model: '' })
  assert.deepEqual(buildSuitePatch({ ...EMPTY_SUITE_FORM, loop: 'off' }).patch, { loop_enabled: false })
  assert.deepEqual(buildSuitePatch({ ...EMPTY_SUITE_FORM, temperature: '0' }).patch, { temperature: 0 })
})

test('values outside the agent limits are refused before any request', () => {
  const cases: Array<[Partial<typeof EMPTY_SUITE_FORM>, RegExp]> = [
    [{ temperature: '1.5' }, /Temperature must be from 0 to 1/],
    [{ maxTokens: '127' }, /Max tokens must be a whole number from 128 to 2000/],
    [{ maxTokens: '300.5' }, /Max tokens/],
    [{ loopInterval: '10' }, /Loop interval/],
    [{ loopMaxRuns: '49' }, /Runs per day/],
    [{ model: 'custom', modelId: '  ' }, /Enter a model id/],
    [{ temperature: 'warm' }, /Temperature/],
  ]
  for (const [change, message] of cases) {
    const { patch, errors } = buildSuitePatch({ ...EMPTY_SUITE_FORM, ...change })
    assert.equal(errors.length, 1, JSON.stringify(change))
    assert.match(errors[0], message)
    assert.deepEqual(patch, {})
  }
})

test('rejection details read as sentences and unknown details pass through', () => {
  assert.match(suiteDetailText('loop_task_required'), /loop task/)
  assert.match(suiteDetailText('unknown_model'), /not installed/)
  assert.match(suiteDetailText('models_unavailable'), /Ollama/)
  assert.match(suiteDetailText('suite_not_found'), /suite/)
  assert.equal(suiteDetailText('brand_new_detail'), 'brand_new_detail')
  assert.equal(suiteDetailText('[object Object]'), 'The suite update was refused.')
  assert.equal(suiteDetailText(''), 'The suite update was refused.')
})

test('current settings summarize agreement and spread across members', () => {
  const member = { temperature: 0.35, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60, loop_max_runs_per_day: 4 }
  assert.deepEqual(describeSuiteSettings([member, member]), {
    temperature: '0.35', maxTokens: '1200', loop: 'off', loopInterval: '60', loopMaxRuns: '4',
  })
  const mixed = describeSuiteSettings([member, { ...member, temperature: 0.6, max_tokens: 1600, loop_enabled: true }])
  assert.equal(mixed.temperature, '0.35–0.6')
  assert.equal(mixed.maxTokens, '1200–1600')
  assert.equal(mixed.loop, 'on for 1 of 2')
  assert.equal(describeSuiteSettings([{ ...member, loop_enabled: true }]).loop, 'on')
  assert.equal(describeSuiteSettings([]).temperature, '—')
})

test('returned records replace only their own agents', () => {
  const agents = [{ id: 'a', v: 1 }, { id: 'b', v: 1 }, { id: 'c', v: 1 }]
  assert.deepEqual(withUpdatedAgents(agents, [{ id: 'b', v: 2 }, { id: 'z', v: 9 }]), [{ id: 'a', v: 1 }, { id: 'b', v: 2 }, { id: 'c', v: 1 }])
})
