import assert from 'node:assert/strict'
import test from 'node:test'
import { runTokens } from '../src/console/runTokens.ts'

test('distinguishes unknown, partial, zero and measured usage', () => {
  assert.equal(runTokens({}), 'Tokens unrecorded')
  assert.equal(runTokens({ input_tokens: 12 }), 'Tokens unrecorded')
  assert.equal(runTokens({ input_tokens: null, output_tokens: null }), 'Tokens unrecorded')
  assert.equal(runTokens({ input_tokens: 0, output_tokens: 0 }), '0 tokens')
  assert.equal(runTokens({ input_tokens: 1200, output_tokens: 300 }), '1,500 tokens')
})
