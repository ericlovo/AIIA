import assert from 'node:assert/strict'
import { test } from 'node:test'
import { captureText, receiptLabel } from '../src/console/memoryText.ts'

test('captureText strips bot mentions and collapses whitespace', () => {
  assert.equal(captureText('<@U0C1DCQFMRC>  log this EPIC for LNS'), 'log this EPIC for LNS')
  assert.equal(captureText('<@U1>'), '')
  assert.equal(captureText('keep <@U1>  middle\n\nlines'), 'keep middle lines')
})

test('receiptLabel distinguishes sent, queued, failed, and no thread', () => {
  assert.deepEqual(receiptLabel('sent', '', 'Save'), { tone: 'sent', text: 'Save receipt sent to Slack' })
  assert.deepEqual(receiptLabel('pending', '', 'Memory'), { tone: 'pending', text: 'Memory receipt queued for Slack' })
  assert.deepEqual(receiptLabel('sending', '', 'Memory').tone, 'pending')
  assert.deepEqual(receiptLabel('failed', 'missing_scope', 'Memory'), { tone: 'failed', text: 'Memory receipt failed: missing_scope' })
  assert.deepEqual(receiptLabel(null, null, 'Memory'), { tone: 'none', text: 'No Slack thread for memory receipt' })
})
