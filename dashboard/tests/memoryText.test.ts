import assert from 'node:assert/strict'
import { test } from 'node:test'
import { captureText, memoryPostLabel, priorityLabel, receiptLabel } from '../src/console/memoryText.ts'

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

test('priorityLabel names each priority and treats unknown values as normal', () => {
  assert.deepEqual(priorityLabel('urgent'), { tone: 'urgent', text: 'Urgent' })
  assert.deepEqual(priorityLabel('high'), { tone: 'high', text: 'High' })
  assert.deepEqual(priorityLabel('low'), { tone: 'low', text: 'Low' })
  assert.deepEqual(priorityLabel('normal'), { tone: 'normal', text: 'Normal' })
  assert.deepEqual(priorityLabel(undefined), { tone: 'normal', text: 'Normal' })
  assert.deepEqual(priorityLabel('critical'), { tone: 'normal', text: 'Normal' })
})

test('memoryPostLabel reports approved posts and stays silent when none was requested', () => {
  assert.deepEqual(memoryPostLabel('sent', ''), { tone: 'sent', text: 'Posted to #aiia-memory' })
  assert.deepEqual(memoryPostLabel('sending', ''), { tone: 'pending', text: 'Post queued for #aiia-memory' })
  assert.deepEqual(memoryPostLabel('pending', 'rate_limited'), { tone: 'pending', text: 'Post queued for #aiia-memory' })
  assert.deepEqual(memoryPostLabel('failed', 'not_in_channel'), { tone: 'failed', text: 'Post to #aiia-memory failed: not_in_channel' })
  assert.equal(memoryPostLabel(null, null), null)
  assert.deepEqual(memoryPostLabel(null, null, true), { tone: 'none', text: 'Same memory already posted to #aiia-memory from another capture' })
  assert.deepEqual(memoryPostLabel('sent', '', true).tone, 'sent')
})
