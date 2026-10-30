import test from 'node:test'
import assert from 'node:assert/strict'
import { groupIdeaEvidence } from '../src/console/ideaEvidence.ts'
test('groups repeated claims but retains every original and changed evidence', () => {
  const a = { source: 'code_review', project: 'aiia', text: 'Issue in repo/path\n\nWhy: Evidence\n\nFound by the code review loop in commit abc123.' }
  const b = { ...a, text: a.text.replace('abc123', 'def456') }
  const c = { ...a, text: a.text.replace('Evidence', 'Different evidence') }
  const d = { ...a, project: 'other' }
  const rows = [a,b,c,d], before = JSON.stringify(rows)
  assert.deepEqual(groupIdeaEvidence(rows).map(g => g.length), [2,1,1])
  assert.equal(JSON.stringify(rows), before)
})
