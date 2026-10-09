import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Agent } from '../src/lib/api.ts'
import {
  assignmentTitle,
  greeting,
  homeCopyHasForbiddenWords,
  kickoffPhrase,
  morningLede,
  pickAgentForAsk,
  type MorningLine,
} from '../src/console/morningNote.ts'

const agent = (id: string, overrides: Partial<Agent> = {}): Agent => ({
  id, name: id, mission: 'Do the job.', persona: '', skills: [], tools: [], repo_id: '',
  temperature: 0.35, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60,
  loop_task: '', loop_max_runs_per_day: 4, loop_runs_today: 0, loop_day: '',
  status: 'idle', last_run_at: null, last_result: '', last_error: '', runs: [],
  created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z',
  ...overrides,
})

const roster = [
  agent('ci', { name: 'CI & Failure Fixer', kind: 'coding', handles: ['ci', 'failure'], use_when: 'When checks are red or tests fail.' }),
  agent('review', { name: 'Code Reviewer', kind: 'coding', handles: ['review', 'pr'], use_when: 'When a pull request needs a second look.' }),
  agent('alumni', { name: 'Alumni Nations Research Scout', kind: 'ops', handles: ['alumni'], use_when: 'When Alumni Nations research is needed.' }),
  agent('client', { name: 'Client Daily Preparer', kind: 'ops', handles: ['trs', 'client'], use_when: 'When a client needs daily prep.' }),
  agent('builder', { name: 'Repo Brief & Implementer', kind: 'coding', handles: ['build'], use_when: 'When something should be built or changed.' }),
  agent('aiia', { name: 'AIIA Product Lead', kind: 'product', handles: ['aiia'], use_when: 'When the ask is about AIIA.' }),
  agent('retired', { name: 'Old Scout', kind: 'product', retired: true, handles: ['ci'] }),
]

test('greeting follows the local hour', () => {
  const at = (hour: number) => {
    const date = new Date('2026-10-08T12:00:00')
    date.setHours(hour, 0, 0, 0)
    return date
  }
  assert.equal(greeting(at(7)), 'Good morning')
  assert.equal(greeting(at(13)), 'Good afternoon')
  assert.equal(greeting(at(20)), 'Good evening')
  assert.equal(greeting(at(3)), 'Good evening')
})

test('lede names good shape, stuck lines, decisions, and the Alumni Nations kickoff', () => {
  const products: MorningLine[] = [
    { id: 'mia', name: 'MIA', kind: 'product', state: 'shipped', note: 'Out.' },
    { id: 'morrow', name: 'Morrow', kind: 'product', state: 'clean', note: 'Quiet.' },
    { id: 'sanction', name: 'Sanction', kind: 'product', state: 'blocked', note: 'Checks are red.' },
  ]
  const customers: MorningLine[] = [
    { id: 'alumni-nations', name: 'Alumni Nations', kind: 'customer', state: 'countdown', note: '7 days', target: '2026-10-15T09:00:00' },
  ]
  const lede = morningLede(products, customers, 7, new Date('2026-10-08T21:00:00'))
  assert.match(lede, /MIA and Morrow are in good shape/)
  assert.match(lede, /Sanction is stuck/)
  assert.match(lede, /Seven things need a quick answer from you/)
  assert.match(lede, /Alumni Nations kicks off/)
  assert.equal(homeCopyHasForbiddenWords(lede), false)
})

test('kickoff phrase uses today, tomorrow, or the calendar date', () => {
  assert.equal(kickoffPhrase('2026-10-08T09:00:00', new Date('2026-10-08T08:00:00')), 'today')
  assert.equal(kickoffPhrase('2026-10-09T09:00:00', new Date('2026-10-08T21:00:00')), 'tomorrow')
  assert.match(kickoffPhrase('2026-10-15T09:00:00', new Date('2026-10-08T21:00:00')) ?? '', /Oct/)
})

test('ask routing uses roster handles, then kind, and skips retired agents', () => {
  assert.equal(pickAgentForAsk('have someone look at why Sanction CI is red', roster)?.agent.name, 'CI & Failure Fixer')
  assert.equal(pickAgentForAsk('review the pull request', roster)?.agent.name, 'Code Reviewer')
  assert.equal(pickAgentForAsk('alumni nations kickoff research', roster)?.agent.name, 'Alumni Nations Research Scout')
  assert.equal(pickAgentForAsk('prep the TRS client notes', roster)?.agent.name, 'Client Daily Preparer')
  assert.equal(pickAgentForAsk('what should we do this week?', roster)?.agent.name, 'AIIA Product Lead')
  assert.equal(pickAgentForAsk('ci is red', [roster[6], roster[0]])?.agent.id, 'ci')
})

test('assignment titles stay within the create-assignment limit', () => {
  assert.equal(assignmentTitle('  merge the digest  '), 'merge the digest')
  assert.equal(assignmentTitle('x'.repeat(130)).length, 120)
})

test('home copy checker flags the words the morning note must not show', () => {
  assert.equal(homeCopyHasForbiddenWords('Seven things need a quick answer from you.'), false)
  assert.equal(homeCopyHasForbiddenWords('3 runs waiting'), true)
  assert.equal(homeCopyHasForbiddenWords('Open the inbox'), true)
  assert.equal(homeCopyHasForbiddenWords('2 loops ok'), true)
  assert.equal(homeCopyHasForbiddenWords('by sources'), true)
})
