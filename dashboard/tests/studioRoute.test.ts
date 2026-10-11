import { test } from 'node:test'
import assert from 'node:assert/strict'
import { formatRoute, parseRoute, type StudioRoute } from '../src/console/studioRoute.ts'

test('every route survives a round trip through the URL', () => {
  const routes: StudioRoute[] = [
    { view: 'inbox' }, { view: 'inbox', source: 'slack' }, { view: 'inbox', source: 'loops' },
    { view: 'inbox', source: 'signals' }, { view: 'inbox', source: 'all' },
    { view: 'jobs' }, { view: 'projects' }, { view: 'history' }, { view: 'history', attention: true },
    { view: 'home' },
    { view: 'switchboard' }, { view: 'switchboard', taskId: 'nightly-sync' }, { view: 'activity' },
    { view: 'agents' }, { view: 'agents', agentId: 'a/b c' }, { view: 'assignments' },
    { view: 'assignments', assignmentId: 'asg-1' }, { view: 'assignments', agentId: 'agent-2' },
    { view: 'handoffs', from: 'asg-1', to: 'agent-3' }, { view: 'memory' }, { view: 'memory', review: 'declined' },
    { view: 'memory', review: 'all' }, { view: 'memory', source: 'signals' }, { view: 'memory', source: 'signals', review: 'open' }, { view: 'world' }, { view: 'signals' }, { view: 'switchboard', attention: true },
  ]
  for (const route of routes) {
    const parsed = parseRoute(formatRoute(route))
    // Absent optional fields read back as undefined, which deepEqual treats as missing.
    assert.deepEqual(JSON.parse(JSON.stringify(parsed)), route, formatRoute(route))
  }
})

test('paths are the names people see, not internal view ids', () => {
  assert.equal(formatRoute({ view: 'home' }), '#/note')
  assert.equal(formatRoute({ view: 'switchboard' }), '#/today')
  assert.equal(formatRoute({ view: 'world' }), '#/map')
  assert.equal(formatRoute({ view: 'agents', agentId: 'a/b c' }), '#/agents/a%2Fb%20c')
})

test('unknown or malformed addresses are rejected, never guessed', () => {
  for (const hash of ['', '#', '#/', '#/nowhere', '#today', '#/memory?review=bogus', '#/inbox?source=bogus', '#/inbox/extra', '#/today/extra', '#/map/1', '#/agents/%E0%A4%A']) {
    assert.equal(parseRoute(hash), null, hash)
  }
})

test('an assignment id wins over an agent preset, since they mean different screens', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(parseRoute('#/assignments/asg-1?agent=x'))), { view: 'assignments', assignmentId: 'asg-1' })
})
