// Inbox navigation, source counts, legacy empty captures, and query invalidation.
// All APIs and WebSockets are synthetic; unexpected traffic fails closed.
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const studioDist = process.env.STUDIO_DIST_DIR
const base = studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/'
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-studio-inbox')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = new Date().toISOString().slice(0, 10)
const widths = [1440, 390, 320]
const proposalSources = ['backlog_steward', 'code_review', 'standup', 'public_signals']
const queueSources = ['slack', 'local_proposals', '/api/assignments', '/api/git-workspaces', '/api/git-writes']
const agents = [{ id: 'agent-synthetic', name: 'Synthetic reviewer', mission: 'Review synthetic work.', persona: 'Direct.', status: 'idle', skills: [], tools: ['Local memory'], repo_id: '', runs: [], loop_enabled: false, loop_interval_minutes: 60, loop_max_runs_per_day: 4, last_result: '', last_error: '' }]
const idea = (id, text, overrides = {}) => ({
  id, text, source: 'slack', project: 'mindmoor', workspace_id: 'T_TEST', channel_id: 'C_TEST', author_id: 'U_TEST',
  created_at: `${date}T12:00:00Z`, status: 'unreviewed', priority: 'normal', memory_id: '', memory_category: '', review_note: '', reviewed_at: '', review_outcome: '', assignment_id: '', post_requested: 0,
  acknowledgement_status: null, acknowledgement_error: null, acknowledgement_ts: null, promotion_status: null, promotion_error: null, promotion_ts: null,
  memory_post_status: null, memory_post_error: null, memory_post_ts: null, ...overrides,
})
const seedIdeas = () => [
  idea('slack-alpha', '<@U123> Keep the capture wording intact when logging decisions.'),
  idea('slack-bravo', 'Queue the synthetic follow-up for a human-approved run.', { project: 'other-project' }),
  idea('slack-empty', ' \n<@U123>\t<@U456> ', { acknowledgement_status: 'sent', acknowledgement_ts: '123.456' }),
  idea('slack-logged', 'Previously logged Slack capture.', { status: 'promoted', memory_id: 'memory-old', memory_category: 'project' }),
  idea('slack-dismissed', 'Previously dismissed Slack capture.', { status: 'dismissed' }),
  idea('loop-backlog', 'Pending backlog proposal.', { source: 'backlog_steward' }),
  idea('loop-review', 'Pending code review proposal.', { source: 'code_review' }),
  idea('signal-pending', 'Pending public signal for verification.', { source: 'public_signals' }),
  idea('loop-accepted', 'Accepted proposal must not enter the pending queue.', { source: 'backlog_steward', assignment_id: 'assignment-old', review_outcome: 'needs_work' }),
  idea('loop-dismissed', 'Closed historical proposal must not enter the pending queue.', { source: 'standup', status: 'dismissed', review_outcome: 'declined' }),
  idea('loop-promoted', 'Previously logged proposal must not enter the pending queue.', { source: 'code_review', status: 'promoted', memory_id: 'memory-proposal', memory_category: 'project' }),
]
const work = (id, overrides = {}) => ({
  id, agent_id: agents[0].id, title: `Synthetic work ${id}`, objective: 'Verify synthetic output.', status: 'completed', result: 'Synthetic evidence.', error: '', priority: 'normal', context: '', success_criteria: '', source_handoff_id: '', review_status: 'unreviewed', dismissed_at: '', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z`, ...overrides,
})
const workspaces = [{ id: 'workspace-synthetic', assignment_id: 'review', agent_id: agents[0].id, repo_id: 'synthetic', status: 'pending', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` }]
const writes = [{ id: 'write-synthetic', workspace_id: 'workspace-synthetic', assignment_id: 'review', op: 'commit', status: 'pending', result: {}, created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` }]
const cleanText = text => text.replace(/<@[A-Z0-9]+>/g, '').replace(/\s+/g, ' ').trim() || '(mention only, no text)'
const belongsTo = (item, source) => !source || (source === 'local_proposals' ? proposalSources.includes(item.source) : item.source === source)

function listing(ideas, params) {
  const scoped = ideas.filter(item => belongsTo(item, params.get('source'))
    && (!params.get('project') || item.project === params.get('project'))
    && (!params.get('query') || item.text.toLowerCase().includes(params.get('query').toLowerCase()))
    && (!params.get('priority') || item.priority === params.get('priority'))
    && (!params.get('outcome') || (params.get('outcome') === 'open' ? item.status === 'unreviewed' && !item.assignment_id && !item.review_outcome : item.review_outcome === params.get('outcome'))))
  const counts = { unreviewed: 0, promoted: 0, dismissed: 0 }
  for (const item of scoped) counts[item.status]++
  const rows = scoped.filter(item => !params.get('status') || item.status === params.get('status'))
    .sort((a, b) => b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id))
  const offset = Number(params.get('offset') || 0)
  return { ideas: rows.slice(offset, offset + 50), total: rows.length, offset, counts }
}

async function open(path, width, { failed = [], held = [], clock = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height: width === 1440 ? 1000 : 844 }, serviceWorkers: 'block' })
  const page = await context.newPage()
  page.setDefaultTimeout(12000)
  if (clock) await page.clock.install()
  const state = {
    ideas: seedIdeas(), failures: new Set(failed), calls: [], mutations: [], unexpected: [], errors: [],
    assignments: [work('review'), work('failure', { status: 'failed', result: '', error: 'Synthetic failure.' }), work('accepted', { review_status: 'accepted' }), work('dismissed', { status: 'failed', result: '', dismissed_at: `${date}T13:00:00Z` })],
  }
  const gates = new Map(held.map(key => {
    let release
    const promise = new Promise(resolve => { release = resolve })
    return [key, { promise, release }]
  }))
  page.on('pageerror', error => state.errors.push(error.message))
  await page.routeWebSocket('**/*', ws => {
    if (new URL(ws.url()).pathname !== '/ws') state.unexpected.push(`WebSocket ${ws.url()}`)
    ws.onMessage(() => {})
  })
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()
    if (url.origin !== new URL(base).origin) {
      state.unexpected.push(`${method} ${url}`)
      return route.abort()
    }
    if (!path.startsWith('/api/')) {
      if (method === 'GET' && (path === '/' || path.startsWith('/assets/'))) {
        if (!studioDist) return route.continue()
        const file = path === '/' ? 'index.html' : path.slice(1)
        return route.fulfill({ contentType: file.endsWith('.html') ? 'text/html' : file.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(studioDist, file)) })
      }
      state.unexpected.push(`${method} ${path}`)
      return route.abort()
    }
    const params = Object.fromEntries(url.searchParams)
    state.calls.push({ path, method, ...params })
    const source = path === '/api/memory-inbox' ? params.source || 'all' : path
    if (gates.has(source)) await gates.get(source).promise
    const respond = (body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    if (state.failures.has(source)) return respond({ detail: 'synthetic_source_unavailable' }, 503)
    const mutation = /^\/api\/memory-inbox\/([^/]+)\/(dismiss|promote|assign)$/.exec(path)
    if (method === 'POST' && mutation) {
      const [, id, action] = mutation
      const item = state.ideas.find(row => row.id === id)
      assert.ok(item, 'only seeded captures can be mutated')
      const sent = request.postDataJSON()
      state.mutations.push({ id, action, body: sent })
      if (action === 'dismiss') Object.assign(item, { status: 'dismissed', reviewed_at: `${date}T13:00:00Z` })
      if (action === 'promote') {
        assert.ok(cleanText(item.text) !== '(mention only, no text)')
        assert.equal(sent.post_to_slack, false, 'this suite never requests Slack posts')
        Object.assign(item, { status: 'promoted', memory_id: `memory-${id}`, memory_category: sent.category, priority: sent.priority, reviewed_at: `${date}T13:00:00Z` })
      }
      if (action === 'assign') {
        assert.ok(cleanText(item.text) !== '(mention only, no text)')
        assert.equal(sent.agent_id, agents[0].id)
        item.assignment_id = `assignment-${id}`
        if (item.source !== 'slack') Object.assign(item, { review_outcome: 'needs_work', review_note: sent.review_note })
        const assignment = work(item.assignment_id, { title: cleanText(item.text), status: 'queued', result: '', source_kind: 'memory_capture', source_id: id })
        state.assignments.push(assignment)
        return respond({ idea: item, assignment })
      }
      return respond({ idea: item, memory_id: item.memory_id })
    }
    if (method === 'GET' && path === '/api/memory-inbox') return respond(listing(state.ideas, url.searchParams))
    const bodies = {
      '/api/agents': { agents },
      '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
      '/api/agents/models': { default: 'synthetic-model', models: [] },
      '/api/assignments': { assignments: state.assignments },
      '/api/handoffs': { handoffs: [] },
      '/api/git-workspaces': { workspaces },
      '/api/git-writes': { writes },
      '/api/tasks': [],
      '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
      '/api/monitor': { services: {} },
      '/api/voice/status': { status: 'not_configured', configured: false, reason: 'synthetic_only', tools: [] },
      '/api/studio/activity': { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] },
      '/api/tokens/today': { date, total_tokens: 0, total_requests: 0, total_cost: 0, by_provider: {}, by_purpose: {} },
      '/api/tokens/recent': { days: [] },
      '/api/public-signals/leads': { leads: [], total: 0, offset: 0, limit: 25 },
      // Deliberately inconsistent with the live pending queue: never use these historical metrics for Inbox counts.
      '/api/memory-inbox/review-health': { window_days: 14, since: `${date}T00:00:00Z`, filed: 88, reviewed: 88, totals: { open: 0, needs_work: 75, already_fixed: 0, declined: 13, external_failure: 0, unclassified: 0 }, by_source: [], by_project: [] },
      '/api/integrations/slack/status': { configured: true, workspace_id: 'T_TEST', channel_ids: ['C_TEST'], outbound_messages: false, acknowledgements_enabled: false, acknowledgements_configured: false, acknowledgements: {}, promotion_acknowledgements: {}, memory_posts_enabled: true, memory_posts_configured: true, memory_post_channel_id: 'C_MEMORY_TEST', memory_posts: {} },
    }
    if (method === 'GET' && Object.hasOwn(bodies, path)) return respond(bodies[path])
    state.unexpected.push(`${method} ${path}`)
    return respond({ detail: 'unexpected_synthetic_request' }, 501)
  })
  const release = () => { for (const gate of gates.values()) gate.release(); gates.clear() }
  const close = async () => {
    release()
    assert.deepEqual(state.errors, [], 'no uncaught browser errors')
    assert.deepEqual(state.unexpected, [], 'no unmocked requests or real mutations')
    await context.close()
  }
  await page.goto(`${base}${path}`)
  return { page, state, release, close }
}

const nav = page => page.getByRole('navigation', { name: 'Studio', exact: true })
const captures = page => page.getByRole('region', { name: 'Inbox captures', exact: true })
const queue = (page, label) => page.getByRole('navigation', { name: 'Inbox queues', exact: true }).getByRole('link', { name: new RegExp(`^${label}: `) })
const row = (page, text) => captures(page).getByRole('listitem').filter({ has: page.getByText(text, { exact: true }) })
const heading = (page, title) => page.getByRole('heading', { level: 1, name: title, exact: true })
const screenshot = async (page, name) => {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `${name}: no horizontal page overflow`)
  await page.screenshot({ path: join(output, `${name}.png`), animations: 'disabled' })
}
const queueCount = async (page, label, value) => {
  const link = queue(page, label)
  await link.getByText(String(value), { exact: true }).waitFor()
  const detail = { Slack: 'Unreviewed captures', Proposals: 'Pending findings', 'Work review': 'Reports, failures, approvals' }[label]
  assert.equal(await link.getAttribute('aria-label'), `${label}: ${value}. ${detail}`)
  if (typeof value === 'string') assert.equal(await link.getByText('0', { exact: true }).count(), 0, `${label} must not report an unavailable source as zero`)
}
const activate = async (page, target, keyboard, key = 'Enter') => {
  if (keyboard) { await target.focus(); await page.keyboard.press(key) }
  else await target.click()
}
async function expectSource(page, source, state) {
  const labels = { slack: 'From Slack', loops: 'From loops', signals: 'Public signals', all: 'All' }
  const apiSource = { slack: 'slack', loops: 'local_proposals', signals: 'public_signals', all: '' }[source]
  const pending = state.ideas.filter(item => belongsTo(item, apiSource) && item.status === 'unreviewed'
    && (source !== 'loops' || (!item.assignment_id && !item.review_outcome)))
  await heading(page, 'Inbox').waitFor()
  await captures(page).getByRole('tablist', { name: 'Capture origin' }).getByRole('tab', { name: labels[source], exact: true, selected: true }).waitFor()
  await captures(page).getByRole('tablist', { name: 'Capture filters' }).getByRole('tab', { name: `${source === 'loops' ? 'Pending' : 'Unreviewed'} ${pending.length}`, exact: true, selected: true }).waitFor()
  assert.equal(await captures(page).locator('li[data-idea-status]').count(), pending.length)
  for (const item of pending) await row(page, cleanText(item.text)).waitFor()
  if (source === 'loops') assert.equal(await row(page, 'Accepted proposal must not enter the pending queue.').count(), 0)
  assert.equal(await captures(page).locator('li:not([data-idea-status="unreviewed"])').count(), 0)
  assert.equal(await nav(page).getByRole('link', { name: 'Inbox', exact: true }).getAttribute('aria-current'), 'page')
}
async function expectCounts(page, slack = 3, proposals = 3) {
  await queueCount(page, 'Slack', slack)
  await queueCount(page, 'Proposals', proposals)
  await queueCount(page, 'Work review', 4)
}

try {
  for (const width of widths) {
    const { page, state, close } = await open('#/today', width)
    await heading(page, 'Today').waitFor()
    await expectCounts(page)
    assert.equal(await nav(page).getByRole('link', { name: 'Inbox', exact: true }).getAttribute('href'), '#/inbox')
    await screenshot(page, `today-queues-${width}`)

    await activate(page, queue(page, 'Work review'), width === 1440)
    await page.waitForFunction(() => location.hash === '#/history?attention=1')
    await heading(page, 'Activity history').waitFor()
    const attention = page.getByRole('region', { name: 'Needs attention', exact: true })
    await attention.getByRole('heading', { name: 'Needs attention (4)', exact: true }).waitFor()
    assert.equal(await attention.evaluate(element => document.activeElement === element), true)
    assert.equal(await attention.getByText('Synthetic work accepted', { exact: true }).count(), 0)
    assert.equal(await attention.getByText('Synthetic work dismissed', { exact: true }).count(), 0)
    await page.reload()
    await attention.getByRole('heading', { name: 'Needs attention (4)', exact: true }).waitFor()
    assert.equal(await attention.evaluate(element => document.activeElement === element), true)
    await screenshot(page, `work-drilldown-${width}`)
    assert.ok(state.calls.some(call => call.path === '/api/memory-inbox/review-health'))
    await page.goBack()
    await heading(page, 'Today').waitFor()

    await activate(page, queue(page, 'Proposals'), width !== 1440)
    await page.waitForFunction(() => location.hash === '#/inbox?source=loops')
    await expectSource(page, 'loops', state)
    await expectCounts(page)
    const proposalReads = state.calls.filter(call => call.path === '/api/memory-inbox' && call.source === 'local_proposals')
    assert.ok(proposalReads.length > 0)
    assert.ok(proposalReads.every(call => call.status === 'unreviewed' && call.outcome === 'open'), 'Proposals opens unassigned pending findings, never the historical review slice')
    await screenshot(page, `proposals-pending-${width}`)
    await page.goBack()
    await heading(page, 'Today').waitFor()
    await activate(page, queue(page, 'Slack'), width === 1440)
    await page.waitForFunction(() => location.hash === '#/inbox?source=slack')
    await expectSource(page, 'slack', state)
    await page.goBack()
    await heading(page, 'Today').waitFor()

    await activate(page, nav(page).getByRole('link', { name: 'Inbox', exact: true }), width === 320)
    await page.waitForFunction(() => location.hash === '#/inbox')
    await expectSource(page, 'slack', state)
    await row(page, cleanText(state.ideas[0].text)).getByRole('checkbox').waitFor()
    await screenshot(page, `inbox-default-${width}`)
    const blank = row(page, '(mention only, no text)')
    assert.equal(await blank.getByRole('combobox').count(), 0)
    assert.equal(await blank.getByRole('checkbox').count(), 0)
    assert.equal(await blank.getByRole('button', { name: 'Log to memory', exact: true }).isDisabled(), true)
    assert.equal(await blank.getByRole('button', { name: 'Queue as work', exact: true }).isDisabled(), true)
    assert.equal(await blank.getByRole('button', { name: 'Dismiss', exact: true }).isEnabled(), true)
    await blank.getByText('Legacy save receipt sent; no idea text captured', { exact: true }).waitFor()
    await blank.scrollIntoViewIfNeeded()
    await screenshot(page, `legacy-empty-${width}`)
    const normal = row(page, cleanText(state.ideas[0].text))
    await normal.scrollIntoViewIfNeeded()
    const textBounds = await normal.locator('p').first().boundingBox()
    assert.ok(textBounds.width > width / 2, 'capture text keeps a readable full-width line, not a squeezed control-side column')
    await screenshot(page, `capture-controls-${width}`)

    const origins = captures(page).getByRole('tablist', { name: 'Capture origin' })
    await activate(page, origins.getByRole('tab', { name: 'From loops', exact: true }), width === 320, 'Space')
    await expectSource(page, 'loops', state)
    await page.goBack()
    await expectSource(page, 'slack', state)
    await page.goForward()
    await expectSource(page, 'loops', state)
    await page.reload()
    await expectSource(page, 'loops', state)
    assert.equal(await page.evaluate(() => location.hash), '#/inbox?source=loops')

    for (const source of ['slack', 'loops', 'signals', 'all']) {
      await page.goto(`${base}#/inbox?source=${source}`)
      await expectSource(page, source, state)
      await page.reload()
      await expectSource(page, source, state)
      assert.equal(await page.evaluate(() => location.hash), `#/inbox?source=${source}`)
      await screenshot(page, `direct-${source}-${width}`)
    }
    for (const [source, label, dismissedText] of [
      ['slack', 'Slack', 'Previously dismissed Slack capture.'],
      ['loops', 'Proposals', 'Closed historical proposal must not enter the pending queue.'],
    ]) {
      await queue(page, label).click()
      await expectSource(page, source, state)
      const filters = captures(page).getByRole('tablist', { name: 'Capture filters' })
      await filters.getByRole('tab', { name: /^Dismissed/ }).click()
      await row(page, dismissedText).waitFor()
      const search = captures(page).getByRole('textbox', { name: 'Search captures' })
      await search.fill('synthetic-query-with-no-matches')
      await page.waitForFunction(() => document.querySelectorAll('li[data-idea-status]').length === 0)
      const routeBefore = await page.evaluate(() => location.hash)
      await activate(page, queue(page, label), width === 320)
      await expectSource(page, source, state)
      assert.equal(await search.inputValue(), '', 'clicking the active queue clears a stale search')
      assert.equal(await page.evaluate(() => location.hash), routeBefore, 'same-route queue click resets the counted slice')
      await screenshot(page, `reset-active-${source}-${width}`)
    }
    const filters = captures(page).getByRole('tablist', { name: 'Capture filters' })
    await filters.getByRole('tab', { name: /^Logged/ }).click()
    await row(page, 'Previously logged proposal must not enter the pending queue.').waitFor()
    await filters.getByRole('tab', { name: 'All', exact: true }).click()
    await row(page, 'Accepted proposal must not enter the pending queue.').waitFor()
    assert.ok(state.calls.some(call => call.path === '/api/memory-inbox' && call.source === 'local_proposals' && call.status === 'promoted' && !call.outcome))
    assert.ok(state.calls.some(call => call.path === '/api/memory-inbox' && call.source === 'local_proposals' && !call.status && !call.outcome))
    await queue(page, 'Proposals').click()
    await expectSource(page, 'loops', state)
    assert.deepEqual(state.mutations, [], 'navigation cannot mutate captures')
    await close()
    console.log(`${width}px: source queues, keyboard/click drilldowns, back/forward/reload, direct routes, blank controls passed`)
  }

  for (const width of widths) {
    const { page, state, release, close } = await open('#/today', width, { held: queueSources, clock: true })
    await heading(page, 'Today').waitFor()
    for (const label of ['Slack', 'Proposals', 'Work review']) await queueCount(page, label, 'Loading')
    await screenshot(page, `queues-pending-${width}`)
    state.failures = new Set(queueSources)
    release()
    for (const label of ['Slack', 'Proposals', 'Work review']) await queueCount(page, label, 'Unavailable')
    await screenshot(page, `queues-unavailable-${width}`)
    state.failures.clear()
    await page.reload()
    await expectCounts(page)
    // A failed refresh must also override previously cached nonzero counts.
    state.failures = new Set(queueSources)
    await nav(page).getByRole('link', { name: 'Inbox', exact: true }).click()
    await heading(page, 'Inbox').waitFor()
    await page.clock.fastForward(16_000)
    for (const label of ['Slack', 'Proposals', 'Work review']) await queueCount(page, label, 'Unavailable')
    await captures(page).getByRole('alert').filter({ hasText: 'Memory inbox unavailable.' }).waitFor()
    await screenshot(page, `cached-counts-unavailable-${width}`)
    await close()
  }

  // Each dependency fails independently: other queues must keep their own counts.
  for (const source of queueSources) {
    const { page, close } = await open('#/today', 1440, { failed: [source] })
    await queueCount(page, 'Slack', source === 'slack' ? 'Unavailable' : 3)
    await queueCount(page, 'Proposals', source === 'local_proposals' ? 'Unavailable' : 3)
    await queueCount(page, 'Work review', source.startsWith('/api/') ? 'Unavailable' : 4)
    await screenshot(page, `unavailable-${source.split('/').at(-1)}`)
    await close()
  }

  for (const width of widths) {
    const { page, state, close } = await open('#/inbox', width)
    await expectSource(page, 'slack', state)
    await expectCounts(page)
    const mutate = async (target, id, action, source = 'slack') => {
      const countRefresh = page.waitForResponse(response => {
        const url = new URL(response.url())
        return url.pathname === '/api/memory-inbox' && url.searchParams.get('source') === source
          && state.mutations.some(item => item.id === id && item.action === action)
      })
      await activate(page, target, width === 320, 'Space')
      await countRefresh
    }
    await mutate(row(page, '(mention only, no text)').getByRole('button', { name: 'Dismiss', exact: true }), 'slack-empty', 'dismiss')
    await page.getByRole('status').filter({ hasText: 'Capture dismissed.' }).waitFor()
    await expectCounts(page, 2)
    await expectSource(page, 'slack', state)
    assert.equal(await row(page, '(mention only, no text)').count(), 0)

    await mutate(row(page, cleanText(state.ideas[0].text)).getByRole('button', { name: 'Log to memory', exact: true }), 'slack-alpha', 'promote')
    await page.getByRole('status').filter({ hasText: 'Logged to AIIA memory' }).waitFor()
    await expectCounts(page, 1)
    await expectSource(page, 'slack', state)
    const pending = row(page, cleanText(state.ideas[1].text))
    await pending.getByRole('combobox', { name: 'Agent for capture slack-br', exact: true }).selectOption(agents[0].id)
    const assignmentRefresh = page.waitForResponse(response => new URL(response.url()).pathname === '/api/assignments' && state.mutations.some(item => item.action === 'assign'))
    await mutate(pending.getByRole('button', { name: 'Queue as work', exact: true }), 'slack-bravo', 'assign')
    await assignmentRefresh
    await page.getByRole('status').filter({ hasText: 'Queued for Synthetic reviewer' }).waitFor()
    await pending.getByText('queued as work assignme', { exact: true }).waitFor()
    assert.equal(await pending.getByRole('button', { name: 'Queue as work', exact: true }).count(), 0)
    // Queueing work does not log/dismiss the capture, and queued work is not review-ready.
    await expectCounts(page, 1)
    assert.equal(state.ideas.find(item => item.id === 'slack-bravo').status, 'unreviewed')
    assert.equal(state.assignments.at(-1).status, 'queued')
    assert.deepEqual(state.mutations.map(({ id, action }) => [id, action]), [['slack-empty', 'dismiss'], ['slack-alpha', 'promote'], ['slack-bravo', 'assign']])
    await screenshot(page, `mutations-invalidated-${width}`)
    await nav(page).getByRole('link', { name: 'Today', exact: true }).click()
    await expectCounts(page, 1)
    await page.reload()
    await expectCounts(page, 1)
    await queue(page, 'Proposals').click()
    await expectSource(page, 'loops', state)
    const proposal = row(page, 'Pending backlog proposal.')
    await proposal.getByRole('combobox', { name: 'Agent for capture loop-bac', exact: true }).selectOption(agents[0].id)
    await proposal.getByRole('textbox', { name: 'Review rationale for capture loop-bac', exact: true }).fill('Synthetic acceptance rationale.')
    await mutate(proposal.getByRole('button', { name: 'Accept as work', exact: true }), 'loop-backlog', 'assign', 'local_proposals')
    await expectCounts(page, 1, 2)
    await expectSource(page, 'loops', state)
    assert.equal(await proposal.count(), 0, 'accepted work leaves the pending proposal queue')
    const accepted = state.ideas.find(item => item.id === 'loop-backlog')
    assert.equal(accepted.status, 'unreviewed')
    assert.equal(accepted.review_outcome, 'needs_work')
    assert.equal(accepted.assignment_id, 'assignment-loop-backlog')
    assert.equal(state.mutations.at(-1).body.review_note, 'Synthetic acceptance rationale.')
    await screenshot(page, `proposal-accepted-${width}`)
    await page.reload()
    await expectSource(page, 'loops', state)
    await expectCounts(page, 1, 2)
    await nav(page).getByRole('link', { name: 'Today', exact: true }).click()
    await expectCounts(page, 1, 2)
    await close()
    console.log(`${width}px: synthetic dismiss/promote/queue invalidation and Today counts passed`)
  }
  console.log('studio inbox: all checks passed')
} catch (error) {
  for (const [index, context] of browser.contexts().entries()) {
    const page = context.pages()[0]
    if (page && !page.isClosed()) await page.screenshot({ path: join(output, `failure-${index}.png`), animations: 'disabled' }).catch(() => {})
  }
  throw error
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
