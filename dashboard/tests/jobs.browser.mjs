// Exercise the built Studio. Every API and WebSocket is mocked; no live records.
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || '/tmp/aiia-jobs-ui'
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
await mkdir(output, { recursive: true })
const base = 'http://studio.test'
let browser
try {
  browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } })
    const page = await context.newPage()
    const errors = []
    const unexpected = []
    const writes = []
    const reads = []
    page.setDefaultTimeout(15_000)
    page.on('pageerror', error => errors.push(error.message))
    const stamp = () => new Date().toISOString()
    const makeAgent = (id, extra = {}) => ({
      id, name: id, mission: 'Review evidence only.', persona: 'Evidence first.', skills: ['Analysis'], tools: ['Repository read'], repo_id: 'qa-project',
      temperature: 0.2, max_tokens: 1600, loop_enabled: true, loop_interval_minutes: 240, loop_max_runs_per_day: 3,
      loop_task: 'Review supplied repository evidence.', loop_runs_today: 0, loop_day: stamp().slice(0, 10),
      loop_checked_at: stamp(), loop_skip_reason: '', status: 'idle', last_run_at: stamp(), last_result: 'Prior evidence', last_error: '', runs: [], created_at: stamp(), updated_at: stamp(), ...extra,
    })
    const agents = [
      makeAgent('QA paused watch', { loop_enabled: false }),
      makeAgent('QA review-blocked watch', { loop_skip_reason: 'awaiting_review' }),
      makeAgent('QA running watch', { status: 'running' }),
      makeAgent('QA failed source', { loop_skip_reason: 'check_incomplete' }),
      makeAgent('Manual only', { loop_task: '' }),
    ]
    const assignments = []
    let failToggle = false
    let failRun = true
    let failResources = false
    let failTasks = false
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/*', async route => {
      const request = route.request()
      const url = new URL(request.url())
      if (url.origin !== base) { unexpected.push(request.url()); await route.abort(); return }
      if (url.pathname === '/') { await route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) }); return }
      if (url.pathname.startsWith('/assets/')) {
        await route.fulfill({ contentType: url.pathname.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, url.pathname.slice(1))) })
        return
      }
      if (!url.pathname.startsWith('/api/')) { await route.fulfill({ status: 204 }); return }
      const path = url.pathname
      const method = request.method()
      if (method !== 'GET') writes.push({ path, method, data: request.postDataJSON() })
      else reads.push(path)
      let status = 200
      let body
      if (path === '/api/agents' && method === 'GET') body = { agents }
      else if (path === '/api/agents/resources') {
        status = failResources ? 503 : 200
        body = failResources ? { detail: 'Repositories offline' } : { repos: [{ id: 'qa-project', name: 'QA project', branch: 'qa-fixtures-only', dirty: false, github_repo: '', path: '/qa/not-a-live-repo' }], github: { status: 'disconnected', mode: 'read_only', provider: 'github', account: '', reason: 'QA' } }
      } else if (path === '/api/tasks') {
        status = failTasks ? 503 : 200
        body = failTasks ? { detail: 'System task source offline' } : [{ task_id: 'qa-task', name: 'QA system collector', description: 'Built-in fixture task, not project CI health.', interval_seconds: 3600, last_run: stamp(), next_run: stamp(), last_status: 'completed', run_count: 1, fail_count: 0, enabled: true }]
      } else if (path === '/api/assignments' && method === 'GET') body = { assignments }
      else if (path === '/api/git-workspaces') body = { workspaces: [] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (path === '/api/memory-inbox' && method === 'GET') body = { ideas: [], total: 0, offset: 0, counts: { unreviewed: 0, promoted: 0, dismissed: 0 } }
      else if (path === '/api/agents/models') body = { default: 'qa-model', models: [] }
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { status: 'not_configured', configured: false, reason: 'missing_xai_api_key', tools: [] }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path.startsWith('/api/assignments/') && path.endsWith('/history')) body = { runs: [], total: 0, offset: 0, limit: 25, attempt_id: '', current_output_saved: true, completed_run_id: '' }
      else if (path === '/api/integrations/typesafe/status') body = { ready: false, enabled: false, configured: false }
      else if (path === '/api/signal-jobs') body = { ready: false, retrieval_enabled: false, screening_enabled: false, configured: false, jobs: [] }
      else if (path === '/api/public-signals/leads') body = { leads: [], total: 0, offset: 0, limit: 25 }
      else if (path === '/api/agents' && method === 'POST') {
        const agent = makeAgent('qa-new', { ...request.postDataJSON(), loop_checked_at: null, last_run_at: null, last_result: '' })
        agents.push(agent)
        body = { agent }
      } else if (path.endsWith('/loop') && method === 'POST') {
        const agent = agents.find(item => `/api/agents/${item.id}/loop` === decodeURI(path))
        assert.ok(agent)
        if (failToggle) { status = 503; body = { detail: 'Schedule unavailable' } }
        else { Object.assign(agent, { loop_enabled: request.postDataJSON().enabled, updated_at: stamp() }); body = { agent } }
      } else if (path === '/api/assignments' && method === 'POST') {
        const assignment = { ...request.postDataJSON(), id: `qa-test-${assignments.length + 1}`, status: 'queued', result: '', error: '', review_status: 'unreviewed', created_at: stamp(), updated_at: stamp(), started_at: null, completed_at: null, source_handoff_id: '' }
        assignments.push(assignment)
        body = { assignment }
      } else if (path.startsWith('/api/assignments/') && path.endsWith('/run')) {
        const assignment = assignments.find(item => `/api/assignments/${item.id}/run` === path)
        assert.ok(assignment)
        if (failRun) {
          failRun = false
          Object.assign(assignment, { status: 'failed', error: 'Mini unavailable', updated_at: stamp() })
          status = 503; body = { detail: 'Mini unavailable' }
        } else {
          Object.assign(assignment, { status: 'completed', error: '', result: 'Evidence\nQA snapshot at revision abc123.\nFindings\nCI and deployment are unknown.\nNext action\nReview the changed paths.', updated_at: stamp(), completed_at: stamp() })
          const agent = agents.find(item => item.id === assignment.agent_id)
          Object.assign(agent, { last_run_at: stamp(), last_result: assignment.result, updated_at: stamp() })
          body = { assignment, agent, model: 'qa-model', latency_ms: 10 }
        }
      } else { unexpected.push(`${method} ${path}`); status = 500; body = { detail: 'Unmocked request blocked' } }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })

    await page.goto(`${base}/#/jobs`)
    await page.getByRole('heading', { name: 'QA paused watch', exact: true }).waitFor()
    assert.equal(await page.getByRole('heading', { name: 'Manual only' }).count(), 0)
    assert.equal(writes.length, 0)
    assert.equal(await page.locator('main').evaluate(element => getComputedStyle(element).overflowY), 'auto')
    await page.getByText('System tasks', { exact: false }).first().click()
    await page.getByText('QA system collector', { exact: true }).waitFor()
    await page.screenshot({ path: join(output, `jobs-list-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.equal(await page.getByRole('button', { name: /run.*system/i }).count(), 0)

    const paused = page.getByRole('listitem').filter({ has: page.getByRole('heading', { name: 'QA paused watch', exact: true }) })
    failToggle = true
    await paused.getByRole('button', { name: 'Resume', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'Schedule unavailable' }).waitFor()
    assert.equal(agents[0].loop_enabled, false)
    failToggle = false
    await paused.getByRole('button', { name: 'Resume', exact: true }).click()
    await paused.getByRole('button', { name: 'Pause', exact: true }).waitFor()
    await paused.getByRole('button', { name: 'Pause', exact: true }).click()
    await paused.getByRole('button', { name: 'Resume', exact: true }).waitFor()

    await page.getByRole('button', { name: 'New job', exact: true }).click()
    await page.getByLabel('Project / repository').selectOption('qa-project')
    await page.getByLabel('Job name', { exact: true }).fill('QA new evidence job')
    await page.screenshot({ path: join(output, `jobs-form-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.getByRole('button', { name: 'Create paused job' }).click()
    await page.getByRole('heading', { name: 'QA new evidence job', exact: true }).waitFor()
    assert.equal(agents.at(-1).loop_enabled, false)
    assert.equal(assignments.length, 0)
    assert.equal(await page.getByRole('button', { name: 'Enable job' }).isDisabled(), true)

    await page.getByRole('button', { name: 'Test once on Mini' }).click()
    await page.getByRole('alert').filter({ hasText: 'Test request failed' }).waitFor()
    assert.equal(assignments.length, 1)
    assert.equal(await page.getByRole('button', { name: 'Enable job' }).isDisabled(), true)
    await page.getByRole('button', { name: 'Retry saved test' }).click()
    await page.getByText('QA snapshot at revision abc123.', { exact: false }).waitFor()
    assert.equal(assignments.length, 1)
    assert.equal(await page.getByRole('button', { name: 'Enable job' }).isDisabled(), true)
    await page.screenshot({ path: join(output, `jobs-result-${width}.png`), fullPage: true })

    await page.reload()
    const created = page.getByRole('listitem').filter({ has: page.getByRole('heading', { name: 'QA new evidence job', exact: true }) })
    await created.waitFor()
    assert.equal(await created.getByRole('button', { name: 'Resume', exact: true }).count(), 0)
    await created.getByRole('button', { name: 'Test / result', exact: true }).click()
    await page.getByText('QA snapshot at revision abc123.', { exact: false }).waitFor()
    const writesBeforeInspection = writes.length
    await page.getByLabel('I reviewed the test evidence.').check()
    assert.equal(writes.length, writesBeforeInspection)
    await page.getByRole('button', { name: 'Enable job', exact: true }).click()
    await page.getByRole('button', { name: 'Pause job', exact: true }).waitFor()
    assert.equal(agents.at(-1).loop_enabled, true)
    await page.getByRole('button', { name: 'Pause job', exact: true }).click()
    await page.getByRole('button', { name: 'Enable job', exact: true }).waitFor()
    assert.equal(agents.at(-1).loop_enabled, false)
    await page.getByRole('button', { name: 'Open saved test' }).click()
    assert.match(page.url(), /#\/(?:assignments|work)\/qa-test-1/)
    await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Jobs', exact: true }).click()
    await page.getByRole('button', { name: 'Open public signals' }).click()
    assert.match(page.url(), /#\/signals/)

    assignments[0].review_status = 'rejected'
    await page.goto(`${base}/#/jobs`)
    await page.reload()
    await created.getByRole('button', { name: 'Test / result', exact: true }).click()
    await page.getByText('This test was rejected or closed.', { exact: false }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Enable job' }).isDisabled(), true)
    assert.equal(await page.getByRole('checkbox').count(), 0)
    assignments[0].review_status = 'unreviewed'
    const originalTask = agents.at(-1).loop_task
    agents.at(-1).loop_task = 'Changed configuration with no matching test'
    agents.at(-1).updated_at = stamp()
    await page.reload()
    await created.getByRole('button', { name: 'Test / result', exact: true }).click()
    await page.getByText('No saved test for this configuration.', { exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Enable job' }).isDisabled(), true)
    agents.at(-1).loop_task = originalTask
    await page.getByRole('button', { name: 'Close job' }).click()

    failResources = true
    failTasks = true
    await page.getByRole('button', { name: 'Refresh jobs' }).click()
    await page.getByText('System tasks', { exact: false }).first().click()
    await page.getByRole('alert').filter({ hasText: 'System task source offline' }).waitFor()
    await page.getByRole('button', { name: 'New job', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'Repositories offline' }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Create paused job' }).isDisabled(), true)
    assert.deepEqual(errors, [])
    assert.deepEqual(unexpected, [])
    assert.equal(writes.filter(item => item.path === '/api/agents').length, 1)
    assert.equal(writes.filter(item => item.path === '/api/assignments').length, 1)
    assert.equal(writes.filter(item => item.path.endsWith('/run')).length, 2)
    assert.equal(writes.filter(item => item.path.includes('/tasks')).length, 0)

    failResources = false
    failTasks = false
    const workFixture = (id, extra) => ({ id, title: id, objective: 'QA evidence', agent_id: agents[0].id, status: 'completed', review_status: 'unreviewed', result: 'QA evidence only', error: '', priority: 'normal', context: '', success_criteria: '', source_handoff_id: '', created_at: stamp(), updated_at: stamp(), started_at: null, completed_at: stamp(), ...extra })
    assignments.push(workFixture('QA source evidence needs review', {}), workFixture('QA unavailable source', { status: 'failed', error: 'Source unavailable', result: '' }), workFixture('QA manual research', { status: 'queued', result: '' }), workFixture('QA accepted brief', { review_status: 'accepted' }))
    await page.goto(`${base}/#/today`)
    await page.reload()
    await page.getByRole('heading', { name: /^Needs attention \(\d+\)$/ }).waitFor()
    await page.getByText('Awaiting manual start', { exact: true }).waitFor()
    const voice = page.getByRole('button', { name: 'Voice', exact: true })
    assert.equal(await voice.getAttribute('aria-expanded'), 'false')
    assert.equal(await page.getByText('Voice Conductor', { exact: true }).count(), 0)
    assert.equal(reads.filter(path => path === '/api/voice/status').length, 0)
    await page.screenshot({ path: join(output, `today-shell-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await voice.click()
    await page.getByText('Voice Conductor', { exact: true }).waitFor()
    assert.equal(await voice.getAttribute('aria-expanded'), 'true')
    await voice.click()
    await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Jobs', exact: true }).click()
    await page.getByRole('heading', { name: 'Jobs', exact: true }).waitFor()
    await page.getByRole('button', { name: 'Open public signals' }).scrollIntoViewIfNeeded()
    assert.equal(await page.getByRole('main').evaluate(element => element.scrollTop > 0), true)
    await page.getByRole('main').evaluate(element => { element.scrollTop = 0 })
    await page.screenshot({ path: join(output, `jobs-shell-${width}.png`), fullPage: true })

    // A production-sized queue must not consume the selected item's mobile row.
    assignments.find(item => item.id === 'QA source evidence needs review').review_version = 'qa-review-v1'
    assignments.push(...Array.from({ length: 132 }, (_, index) => workFixture(`QA archived work ${index}`, { review_status: 'accepted' })))
    await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Today', exact: true }).click()
    await page.getByText('QA source evidence needs review', { exact: true }).click()
    const inspector = page.getByRole('complementary', { name: 'Selected work', exact: true })
    await inspector.getByText('QA evidence only', { exact: true }).waitFor()
    await page.getByText('QA archived work 131', { exact: true }).waitFor({ state: 'attached' })
    const box = await inspector.boundingBox()
    assert.ok(box && box.height > 100, `Selected work row collapsed at ${width}px`)
    assert.equal(await inspector.evaluate(element => {
      const rect = element.getBoundingClientRect()
      return rect.top >= 0 && rect.top < innerHeight && rect.right <= innerWidth
    }), true)
    await inspector.getByRole('button', { name: 'Accept output', exact: true }).scrollIntoViewIfNeeded()
    assert.equal(await inspector.getByRole('button', { name: 'Accept output', exact: true }).isEnabled(), true)
    assert.equal(await inspector.getByRole('button', { name: 'Accept output', exact: true }).evaluate(element => {
      const rect = element.getBoundingClientRect()
      return rect.top >= 0 && rect.bottom <= innerHeight
    }), true)
    await page.screenshot({ path: join(output, `work-large-queue-${width}.png`), fullPage: true })
    assert.deepEqual(errors, [])
    assert.deepEqual(unexpected, [])
    await context.close()
    console.log(`${width}px: Jobs lifecycle, rejected/stale gates, scrolling, outages, Today shell and opt-in voice passed`)
  }
} finally {
  await browser?.close()
}
