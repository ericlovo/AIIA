// Studio routing: every view is an address. Deep links open the right record,
// back and forward walk the views, selection inside a view keeps the address
// current without adding history, and unknown addresses land on Today.
//
// Every API response and the Studio WebSocket are intercepted with synthetic
// data; nothing is written to a real Brain or Command Center.
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const studioDist = process.env.STUDIO_DIST_DIR
const base = studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/'
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = new Date().toISOString().slice(0, 10)

const agent = (id, name) => ({ id, name, mission: `${name} mission.`, persona: 'Direct.', status: 'idle', skills: [], tools: ['Local memory'], repo_id: '', runs: [], temperature: 0.35, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60, loop_task: '', loop_max_runs_per_day: 4, last_result: '', last_error: '' })
const agents = [agent('a0', 'Signal Scout'), agent('a1', 'Repo Warden')]
const work = (id, title, agentId) => ({ id, agent_id: agentId, title, objective: `Objective for ${title}.`, status: 'queued', result: '', error: '', priority: 'normal', context: '', success_criteria: '', source_handoff_id: '', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` })
const assignments = [work('asg-1', 'Scan the repository', 'a0'), work('asg-2', 'Draft the partner brief', 'a1')]
const tasks = [{ task_id: 'nightly-sync', name: 'Nightly sync', description: 'Syncs memory overnight.', interval_seconds: 86400, last_run: null, next_run: null, last_status: 'success', run_count: 3, fail_count: 0, enabled: true }]

async function open(path = '', { agentsDelayMs = 0, width = 1440, workItems = assignments } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 } })
  const page = await context.newPage()
  page.setDefaultTimeout(12000)
  const pageErrors = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
  if (studioDist) {
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(studioDist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(studioDist, asset)) })
    })
  }
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/agents' && agentsDelayMs) await new Promise(resolve => setTimeout(resolve, agentsDelayMs))
    const bodies = {
      '/api/agents': { agents },
      '/api/public-signals/leads': { leads: [], total: 0, offset: 0, limit: 25 },
      '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
      '/api/agents/models': { default: 'synthetic-model:1b', models: [] },
      '/api/assignments': { assignments: workItems },
      '/api/handoffs': { handoffs: [] },
      '/api/git-workspaces': { workspaces: [] },
      '/api/git-writes': { writes: [] },
      '/api/studio/activity': { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] },
      '/api/memory-inbox': { ideas: [], total: 0, offset: 0, counts: { unreviewed: 0, promoted: 0, dismissed: 0 } },
      '/api/memory-inbox/review-health': { window_days: 14, since: `${date}T00:00:00+00:00`, filed: 0, reviewed: 0, totals: { open: 0, needs_work: 0, already_fixed: 0, declined: 0, external_failure: 0, unclassified: 0 }, by_source: [], by_project: [] },
      '/api/agent-world/layout': { layout: { version: 1, revision: 0, positions: {}, updated_at: null } },
      '/api/tasks': tasks,
      '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
      '/api/monitor': { services: {} },
      '/api/voice/status': { status: 'not_configured', configured: false, reason: 'missing_xai_api_key', tools: [] },
    }
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify(bodies[path] ?? {}) })
  })
  await page.goto(`${base}${path}`)
  return { context, page, pageErrors }
}

const nav = page => page.getByRole('navigation', { name: 'Studio' })
const hash = page => page.evaluate(() => window.location.hash)
const heading = (page, name) => page.getByRole('heading', { level: 1, name, exact: true })

try {
  for (const width of [1440, 390]) {
    const workItems = Array.from({ length: 3 }, (_, i) => ({ ...work(`failure-${i}`, `Investigate failure ${i}`, 'a0'), status: 'failed', error: `failure evidence ${i}`, review_status: 'unreviewed' }))
    workItems.push({ ...work('dismissed', 'Already dismissed', 'a0'), status: 'failed', dismissed_at: `${date}T13:00:00Z` })
    const { context, page, pageErrors } = await open('#/overview', { width, workItems })
    const link = page.getByRole('link', { name: 'Needs attention: 3. Investigate', exact: true })
    await link.waitFor()
    const output = process.env.SCREENSHOT_DIR || '/tmp/aiia-overview-attention'
    await mkdir(output, { recursive: true })
    await link.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `attention-link-${width}.png`) })
    if (width === 1440) { await link.focus(); await page.keyboard.press('Enter') }
    else await link.click()
    assert.equal(await hash(page), '#/today?attention=1')
    const queue = page.getByRole('region', { name: 'Needs attention', exact: true })
    await queue.getByRole('heading', { name: 'Needs attention (3)', exact: true }).waitFor()
    assert.equal(await queue.evaluate(element => document.activeElement === element), true)
    assert.equal(await queue.getByText('Already dismissed', { exact: true }).count(), 0)
    await page.screenshot({ path: join(output, `attention-queue-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await queue.getByRole('button', { name: /Investigate failure 0/ }).click()
    assert.equal(await hash(page), '#/assignments/failure-0')
    const evidence = page.getByRole('region', { name: 'Failure evidence' })
    await evidence.getByText('failure evidence 0', { exact: true }).waitFor()
    await evidence.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `work-evidence-${width}.png`), animations: 'disabled' })
    assert.equal(await page.getByRole('button', { name: 'Retry assignment', exact: true }).count(), 1)
    await page.goBack()
    await queue.waitFor()
    await page.reload()
    await queue.waitFor()
    assert.equal(await queue.evaluate(element => document.activeElement === element), true)
    assert.deepEqual(pageErrors, [])
    await context.close()
  }
  for (const width of [320, 390, 653]) {
    const { context, page, pageErrors } = await open('#/today', { width })
    await heading(page, 'Today').waitFor()
    assert.deepEqual(await nav(page).getByRole('link').allTextContents(), ['Today', 'Agents', 'Work'])
    const tools = nav(page).getByText('Studio', { exact: true })
    await tools.click()
    await nav(page).getByRole('link', { name: 'Signals', exact: true }).waitFor()
    await heading(page, 'Today').click({ position: { x: 5, y: 5 } })
    assert.equal(await nav(page).locator('details').getAttribute('open'), null)
    await tools.click()
    await page.keyboard.press('Escape')
    assert.equal(await tools.evaluate(element => element === document.activeElement), true)
    assert.equal(await nav(page).getByRole('link', { name: 'Signals', exact: true }).count(), 0)
    await tools.click()
    const output = process.env.SCREENSHOT_DIR || '/tmp/aiia-overview-attention'
    await page.screenshot({ path: join(output, `studio-menu-${width}.png`), animations: 'disabled' })
    await nav(page).getByRole('link', { name: 'Map', exact: true }).click()
    await heading(page, 'Agent control map').waitFor()
    assert.equal(await nav(page).locator('details').getAttribute('open'), null)
    await nav(page).getByRole('link', { name: 'Work', exact: true }).click()
    await heading(page, 'Assignment queue').waitFor()
    assert.equal(await nav(page).getByRole('link', { name: 'Work', exact: true }).getAttribute('aria-current'), 'page')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(pageErrors, [])
    await context.close()
  }
  // 1. Every nav item is a real link to its own address, and one is current.
  {
    const { context, page, pageErrors } = await open()
    await heading(page, 'Today').waitFor()
    assert.equal(await hash(page), '#/today', 'an empty address lands on Today')
    const links = await nav(page).getByRole('link').evaluateAll(items => items.map(item => [item.textContent, item.getAttribute('href'), item.getAttribute('aria-current')]))
    assert.deepEqual(links, [
      ['Today', '#/today', 'page'], ['Overview', '#/overview', null], ['Agents', '#/agents', null],
      ['Signals', '#/signals', null],
      ['Assignments', '#/assignments', null], ['Handoffs', '#/handoffs', null], ['Memory', '#/memory', null], ['Map', '#/map', null],
    ])
    assert.equal(await page.getByRole('tab', { name: 'Today', exact: true }).count(), 0, 'navigation is not an ARIA tab list')
    assert.deepEqual(pageErrors, [])
    await context.close()
  }

  // 2. Back and forward walk the views in the order they were opened.
  {
    const { context, page } = await open('#/today')
    await heading(page, 'Today').waitFor()
    await nav(page).getByRole('link', { name: 'Map', exact: true }).click()
    await heading(page, 'Agent control map').waitFor()
    await nav(page).getByRole('link', { name: 'Memory', exact: true }).click()
    await heading(page, 'Memory log').waitFor()
    assert.equal(await hash(page), '#/memory')
    await page.goBack()
    await heading(page, 'Agent control map').waitFor()
    assert.equal(await nav(page).getByRole('link', { name: 'Map', exact: true }).getAttribute('aria-current'), 'page')
    await page.goBack()
    await heading(page, 'Today').waitFor()
    await page.goForward()
    await heading(page, 'Agent control map').waitFor()
    await context.close()
  }

  // 3. A deep link to an assignment opens it; selecting another rewrites the
  //    address without a history entry, so back leaves the view instead.
  {
    const { context, page } = await open('#/today')
    await heading(page, 'Today').waitFor()
    await page.evaluate(() => { window.location.hash = '#/assignments/asg-2' })
    await heading(page, 'Assignment queue').waitFor()
    await page.getByText('Objective for Draft the partner brief.').first().waitFor()
    await page.getByText('Scan the repository', { exact: true }).first().click()
    await page.waitForFunction(() => window.location.hash === '#/assignments/asg-1')
    await page.goBack()
    await heading(page, 'Today').waitFor()
    await context.close()
  }

  // 4. A reloaded deep link survives, even when agents load after the page.
  {
    const { context, page, pageErrors } = await open('#/agents/a1', { agentsDelayMs: 800 })
    await heading(page, 'Agents').waitFor()
    await page.getByRole('region', { name: 'Repo Warden activity' }).waitFor()
    await page.locator('aside').getByText('Repo Warden', { exact: true }).waitFor()
    assert.equal(await hash(page), '#/agents/a1', 'a slow agent list must not erase the deep link')
    // Choosing another agent keeps the address in step.
    await page.getByRole('button', { name: /Signal Scout/ }).first().click()
    await page.waitForFunction(() => window.location.hash === '#/agents/a0')
    assert.deepEqual(pageErrors, [])
    await context.close()
  }

  // 5. Unknown addresses land on Today and leave no dead entry behind.
  {
    const { context, page } = await open('#/nowhere/at/all')
    await heading(page, 'Today').waitFor()
    await page.waitForFunction(() => window.location.hash === '#/today')
    // Following a bad link from the Map, back returns to the Map, not the bad link.
    await nav(page).getByRole('link', { name: 'Map', exact: true }).click()
    await heading(page, 'Agent control map').waitFor()
    await page.evaluate(() => { window.location.hash = '#/bogus' })
    await page.waitForFunction(() => window.location.hash === '#/today')
    await heading(page, 'Today').waitFor()
    await page.goBack()
    await heading(page, 'Agent control map').waitFor()
    assert.equal(await hash(page), '#/map', 'the redirect replaces the bad entry, it does not add one')
    await context.close()
  }

  // 6. Cross-view doorways are addresses: the review inbox, and a Pulse loop.
  {
    const { context, page } = await open('#/today')
    await heading(page, 'Today').waitFor()
    await page.getByRole('button', { name: 'Open the review inbox' }).click()
    await heading(page, 'Memory log').waitFor()
    assert.equal(await hash(page), '#/memory?review=all')
    await page.getByRole('button', { name: /^Open Nightly sync/ }).click()
    await heading(page, 'Today').waitFor()
    assert.equal(await hash(page), '#/today?task=nightly-sync')
    await page.getByRole('heading', { level: 2, name: 'Nightly sync' }).waitFor()
    await context.close()
  }

  console.log('studio routing: all checks passed')
} finally {
  await browser.close()
}
