import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || '/tmp/aiia-routing-advisor'
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let calls = 0
    let unavailable = false
    let noMatch = false
    let ready = true
    const agents = [{ id: 'ci', name: 'CI reviewer', skills: ['Analysis'], mission: 'PRIVATE', persona: 'PRIVATE', status: 'idle', runs: [], tools: [], loop_enabled: false }]
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('http://studio.test/', route => readFile(join(dist, 'index.html')).then(body => route.fulfill({ contentType: 'text/html', body })))
    await page.route('http://studio.test/assets/**', async route => {
      const path = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: path.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, path)) })
    })
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') {
        assert.equal(route.request().method(), 'GET', 'Advice must not create assignments')
        body = { assignments: [] }
      } else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/git-workspaces') body = { workspaces: [] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (path === '/api/integrations/typesafe/status') body = { ready, enabled: ready, configured: ready }
      else if (path === '/api/assignments/suggest-agent') {
        calls++
        assert.deepEqual(route.request().postDataJSON(), { brief: 'Review CI', candidate_agent_ids: ['ci'], allow_external: true })
        status = unavailable ? 503 : 200
        body = unavailable ? { detail: 'typesafe_unavailable' } : { status: noMatch ? 'no_match' : 'suggested', agent_id: noMatch ? null : 'ci', confidence: 0.8, model: 'jev-test', usage: { input_tokens: 100, output_tokens: 10 }, requires_confirmation: true }
      } else if (path === '/api/tasks') body = []
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      else if (path === '/api/tokens/today') body = { total_tokens: 0, total_requests: 0, total_cost: 0, by_provider: {}, by_purpose: {} }
      else if (path === '/api/tokens/recent') body = { days: [] }
      else if (path === '/api/studio/activity') body = { days: [], agent_days: [], runs: [], total: 0, usage_by_agent: [] }
      else if (path === '/api/memory-inbox') body = { ideas: [], counts: { unreviewed: 0, promoted: 0, dismissed: 0 } }
      else if (path === '/api/memory-inbox/review-health') body = { window_days: 14, since: '2026-09-24', filed: 0, reviewed: 0, totals: { open: 0, needs_work: 0, already_fixed: 0, declined: 0, external_failure: 0, unclassified: 0 }, by_source: [], by_project: [] }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto('http://studio.test/')
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    const advisor = page.getByRole('region', { name: 'Jev routing advisor' })
    const suggest = advisor.getByRole('button', { name: 'Suggest specialist' })
    await advisor.getByRole('textbox', { name: 'Routing brief' }).fill('Review CI')
    assert.equal(await suggest.isDisabled(), true)
    assert.equal(calls, 0)
    await advisor.getByRole('checkbox').check()
    await suggest.click()
    await advisor.getByText('Suggested: CI reviewer', { exact: true }).waitFor()
    const assigned = page.getByRole('combobox').filter({ has: page.locator('option[value="ci"]') })
    assert.equal(await assigned.inputValue(), '')
    await advisor.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `advice-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await advisor.getByRole('button', { name: 'Use suggested agent' }).click()
    assert.equal(await assigned.inputValue(), 'ci')
    await advisor.getByRole('textbox').fill('Changed brief')
    assert.equal(await advisor.getByRole('button', { name: 'Use suggested agent' }).count(), 0)
    await advisor.getByRole('textbox').fill('Review CI')
    noMatch = true
    await suggest.click()
    await advisor.getByText('No suitable specialist identified.', { exact: true }).waitFor()
    assert.equal(await assigned.inputValue(), 'ci')
    unavailable = true
    await suggest.click()
    await advisor.getByRole('alert').waitFor()
    assert.equal(await assigned.inputValue(), 'ci')
    ready = false
    await page.reload()
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await advisor.getByText('Not connected. Manual assignment is available.', { exact: true }).waitFor()
    assert.deepEqual(errors, [])
    console.log(`${width}px: consent, explicit selection, stale advice, no-match, outage, disconnected passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
