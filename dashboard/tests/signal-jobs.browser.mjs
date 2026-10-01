import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || '/tmp/aiia-signal-jobs'
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } })
    const page = await context.newPage()
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let ready = true
    let fail = false
    let calls = 0
    const jobs = [
      { id: 'market_news', name: 'Market Signal Scout', specialty: 'Family capital and private equity', enabled: false, interval_hours: 12, last_run: null },
      { id: 'lead_signals', name: 'Lead Signal Scout', specialty: 'Expansion, acquisitions and leadership changes', enabled: true, interval_hours: 12,
        last_run: { started: Date.now() / 1000, status: 'review_ready', result: { created: 3, usage: { input_tokens: 1200, output_tokens: 130 } } } },
    ]
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const path = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: path.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, path)) })
    })
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/signal-jobs') {
        body = { ready, retrieval_enabled: ready, screening_enabled: ready, configured: ready, jobs }
        if (fail) { status = 503; body = { detail: 'unavailable' } }
      } else if (path === '/api/signal-jobs/market_news/run') {
        calls++
        assert.equal(route.request().method(), 'POST')
        jobs[0].last_run = { started: Date.now() / 1000, status: 'no_signal', result: { created: 0, usage: { input_tokens: 100, output_tokens: 20 } } }
        body = { status: 'no_signal' }
      } else if (path === '/api/signal-jobs/lead_signals') {
        assert.equal(route.request().method(), 'PUT')
        assert.deepEqual(route.request().postDataJSON(), { enabled: false })
        jobs[1].enabled = false
        body = { ready, jobs }
      } else if (path === '/api/public-signals/leads') body = { leads: [], total: 0, offset: 0, limit: 25 }
      else if (path === '/api/agents') body = { agents: [] }
      else if (path === '/api/tasks') body = []
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      else if (path === '/api/memory-inbox') body = { ideas: [], counts: { unreviewed: 0, promoted: 0, dismissed: 0 } }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto('http://studio.test/#/signals')
    await page.getByText('Ready for review', { exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run Lead Signal Scout' }).isDisabled(), true)
    assert.equal(calls, 0)
    await page.screenshot({ path: join(output, `signals-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.getByRole('button', { name: 'Run Market Signal Scout' }).click()
    await page.getByText('No qualifying signals', { exact: true }).waitFor()
    assert.equal(calls, 1)
    assert.equal(await page.getByRole('button', { name: 'Run Market Signal Scout' }).isDisabled(), true)
    ready = false
    await page.reload()
    await page.getByText('Jev not configured', { exact: true }).waitFor()
    await page.getByRole('checkbox').nth(1).click()
    await page.waitForFunction(() => !document.querySelectorAll('input[type="checkbox"]')[1].checked)
    assert.equal(jobs[1].enabled, false)
    await page.getByRole('button', { name: 'Review inbox' }).click()
    assert.match(page.url(), /#\/memory\?source=signals/)
    fail = true
    await page.goto('http://studio.test/#/signals')
    await page.reload()
    await page.getByText('Signal jobs unavailable.', { exact: false }).waitFor()
    assert.deepEqual(errors, [])
    await context.close()
    console.log(`${width}px: real controls, cooldown, disconnected, pause, inbox link and outage passed`)
  }
} finally {
  await browser.close()
}
