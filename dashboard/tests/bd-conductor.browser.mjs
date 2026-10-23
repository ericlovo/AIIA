// Synthetic fixtures only. Actual runtime integration is exercised separately.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { chromium } from 'playwright'
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-bd-ui')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const budget = { tokens: { used: 1250, limit: 40000 }, calls: { used: 4, limit: 100 }, seconds: { used: 40, limit: 3600 } }
const at = new Date().toISOString()
const bots = ['Healthcare', 'PE technical diligence', 'Alumni & associations', 'Client BD'].map((name, i) => ({ id: `bot-${i}`, name, segment: name, tenant_id: `synthetic-${i}`, counts: { queued: 2, running: i === 0 ? 1 : 0, completed: 1, failed: 0 }, budgets: budget }))
const snapshot = {
  mode: 'replay', schedule_enabled: false, kill_switch: false, bots, budget,
  jobs: [{ id: 'job-1', bot_id: 'bot-0', tenant_id: 'synthetic-0', segment: 'Healthcare', state: 'running', step: 'Extracting public evidence', created_at: at, started_at: at, heartbeat_at: at }],
  events: [{ id: 'event-1', kind: 'job_running', message: 'Research started from recorded public evidence.', created_at: at, bot_id: 'bot-0', job_id: 'job-1' }],
  packets: [{ id: 'packet-1', index: 0, job_id: 'previous-job', bot_id: 'bot-0', tenant_id: 'synthetic-0', segment: 'Healthcare', created_at: at, mode: 'replay', review_status: 'human_review_required', organization: 'Synthetic Health Group', source_url: 'https://example.com/research', emails: ['research@example.com'], phones: ['+1 202-555-0100'], signals: ['Expansion announcement in recorded source.'], insights: ['Explore demand for intake automation.'], summary: 'Synthetic evidence packet for review. This is not a real prospect.' }],
  health: { qwen: { status: 'replay', detail: 'Recorded extraction' }, jev: { status: 'replay', detail: 'Recorded verdicts' }, managed_leads: { status: 'data_unavailable', detail: 'Staging API is not connected' } },
  jev: { calls_per_minute: 2, verdict_mix: { Choice: 1, Score: 1, Noul: 0 }, latency_ms: null, errors: null },
}
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } })
    const page = await context.newPage()
    const errors = [], posts = []
    let current = structuredClone(snapshot), unavailable = false, actionFail = false
    page.on('pageerror', error => errors.push(error.message))
    await context.addInitScript(() => {
      class Source {
        constructor() { window.bdSource = this; setTimeout(() => this.onopen?.(), 50) }
        addEventListener(name, listener) { this[name] = listener }
        close() {}
      }
      window.EventSource = Source
    })
    await context.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let data = {}
      if (path === '/api/bd/snapshot') {
        if (unavailable) return route.fulfill({ status: 503, json: { detail: 'fixture_data_unavailable' } })
        data = current
      } else if (path.startsWith('/api/bd/') && route.request().method() === 'POST') {
        posts.push({ path, body: route.request().postDataJSON() })
        if (actionFail) return route.fulfill({ status: 503, json: { detail: 'fixture_action_unavailable' } })
        if (path.endsWith('/review')) current.packets[0].review_status = route.request().postDataJSON().status
        if (path.endsWith('kill-switch')) current.kill_switch = route.request().postDataJSON().enabled
        data = { ok: true }
      } else if (path === '/api/agents') data = { agents: [] }
      else if (path === '/api/agents/resources') data = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/agents/models') data = { models: [], default: '' }
      else if (path === '/api/health') data = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') data = { services: {} }
      else if (path === '/api/integrations/typesafe/status') data = { ready: false }
      return route.fulfill({ json: data })
    })
    await page.goto(`${process.env.STUDIO_URL || 'http://127.0.0.1:5187/'}#/bd`)
    await page.getByRole('heading', { name: 'The research desk.' }).waitFor()
    await page.getByText('Synthetic Health Group', { exact: true }).waitFor()
    await page.getByText('Live', { exact: true }).waitFor()
    assert.equal(await page.getByRole('link', { name: 'BD conductor', exact: true }).getAttribute('aria-current'), 'page')
    assert.equal(await page.getByLabel('Jev router', { exact: true }).getByText('Unavailable').count(), 2)
    await page.getByText('Review evidence & contacts', { exact: true }).click()
    assert.equal(await page.getByRole('link', { name: 'Open source evidence' }).getAttribute('href'), 'https://example.com/research')
    await page.getByText('Email: research@example.com', { exact: true }).waitFor()
    await page.screenshot({ path: join(output, `bd-evidence-${width}.png`), fullPage: true })
    await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bd-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.equal(await page.locator('.bd-page').evaluate(el => el.scrollWidth > el.clientWidth), false)
    await page.getByRole('combobox', { name: 'Show' }).selectOption('bot-1')
    await page.getByText('No packets in this view yet.', { exact: false }).waitFor()
    await page.getByRole('combobox', { name: 'Show' }).selectOption('all')
    await page.getByRole('button', { name: 'Pause feed' }).click()
    const next = structuredClone(current)
    next.events.unshift({ ...next.events[0], id: 'event-2', kind: 'packet_assembled', message: 'SSE packet arrived without refreshing.' })
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), next)
    assert.equal(await page.getByText('SSE packet arrived without refreshing.').count(), 0)
    await page.getByRole('button', { name: 'Resume feed' }).click()
    await page.getByText('SSE packet arrived without refreshing.').waitFor()
    await page.getByRole('button', { name: 'Run shadow batch' }).click()
    await page.getByText('Shadow batch accepted.', { exact: false }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/shadow', body: {} })
    await page.getByRole('button', { name: 'Stop conductor' }).click()
    await page.getByRole('button', { name: 'Release kill switch' }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run shadow batch' }).isDisabled(), true)
    await page.getByRole('button', { name: 'Release kill switch' }).click()
    await page.getByRole('button', { name: 'Stop conductor' }).waitFor()
    await page.getByRole('button', { name: 'Research next', exact: true }).click()
    await page.getByText('Local research triage saved.', { exact: true }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/packets/previous-job/0/review', body: { status: 'research_next' } })
    await page.getByRole('button', { name: 'Dismiss', exact: true }).click()
    await page.getByText('No packets in this view yet.', { exact: false }).waitFor()
    await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('dismissed')
    await page.getByRole('button', { name: 'Restore', exact: true }).click()
    await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('human_review_required')
    await page.getByText('Synthetic Health Group', { exact: true }).waitFor()
    actionFail = true
    await page.getByRole('button', { name: 'Run shadow batch' }).click()
    await page.getByRole('alert').filter({ hasText: 'fixture_action_unavailable' }).waitFor()
    unavailable = true
    await page.reload()
    await page.getByRole('alert').filter({ hasText: 'fixture_data_unavailable' }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run shadow batch' }).isDisabled(), true)
    assert.equal(await page.getByRole('heading', { name: 'Research lanes' }).count(), 0)
    await page.screenshot({ path: join(output, `bd-unavailable-${width}.png`), fullPage: true })
    assert.deepEqual(errors, [])
    console.log(`BD ${width}px: source evidence, filter, SSE, pause, shadow action, kill switch, unavailable states and overflow passed`)
    await context.close()
  }
} finally { await browser.close() }
