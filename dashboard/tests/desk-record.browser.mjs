// Browser contract for a human decision. Local assets and synthetic API responses only.
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from 'playwright'

const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-desk-record')
await mkdir(output, { recursive: true })

const packet = {
  id: 'bd-fixture-42:1', job_id: 'bd-fixture-42', index: 1, bot_id: 'synthetic-research',
  tenant_id: 'synthetic-cooperative', segment: 'healthcare', mode: 'replay',
  organization: 'Cedar Valley Cooperative', source_url: 'https://example.org/public-notice',
  summary: 'A cooperative plans to solicit proposals for a shared purchasing platform.',
  source_quote: 'The cooperative will solicit proposals for a shared purchasing platform this winter.',
  candidate_status: 'supported', candidate_reason: 'source_supported_candidate', disposition: 'research',
  review_status: 'human_review_required', created_at: '2026-11-12T12:00:00Z',
  emails: [], phones: [], signals: ['Upcoming procurement'], insights: [],
}
const stages = ['qwen_replayed', 'jev_replayed'].map((kind, index) => ({
  id: index + 1, kind, created_at: packet.created_at, mode: 'replay',
  message: index ? 'Recorded judgment supported further research.' : 'Recorded source passage extracted.',
  job_id: packet.job_id, bot_id: packet.bot_id, tenant_id: packet.tenant_id, provider: index ? 'jev' : 'qwen',
}))
const contextData = {
  '/api/agents': { agents: [] }, '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
  '/api/agents/models': { default: '', models: [] }, '/api/monitor': { services: {} },
  '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
  '/api/integrations/typesafe/status': { ready: false },
}
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const [width, status, button] of [[1440, 'research_next', 'Research next'], [768, 'watch', 'Watch'], [390, 'dismissed', 'Dismiss']]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, serviceWorkers: 'block' })
    const page = await context.newPage()
    page.setDefaultTimeout(6000)
    const writes = [], blocked = [], errors = [], detailReads = []
    let receipt, failNext = width === 390, unavailable = false, mismatched = false
    page.on('pageerror', e => errors.push(e.message))
    await page.routeWebSocket('**/*', ws => ws.onMessage(() => {}))
    await context.route('**/*', async route => {
      const req = route.request()
      const url = new URL(req.url())
      const path = url.pathname
      if (url.origin !== 'http://studio.test') { blocked.push(req.url()); return route.abort() }
      if (path === '/') return route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) })
      if (/^\/assets\/[\w.-]+\.(css|js)$/.test(path)) return route.fulfill({
        contentType: path.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, path.slice(1))),
      })
      if (req.method() === 'POST' && path === '/api/bd/desk/synthetic-research/bd-fixture-42/1/review') {
        writes.push(req.postDataJSON())
        if (failNext) {
          failNext = false
          return route.fulfill({ status: 409, json: { detail: 'packet_review_changed' } })
        }
        receipt = {
          id: 105, job_id: packet.job_id, packet_index: packet.index, bot_id: packet.bot_id,
          tenant_id: packet.tenant_id, status, created_at: '2026-11-12T14:30:00Z', mode: 'replay',
          organization: packet.organization,
        }
        return route.fulfill({ json: { receipt } })
      }
      if (req.method() === 'GET') {
        if (path === '/api/bd/desk') {
          if (unavailable) return route.fulfill({ status: 503, json: { detail: 'synthetic_queue_unavailable' } })
          return route.fulfill({ json: { packets: receipt ? [] : [packet], has_more: false } })
        }
        if (path === '/api/bd/record') return route.fulfill({ json: { receipts: receipt ? [receipt] : [], has_more: false } })
        if (path === '/api/bd/record/105' && receipt) {
          detailReads.push(path)
          return route.fulfill({ json: { receipt, packet: { ...packet, tenant_id: mismatched ? 'unrelated-workspace' : packet.tenant_id, review_status: status }, stages } })
        }
        if (path === '/api/bd/record/999') return route.fulfill({ status: 404, json: { detail: 'receipt_not_found' } })
        if (path in contextData) return route.fulfill({ json: contextData[path] })
      }
      blocked.push(req.method() + ' ' + path)
      await route.fulfill({ status: 503, json: { detail: 'unexpected_fixture_request' } })
    })
    await page.goto('http://studio.test/')
    await page.getByRole('heading', { level: 1, name: 'Desk', exact: true }).waitFor()
    assert.equal(new URL(page.url()).hash, '#/desk')
    await page.getByRole('heading', { name: packet.organization, exact: true }).waitFor()
    await page.getByText(packet.source_quote, { exact: true }).waitFor()
    assert.equal(await page.getByRole('link', { name: 'Open source' }).getAttribute('href'), packet.source_url)
    assert.equal(writes.length, 0)
    await page.screenshot({ path: join(output, 'desk-' + width + '.png'), animations: 'disabled' })
    const choice = page.getByRole('button', { name: button, exact: true })
    await choice.click()
    if (width === 390) {
      await page.getByRole('alert').first().waitFor()
      assert.equal(await page.getByRole('link', { name: 'View receipt in Record' }).count(), 0)
      await page.getByRole('button', { name: 'Refresh packets' }).click()
      await choice.click()
    }
    const link = page.getByRole('link', { name: 'View receipt in Record' })
    await link.waitFor()
    assert.equal(await link.getAttribute('href'), '#/record/105')
    assert.deepEqual(writes, Array.from({ length: width === 390 ? 2 : 1 }, () => ({ status, expected_tenant_id: packet.tenant_id })))
    await link.click()
    const run = page.getByRole('region', { name: 'Run evidence' })
    await run.waitFor()
    assert.ok((await page.getByRole('main').innerText()).includes(packet.job_id))
    assert.ok((await run.innerText()).includes(packet.source_quote))
    const runLink = page.getByRole('link', { name: 'Inspect run ' + packet.job_id })
    assert.equal(await runLink.getAttribute('href'), '#/record/105?show=run')
    await runLink.click()
    await page.getByRole('heading', { name: 'Run record', exact: true }).waitFor()
    await run.getByText(stages[0].message, { exact: true }).waitFor()
    await run.getByText(stages[1].message, { exact: true }).waitFor()
    await page.reload()
    await run.waitFor()
    assert.equal(new URL(page.url()).hash, '#/record/105?show=run')
    await run.getByText(stages[1].message, { exact: true }).waitFor()
    await page.screenshot({ path: join(output, 'run-' + width + '.png'), animations: 'disabled' })
    await page.getByRole('link', { name: /Back to receipt/ }).click()
    await page.getByRole('heading', { name: 'Decision receipt', exact: true }).waitFor()
    await run.waitFor()
    assert.ok(detailReads.length >= 2)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.screenshot({ path: join(output, 'receipt-' + width + '.png'), animations: 'disabled' })
    const nav = page.getByRole('navigation', { name: 'Studio', exact: true })
    await nav.getByRole('link', { name: 'Record', exact: true }).click()
    await page.getByRole('searchbox', { name: 'Find a receipt' }).fill(packet.job_id)
    await page.getByRole('main').getByRole('link').filter({ hasText: packet.organization }).click()
    await run.waitFor()
    await nav.getByRole('link', { name: 'Desk', exact: true }).click()
    await page.getByRole('heading', { name: 'No packets awaiting review' }).waitFor()
    if (width === 390) {
      mismatched = true
      await page.goto('http://studio.test/#/record/105')
      await page.reload()
      await page.getByRole('alert').first().waitFor()
      assert.equal(await run.count(), 0)
      unavailable = true
      await page.goto('http://studio.test/#/desk')
      await page.reload()
      await page.getByRole('alert').first().waitFor()
      assert.equal(await choice.count(), 0)
      await page.goto('http://studio.test/#/record/999')
      await page.getByRole('alert').first().waitFor()
      assert.equal(await run.count(), 0)
    }
    assert.deepEqual(blocked, [])
    assert.deepEqual(errors, [])
    await context.close()
    console.log(width + 'px: ' + status + ' persisted through Record and reload')
  }
} finally { await browser.close() }
