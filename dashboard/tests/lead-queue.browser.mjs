import assert from 'node:assert/strict'
import { readFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
const output = '/tmp/aiia-lead-queue'
await mkdir(output, { recursive: true })
try {
  for (const width of [390, 1440]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, asset)) })
    })
    const review = { status: 'qualified', company: 'Example Manufacturing', evidence_url: 'https://example.com/news', account_fit: 'Family-owned manufacturer in Iowa', observed_change: 'New facility announced', note: 'Verify expansion timing', version: 1, updated_at: '2026-10-01T12:00:00Z' }
    const leads = Array.from({ length: 26 }, (_, i) => ({ id: `signal-${i}`, company: i < 2 ? review.company : '', text: `Public expansion signal ${i}`, created_at: '2026-10-01T12:00:00Z', inbox_status: i === 1 ? 'dismissed' : 'unreviewed', assignment_id: '', decision: i < 2 ? 'qualified' : 'unreviewed', review: i < 2 ? review : null }))
    let fail = false
    let postCount = 0
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      if (url.pathname === '/api/public-signals/leads') {
        if (fail) return route.fulfill({ status: 503, json: { detail: 'lead_queue_unavailable' } })
        const status = url.searchParams.get('status')
        const company = url.searchParams.get('company')
        const offset = Number(url.searchParams.get('offset'))
        const filtered = leads.filter(lead => (status === 'all' || status === lead.decision) && lead.company.toLowerCase().includes(company.toLowerCase()))
        return route.fulfill({ json: { leads: filtered.slice(offset, offset + 25), total: filtered.length, offset, limit: 25 } })
      }
      if (url.pathname === '/api/memory-inbox/signal-0/assign') {
        assert.equal(route.request().method(), 'POST')
        assert.equal(route.request().postDataJSON().agent_id, 'a1')
        leads[0].assignment_id = 'work-1'
        postCount++
        return route.fulfill({ json: { assignment: { id: 'work-1' }, idea: leads[0] } })
      }
      if (url.pathname.endsWith('/qualification')) {
        if (route.request().method() === 'PUT') {
          const { expected_version, ...updated } = route.request().postDataJSON()
          assert.equal(expected_version, 1)
          leads[0].decision = updated.status
          leads[0].review = { ...updated, version: 2, updated_at: review.updated_at }
          return route.fulfill({ json: { review: leads[0].review } })
        }
        return route.fulfill({ json: { review, history: [review] } })
      }
      assert.equal(route.request().method(), 'GET', 'No unexpected writes or runs')
      const bodies = {
        '/api/agents': { agents: [{ id: 'a1', name: 'Market Researcher' }] },
        '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
        '/api/agents/models': { models: [] }, '/api/tasks': [],
        '/api/signal-jobs': { jobs: [{ id: 'lead_signals', name: 'Lead Signal Scout', specialty: 'Public expansion evidence', enabled: false, interval_hours: 12, last_run: null }], ready: false, configured: true, retrieval_enabled: false, screening_enabled: false },
        '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
        '/api/monitor': { services: {} },
      }
      return route.fulfill({ json: bodies[url.pathname] ?? {} })
    })
    await page.goto('http://studio.test/#/signals')
    const queue = page.getByRole('region', { name: 'Lead queue', exact: true })
    await queue.getByText(/26 matching signals/).waitFor()
    await page.getByText('Jev disabled', { exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run Lead Signal Scout' }).isVisible(), false)
    await page.getByText('Discovery automation (0/1 scheduled)', { exact: true }).click()
    await page.getByText('Discovery is paused.', { exact: false }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run Lead Signal Scout' }).isDisabled(), true)
    await page.getByText('Discovery automation (0/1 scheduled)', { exact: true }).click()
    assert.equal(await queue.getByRole('region', { name: 'Example Manufacturing', exact: true }).count(), 1)
    await queue.getByRole('button', { name: 'Next lead page' }).click()
    await queue.getByText('Public expansion signal 25', { exact: true }).waitFor()
    await queue.getByLabel('Qualification filter').selectOption('qualified')
    await queue.getByText(/2 matching signals/).waitFor()
    assert.equal(await queue.getByRole('button', { name: 'Previous lead page' }).isDisabled(), true)
    assert.equal(await queue.getByRole('button', { name: 'Queue research', exact: true }).count(), 1)
    await queue.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `qualified-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await queue.getByLabel('Research agent for signal-0').selectOption('a1')
    await queue.getByRole('button', { name: 'Queue research', exact: true }).click()
    await queue.getByRole('link', { name: 'Open research assignment' }).waitFor()
    assert.equal(postCount, 1)
    assert.equal(await queue.getByRole('link', { name: 'Open research assignment' }).getAttribute('href'), '#/assignments/work-1')
    await queue.getByText('Lead qualification', { exact: true }).first().click()
    await queue.getByLabel('Decision', { exact: true }).selectOption('watch')
    await queue.getByRole('button', { name: 'Save qualification', exact: true }).click()
    await queue.getByText(/1 matching signals/).waitFor()
    assert.equal(await queue.getByText('Public expansion signal 0', { exact: true }).count(), 0)
    await queue.getByLabel('Company', { exact: true }).fill('missing')
    await queue.getByRole('button', { name: 'Search', exact: true }).click()
    await queue.getByText('No signals match this view.').waitFor()
    fail = true
    await queue.getByRole('button', { name: 'Refresh lead queue' }).click()
    await queue.getByRole('alert').waitFor()
    assert.equal(await queue.getByText('No signals match this view.').count(), 0)
    fail = false
    await queue.getByRole('button', { name: 'Refresh lead queue' }).click()
    await queue.getByText('No signals match this view.').waitFor()
    assert.deepEqual(errors, [])
    await page.close()
  }
  console.log('PASS: qualification filter, company grouping/search, pagination, research handoff, dismissed guard, empty/error recovery, responsive layouts')
} finally { await browser.close() }
