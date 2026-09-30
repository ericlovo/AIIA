import assert from 'node:assert/strict'
import { readFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
const output = '/tmp/aiia-timelines-leads'
await mkdir(output, { recursive: true })
try {
  for (const width of [390, 1440]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, asset)) })
    })
    let review = null
    let history = []
    let stale = false
    const run = { id: 'run-1', agent_id: 'a1', agent_name: 'Market Scout', status: 'failed', at: '2026-09-30T12:00:00Z', trigger: 'manual', model: 'local-test', input_tokens: 120, output_tokens: 30, assignment_id: 'work-1', task: 'Check primary evidence', error: 'Source unavailable', result: '' }
    const agent = { id: 'a1', name: 'Market Scout', mission: 'Review Iowa growth signals', persona: 'Evidence first', status: 'idle', skills: [], tools: [], runs: [], repo_id: '', temperature: 0.3, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60, loop_task: '', loop_max_runs_per_day: 4 }
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      if (url.pathname.endsWith('/qualification')) {
        if (route.request().method() === 'PUT') {
          if (stale) return route.fulfill({ status: 409, json: { detail: 'qualification_changed_reload_before_saving' } })
          const { expected_version, ...draft } = route.request().postDataJSON()
          assert.equal(expected_version, review?.version ?? 0)
          review = { ...draft, version: expected_version + 1, updated_at: '2026-09-30T12:00:00Z' }
          history = [review, ...history]
          return route.fulfill({ json: { review } })
        }
        return route.fulfill({ json: { review, history } })
      }
      assert.equal(route.request().method(), 'GET', 'No unexpected writes')
      const bodies = {
        '/api/agents': { agents: [agent] }, '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
        '/api/agents/models': { default: 'local-test', models: [] },
        '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
        '/api/monitor': { services: {} }, '/api/tasks': [], '/api/assignments': { assignments: [] },
        '/api/studio/activity': { runs: url.searchParams.get('status') === 'completed' ? [] : [run] },
        '/api/studio/runs/run-1': { run },
        '/api/memory-inbox': { ideas: [{ id: 'signal-1', text: 'Iowa family-owned manufacturer expands capacity. Verify primary evidence.', source: 'public_signals', status: 'unreviewed', project: 'pl', created_at: '2026-09-30T12:00:00Z' }], total: 1, offset: 0, counts: { unreviewed: 1, promoted: 0, dismissed: 0 } },
      }
      await route.fulfill({ json: bodies[url.pathname] ?? {} })
    })
    await page.goto('http://studio.test/#/agents/a1')
    const timeline = page.getByRole('region', { name: 'Market Scout activity' })
    await timeline.getByText('150 tokens', { exact: false }).waitFor()
    await timeline.locator('summary').click()
    await timeline.getByText('Source unavailable', { exact: true }).waitFor()
    assert.equal(await timeline.getByRole('link', { name: 'Open assignment' }).getAttribute('href'), '#/assignments/work-1')
    await timeline.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `timeline-${width}.png`) })
    await timeline.getByLabel('Run status').selectOption('completed')
    await timeline.getByText('No recorded runs in this view.').waitFor()
    await page.getByRole('button', { name: 'Configuration', exact: true }).click()
    await page.getByText('Agent controls', { exact: true }).waitFor()
    await page.goto('http://studio.test/#/memory?source=signals')
    assert.equal(await page.getByRole('tab', { name: 'Public signals', exact: true }).getAttribute('aria-selected'), 'true')
    await page.getByText('Lead qualification', { exact: true }).click()
    await page.getByLabel('Decision', { exact: true }).selectOption('qualified')
    await page.getByLabel('Company', { exact: true }).fill('Example Manufacturing')
    await page.getByLabel('Primary-source URL (HTTPS)').fill('https://example.com/expansion')
    await page.getByLabel('Account fit').fill('Family-owned manufacturer in Iowa')
    await page.getByLabel('Observed change').fill('New production facility')
    await page.getByLabel('Decision rationale').fill('Primary announcement reviewed; research next')
    await page.getByRole('button', { name: 'Save qualification', exact: true }).click()
    await page.getByText(/Saved revision 1/).waitFor()
    assert.equal(review.status, 'qualified')
    await page.getByLabel('Decision', { exact: true }).scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `qualification-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    stale = true
    await page.getByLabel('Decision rationale').fill('Unsaved local draft')
    await page.getByRole('button', { name: 'Save qualification', exact: true }).click()
    await page.getByText('Someone saved a newer review. Your draft has not been saved.').waitFor()
    assert.equal(await page.getByLabel('Decision rationale').inputValue(), 'Unsaved local draft')
    assert.deepEqual(errors, [])
    await page.close()
  }
  console.log('PASS: timeline evidence/filter/configuration; qualification save, source routing, conflict retention, responsive layouts')
} finally { await browser.close() }
