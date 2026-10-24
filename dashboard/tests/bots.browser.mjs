// Synthetic only: all requests are intercepted; no live bots or models run.
import assert from 'node:assert/strict'
import { readFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-bots')
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, serviceWorkers: 'block' })
    const page = await context.newPage()
    const errors = [], blocked = [], posts = []
    let fail = false, failAfterRelease = false, historyFail = false, release
    let agents = [
      { id: 'research', name: 'Research Scout', mission: 'Find evidence for product decisions.', one_liner: 'Find evidence for product decisions.', status: 'idle', last_result: 'Three sources support testing the simpler onboarding flow.', last_error: '', last_run_at: '2026-10-14T09:00:00Z', use_cases: [{ label: 'Research brief', task: 'Prepare a sourced research brief.' }, { label: 'Check evidence', task: 'Assess the supplied evidence.' }, { label: 'Find research gaps', task: 'Identify three useful research gaps.' }], runs: [], skills: [], tools: [] },
      { id: 'code', name: 'Code Partner', mission: 'Inspect code and explain changes.', status: 'idle', last_result: '', last_error: '', runs: [], skills: [], tools: [] },
    ]
    let history = Array.from({ length: 25 }, (_, i) => ({ id: `history-${i}`, agent_id: 'research', at: new Date(Date.UTC(2026, 9, 14, 9, 0, -i)).toISOString(), task: `Research task ${25 - i}`, status: i === 0 ? 'failed' : 'completed', error: i === 0 ? 'Source unavailable; retained for review.' : '', result: i === 0 ? '' : `Evidence for task ${25 - i}.` }))
    for (const [name, mission] of [
      ['Project Guide', 'Turn a project goal into clear next steps.'], ['Review Partner', 'Find actionable issues in proposed code changes.'],
      ['Writing Partner', 'Shape rough notes into clear drafts.'], ['Operations Scout', 'Surface changes that need an operator decision.'],
      ['Knowledge Keeper', 'Find the context behind earlier decisions.'], ['Customer Researcher', 'Summarize supplied customer feedback.'],
      ['Quality Partner', 'Check behavior against acceptance criteria.'], ['Release Guide', 'Collect evidence for a release review.'],
    ]) agents.push({ ...agents[1], id: name.toLowerCase().replaceAll(' ', '-'), name, mission })
    page.on('pageerror', e => errors.push(e.message))
    await context.routeWebSocket('**/*', socket => socket.onMessage(() => {}))
    await context.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url())
      if (url.origin !== 'http://studio.test') { blocked.push(request.url()); return route.abort() }
      if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) })
      if (/^\/assets\/[A-Za-z0-9_.-]+\.(css|js)$/.test(url.pathname)) return route.fulfill({ contentType: url.pathname.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, url.pathname.slice(1))) })
      if (request.method() === 'POST' && url.pathname === '/api/agents/research/run') {
        posts.push(request.postDataJSON())
        if (fail) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'fixture_offline' }) })
        await new Promise(resolve => { release = resolve })
        if (failAfterRelease) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'fixture_delayed_failure' }) })
        history.unshift({ id: `new-${posts.length}`, agent_id: 'research', at: '2026-10-14T10:00:00Z', task: request.postDataJSON().task, status: 'completed', result: 'New evidence: simplify the first step.', error: '' })
        agents = agents.map(a => a.id === 'research' ? { ...a, last_result: 'New evidence: simplify the first step.', last_run_at: '2026-10-14T10:00:00Z' } : a)
        return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ agent: agents[0], model: 'fixture', latency_ms: 100 }) })
      }
      if (request.method() === 'GET' && /^\/api\/agents\/[^/]+\/history$/.test(url.pathname)) {
        if (historyFail) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'fixture_history_unavailable' }) })
        const offset = Number(url.searchParams.get('offset') || 0)
        const runs = url.pathname.includes('/research/') ? history : []
        return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ runs: runs.slice(offset, offset + 20), total: runs.length, offset, limit: 20 }) })
      }
      if (request.method() !== 'GET') { blocked.push(request.method() + url.pathname); return route.abort() }
      const bodies = {
        '/api/agents': { agents }, '/api/agents/resources': { repos: [], github: { status: 'disconnected' } }, '/api/agents/models': { models: [], default: '' },
        '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } }, '/api/monitor': { services: {} }, '/api/integrations/typesafe/status': { ready: false },
      }
      if (!(url.pathname in bodies)) { blocked.push(url.pathname); return route.fulfill({ status: 503, body: '{}' }) }
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify(bodies[url.pathname]) })
    })
    await page.goto('http://studio.test/#/bots')
    await page.getByRole('heading', { name: 'Bots', exact: true }).waitFor()
    assert.equal(new URL(page.url()).hash, '#/bots')
    const nav = page.getByRole('navigation', { name: 'Studio', exact: true })
    assert.deepEqual(await nav.getByRole('link').allTextContents(), ['Research', 'Bots', 'Work'])
    assert.equal(posts.length, 0)
    await page.getByRole('link', { name: 'Open Research Scout' }).focus()
    await page.keyboard.press('Enter')
    const task = page.getByLabel('Additional context', { exact: true })
    await task.waitFor()
    assert.equal(new URL(page.url()).hash, '#/bots/research')
    assert.equal(await page.getByLabel('Temperature').count(), 0)
    assert.equal(await task.getAttribute('maxlength'), '8000')
    const historyRegion = page.getByRole('region', { name: 'History', exact: true })
    await historyRegion.getByText('1–20 of 25 runs · Newest first').waitFor()
    assert.equal(await historyRegion.locator('summary').first().textContent().then(t => t.includes('Research task 25')), true)
    await historyRegion.locator('summary').first().click()
    await historyRegion.getByText('Source unavailable; retained for review.', { exact: true }).waitFor()
    await historyRegion.locator('summary').nth(1).click()
    await historyRegion.getByText('Evidence for task 24.', { exact: true }).waitFor()
    await page.getByRole('main').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bots-detail-${width}.png`), animations: 'disabled' })
    await historyRegion.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `bots-history-${width}.png`), animations: 'disabled' })
    await historyRegion.getByRole('button', { name: 'Older runs' }).click()
    await historyRegion.getByText('21–25 of 25 runs · Newest first').waitFor()
    assert.equal(await historyRegion.locator('summary').count(), 5)
    await historyRegion.getByRole('button', { name: 'Newer runs' }).click()
    await historyRegion.getByText('1–20 of 25 runs · Newest first').waitFor()
    assert.equal(await page.getByRole('region', { name: 'Use cases' }).getByRole('button').count(), 3)
    await task.fill('  Find the next useful experiment  ')
    await page.getByRole('button', { name: 'Send task', exact: true }).click()
    await page.getByRole('button', { name: 'Working...', exact: true }).waitFor()
    await page.getByRole('link', { name: 'Open Code Partner' }).click()
    await page.getByRole('region', { name: 'Code Partner interaction' }).waitFor()
    assert.equal(await task.inputValue(), '')
    await page.getByRole('region', { name: 'History', exact: true }).getByText('No runs yet. Run a use case or send a task to start.').waitFor()
    await page.goBack()
    await page.getByRole('button', { name: 'Working...', exact: true }).waitFor()
    assert.equal(await task.isDisabled(), true)
    assert.equal(posts.length, 1)
    release()
    await page.getByRole('region', { name: 'Latest result' }).getByText('New evidence: simplify the first step.', { exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Send task' }).isDisabled(), true)
    assert.equal(await task.inputValue(), '')
    await page.getByRole('main').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bots-success-${width}.png`), animations: 'disabled' })
    await page.getByRole('button', { name: 'Research brief', exact: true }).click()
    await page.getByRole('button', { name: 'Working...', exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Check evidence', exact: true }).isDisabled(), true)
    assert.deepEqual(posts[1], { task: 'Prepare a sourced research brief.' })
    release()
    await page.getByRole('button', { name: 'Send task', exact: true }).waitFor()
    await task.fill('  Focus on onboarding  ')
    await page.getByRole('button', { name: 'Check evidence', exact: true }).click()
    await page.getByRole('button', { name: 'Working...', exact: true }).waitFor()
    assert.deepEqual(posts[2], { task: 'Assess the supplied evidence.\n\nAdditional context:\nFocus on onboarding' })
    release()
    await page.getByRole('button', { name: 'Send task', exact: true }).waitFor()
    assert.equal(await task.inputValue(), '')
    await historyRegion.getByText('1–20 of 28 runs · Newest first').waitFor()
    await task.fill('x'.repeat(8000))
    assert.equal(await page.getByRole('button', { name: 'Research brief', exact: true }).isDisabled(), true)
    fail = true
    await task.fill('Try another experiment')
    await page.getByRole('button', { name: 'Send task' }).click()
    await page.getByRole('alert').getByText(/Task request failed/).waitFor()
    assert.equal(await task.inputValue(), 'Try another experiment')
    assert.equal(posts.length, 4)
    fail = false; failAfterRelease = true
    await page.getByRole('button', { name: 'Send task' }).click()
    await page.getByRole('button', { name: 'Working...', exact: true }).waitFor()
    await page.getByRole('link', { name: 'Open Code Partner' }).click()
    await page.getByRole('region', { name: 'Code Partner interaction' }).waitFor()
    await task.fill('Keep this other bot draft')
    release()
    await page.getByRole('link', { name: 'Open Research Scout' }).click()
    await page.getByRole('alert').getByText(/fixture_delayed_failure/).waitFor()
    assert.equal(await task.inputValue(), 'Try another experiment')
    await page.getByRole('link', { name: 'Open Code Partner' }).click()
    await page.getByRole('region', { name: 'Code Partner interaction' }).waitFor()
    assert.equal(await task.inputValue(), 'Keep this other bot draft')
    assert.equal(await page.getByRole('alert').count(), 0)
    await page.getByRole('link', { name: 'Open Research Scout' }).click()
    await page.getByRole('region', { name: 'Research Scout interaction' }).waitFor()
    assert.deepEqual(posts[0], { task: 'Find the next useful experiment' })
    await page.getByRole('main').evaluate(el => { el.scrollTop = 0 })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.screenshot({ path: join(output, `bots-task-${width}.png`), animations: 'disabled' })
    await nav.getByRole('link', { name: 'Bots', exact: true }).click()
    await page.getByRole('region', { name: 'Research Scout interaction' }).waitFor({ state: 'detached' })
    await page.getByRole('link', { name: 'Open Research Scout' }).waitFor()
    await page.screenshot({ path: join(output, `bots-home-${width}.png`), animations: 'disabled' })
    if (width === 390) {
      const lowerBot = page.getByRole('link', { name: 'Open Release Guide' })
      await lowerBot.scrollIntoViewIfNeeded()
      await lowerBot.click()
      const selectedHeading = page.getByRole('region', { name: 'Release Guide interaction' }).getByRole('heading', { name: 'Release Guide' })
      await selectedHeading.waitFor()
      assert.equal(await selectedHeading.evaluate(el => document.activeElement === el), true)
      const box = await selectedHeading.boundingBox()
      assert.ok(box.y > 0 && box.y < 400, 'Selecting a lower bot scrolls the interaction into view')
      assert.equal(await task.evaluate(el => document.activeElement === el), false, 'Mobile selection does not open the keyboard')
    }
    await nav.locator('summary').click()
    assert.equal(await nav.getByRole('link', { name: 'Today', exact: true }).isVisible(), true)
    await page.keyboard.press('Escape')
    assert.equal(await nav.getByRole('link', { name: 'Today', exact: true }).isVisible(), false)
    historyFail = true
    await page.getByRole('link', { name: 'Open Customer Researcher' }).click()
    await page.getByRole('region', { name: 'History', exact: true }).getByText(/History unavailable/).waitFor()
    historyFail = false
    await page.getByRole('button', { name: 'Retry history', exact: true }).click()
    await page.getByRole('region', { name: 'History', exact: true }).getByText('No runs yet. Run a use case or send a task to start.').waitFor()
    assert.deepEqual(errors, [])
    assert.deepEqual(blocked, [])
    console.log(`${width}px: Bots default, keyboard selection, task success/failure, switch/back pending guard, no automatic execution, More and layout passed`)
    await context.close()
  }
} finally { await browser.close() }
console.log(`Screenshots: ${output}`)
