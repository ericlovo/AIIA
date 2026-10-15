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
    let fail = false, failAfterRelease = false, release
    let agents = [
      { id: 'research', name: 'Research Scout', mission: 'Find evidence for product decisions.', one_liner: 'Find evidence for product decisions.', status: 'idle', last_result: 'Three sources support testing the simpler onboarding flow.', last_error: '', last_run_at: '2026-10-14T09:00:00Z', runs: [], skills: [], tools: [] },
      { id: 'code', name: 'Code Partner', mission: 'Inspect code and explain changes.', status: 'idle', last_result: '', last_error: '', runs: [], skills: [], tools: [] },
    ]
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
        agents = agents.map(a => a.id === 'research' ? { ...a, last_result: 'New evidence: simplify the first step.', last_run_at: '2026-10-14T10:00:00Z' } : a)
        return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ agent: agents[0], model: 'fixture', latency_ms: 100 }) })
      }
      if (request.method() !== 'GET') { blocked.push(request.method() + url.pathname); return route.abort() }
      const bodies = {
        '/api/agents': { agents }, '/api/agents/resources': { repos: [], github: { status: 'disconnected' } }, '/api/agents/models': { models: [], default: '' },
        '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } }, '/api/monitor': { services: {} }, '/api/integrations/typesafe/status': { ready: false },
      }
      if (!(url.pathname in bodies)) { blocked.push(url.pathname); return route.fulfill({ status: 503, body: '{}' }) }
      return route.fulfill({ contentType: 'application/json', body: JSON.stringify(bodies[url.pathname]) })
    })
    await page.goto('http://studio.test/')
    await page.getByRole('heading', { name: 'Bots', exact: true }).waitFor()
    assert.equal(new URL(page.url()).hash, '#/bots')
    const nav = page.getByRole('navigation', { name: 'Studio', exact: true })
    assert.deepEqual(await nav.getByRole('link').allTextContents(), ['Bots', 'Work', 'Projects'])
    assert.equal(posts.length, 0)
    await page.getByRole('link', { name: 'Open Research Scout' }).focus()
    await page.keyboard.press('Enter')
    const task = page.getByLabel('Give task', { exact: true })
    await task.waitFor()
    assert.equal(new URL(page.url()).hash, '#/bots/research')
    assert.equal(await page.getByLabel('Temperature').count(), 0)
    assert.equal(await task.getAttribute('maxlength'), '8000')
    await task.fill('  Find the next useful experiment  ')
    await page.getByRole('button', { name: 'Send task', exact: true }).click()
    await page.getByRole('button', { name: 'Working...', exact: true }).waitFor()
    await page.getByRole('link', { name: 'Open Code Partner' }).click()
    await page.getByRole('region', { name: 'Code Partner interaction' }).waitFor()
    assert.equal(await task.inputValue(), '')
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
    fail = true
    await task.fill('Try another experiment')
    await page.getByRole('button', { name: 'Send task' }).click()
    await page.getByRole('alert').getByText(/Task request failed/).waitFor()
    assert.equal(await task.inputValue(), 'Try another experiment')
    assert.equal(posts.length, 2)
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
    assert.deepEqual(errors, [])
    assert.deepEqual(blocked, [])
    console.log(`${width}px: Bots default, keyboard selection, task success/failure, switch/back pending guard, no automatic execution, More and layout passed`)
    await context.close()
  }
} finally { await browser.close() }
console.log(`Screenshots: ${output}`)
