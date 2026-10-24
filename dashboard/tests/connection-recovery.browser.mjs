// Local synthetic session/transport failures only; no public origin or login.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from 'playwright'
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-connection-recovery')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    for (const kind of ['session', 'response', 'network']) {
      const context = await browser.newContext({ viewport: { width, height: 950 } })
      const page = await context.newPage()
      let failed = true, documents = 0
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      page.on('request', request => { if (request.isNavigationRequest()) documents++ })
      await context.route('**/api/**', route => {
        const path = new URL(route.request().url()).pathname
        if (failed && ['/api/projects', '/api/health'].includes(path)) {
          if (kind === 'network') return route.abort('failed')
          return route.fulfill(kind === 'session' ? { status: 401, json: { detail: 'access required' } } : { contentType: 'text/html', body: '<html>Session page</html>' })
        }
        const data = path === '/api/projects' ? { projects: [] } : path === '/api/agents' ? { agents: [] } : path === '/api/agents/resources' ? { repos: [], github: { status: 'disconnected' } } : path === '/api/agents/models' ? { models: [], default: '' } : path === '/api/health' ? { aiia: { status: 'online' }, ollama: { status: 'online' } } : path === '/api/monitor' ? { services: {} } : {}
        return route.fulfill({ json: data })
      })
      await page.goto(`${process.env.STUDIO_URL}#/projects`)
      const banner = page.getByRole('alert').filter({ has: page.getByRole('button', { name: 'Reconnect Studio' }) })
      await banner.waitFor()
      await banner.getByText(kind === 'session' ? 'Studio access is blocked or your session has expired.' : kind === 'network' ? 'This browser could not reach Studio.' : 'Studio returned a page instead of data.', { exact: true }).waitFor()
      assert.equal(await page.getByText('brain down', { exact: true }).count(), 0)
      assert.equal(await page.getByRole('link', { name: 'Open research desk', exact: true }).getAttribute('href'), '#/bd')
      await page.screenshot({ path: `${output}/${kind}-${width}.png`, animations: 'disabled' })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      failed = false
      if (kind === 'session') {
        await banner.getByRole('button', { name: 'Reconnect Studio' }).click()
        await page.waitForFunction(() => !document.querySelector('[aria-label="Studio connection"]'))
        assert.equal(new URL(page.url()).hash, '#/projects')
        assert.equal(documents, 2)
      } else {
        await banner.getByRole('button', { name: 'Retry connection', exact: true }).click()
        await banner.waitFor({ state: 'hidden' })
      }
      assert.deepEqual(errors, [])
      await context.close()
    }
  }
} finally { await browser.close() }
console.log(`Connection recovery passed desktop/mobile session, HTML and network fixtures. Screenshots: ${output}`)
