import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { chromium } from 'playwright'

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } })
    let fallback = true
    await page.route('**/api/**', route => {
      const path = new URL(route.request().url()).pathname
      const services = Object.fromEntries([['default', 'App Analyst'], ['platform', 'AIIA Platform']].map(
        ([id, name]) => [id, { id, name, status: fallback ? 'not_configured' : 'offline', configured: !fallback }],
      ))
      const data = path === '/api/monitor' ? { services } : path === '/api/health'
        ? { aiia: { status: 'online' }, ollama: { status: 'online' } } : { projects: [], agents: [] }
      return route.fulfill({ json: data })
    })
    await page.goto(process.env.STUDIO_URL + '#/projects')
    const header = page.locator('header').first()
    for (const label of ['brain online', 'ollama ready', 'App Analyst: Not configured', 'AIIA Platform: Not configured']) {
      await header.getByText(label, { exact: true }).waitFor()
    }
    assert.ok(await header.evaluate(el => el.scrollWidth <= innerWidth))
    if (process.env.SCREENSHOT_DIR) {
      await mkdir(process.env.SCREENSHOT_DIR, { recursive: true })
      await header.screenshot({ path: join(process.env.SCREENSHOT_DIR, width + '.png') })
    }
    fallback = false
    await page.reload()
    await header.getByText('App Analyst: offline', { exact: true }).waitFor()
    await page.close()
  }
} finally {
  await browser.close()
}
