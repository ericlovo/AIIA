import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-studio-ux')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-14'
const agents = Array.from({ length: 24 }, (_, i) => ({
  id: `synthetic-${i}`, name: `CI specialist ${i + 1}`, mission: 'Review local repository evidence and identify the next verification task.',
  status: i === 0 ? 'running' : 'idle', skills: ['Analysis'], tools: ['Repository read'],
  repo_id: 'aiia', loop_enabled: false, loop_max_runs_per_day: 4, runs: [],
}))
const assignments = agents.flatMap(agent => Array.from({ length: 3 }, (_, i) => ({
  id: `${agent.id}-work-${i}`, agent_id: agent.id, title: `Verify contract ${agent.id} ${i}`,
  objective: 'Verify the isolated test fixture.', status: i === 0 ? 'queued' : 'completed',
  result: i ? 'Synthetic evidence only.' : '', priority: 'normal', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z`,
})))
const usage = {
  date, total_tokens: 84256, total_requests: 29, total_cost: 0,
  by_provider: { local: { tokens: 84256, input_tokens: 69950, output_tokens: 14306, requests: 29, cost: 0 } },
  by_purpose: { agent_studio_loop: { tokens: 84256, requests: 29, providers: ['local'], model: 'qwen3:8b' } },
}

try {
  for (const width of [1440, 653, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let tokenFailure = false
    let emptyUsage = false
    let layoutFailure = false
    let layout = { version: 1, revision: 0, positions: {}, updated_at: null }
    let saves = 0
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/agent-world/layout') {
        if (route.request().method() === 'PUT') {
          if (layoutFailure) status = 503
          else {
            saves++
            layout = { ...layout, positions: { ...layout.positions, ...route.request().postDataJSON().positions }, revision: saves }
          }
        } else if (route.request().method() === 'DELETE') layout = { ...layout, positions: {} }
        body = status === 503 ? { detail: 'Synthetic storage failure' } : { layout }
      } else if (path === '/api/tokens/today') {
        status = tokenFailure ? 503 : 200
        body = tokenFailure ? { detail: 'Synthetic usage outage' } : emptyUsage ? { ...usage, total_tokens: 0, total_requests: 0, by_provider: {}, by_purpose: {} } : usage
      } else if (path === '/api/tokens/recent') body = { days: Array.from({ length: 14 }, (_, i) => ({ date: `2026-09-${String(14 - i).padStart(2, '0')}`, total_tokens: i * 1234, total_requests: i, total_cost: 0 })) }
      else if (path === '/api/studio/activity') body = { today: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0 }
      else if (path === '/api/tasks') body = []
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5184/')
    const tokens = page.getByRole('region', { name: 'Platform token usage' })
    await tokens.getByText('84,256', { exact: true }).first().waitFor().catch(async error => {
      console.error(await page.locator('body').innerText())
      throw error
    })
    await tokens.locator('summary').click()
    await tokens.getByRole('rowheader').filter({ hasText: 'agent_studio_loop' }).waitFor()
    await tokens.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `usage-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)

    tokenFailure = true
    await tokens.getByRole('button', { name: 'Refresh token usage' }).click()
    await tokens.getByRole('alert').waitFor({ timeout: 15000 })
    assert.ok((await tokens.innerText()).includes('84,256'))
    tokenFailure = false
    emptyUsage = true
    await tokens.getByRole('button', { name: 'Refresh token usage' }).click()
    await tokens.getByText('No attributed usage reported today.').waitFor()

    await page.getByRole('tab', { name: 'Map', exact: true }).click()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 48)
    const assertNoOverlaps = async count => {
      const rects = await page.locator('[data-graph-node]').evaluateAll(nodes => nodes.map(node => {
        const r = node.getBoundingClientRect(); return { x: r.x, y: r.y, right: r.right, bottom: r.bottom }
      }))
      assert.equal(rects.length, count)
      for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) {
        const a = rects[i], b = rects[j]
        assert.ok(a.right <= b.x || b.right <= a.x || a.bottom <= b.y || b.bottom <= a.y, `Node overlap at ${width}: ${i}, ${j}`)
      }
    }
    await assertNoOverlaps(48)
    await page.getByRole('checkbox', { name: 'Completed assignments', exact: true }).check()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 96)
    await assertNoOverlaps(96)
    await page.getByRole('button', { name: 'Fit map', exact: true }).click()
    await page.screenshot({ path: join(output, `map-fit-${width}.png`) })
    const node = page.locator('[data-agent-target="synthetic-0"]')
    await node.press('Enter')
    const inspector = page.getByRole('complementary', { name: 'Node controls' })
    await inspector.waitFor()
    const rect = await inspector.boundingBox()
    assert.ok(rect.x >= 0 && rect.x + rect.width <= width)
    await page.screenshot({ path: join(output, `map-inspector-${width}.png`) })
    await inspector.getByRole('button', { name: 'Close node controls' }).click()
    const saved = page.waitForResponse(response => response.url().endsWith('/api/agent-world/layout') && response.request().method() === 'PUT')
    await node.press('ArrowRight')
    await saved
    assert.ok(saves > 0)
    layoutFailure = true
    await node.press('ArrowLeft')
    await page.getByText('Layout save failed', { exact: true }).waitFor()
    layoutFailure = false
    await page.getByRole('button', { name: 'Reset layout' }).click()
    await page.getByText('Layout synced', { exact: true }).waitFor()
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: usage, refresh failure/recovery, 96 nodes, zoom, inspector, keyboard save, save failure/recovery passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
