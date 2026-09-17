// The Map's agent inspector shows the agent's real configuration. Every
// response is synthetic; no Command Center, Brain or model is reached.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-agent-inspector')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-17'

function seedAgents() {
  const base = {
    persona: 'Focused, pragmatic, and direct.', skills: [], tools: [], repo_id: '',
    temperature: 0.35, max_tokens: 1200, model: '', suite: '', memory_namespace: '',
    loop_enabled: false, loop_interval_minutes: 60, loop_task: '', loop_max_runs_per_day: 4,
    loop_runs_today: 0, loop_day: date, status: 'idle', last_run_at: null, last_result: '',
    last_error: '', runs: [], created_at: `${date}T09:00:00Z`, updated_at: `${date}T09:00:00Z`,
  }
  return [
    {
      ...base, id: 'agent-1', name: 'Release Gate Reviewer', mission: 'Gate release candidates.',
      persona: 'Skeptical reviewer who cites evidence.', skills: ['Analysis', 'Research'],
      tools: ['Repository read', 'Local memory'], repo_id: 'aiia', temperature: 0.2, max_tokens: 1600,
      suite: 'release', loop_enabled: true, loop_interval_minutes: 120, loop_task: 'Review the latest candidate.',
      loop_max_runs_per_day: 6, loop_runs_today: 2,
      last_result: `GO verdict. ${'Evidence line. '.repeat(60)}TAIL_MARKER`,
      last_error: 'local_model_unavailable',
    },
    { ...base, id: 'agent-2', name: 'Docs Scout', mission: 'Find stale docs.', model: 'llama3.1:8b' },
  ]
}

const models = {
  default: 'qwen3:8b',
  models: [
    { id: 'qwen3:8b', label: 'qwen3:8b', family: 'qwen3', parameter_size: '8.2B', size_gb: 5.2, default: true },
    { id: 'llama3.1:8b', label: 'llama3.1:8b', family: 'llama', parameter_size: '8.0B', size_gb: 4.9, default: false },
  ],
}

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    const agents = seedAgents()
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/models') body = models
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments: [] }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/agent-world/layout') body = { layout: { version: 1, revision: 0, positions: {}, updated_at: null } }
      else if (path === '/api/tasks') body = []
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      else if (path === '/api/tokens/today') body = { date, total_tokens: 0, total_requests: 0, total_cost: 0, by_provider: {}, by_purpose: {} }
      else if (path === '/api/tokens/recent') body = { days: [] }
      else if (path === '/api/studio/activity') body = { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] }
      else if (path === '/api/memory-inbox') body = { ideas: [], total: 0, offset: 0, counts: { unreviewed: 0, promoted: 0, dismissed: 0 } }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5191/')
    await page.getByRole('tab', { name: 'Map', exact: true }).click()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 2)
    const inspector = page.getByRole('complementary', { name: 'Node controls' })
    const config = inspector.getByRole('definition')
    const row = async label => {
      const terms = await inspector.locator('dt').allInnerTexts()
      const index = terms.indexOf(label)
      assert.ok(index >= 0, `missing config row ${label}`)
      return (await config.nth(index).innerText()).trim()
    }

    // A3: the real configuration, not just name and mission.
    await page.locator('[data-agent-target="agent-1"]').press('Enter')
    await inspector.waitFor()
    await inspector.getByText('Release Gate Reviewer', { exact: true }).waitFor()
    await page.waitForFunction(() => document.querySelector('[aria-label="Node controls"]')?.textContent?.includes('Task default: qwen3:8b'))
    assert.equal(await row('Model'), 'Task default: qwen3:8b')
    assert.equal(await row('Temperature'), '0.2')
    assert.equal(await row('Max tokens'), '1600')
    assert.equal(await row('Tools'), 'Repository read, Local memory')
    assert.equal(await row('Skills'), 'Analysis, Research')
    assert.equal(await row('Repo'), 'aiia')
    assert.equal(await row('Suite'), 'release')
    assert.equal(await row('Loop'), 'On · every 2h · 2 of 6 runs today')
    assert.equal(await row('Persona'), 'Skeptical reviewer who cites evidence.')
    const text = await inspector.innerText()
    assert.ok(text.includes('GO verdict.'))
    assert.ok(!text.includes('TAIL_MARKER'), 'last result must be truncated')
    assert.ok(text.includes('local_model_unavailable'))
    const rect = await inspector.boundingBox()
    assert.ok(rect.x >= 0 && rect.x + rect.width <= width)
    await page.screenshot({ path: join(output, `inspector-config-${width}.png`) })

    // A pinned model is shown by name, and switching nodes resets the inspector.
    await inspector.getByRole('button', { name: 'Close node controls' }).click()
    await page.locator('[data-agent-target="agent-2"]').press('Enter')
    await inspector.getByText('Docs Scout', { exact: true }).waitFor()
    assert.equal(await row('Model'), 'llama3.1:8b')
    assert.equal(await row('Loop'), 'Off')
    assert.equal(await row('Tools'), 'None')
    assert.ok(!(await inspector.innerText()).includes('Last error'))

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: inspector configuration display passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
