// Attempt history on an assignment: saved attempts with model, duration and
// measured tokens, expanding one to read its task and output, paging, the
// applied-attempt marker, and an explicit failure when history is unavailable.
// Every response is synthetic; nothing reaches the Mini.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-assignment-history')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-16'
const agents = [{
  id: 'agent-1', name: 'Evidence Auditor', mission: 'Audit supplied evidence.',
  status: 'idle', skills: ['Analysis'], tools: [], repo_id: 'aiia',
  loop_enabled: false, loop_max_runs_per_day: 4, runs: [],
}]
const assignments = [{
  id: 'work-1', agent_id: 'agent-1', title: 'Audit the FLOW-01 fixture',
  objective: 'Identify whether every API validates repository mounts.',
  status: 'completed', result: 'Validation cannot be determined.', error: '',
  priority: 'normal', context: '', success_criteria: '', source_handoff_id: '',
  created_at: `${date}T11:00:00Z`, updated_at: `${date}T12:00:00Z`,
  review_status: 'unreviewed', review_note: '', review_version: 'v1', reviewed_at: null,
}]
// 25 attempts so the pager appears; the newest carries measured tokens, an
// older one predates token capture and must read as unrecorded, not zero.
const attempts = Array.from({ length: 25 }, (_, i) => ({
  id: `run-${String(24 - i).padStart(2, '0')}`, agent_id: 'agent-1', agent_name: 'Evidence Auditor',
  repo_id: 'aiia', at: `${date}T1${i % 10}:00:00Z`, status: i === 1 ? 'failed' : 'completed',
  trigger: 'assignment', assignment_id: 'work-1', model: 'qwen3:8b', latency_ms: 20000, legacy: 0,
  input_tokens: i === 2 ? null : 1234, output_tokens: i === 2 ? null : 56,
  task: `Synthetic task ${i}`, result: i === 1 ? '' : `Synthetic output ${i}`,
  error: i === 1 ? 'empty_agent_result' : '',
}))

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let historyFails = false
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      const path = url.pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/git-workspaces') body = { workspaces: [] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (/^\/api\/assignments\/[^/]+\/history$/.test(path)) {
        if (historyFails) { status = 503; body = { detail: 'assignment_history_unavailable' } }
        else {
          const offset = Number(url.searchParams.get('offset') || 0)
          body = {
            runs: attempts.slice(offset, offset + 20), total: attempts.length, offset, limit: 20,
            attempt_id: 'run-24', current_output_saved: true, completed_run_id: 'run-24',
          }
        }
      }
      else if (/^\/api\/studio\/runs\//.test(path)) {
        const run = attempts.find(a => a.id === path.split('/').pop())
        body = { run }
      }
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

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5186/')
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Audit the FLOW-01 fixture', { exact: true }).click()

    const history = page.getByRole('region', { name: 'Assignment attempt history' })
    await history.getByText('25 saved attempts').waitFor()

    // Measured tokens ride alongside model and duration; the applied attempt is marked.
    const rows = history.getByRole('listitem')
    assert.ok((await rows.first().innerText()).includes('1,290 tokens'))
    assert.ok((await rows.first().innerText()).includes('Applied to assignment'))
    // An attempt recorded before token capture reads as unrecorded, never as zero.
    assert.ok((await rows.nth(2).innerText()).includes('Tokens unrecorded'))
    assert.ok(!(await rows.nth(2).innerText()).includes('0 tokens'))
    // A failed attempt is labelled as such rather than as saved output.
    assert.ok((await rows.nth(1).innerText()).includes('Failed attempt'))
    await history.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `history-${width}.png`) })

    // Expanding one attempt loads its recorded task and saved output.
    await rows.first().getByRole('button').first().click()
    await history.getByText('Recorded task').waitFor()
    await history.getByText('Synthetic output 0', { exact: true }).waitFor()

    // Paging moves through the saved attempts.
    await history.getByRole('navigation', { name: 'Attempt history pages' }).waitFor()
    assert.ok((await history.innerText()).includes('1–20 of 25'))
    await history.getByRole('button', { name: 'Older attempts' }).click()
    await history.getByText('21–25 of 25').waitFor()
    await history.getByRole('button', { name: 'Newer attempts' }).click()
    await history.getByText('1–20 of 25').waitFor()

    // An unavailable ledger says so instead of rendering an empty history.
    historyFails = true
    await history.getByRole('button', { name: 'Refresh attempt history' }).click()
    await history.getByRole('alert').waitFor({ timeout: 15000 })
    assert.ok((await history.innerText()).includes('Attempt history unavailable'))
    historyFails = false

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: attempt list, measured tokens, unrecorded, expand, paging, failure passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
