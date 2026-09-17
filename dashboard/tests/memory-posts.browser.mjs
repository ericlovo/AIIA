// Memory log priority: a human sets it when logging, and the log can filter and
// sort by it. Every response is synthetic; the server's filter and sort are mirrored.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-memory-posts')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-17'
const RANK = { urgent: 0, high: 1, normal: 2, low: 3 }

const blank = {
  source: 'slack', project: 'mindmoor', workspace_id: 'T_TEST', channel_id: 'C_ONE', author_id: 'U_AUTHOR',
  memory_id: '', memory_category: '', review_note: '', reviewed_at: '', priority: 'normal', post_requested: 0,
  acknowledgement_status: null, acknowledgement_error: null, acknowledgement_ts: null,
  promotion_status: null, promotion_error: null, promotion_ts: null,
}
const seed = () => [
  { ...blank, id: 'idea-alpha-0000000', text: '<@U0BOT1> ship the cron contract first', created_at: `${date}T17:00:00Z`, status: 'unreviewed' },
  { ...blank, id: 'idea-bravo-0000000', text: 'older low priority note', created_at: `${date}T15:00:00Z`, status: 'promoted', memory_id: 'lessons_1_1', memory_category: 'lessons', reviewed_at: `${date}T16:00:00Z`, priority: 'low' },
]

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    const ideas = seed()
    const promotes = []
    const listings = []
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      const path = url.pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents: [] }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments: [] }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/tasks') body = []
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      else if (path === '/api/tokens/today') body = { date, total_tokens: 0, total_requests: 0, total_cost: 0, by_provider: {}, by_purpose: {} }
      else if (path === '/api/tokens/recent') body = { days: [] }
      else if (path === '/api/studio/activity') body = { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] }
      else if (path === '/api/integrations/slack/status') body = { configured: true, workspace_id: 'T_TEST', channel_ids: ['C_ONE'], outbound_messages: false, acknowledgements_enabled: false, acknowledgements_configured: false, acknowledgements: {}, promotion_acknowledgements: {} }
      else if (path === '/api/memory-inbox') {
        const params = url.searchParams
        listings.push(params.toString())
        let rows = ideas.filter(idea => (!params.get('status') || idea.status === params.get('status')) && (!params.get('priority') || idea.priority === params.get('priority')))
        rows = [...rows].sort((a, b) => (params.get('sort') === 'priority' ? RANK[a.priority] - RANK[b.priority] : 0) || b.created_at.localeCompare(a.created_at))
        const counts = { unreviewed: 0, promoted: 0, dismissed: 0 }
        for (const idea of ideas) counts[idea.status]++
        body = { ideas: rows, total: rows.length, offset: 0, counts }
      } else if (/^\/api\/memory-inbox\/[^/]+\/promote$/.test(path)) {
        const idea = ideas.find(item => item.id === path.split('/')[3])
        const sent = route.request().postDataJSON()
        promotes.push(sent)
        Object.assign(idea, { status: 'promoted', memory_id: 'decisions_9_1', memory_category: sent.category, reviewed_at: `${date}T18:00:00Z`, priority: sent.priority })
        body = { idea, memory_id: idea.memory_id }
      }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5193/')
    await page.getByRole('tab', { name: 'Memory', exact: true }).click()
    const memory = page.getByRole('region', { name: 'Memory log' })
    await memory.getByText('ship the cron contract first', { exact: true }).waitFor()

    // Priority is chosen at log time and sent with the promote request.
    const row = memory.getByRole('listitem').filter({ hasText: 'ship the cron contract first' })
    assert.equal(await row.getByLabel('Priority for capture idea-alp').inputValue(), 'normal')
    await row.getByLabel('Memory category for capture idea-alp').selectOption('decisions')
    await row.getByLabel('Priority for capture idea-alp').selectOption('urgent')
    await page.screenshot({ path: join(output, `priority-select-${width}.png`) })
    await row.getByRole('button', { name: 'Log to memory' }).click()
    await page.getByRole('status').filter({ hasText: 'at urgent priority' }).waitFor()
    assert.deepEqual(promotes.at(-1), { category: 'decisions', note: '', priority: 'urgent' })

    // Logged captures carry a priority badge; sorting by priority and filtering both reach the server.
    await memory.getByRole('tab', { name: /^Logged/ }).click()
    await memory.getByLabel('Priority Urgent').waitFor()
    await memory.getByLabel('Priority Low').waitFor()
    assert.deepEqual(await memory.locator('li[data-idea-priority]').evaluateAll(rows => rows.map(r => r.dataset.ideaPriority)), ['urgent', 'low'])
    await memory.getByLabel('Sort captures').selectOption('priority')
    await page.waitForFunction(() => document.body.innerText.includes('Highest priority first'))
    assert.ok(listings.some(query => query.includes('sort=priority')))
    await memory.getByLabel('Filter by priority').selectOption('low')
    await page.waitForFunction(() => document.querySelectorAll('li[data-idea-priority]').length === 1)
    assert.ok(listings.at(-1).includes('priority=low'))
    await memory.getByLabel('Priority Low').waitFor()
    await page.screenshot({ path: join(output, `priority-filter-${width}.png`) })
    await memory.getByLabel('Filter by priority').selectOption('')
    await page.waitForFunction(() => document.querySelectorAll('li[data-idea-priority]').length === 2)

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: priority at log time, badge, sort and filter passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
