import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-studio-ux')
const studioDist = process.env.STUDIO_DIST_DIR
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
const agentUsage = [
  { agent_id: agents[0].id, agent_name: agents[0].name, runs: 2, measured_runs: 1, input_tokens: 1234, output_tokens: 56 },
  { agent_id: agents[1].id, agent_name: agents[1].name, runs: 1, measured_runs: 1, input_tokens: 0, output_tokens: 0 },
  { agent_id: agents[2].id, agent_name: agents[2].name, runs: 3, measured_runs: 0, input_tokens: null, output_tokens: null },
]
const measuredRun = {
  id: 'synthetic-run', agent_id: agents[0].id, agent_name: agents[0].name,
  repo_id: 'aiia', at: `${date}T12:00:00Z`, status: 'completed', trigger: 'manual',
  assignment_id: '', model: 'qwen3:8b', latency_ms: 1000, legacy: 0,
  input_tokens: 1234, output_tokens: 56, result: 'Synthetic result', task: 'Synthetic task',
}

const makeIdeas = () => [
  { id: 'idea-one-00000000', text: '<@U0C1DCQFMRC> log this EPIC for LNS', source: 'slack', project: 'mindmoor', workspace_id: 'T_TEST', channel_id: 'C_ONE', author_id: 'U_AUTHOR', created_at: `${date}T17:22:31Z`, status: 'unreviewed', memory_id: '', memory_category: '', review_note: '', reviewed_at: '', acknowledgement_status: 'sent', acknowledgement_error: '', acknowledgement_ts: '1.1', promotion_status: null, promotion_error: null, promotion_ts: null },
  { id: 'idea-two-00000000', text: 'capture milestone from the slash command', source: 'slack', project: 'mindmoor', workspace_id: 'T_TEST', channel_id: 'C_ONE', author_id: 'U_AUTHOR', created_at: `${date}T16:43:24Z`, status: 'unreviewed', memory_id: '', memory_category: '', review_note: '', reviewed_at: '', acknowledgement_status: null, acknowledgement_error: null, acknowledgement_ts: null, promotion_status: null, promotion_error: null, promotion_ts: null },
  { id: 'idea-three-0000000', text: '<@U0C1DCQFMRC> channel verification test only', source: 'slack', project: 'mindmoor', workspace_id: 'T_TEST', channel_id: 'C_ONE', author_id: 'U_AUTHOR', created_at: `${date}T16:35:46Z`, status: 'dismissed', memory_id: '', memory_category: '', review_note: 'test noise', reviewed_at: `${date}T18:00:00Z`, acknowledgement_status: 'sent', acknowledgement_error: '', acknowledgement_ts: '1.2', promotion_status: null, promotion_error: null, promotion_ts: null },
]

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
    const ideas = makeIdeas()
    let promoteCalls = 0
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    if (studioDist) {
      await page.route('http://studio.test/', async route => route.fulfill({
        contentType: 'text/html',
        body: await readFile(join(studioDist, 'index.html')),
      }))
      await page.route('http://studio.test/assets/**', async route => {
        const asset = new URL(route.request().url()).pathname.slice(1)
        await route.fulfill({
          contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript',
          body: await readFile(join(studioDist, asset)),
        })
      })
    }
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
      else if (path === '/api/studio/activity') {
        const selected = new URL(route.request().url()).searchParams.get('agent_id')
        body = { today: date, start: '2026-06-17', days: [], agent_days: [], runs: [measuredRun], total: 6, matching: 6, imported: 0, usage_by_agent: agentUsage.filter(row => !selected || row.agent_id === selected) }
      }
      else if (path === '/api/studio/runs/synthetic-run') body = { run: measuredRun }
      else if (path === '/api/memory-inbox') {
        const wanted = new URL(route.request().url()).searchParams.get('status')
        const rows = ideas.filter(idea => !wanted || idea.status === wanted)
        const counts = { unreviewed: 0, promoted: 0, dismissed: 0 }
        for (const idea of ideas) counts[idea.status]++
        body = { ideas: rows, total: rows.length, offset: 0, counts }
      } else if (path.startsWith('/api/memory-inbox/')) {
        const [, , , id, action] = path.split('/')
        const idea = ideas.find(item => item.id === id)
        if (!idea) status = 404
        else if (action === 'promote') {
          promoteCalls++
          Object.assign(idea, { status: 'promoted', memory_id: 'decisions_9_1789', memory_category: route.request().postDataJSON().category, reviewed_at: `${date}T18:30:00Z`, promotion_status: idea.acknowledgement_status ? 'pending' : null })
          body = { idea, memory_id: idea.memory_id }
        } else if (action === 'dismiss') { Object.assign(idea, { status: 'dismissed', reviewed_at: `${date}T18:30:00Z` }); body = { idea } }
        else if (action === 'restore') { Object.assign(idea, { status: 'unreviewed', reviewed_at: '', review_note: '' }); body = { idea } }
        if (status === 404) body = { detail: 'idea_not_found' }
      } else if (path === '/api/integrations/slack/status') body = { configured: true, workspace_id: 'T_TEST', channel_ids: ['C_ONE', 'C_TWO'], outbound_messages: true, acknowledgements_enabled: true, acknowledgements_configured: true, acknowledgements: { sent: 2 }, promotion_acknowledgements: {} }
      else if (path === '/api/tasks') body = []
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto(studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/')
    await page.getByText('Token usage and agent attribution', { exact: true }).click()
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

    const attribution = page.getByRole('region', { name: 'Agent token attribution' })
    await attribution.getByRole('button', { name: 'CI specialist 1', exact: true }).waitFor()
    assert.ok((await attribution.innerText()).includes('1,234'))
    assert.ok((await attribution.innerText()).includes('0 / 3'))
    await attribution.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `agent-tokens-${width}.png`) })
    await attribution.getByRole('button', { name: 'CI specialist 1', exact: true }).click()
    await page.waitForFunction(() => document.querySelectorAll('[aria-label="Agent token attribution"] tbody tr').length === 1)
    await attribution.getByRole('button', { name: 'CI specialist 1', exact: true }).click()
    await page.getByRole('button').filter({ hasText: '1,290 tokens' }).click()
    const runInspector = page.getByRole('complementary', { name: 'Activity inspector' })
    await runInspector.getByText('Input tokens', { exact: true }).waitFor()
    assert.ok((await runInspector.innerText()).includes('1,234'))
    await runInspector.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `run-tokens-${width}.png`) })

    await page.getByRole('tab', { name: 'Memory', exact: true }).click()
    const memory = page.getByRole('region', { name: 'Memory log' })
    await memory.getByText('log this EPIC for LNS', { exact: true }).waitFor()
    assert.ok(!(await memory.innerText()).includes('<@U0C1DCQFMRC>'))
    assert.ok((await page.locator('main header').first().innerText()).includes('2 unreviewed'))
    assert.ok((await memory.innerText()).includes('Save receipt sent to Slack'))
    assert.ok((await memory.innerText()).includes('No Slack thread for save receipt'))
    await page.screenshot({ path: join(output, `memory-${width}.png`) })
    await memory.getByLabel('Memory category for capture idea-one').selectOption('decisions')
    await memory.getByRole('listitem').filter({ hasText: 'log this EPIC for LNS' }).getByRole('button', { name: 'Log to memory' }).click()
    await page.getByRole('status').filter({ hasText: 'Logged to AIIA memory as decisions' }).waitFor()
    await page.waitForFunction(() => document.querySelectorAll('[data-idea-status="unreviewed"]').length === 1)
    assert.equal(promoteCalls, 1)
    await memory.getByRole('tab', { name: /^Logged/ }).click()
    await memory.getByText('Memory receipt queued for Slack').waitFor()
    assert.ok((await memory.innerText()).includes('decisions · decisions_9_1789'))
    await memory.getByRole('tab', { name: /^Dismissed/ }).click()
    await memory.getByRole('button', { name: 'Restore' }).click()
    await page.getByRole('status').filter({ hasText: 'Capture restored to Unreviewed' }).waitFor()
    await page.screenshot({ path: join(output, `memory-logged-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)

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
    console.log(`${width}px: platform/agent/run usage, attribution filtering, memory log/promote/restore, refresh failure/recovery, 96 nodes, zoom, inspector, keyboard save, save failure/recovery passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
