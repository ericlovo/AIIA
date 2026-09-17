// Map relationships: the handoff wire creates the handoff in place, edges can be
// inspected and removed, and suites colour, filter and bulk-tune the Map.
// Every response is synthetic; nothing reaches a running Command Center.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-map-relationships')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-16'

const agentBase = {
  persona: '', skills: ['Analysis'], tools: [], repo_id: 'aiia', temperature: 0.35, max_tokens: 1200,
  loop_enabled: false, loop_interval_minutes: 60, loop_task: '', loop_max_runs_per_day: 4, loop_runs_today: 0,
  loop_day: date, memory_namespace: '', status: 'idle', last_run_at: null, last_result: '', last_error: '',
  runs: [], created_at: `${date}T10:00:00Z`, updated_at: `${date}T10:00:00Z`,
}
function seedAgents() {
  return [
    { ...agentBase, id: 'agent-scout', name: 'Signal Scout', mission: 'Find signal in the repository.', suite: 'research', loop_task: 'Scan for changes.' },
    { ...agentBase, id: 'agent-writer', name: 'Brief Writer', mission: 'Turn findings into briefs.', suite: 'research' },
    { ...agentBase, id: 'agent-gate', name: 'Review Gate', mission: 'Gate risky changes.', suite: 'review' },
    { ...agentBase, id: 'agent-solo', name: 'Solo Operator', mission: 'Handle one-off work.', suite: '' },
  ]
}
const assignmentBase = {
  priority: 'normal', context: '', success_criteria: '', source_handoff_id: '', review_status: 'unreviewed',
  error: '', created_at: `${date}T11:00:00Z`, updated_at: `${date}T11:00:00Z`, started_at: null, completed_at: null,
}
function seedAssignments() {
  return [
    { ...assignmentBase, id: 'asg-scan', agent_id: 'agent-scout', title: 'Scan repository', objective: 'Scan it.', status: 'completed', result: 'Synthetic findings only.' },
    { ...assignmentBase, id: 'asg-draft', agent_id: 'agent-writer', title: 'Draft summary', objective: 'Draft it.', status: 'queued', result: '' },
  ]
}
function seedHandoffs() {
  return [{
    id: 'hof-existing', source_assignment_id: 'asg-draft', target_assignment_id: 'asg-gate-review', from_agent_id: 'agent-writer',
    to_agent_id: 'agent-gate', artifact_type: 'brief', artifact: 'Synthetic.', instructions: 'Review the synthetic draft.',
    status: 'queued', created_at: `${date}T12:05:00Z`, updated_at: `${date}T12:05:00Z`,
  }]
}

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    const agents = seedAgents()
    const assignments = seedAssignments()
    let handoffs = seedHandoffs()
    const handoffPosts = []
    const handoffDeletes = []
    let failNextHandoff = true
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const request = route.request()
      const path = new URL(request.url()).pathname
      const method = request.method()
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments }
      else if (path === '/api/handoffs' && method === 'POST') {
        const sent = request.postDataJSON()
        handoffPosts.push(sent)
        if (failNextHandoff) {
          failNextHandoff = false
          status = 422
          body = { detail: 'handoff_capacity_reached' }
        } else {
          const target = { ...assignmentBase, id: 'asg-handoff-new', agent_id: sent.to_agent_id, title: 'Handoff: Scan repository', objective: sent.instructions, status: 'queued', result: '', source_handoff_id: 'hof-new' }
          const source = assignments.find(item => item.id === sent.source_assignment_id)
          const handoff = { id: 'hof-new', source_assignment_id: sent.source_assignment_id, target_assignment_id: target.id, from_agent_id: source.agent_id, to_agent_id: sent.to_agent_id, artifact_type: sent.artifact_type, artifact: source.result, instructions: sent.instructions, status: 'queued', created_at: `${date}T13:30:00Z`, updated_at: `${date}T13:30:00Z` }
          assignments.push(target)
          handoffs = [handoff, ...handoffs]
          body = { handoff, assignment: target }
        }
      }
      else if (path.startsWith('/api/handoffs/') && method === 'DELETE') {
        const id = path.split('/')[3]
        handoffDeletes.push(id)
        handoffs = handoffs.filter(item => item.id !== id)
        body = { deleted: true }
      }
      else if (path === '/api/handoffs') body = { handoffs }
      else if (path === '/api/agent-world/layout') body = { layout: { version: 1, revision: 0, positions: {}, updated_at: null } }
      else if (path === '/api/git-workspaces') body = { workspaces: [] }
      else if (path === '/api/git-writes') body = { writes: [] }
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

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5192/')
    await page.getByRole('tab', { name: 'Map', exact: true }).click()
    await page.getByRole('checkbox', { name: 'Completed assignments', exact: true }).check()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 6)
    await page.locator('[data-edge="handoff:hof-existing"]').waitFor({ state: 'attached' })

    // A6: the wire opens an inline confirm, not the Handoffs tab.
    const wire = page.getByRole('button', { name: 'Wire Scan repository to another agent' })
    if (width >= 1024) {
      await page.getByRole('button', { name: 'Fit map', exact: true }).click()
      const from = await wire.boundingBox()
      const to = await page.locator('[data-agent-target="agent-gate"]').boundingBox()
      await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2)
      await page.mouse.down()
      await page.mouse.move((from.x + to.x) / 2, (from.y + to.y) / 2, { steps: 6 })
      await page.mouse.move(to.x + to.width / 2, to.y + to.height / 2, { steps: 6 })
      await page.getByText('Drop on a target agent').waitFor()
      await page.mouse.up()
    } else {
      await wire.focus()
      await page.keyboard.press('Enter')
      await page.getByText('Select a target agent for “Scan repository”').waitFor()
      await page.locator('[data-agent-target="agent-gate"]').focus()
      await page.keyboard.press('Enter')
    }
    const composer = page.getByRole('complementary', { name: 'Confirm handoff' })
    await composer.waitFor()
    assert.ok((await composer.innerText()).includes('Signal Scout → Review Gate'))
    await page.getByRole('heading', { name: 'Agent control map' }).waitFor()
    const instructions = composer.getByRole('textbox')
    assert.ok((await instructions.inputValue()).includes('Scan repository'))
    await instructions.fill('Gate the synthetic findings before anyone acts on them.')
    await page.screenshot({ path: join(output, `handoff-confirm-${width}.png`) })
    await composer.getByRole('button', { name: 'Create handoff' }).click()
    await composer.getByRole('alert').filter({ hasText: 'The handoff ledger is full' }).waitFor()
    assert.equal(await page.locator('[data-edge="handoff:hof-new"]').count(), 0)
    await composer.getByRole('button', { name: 'Create handoff' }).click()
    await composer.waitFor({ state: 'detached' })
    await page.locator('[data-edge="handoff:hof-new"]').waitFor({ state: 'attached' })
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 7)
    assert.equal(handoffPosts.length, 2)
    assert.deepEqual(handoffPosts[1], {
      source_assignment_id: 'asg-scan', to_agent_id: 'agent-gate', artifact_type: 'brief',
      instructions: 'Gate the synthetic findings before anyone acts on them.',
    })
    await page.getByRole('heading', { name: 'Agent control map' }).waitFor()
    await page.screenshot({ path: join(output, `handoff-created-${width}.png`) })

    // A6b: the created edge is selected and describes itself.
    const edgeControls = page.getByRole('complementary', { name: 'Handoff controls' })
    await edgeControls.waitFor()
    let edgeText = await edgeControls.innerText()
    assert.ok(edgeText.includes('Signal Scout · Scan repository'), edgeText)
    assert.ok(edgeText.includes('Review Gate · Handoff: Scan repository'), edgeText)
    assert.ok(edgeText.includes('2026-09-16 13:30 UTC'), edgeText)
    await edgeControls.getByRole('button', { name: 'Close handoff controls' }).click()
    await edgeControls.waitFor({ state: 'detached' })

    // Keyboard reaches an existing handoff edge; removal needs a confirm.
    const existingEdge = page.getByRole('button', { name: 'Handoff from Brief Writer to Review Gate, queued, created 2026-09-16 12:05 UTC' })
    await existingEdge.focus()
    await page.keyboard.press('Enter')
    await edgeControls.waitFor()
    edgeText = await edgeControls.innerText()
    assert.ok(edgeText.includes('Brief Writer → Review Gate'), edgeText)
    assert.ok(edgeText.includes('queued'), edgeText)
    assert.ok(edgeText.includes('Brief Writer · Draft summary'), edgeText)
    await edgeControls.getByRole('button', { name: 'Remove handoff' }).click()
    await edgeControls.getByText('Remove this handoff? The target assignment stays in the queue.').waitFor()
    await page.screenshot({ path: join(output, `handoff-remove-confirm-${width}.png`) })
    await edgeControls.getByRole('button', { name: 'Keep handoff' }).click()
    assert.deepEqual(handoffDeletes, [])
    await edgeControls.getByRole('button', { name: 'Remove handoff' }).click()
    await edgeControls.getByRole('button', { name: 'Confirm remove' }).click()
    await edgeControls.waitFor({ state: 'detached' })
    await page.locator('[data-edge="handoff:hof-existing"]').waitFor({ state: 'detached' })
    assert.deepEqual(handoffDeletes, ['hof-existing'])
    assert.equal(await page.locator('[data-edge="handoff:hof-new"]').count(), 1)

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)

    // A7: suites carry one colour per suite, and the legend filters the Map.
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 7)
    const badgeColours = await page.locator('[data-suite-badge]').evaluateAll(nodes => nodes.map(node => [node.dataset.suiteBadge, getComputedStyle(node).color]))
    assert.equal(badgeColours.length, 3)
    const research = badgeColours.filter(([slug]) => slug === 'research').map(([, colour]) => colour)
    const review = badgeColours.filter(([slug]) => slug === 'review').map(([, colour]) => colour)
    assert.equal(research.length, 2)
    assert.equal(research[0], research[1])
    assert.notEqual(research[0], review[0])
    assert.equal(await page.locator('[data-agent-target="agent-solo"] [data-suite-badge]').count(), 0)
    const legend = page.getByRole('group', { name: 'Suite legend' })
    await legend.getByRole('button', { name: 'research suite, 2 agents' }).click()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 4)
    assert.equal(await legend.getByRole('button', { name: 'research suite, 2 agents' }).getAttribute('aria-pressed'), 'true')
    assert.equal(await page.locator('[data-agent-target="agent-gate"]').count(), 0)
    assert.equal(await page.locator('[data-agent-target="agent-solo"]').count(), 0)
    assert.equal(await page.locator('[data-edge="handoff:hof-new"]').count(), 0)
    await page.getByText('2 of 4 agents', { exact: false }).waitFor()
    await page.screenshot({ path: join(output, `suite-filter-${width}.png`) })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await legend.getByRole('button', { name: 'All agents' }).click()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 7)
    assert.equal(await page.locator('[data-edge="handoff:hof-new"]').count(), 1)

    // Selecting an agent-to-assignment edge opens that assignment.
    await page.locator('[data-edge="hierarchy:asg-draft"]').click()
    await page.getByRole('heading', { name: 'Assignment queue' }).waitFor()
    await page.getByText('Assignment controls', { exact: true }).waitFor()
    await page.locator('div.text-lg', { hasText: 'Draft summary' }).waitFor()

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: wire to confirmed handoff, inline error, edge inspect, keyboard edge remove with confirm, edge opens assignment, suite colours and filter passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
