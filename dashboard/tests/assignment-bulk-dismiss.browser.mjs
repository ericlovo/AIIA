// Bulk dismissal in Activity history: a person ticks items, gives a reason, and dismisses them
// together or not at all. Dismissal never records a verdict, a changed item stops
// the whole batch, and the error names it. Every response is synthetic.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-assignment-bulk-dismiss')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-24'
const agents = [
  { id: 'agent-ci', name: 'CI Signal Officer', mission: 'Report CI state.', status: 'idle', skills: [], tools: [], repo_id: 'aiia', loop_enabled: true, loop_max_runs_per_day: 12, loop_interval_minutes: 30, loop_runs_today: 0, runs: [] },
  { id: 'agent-dw', name: 'Mindmoor Delivery Watch', mission: 'Watch delivery.', status: 'idle', skills: [], tools: [], repo_id: 'mindmoor', loop_enabled: true, loop_max_runs_per_day: 8, loop_interval_minutes: 60, loop_runs_today: 0, runs: [] },
]

function seed() {
  const items = []
  for (let i = 0; i < 12; i++) {
    const agent = i < 8 ? agents[0] : agents[1]
    items.push({
      id: `asg-${i}`, title: `Scheduled: ${agent.name} #${i}`, objective: 'Inspect.', agent_id: agent.id,
      priority: 'normal', context: '', success_criteria: '', source_handoff_id: '', trigger: 'interval',
      source_kind: 'loop_schedule', status: 'completed', result: `Report ${i}`, error: '',
      created_at: `${date}T0${i % 10}:00:00Z`, updated_at: `${date}T0${i % 10}:00:00Z`, started_at: null, completed_at: null,
      review_status: i === 3 ? 'rejected' : 'unreviewed', review_note: i === 3 ? 'Invented a failure.' : '', reviewed_at: null,
      review_version: `v-${i}`, dismissed_at: null, dismiss_note: '',
    })
  }
  // Still running: never in attention, never selectable.
  items.push({ ...items[0], id: 'asg-running', title: 'Running work', status: 'running', result: '', review_version: 'v-run' })
  return items
}

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let assignments = seed()
    const bulkCalls = []
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments }
      else if (path === '/api/assignments/dismiss') {
        const sent = route.request().postDataJSON()
        bulkCalls.push(sent)
        // Mirror the server: validate every item first, then change all or none.
        const blocking = sent.items.find(pick => assignments.find(a => a.id === pick.id)?.review_version !== pick.expected_version)
        if (blocking) {
          status = 409
          body = { detail: `review_changed_refresh_required:${blocking.id}` }
        } else {
          for (const pick of sent.items) {
            Object.assign(assignments.find(a => a.id === pick.id), { dismissed_at: `${date}T19:00:00Z`, dismiss_note: sent.note, review_version: `${pick.expected_version}-next` })
          }
          body = { assignments: [], dismissed: sent.items.length }
        }
      }
      else if (path === '/api/handoffs') body = { handoffs: [] }
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

    await page.goto(`${process.env.STUDIO_URL || 'http://127.0.0.1:5188/'}#/today`)
    await page.getByRole('heading', { name: /Needs attention \(12\)/ }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Select to dismiss' }).count(), 0)
    await page.getByRole('region', { name: 'Needs attention', exact: true }).getByRole('link', { name: 'Review all', exact: true }).click()
    await page.getByRole('heading', { name: 'Activity history', exact: true }).waitFor()
    assert.equal(await page.evaluate(() => window.location.hash), '#/history?attention=1')
    const attention = page.getByRole('region', { name: 'Needs attention' })
    assert.equal(await attention.evaluate(element => document.activeElement === element), true)
    // Rows by title: the list sorts rejected work first, so positions are not stable.
    const row = title => attention.getByRole('checkbox', { name: new RegExp(`^${title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?!\\d)`) })

    await attention.getByRole('button', { name: 'Select to dismiss' }).click()
    const toolbar = attention.getByRole('group', { name: 'Dismiss selected assignments' })
    const submit = toolbar.getByRole('button', { name: /^Dismiss \d+ items?$/ })
    assert.ok((await toolbar.innerText()).includes('nothing is accepted'))
    assert.equal(await attention.getByRole('checkbox').count(), 12) // running work is not offered
    assert.equal(await submit.isDisabled(), true)

    // Ticking alone is not enough: a reason is required.
    await row(assignments[0].title).check()
    await row(assignments[1].title).check()
    await toolbar.getByText('2 selected').waitFor()
    assert.equal(await toolbar.getByRole('button', { name: 'Dismiss 2 items' }).isDisabled(), true)
    await toolbar.getByLabel(/Reason/).fill('Stale pre-guard backlog')
    assert.equal(await toolbar.getByRole('button', { name: 'Dismiss 2 items' }).isDisabled(), false)

    // One ticked item changes before submit: nothing is dismissed and it is named.
    const ticked = assignments[0]
    ticked.review_version = 'v-changed-elsewhere'
    const refetched = page.waitForResponse(response => new URL(response.url()).pathname === '/api/assignments' && response.request().method() === 'GET' && bulkCalls.length > 0)
    await toolbar.getByRole('button', { name: 'Dismiss 2 items' }).click()
    const alert = toolbar.getByRole('alert')
    await alert.waitFor()
    await refetched
    assert.ok((await alert.innerText()).startsWith('Nothing was dismissed.'))
    assert.ok((await alert.innerText()).includes(ticked.title))
    assert.equal(assignments.filter(a => a.dismissed_at).length, 0)
    // The changed item is unticked; the untouched one stays selected for the retry.
    await toolbar.getByText('1 selected').waitFor()
    assert.equal(await row(ticked.title).isChecked(), false)
    assert.equal(await row(assignments[1].title).isChecked(), true)
    await page.screenshot({ path: join(output, `bulk-refused-${width}.png`), fullPage: true })

    // Re-tick everything (fresh versions) and dismiss the lot.
    await toolbar.getByRole('button', { name: 'Clear' }).click()
    await toolbar.getByRole('button', { name: 'Select all shown (12)' }).click()
    await toolbar.getByText('12 selected').waitFor()
    assert.equal(await toolbar.getByRole('alert').count(), 0) // the old refusal does not linger
    await page.screenshot({ path: join(output, `bulk-selected-${width}.png`), fullPage: true })
    await toolbar.getByRole('button', { name: 'Dismiss 12 items' }).click()
    await attention.getByText('Dismissed 12 items. No verdict was recorded.').waitFor()
    await page.getByRole('heading', { name: /Needs attention \(0\)/ }).waitFor()

    const last = bulkCalls.at(-1)
    assert.equal(last.note, 'Stale pre-guard backlog')
    assert.equal(last.items.length, 12)
    // Every item went up with the version currently shown, including the changed one.
    assert.ok(last.items.every(pick => pick.expected_version === (pick.id === ticked.id ? 'v-changed-elsewhere' : `v-${pick.id.split('-')[1]}`)))
    assert.equal(assignments.find(a => a.id === 'asg-3').review_status, 'rejected') // verdict survives
    assert.equal(assignments.filter(a => a.review_status === 'accepted').length, 0) // nothing accepted
    assert.equal(assignments.find(a => a.id === 'asg-running').dismissed_at, null)

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: reason required, changed item refuses the batch and is named, all-or-nothing dismiss, verdicts kept`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
