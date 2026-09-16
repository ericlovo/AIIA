// Dismissal is its own decision, separate from the review verdict. A failed run
// has no verdict to give but must still be clearable; rejected work must keep
// its rejection through a dismiss and a restore. Every response is synthetic.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-assignment-dismiss')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-16'
const agents = [{
  id: 'agent-1', name: 'Mindmoor Cron Review Gate', mission: 'Gate cron contract patches.',
  status: 'idle', skills: ['Analysis'], tools: [], repo_id: 'aiia',
  loop_enabled: false, loop_max_runs_per_day: 4, runs: [],
}]

const base = {
  agent_id: 'agent-1', priority: 'normal', context: '', success_criteria: '',
  source_handoff_id: '', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z`,
  review_note: '', reviewed_at: null, dismissed_at: null, dismiss_note: '',
}
function seed() {
  return [
    {
      ...base, id: 'failed-1', title: 'Review verified cron contract patch',
      objective: 'Issue a GO or HOLD verdict.', status: 'failed', result: '',
      error: 'empty_agent_result', review_status: 'unreviewed', review_version: 'v-failed-1',
    },
    {
      ...base, id: 'rejected-1', title: 'Cron contract test slice',
      objective: 'Define the smallest PR-ready slice.', status: 'completed',
      result: 'Synthetic evidence only.', error: '', review_status: 'rejected',
      review_note: 'Truncated and invented a helper.', reviewed_at: `${date}T13:00:00Z`,
      review_version: 'v-rejected-1',
    },
  ]
}

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    let assignments = seed()
    const dismissCalls = []
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/git-workspaces') body = { workspaces: [] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (/^\/api\/assignments\/[^/]+\/dismiss$/.test(path)) {
        const id = path.split('/')[3]
        const sent = route.request().postDataJSON()
        dismissCalls.push({ id, ...sent })
        const item = assignments.find(a => a.id === id)
        if (sent.expected_version !== item.review_version) {
          status = 409
          body = { detail: 'review_changed_refresh_required' }
        } else {
          // Mirror the server: only the tracking fields move.
          Object.assign(item, {
            dismissed_at: sent.dismissed ? `${date}T19:00:00Z` : null,
            dismiss_note: sent.dismissed ? (sent.note ?? '') : '',
            review_version: `${item.review_version}-next`,
          })
          body = { assignment: item }
        }
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

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5187/')
    await page.getByRole('heading', { name: /Needs attention \(2\)/ }).waitFor()

    // A failed run has no work product, so it gets tracking but no review panel.
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Review verified cron contract patch', { exact: true }).click()
    const tracking = page.getByRole('region', { name: 'Attention tracking' })
    await tracking.waitFor()
    assert.equal(await page.getByRole('region', { name: 'Artifact review' }).count(), 0)
    assert.ok((await tracking.innerText()).includes('without judging the work'))
    await page.screenshot({ path: join(output, `failed-tracking-${width}.png`) })

    await tracking.getByRole('button', { name: 'Dismiss' }).click()
    await tracking.getByText('Dismissed', { exact: true }).waitFor()
    assert.equal(dismissCalls.at(-1).dismissed, true)
    assert.equal(dismissCalls.at(-1).expected_version, 'v-failed-1')

    await page.getByRole('tab', { name: 'Switchboard', exact: true }).click()
    await page.getByRole('heading', { name: /Needs attention \(1\)/ }).waitFor()

    // Rejected work keeps its verdict through dismissal; both states show together.
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Cron contract test slice', { exact: true }).click()
    const review = page.getByRole('region', { name: 'Artifact review' })
    await review.getByText('Rejected output', { exact: true }).waitFor()
    await page.getByRole('region', { name: 'Attention tracking' }).getByRole('button', { name: 'Dismiss' }).click()
    await page.getByRole('region', { name: 'Attention tracking' }).getByText('Dismissed', { exact: true }).waitFor()
    // The verdict panel is untouched by the dismissal.
    await review.getByText('Rejected output', { exact: true }).waitFor()
    assert.equal(assignments.find(a => a.id === 'rejected-1').review_status, 'rejected')
    assert.equal(assignments.find(a => a.id === 'rejected-1').review_note, 'Truncated and invented a helper.')
    await page.screenshot({ path: join(output, `rejected-dismissed-${width}.png`) })

    await page.getByRole('tab', { name: 'Switchboard', exact: true }).click()
    await page.getByRole('heading', { name: /Needs attention \(0\)/ }).waitFor()

    // Restoring brings it back to attention still carrying the rejection.
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Cron contract test slice', { exact: true }).click()
    await page.getByRole('region', { name: 'Attention tracking' }).getByRole('button', { name: 'Restore to attention' }).click()
    await page.getByRole('region', { name: 'Attention tracking' }).getByText('In the attention list', { exact: true }).waitFor()
    await review.getByText('Rejected output', { exact: true }).waitFor()
    await page.getByRole('tab', { name: 'Switchboard', exact: true }).click()
    await page.getByRole('heading', { name: /Needs attention \(1\)/ }).waitFor()

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: failed dismiss, verdict survives dismiss and restore, attention counts passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
