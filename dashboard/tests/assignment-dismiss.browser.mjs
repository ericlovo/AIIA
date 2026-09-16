// Dismissing a failed assignment: the panel has to appear for a run with no work
// product, offer only the decisions that apply, and drop the record out of the
// attention count. Every response is synthetic; nothing reaches the Mini.
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

function seed() {
  return [
    {
      id: 'failed-1', agent_id: 'agent-1', title: 'Review verified cron contract patch',
      objective: 'Issue a GO or HOLD verdict.', status: 'failed', result: '', error: 'empty_agent_result',
      priority: 'normal', context: '', success_criteria: '', source_handoff_id: '',
      created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z`,
      review_status: 'unreviewed', review_note: '', review_version: 'v-failed-1', reviewed_at: null,
    },
    {
      id: 'done-1', agent_id: 'agent-1', title: 'Cron contract test slice',
      objective: 'Define the smallest PR-ready slice.', status: 'completed',
      result: 'Synthetic evidence only.', error: '',
      priority: 'normal', context: '', success_criteria: '', source_handoff_id: '',
      created_at: `${date}T11:00:00Z`, updated_at: `${date}T11:00:00Z`,
      review_status: 'unreviewed', review_note: '', review_version: 'v-done-1', reviewed_at: null,
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
    const reviewCalls = []
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
      else if (/^\/api\/assignments\/[^/]+\/review$/.test(path)) {
        const id = path.split('/')[3]
        const sent = route.request().postDataJSON()
        reviewCalls.push({ id, ...sent })
        const item = assignments.find(a => a.id === id)
        if (sent.expected_version !== item.review_version) {
          status = 409
          body = { detail: 'review_changed_refresh_required' }
        } else {
          // Mirror the server: the decision is recorded, the outcome is not rewritten.
          Object.assign(item, {
            review_status: sent.decision,
            review_note: sent.note ?? '',
            reviewed_at: sent.decision === 'unreviewed' ? null : `${date}T18:00:00Z`,
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

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5184/')

    // Attention starts at two: one failed run and one unreviewed output.
    const switchboard = page.getByRole('region', { name: 'Needs attention' })
    await switchboard.getByRole('heading', { name: /Needs attention \(2\)/ }).waitFor()

    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Review verified cron contract patch', { exact: true }).click()

    // The panel must exist for a failed run, which previously never showed one.
    const review = page.getByRole('region', { name: 'Artifact review' })
    await review.waitFor()
    await review.getByText('Failed run', { exact: true }).waitFor()
    assert.ok((await review.innerText()).includes('never claims the run succeeded'))

    // Only the decisions that apply to a run with no work product.
    await review.getByRole('button', { name: 'Dismiss' }).waitFor()
    assert.equal(await review.getByRole('button', { name: 'Accept output' }).count(), 0)
    assert.equal(await review.getByRole('button', { name: 'Reject output' }).count(), 0)
    await page.screenshot({ path: join(output, `failed-review-${width}.png`) })

    await review.getByRole('button', { name: 'Dismiss' }).click()
    await review.getByText('Dismissed', { exact: true }).waitFor()
    assert.equal(reviewCalls.length, 1)
    assert.equal(reviewCalls[0].decision, 'dismissed')
    assert.equal(reviewCalls[0].expected_version, 'v-failed-1')

    // The dismissed run leaves attention; the remaining unreviewed output stays.
    await page.getByRole('tab', { name: 'Switchboard', exact: true }).click()
    await page.getByRole('heading', { name: /Needs attention \(1\)/ }).waitFor()

    // Reopening puts it back, so dismissal is never a one-way door.
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Review verified cron contract patch', { exact: true }).click()
    await review.getByRole('button', { name: 'Reopen review' }).click()
    await review.getByText('Failed run', { exact: true }).waitFor()
    await page.getByRole('tab', { name: 'Switchboard', exact: true }).click()
    await page.getByRole('heading', { name: /Needs attention \(2\)/ }).waitFor()

    // A completed output still offers the full set, dismissal included.
    await page.getByRole('tab', { name: 'Assignments', exact: true }).click()
    await page.getByText('Cron contract test slice', { exact: true }).click()
    await review.getByRole('button', { name: 'Accept output' }).waitFor()
    await review.getByRole('button', { name: 'Reject output' }).waitFor()
    await review.getByRole('button', { name: 'Dismiss' }).waitFor()
    await page.screenshot({ path: join(output, `completed-review-${width}.png`) })

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: failed-run dismiss, attention drop, reopen, completed options passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
