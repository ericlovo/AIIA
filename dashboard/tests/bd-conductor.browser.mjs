// Synthetic fixtures only. Actual runtime integration is exercised separately.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { chromium } from 'playwright'
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-bd-ui')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const budget = { tokens: { used: 1250, limit: 40000 }, calls: { used: 4, limit: 100 }, seconds: { used: 40, limit: 3600 } }
const at = new Date().toISOString()
const bots = ['Healthcare', 'PE technical diligence', 'Alumni & associations', 'Client BD'].map((name, i) => ({ id: `bot-${i}`, name, segment: name, tenant_id: `synthetic-${i}`, counts: { queued: 2, running: i === 0 ? 1 : 0, completed: 1, failed: 0 }, budgets: budget }))
const snapshot = {
  mode: 'replay', schedule_enabled: false, kill_switch: false,
  status_banner: { state: 'paused_schedule_off', label: 'Paused: schedule off' },
  daily_summary: { day: '2026-10-27', timezone: 'America/Chicago', candidates_proposed: 1, inconclusive: 2, failed: 0, awaiting_human_review: 1 },
  bots, budget,
  jobs: [{ id: 'queue-1', bot_id: 'bot-1', tenant_id: 'synthetic-1', segment: 'PE technical diligence', state: 'queued', step: 'queued', created_at: at, started_at: null, heartbeat_at: null }, { id: 'job-1', bot_id: 'bot-0', tenant_id: 'synthetic-0', segment: 'Healthcare', state: 'running', step: 'Extracting public evidence', created_at: at, started_at: at, heartbeat_at: at }],
  events: [{ id: 'event-1', kind: 'job_running', message: 'Research started from recorded public evidence.', created_at: at, bot_id: 'bot-0', job_id: 'job-1' }],
  packets: [{ id: 'packet-1', index: 0, candidate_status: 'supported', candidate_reason: 'source_supported_candidate', job_id: 'previous-job', bot_id: 'bot-0', tenant_id: 'synthetic-0', segment: 'Healthcare', created_at: at, mode: 'replay', review_status: 'human_review_required', organization: 'Synthetic Health Group', source_url: 'https://example.com/research', source_quote: 'Synthetic source quotation for review.', emails: ['research@example.com'], phones: ['+1 202-555-0100'], signals: ['Expansion announcement in recorded source.'], insights: ['Explore demand for intake automation.'], summary: 'Synthetic evidence packet for review. This is not a real prospect.' }],
  health: { qwen: { status: 'replay', detail: 'Recorded extraction' }, jev: { status: 'replay', detail: 'Recorded verdicts' }, managed_leads: { status: 'data_unavailable', detail: 'Staging API is not connected' } },
  model_activity: { qwen: { status: 'replay', active_jobs: [], calls: 0, completed: 0, replayed: 3, errors: 0, last_event_at: at, latency_ms: null }, jev: { status: 'replay', active_jobs: [], calls: 0, completed: 0, replayed: 2, errors: 0, last_event_at: at, latency_ms: null } },
  jev: { calls_per_minute: 2, verdict_mix: { Choice: 1, Score: 1, Noul: 0 }, latency_ms: null, errors: null },
}
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } })
    const page = await context.newPage()
    const errors = [], posts = []
    let current = structuredClone(snapshot), unavailable = false, actionFail = false
    page.on('pageerror', error => errors.push(error.message))
    await context.addInitScript(() => {
      class Source {
        constructor() { window.bdSource = this; setTimeout(() => this.onopen?.(), 50) }
        addEventListener(name, listener) { this[name] = listener }
        close() {}
      }
      window.EventSource = Source
    })
    await context.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let data = {}
      if (path === '/api/bd/snapshot') {
        if (unavailable) return route.fulfill({ status: 503, json: { detail: 'fixture_data_unavailable' } })
        data = current
      } else if (path.startsWith('/api/bd/') && route.request().method() === 'POST') {
        posts.push({ path, body: route.request().postDataJSON() })
        if (actionFail) return route.fulfill({ status: 503, json: { detail: 'fixture_action_unavailable' } })
        if (path.endsWith('/review')) current.packets[0].review_status = route.request().postDataJSON().status
        if (path.endsWith('kill-switch')) {
          current.kill_switch = route.request().postDataJSON().enabled
          current.status_banner = current.kill_switch
            ? { state: 'paused_kill_switch', label: 'Paused: kill switch on' }
            : current.schedule_enabled
              ? { state: 'running', label: 'Running' }
              : { state: 'paused_schedule_off', label: 'Paused: schedule off' }
        }
        if (path.endsWith('schedule')) {
          current.schedule_enabled = route.request().postDataJSON().enabled
          current.status_banner = current.kill_switch
            ? { state: 'paused_kill_switch', label: 'Paused: kill switch on' }
            : current.schedule_enabled
              ? { state: 'running', label: 'Running' }
              : { state: 'paused_schedule_off', label: 'Paused: schedule off' }
        }
        data = { ok: true }
      } else if (path === '/api/agents') data = { agents: [] }
      else if (path === '/api/agents/resources') data = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/agents/models') data = { models: [], default: '' }
      else if (path === '/api/health') data = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') data = { services: {} }
      else if (path === '/api/integrations/typesafe/status') data = { ready: false }
      return route.fulfill({ json: data })
    })
    await page.goto(`${process.env.STUDIO_URL || 'http://127.0.0.1:5187/'}#/bd`)
    await page.getByRole('heading', { name: 'The research desk.' }).waitFor()
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Paused: schedule off', { exact: true }).waitFor()
    await page.getByRole('region', { name: 'Daily research summary' }).getByText('Candidates proposed', { exact: true }).waitFor()
    await page.locator('.bd-packet').getByText('Synthetic Health Group', { exact: true }).waitFor()
    await page.getByText('Live', { exact: true }).waitFor()
    await page.getByRole('region', { name: 'Work queue', exact: true }).getByText('PE technical diligence', { exact: true }).first().waitFor()
    assert.equal(await page.locator('.bd-model-working').count(), 0, 'replay never claims active inference')
    await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('dismissed')
    await page.getByRole('region', { name: 'Completed research', exact: true }).getByRole('button', { name: 'Review packet' }).click()
    assert.equal(await page.locator('.bd-packet').evaluate(el => document.activeElement === el), true, 'review jump focuses the evidence card')
    await page.getByText('Synthetic source quotation for review.', { exact: true }).waitFor({ state: 'visible' })
    await page.screenshot({ path: join(output, `bd-review-open-${width}.png`), animations: 'disabled' })
    await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bd-action-${width}.png`), fullPage: true })
    assert.equal(await page.getByRole('link', { name: 'Research', exact: true }).getAttribute('aria-current'), 'page')
    assert.equal(await page.getByLabel('Jev router', { exact: true }).getByText('Unavailable').count(), 2)
    assert.equal(await page.locator('.bd-packet details').getAttribute('open'), '')
    assert.equal(await page.getByRole('link', { name: 'Open source evidence' }).getAttribute('href'), 'https://example.com/research')
    await page.getByText('Email: research@example.com', { exact: true }).waitFor()
    await page.screenshot({ path: join(output, `bd-evidence-${width}.png`), fullPage: true })
    await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bd-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.equal(await page.locator('.bd-page').evaluate(el => el.scrollWidth > el.clientWidth), false)
    await page.getByRole('combobox', { name: 'Show' }).selectOption('bot-1')
    await page.getByText('No packets in this view yet.', { exact: false }).waitFor()
    await page.getByRole('combobox', { name: 'Show' }).selectOption('all')
    await page.getByRole('button', { name: 'Pause feed' }).click()
    const next = structuredClone(current)
    next.events.unshift({ ...next.events[0], id: 'event-2', kind: 'packet_assembled', message: 'SSE packet arrived without refreshing.' })
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), next)
    assert.equal(await page.getByText('SSE packet arrived without refreshing.').count(), 0)
    await page.getByRole('button', { name: 'Resume feed' }).click()
    await page.getByText('SSE packet arrived without refreshing.').waitFor()
    await page.getByRole('button', { name: 'Activate daily research' }).click()
    await page.getByText('Daily research schedule is on.', { exact: false }).waitFor()
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Running', { exact: true }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/schedule', body: { enabled: true } })
    await page.getByRole('button', { name: 'Pause daily schedule' }).click()
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Paused: schedule off', { exact: true }).waitFor()
    await page.getByRole('button', { name: 'Run shadow batch' }).click()
    await page.getByText('Shadow batch accepted.', { exact: false }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/shadow', body: {} })
    await page.getByRole('button', { name: 'Stop conductor' }).click()
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Paused: kill switch on', { exact: true }).waitFor()
    await page.getByRole('button', { name: 'Release kill switch' }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run shadow batch' }).isDisabled(), true)
    await page.getByRole('button', { name: 'Release kill switch' }).click()
    await page.getByRole('button', { name: 'Stop conductor' }).waitFor()
    await page.getByRole('button', { name: 'Research next', exact: true }).click()
    await page.getByText('Local research triage saved.', { exact: true }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/packets/previous-job/0/review', body: { status: 'research_next' } })
    await page.getByRole('button', { name: 'Dismiss', exact: true }).click()
    await page.getByText('No packets in this view yet.', { exact: false }).waitFor()
    await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('dismissed')
    await page.getByRole('button', { name: 'Restore', exact: true }).click()
    await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('human_review_required')
    await page.locator('.bd-packet').getByText('Synthetic Health Group', { exact: true }).waitFor()
    const live = structuredClone(current)
    live.mode = 'live'
    live.health.qwen.status = 'available'
    live.health.jev.status = 'available'
    live.jobs[1].step = 'qwen_called'
    live.jobs[1].heartbeat_at = new Date().toISOString()
    live.model_activity.qwen = { ...live.model_activity.qwen, status: 'running', calls: 1, active_jobs: ['job-1'] }
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.locator('.bd-model-qwen.bd-model-working').waitFor()
    assert.equal(await page.locator('.bd-model-jev.bd-model-working').count(), 0)
    assert.equal(await page.getByRole('button', { name: 'Run shadow batch' }).isDisabled(), true)
    await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bd-qwen-working-${width}.png`), fullPage: true })
    live.model_activity.qwen.active_jobs = []
    live.model_activity.qwen.completed = 1
    live.model_activity.jev.active_jobs = ['job-1']
    live.jobs[1].step = 'jev_called'
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.locator('.bd-model-jev.bd-model-working').waitFor()
    assert.equal(await page.locator('.bd-model-qwen.bd-model-working').count(), 0)
    live.jobs[1].state = 'completed'
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.waitForFunction(() => document.querySelectorAll('.bd-model-working').length === 0)
    live.jobs = []
    live.kill_switch = true
    live.schedule_enabled = true
    live.schedule = { state: 'waiting', start_at: new Date(Date.now() + 3600000).toISOString(), next_run_at: new Date(Date.now() + 3600000).toISOString(), stop_at: new Date(Date.now() + 172800000).toISOString(), timezone: 'America/Chicago', display_only: true }
    live.health.qwen = { status: 'stale', detail: 'Last successful observation is old', observed_at: new Date(Date.now() - 10800000).toISOString(), last_status: 'available' }
    live.health.jev = { status: 'data_unavailable', detail: 'Provider returned an error', observed_at: new Date(Date.now() - 60000).toISOString(), last_status: 'data_unavailable' }
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.locator('.bd-flow-state').getByText('Waiting for planned start', { exact: true }).waitFor()
    const schedulePanel = page.getByRole('region', { name: 'Research schedule', exact: true })
    await schedulePanel.getByText('Planned first run', { exact: true }).waitFor()
    await schedulePanel.getByText('Scheduled stop', { exact: true }).waitFor()
    assert.match(await schedulePanel.textContent(), /Currently paused/)
    assert.match(await schedulePanel.textContent(), /Planned times do not resume paused work/)
    assert.match(await schedulePanel.textContent(), /CDT|CST/)
    await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bd-waiting-${width}.png`), animations: 'disabled' })
    await schedulePanel.screenshot({ path: join(output, `bd-schedule-${width}.png`), animations: 'disabled' })
    await page.locator('.bd-model-qwen').getByText('Last run succeeded', { exact: true }).waitFor()
    await page.locator('.bd-model-jev').getByText('Last call failed', { exact: true }).waitFor()
    assert.equal(await page.locator('.bd-model-working').count(), 0)
    await page.locator('.bd-model-stack').evaluate(el => el.scrollIntoView({ block: 'start' }))
    await page.screenshot({ path: join(output, `bd-provider-observations-${width}.png`), animations: 'disabled' })
    live.schedule = { state: 'paused', start_at: null, next_run_at: null, stop_at: null, timezone: 'America/Chicago', display_only: true }
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.getByText('Conductor paused', { exact: true }).waitFor()
    assert.equal(await page.getByText('Waiting for planned start', { exact: true }).count(), 0)
    await schedulePanel.getByText('Paused', { exact: true }).waitFor()
    await schedulePanel.screenshot({ path: join(output, `bd-manual-paused-${width}.png`), animations: 'disabled' })
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), current)
    const quality = structuredClone(current)
    quality.packets.push(...['reject', 'watch', 'uncertain', 'blank'].map((disposition, index) => ({
      ...quality.packets[0], id: `history-${index}`, job_id: `history-job-${index}`,
      organization: disposition === 'blank' ? '' : `Synthetic ${disposition} result`, disposition,
      candidate_status: disposition === 'reject' ? 'rejected' : undefined,
      candidate_reason: disposition === 'reject' ? 'segment_rejected' : 'segment_entity_or_event_unconfirmed',
      review_status: disposition === 'watch' ? 'research_next' : 'human_review_required',
    })))
    quality.jobs.push({ ...quality.jobs[0], id: 'withheld-job', state: 'completed', finished_at: at, research_outcome: 'inconclusive', result_reason: 'no_new_evidence' })
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), quality)
    const delivered = page.getByRole('region', { name: 'Completed research', exact: true })
    assert.equal(await delivered.getByRole('button', { name: 'Review packet' }).count(), 1)
    assert.equal(await delivered.getByText(/Synthetic reject|Synthetic watch|Synthetic uncertain/).count(), 0)
    assert.equal(await page.locator('.bd-packet').count(), 1)
    const history = page.getByRole('region', { name: 'Research history', exact: true })
    await history.getByText('Inspect research outcomes (5)', { exact: true }).click()
    await history.getByText('No new evidence was found in this run.', { exact: true }).waitFor()
    await history.getByText('Unnamed research result', { exact: true }).waitFor()
    await history.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `bd-outcome-history-${width}.png`), animations: 'disabled' })
    const writesBeforeInspection = posts.length
    await history.getByRole('listitem').filter({ hasText: 'Synthetic watch result' }).getByRole('button', { name: 'Inspect evidence' }).click()
    await page.locator('#bd-packet-history-1').getByRole('link', { name: 'Open source evidence' }).waitFor({ state: 'visible' })
    assert.equal(await page.getByRole('combobox', { name: 'Research outcome' }).inputValue(), 'history')
    assert.equal(posts.length, writesBeforeInspection, 'inspecting history does not mutate reviews')
    assert.equal(await page.locator('.bd-page').evaluate(el => el.scrollWidth > el.clientWidth), false)
    await page.getByRole('combobox', { name: 'Research outcome' }).selectOption('all')
    assert.equal(await page.locator('.bd-packet').count(), 5)
    await page.getByRole('combobox', { name: 'Research outcome' }).selectOption('supported')
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), current)
    actionFail = true
    await page.getByRole('button', { name: 'Run shadow batch' }).click()
    await page.getByRole('alert').filter({ hasText: 'fixture_action_unavailable' }).waitFor()
    unavailable = true
    await page.reload()
    await page.getByRole('alert').filter({ hasText: 'fixture_data_unavailable' }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Run shadow batch' }).isDisabled(), true)
    assert.equal(await page.getByRole('heading', { name: 'Research lanes' }).count(), 0)
    await page.screenshot({ path: join(output, `bd-unavailable-${width}.png`), fullPage: true })
    assert.deepEqual(errors, [])
    console.log(`BD ${width}px: source evidence, filter, SSE, pause, shadow action, kill switch, unavailable states and overflow passed`)
    await context.close()
  }
} finally { await browser.close() }
