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
const bots = [
  { id: 'healthcare', name: 'Healthcare research', segment: 'healthcare', tenant_id: 'synthetic-0', kind: 'segment', counts: { queued: 2, running: 1, completed: 1, failed: 0 }, budgets: budget, paused: false, focused: false, around_the_clock: false },
  { id: 'pe-diligence', name: 'PE technical diligence', segment: 'pe-technical-diligence', tenant_id: 'synthetic-1', kind: 'segment', counts: { queued: 2, running: 0, completed: 1, failed: 0 }, budgets: budget, paused: false, focused: false, around_the_clock: false },
  { id: 'alumni-associations', name: 'Alumni and associations', segment: 'alumni-associations', tenant_id: 'synthetic-2', kind: 'segment', counts: { queued: 2, running: 0, completed: 1, failed: 0 }, budgets: budget, paused: false, focused: false, around_the_clock: false },
  { id: 'srg', name: 'SRG BD research', segment: 'client-bd', tenant_id: 'synthetic-3', kind: 'client', counts: { queued: 2, running: 0, completed: 1, failed: 0 }, budgets: budget, paused: false, focused: false, around_the_clock: false },
  { id: 'comcast', name: 'Comcast Business leads', segment: 'client-bd', tenant_id: 'synthetic-comcast', kind: 'client', research_segment: 'commercial-fit-out', counts: { queued: 2, running: 0, completed: 1, failed: 0 }, budgets: budget, paused: false, focused: false, around_the_clock: false },
]
const snapshot = {
  mode: 'replay', schedule_enabled: false, kill_switch: false,
  status_banner: { state: 'paused_schedule_off', label: 'Paused: schedule off' },
  daily_summary: { day: '2026-10-27', timezone: 'America/Chicago', candidates_proposed: 1, inconclusive: 2, failed: 0, awaiting_human_review: 1 },
  cadence: { timezone: 'America/Chicago', weekday: { interval_seconds: 1800 }, overnight: { enabled: false, interval_seconds: 3600 } },
  bots, budget,
  jobs: [{ id: 'queue-1', bot_id: 'pe-diligence', tenant_id: 'synthetic-1', segment: 'pe-technical-diligence', state: 'queued', step: 'queued', stage: 'queued', created_at: at, started_at: null, heartbeat_at: null }, { id: 'job-1', bot_id: 'healthcare', tenant_id: 'synthetic-0', segment: 'healthcare', state: 'running', step: 'Extracting public evidence', stage: 'extract', created_at: at, started_at: at, heartbeat_at: at }],
  events: [{ id: 'event-1', kind: 'job_running', message: 'Research started from recorded public evidence.', created_at: at, bot_id: 'healthcare', job_id: 'job-1' }],
  packets: [{ id: 'packet-1', index: 0, candidate_status: 'supported', candidate_reason: 'source_supported_candidate', job_id: 'previous-job', bot_id: 'healthcare', tenant_id: 'synthetic-0', segment: 'Healthcare', created_at: at, mode: 'replay', review_status: 'human_review_required', organization: 'Synthetic Health Group', source_url: 'https://example.com/research', source_quote: 'Synthetic source quotation for review.', emails: ['research@example.com'], phones: ['+1 202-555-0100'], signals: ['Expansion announcement in recorded source.'], insights: ['Explore demand for intake automation.'], summary: 'Synthetic evidence packet for review. This is not a real prospect.', verdict: 'Choice', evidence_score: 1.8 }],
  health: { qwen: { status: 'replay', detail: 'Recorded extraction' }, jev: { status: 'replay', detail: 'Recorded verdicts' }, managed_leads: { status: 'data_unavailable', detail: 'Staging API is not connected' } },
  model_activity: { qwen: { status: 'replay', active_jobs: [], calls: 0, completed: 0, replayed: 3, errors: 0, last_event_at: at, latency_ms: null }, jev: { status: 'replay', active_jobs: [], calls: 0, completed: 0, replayed: 2, errors: 0, last_event_at: at, latency_ms: null } },
  jev: { calls_per_minute: 2, verdict_mix: { Choice: 1, Score: 1, Noul: 0 }, latency_ms: null, errors: null },
  steering: { paused_bots: [], focus: null, overnight_enabled: false, around_the_clock: {}, audit: [] },
  assignments: {
    qwen: { assigned: [], queue: ['queue-1'], recent: [], status: 'replay', calls: 0, errors: 0, latency_ms: null },
    jev: { assigned: [], queue: [], recent: [], status: 'replay', calls: 0, errors: 0, latency_ms: null, verdict_mix: { Choice: 1, Score: 1, Noul: 0 } },
    pipeline: [],
  },
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
        if (path.endsWith('steer')) {
          const body = route.request().postDataJSON()
          current.steering = current.steering || { paused_bots: [], focus: null, overnight_enabled: false, around_the_clock: {}, audit: [] }
          if (body.action === 'pause') current.schedule_enabled = false
          if (body.action === 'resume') current.schedule_enabled = true
          if (body.action === 'pause_bot') {
            current.steering.paused_bots = [body.bot_id]
            const bot = current.bots.find(item => item.id === body.bot_id)
            if (bot) bot.paused = true
          }
          if (body.action === 'resume_bot') {
            current.steering.paused_bots = []
            const bot = current.bots.find(item => item.id === body.bot_id)
            if (bot) bot.paused = false
          }
          if (body.action === 'focus') {
            current.steering.focus = { kind: 'bot', id: body.bot_id }
            current.bots.forEach(bot => { bot.focused = bot.id === body.bot_id })
          }
          if (body.action === 'clear_focus') {
            current.steering.focus = null
            current.bots.forEach(bot => { bot.focused = false })
          }
          if (body.action === 'queue') {
            current.jobs.unshift({ id: 'steered-1', bot_id: body.bot_id, tenant_id: 'synthetic-comcast', segment: 'client-bd', state: 'queued', step: 'queued', stage: 'queued', created_at: at, started_at: null, heartbeat_at: null, priority: 1, operator_query: body.query })
          }
          if (body.action === 'cancel') {
            const job = current.jobs.find(item => item.id === body.job_id)
            if (job) job.state = 'cancelled'
          }
          if (body.action === 'bump' && current.jobs[0]) current.jobs[0].priority = 9
          if (body.action === 'set_cadence' && body.weekday_interval_seconds) current.cadence.weekday.interval_seconds = body.weekday_interval_seconds
          if (body.action === 'set_cadence' && body.overnight_enabled != null) current.steering.overnight_enabled = body.overnight_enabled
          if (body.action === 'set_budget' && body.tokens) current.budget.tokens.limit = body.tokens
          if (body.action === 'set_budget' && body.calls) current.budget.calls.limit = body.calls
          current.steering.audit.unshift({ id: Date.now(), action: body.action, created_at: at, bot_id: body.bot_id, job_id: body.job_id, query: body.query, enabled: body.enabled })
        }
        if (path.endsWith('shadow')) data = { ok: true, job_ids: ['shadow-1'] }
        else data = current
      } else if (path === '/api/agents') data = { agents: [] }
      else if (path === '/api/agents/resources') data = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/agents/models') data = { models: [], default: '' }
      else if (path === '/api/health') data = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') data = { services: {} }
      else if (path === '/api/integrations/typesafe/status') data = { ready: false }
      return route.fulfill({ json: data })
    })
    await page.goto(`${process.env.STUDIO_URL || 'http://127.0.0.1:5187/'}#/bd`)
    const toast = page.locator('.bd-toast')
    await page.getByRole('heading', { name: 'Lead finding' }).waitFor()
    await page.getByText('Bots only propose.', { exact: false }).waitFor()
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Paused by you', { exact: true }).waitFor()
    await page.getByRole('heading', { name: /Comcast · Minneapolis permits/ }).waitFor()
    await page.getByRole('heading', { name: 'Synthetic Health Group' }).waitFor()
    assert.equal(await page.locator('.bd-model-working').count(), 0, 'replay never claims active inference')
    assert.equal(await page.getByText('qwen2.5-14b-instruct').count(), 0)
    assert.equal(await page.getByText('$0.03').count(), 0)
    await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
    await page.screenshot({ path: join(output, `bd-${width}.png`), fullPage: true })
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.equal(await page.locator('.bd-page').evaluate(el => el.scrollWidth > el.clientWidth), false)

    if (width === 390) {
      assert.equal(await page.locator('.bd-chips').first().evaluate(el => getComputedStyle(el).display), 'none')
      await page.getByRole('button', { name: 'Steer', exact: true }).click()
      await page.locator('#h-steer-m').waitFor()
      await page.screenshot({ path: join(output, `bd-steer-sheet-${width}.png`) })
      await page.getByRole('button', { name: 'Done' }).click()
    }

    const comcast = page.locator('.bd-lane').filter({ hasText: 'Comcast · Minneapolis permits' })
    await comcast.getByRole('button', { name: 'Pause Comcast · Minneapolis permits' }).click()
    await toast.getByText('You paused Comcast · Minneapolis permits.', { exact: false }).waitFor()
    assert.equal(posts.at(-1).path, '/api/bd/steer')
    assert.deepEqual(posts.at(-1).body, { action: 'pause_bot', bot_id: 'comcast' })
    await toast.getByRole('button', { name: 'Undo' }).click()
    await toast.getByText('Undone', { exact: true }).waitFor()
    assert.deepEqual(posts.at(-1).body, { action: 'resume_bot', bot_id: 'comcast' })

    await comcast.getByRole('button', { name: 'Focus Comcast · Minneapolis permits' }).click()
    assert.deepEqual(posts.at(-1).body, { action: 'focus', bot_id: 'comcast' })
    if (width !== 390) {
      await page.getByRole('button', { name: 'Uptown restaurant build-outs' }).click()
      assert.equal(posts.at(-1).body.action, 'queue')
      assert.equal(posts.at(-1).body.bot_id, 'comcast')
      await page.getByRole('button', { name: 'Hourly' }).click()
      assert.deepEqual(posts.at(-1).body, { action: 'set_cadence', weekday_interval_seconds: 3600 })
    }

    await page.getByRole('button', { name: 'Approve' }).click()
    await toast.getByText('Approved · Synthetic Health Group', { exact: false }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/packets/previous-job/0/review', body: { status: 'research_next' } })
    current.packets[0].review_status = 'human_review_required'
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), current)
    await page.getByRole('heading', { name: 'Synthetic Health Group' }).waitFor()
    await page.getByRole('button', { name: 'Fix' }).click()
    await page.getByLabel('Contact').fill('J. Example (held)')
    await page.getByRole('button', { name: 'Save' }).click()
    await toast.getByText('Saved your fix.', { exact: false }).waitFor()
    assert.equal(posts.filter(item => item.path.includes('/review')).length, 1, 'Fix does not write a review or CRM')

    await page.getByRole('button', { name: 'Details' }).click()
    await page.getByRole('heading', { name: 'Details' }).waitFor()
    await page.getByText('qwen3:8b', { exact: false }).waitFor()
    assert.equal(await page.getByText('qwen2.5-14b-instruct').count(), 0)
    await page.getByRole('button', { name: 'Run shadow batch' }).click()
    await toast.getByText('Shadow batch accepted.', { exact: false }).waitFor()
    assert.deepEqual(posts.at(-1), { path: '/api/bd/shadow', body: {} })
    await page.getByRole('button', { name: 'Close' }).click()

    await page.getByRole('status', { name: 'Conductor status' }).getByRole('button', { name: 'Resume all' }).click()
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Running', { exact: true }).waitFor()
    assert.equal(posts.at(-1).body.action, 'resume')

    if (width !== 390) {
      await page.getByRole('button', { name: 'Stop all bots' }).click()
      await page.getByRole('button', { name: 'Yes, stop everything' }).click()
      await page.getByRole('status', { name: 'Conductor status' }).getByText('Stopped by you', { exact: true }).waitFor()
      assert.deepEqual(posts.at(-1), { path: '/api/bd/kill-switch', body: { enabled: true } })
      await page.screenshot({ path: join(output, `bd-kill-${width}.png`), fullPage: true })
      await page.getByRole('status', { name: 'Conductor status' }).getByRole('button', { name: 'Resume everything' }).click()
      await page.getByRole('status', { name: 'Conductor status' }).getByText('Running', { exact: true }).waitFor()
      assert.equal(posts.some(item => item.path === '/api/bd/kill-switch' && item.body.enabled === false), true)
    }

    const live = structuredClone(current)
    live.mode = 'live'
    live.kill_switch = false
    live.schedule_enabled = true
    live.health.qwen.status = 'available'
    live.health.jev.status = 'available'
    const running = live.jobs.find(job => job.id === 'job-1')
    running.state = 'running'
    running.step = 'qwen_called'
    running.heartbeat_at = new Date(Date.now() - 2000).toISOString()
    live.model_activity.qwen = { ...live.model_activity.qwen, status: 'running', calls: 1, active_jobs: ['job-1'] }
    live.model_activity.jev = { ...live.model_activity.jev, status: 'idle', active_jobs: [] }
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    assert.equal(await page.locator('.bd-model-jev.bd-model-working').count(), 0)
    live.model_activity.qwen.active_jobs = []
    live.model_activity.jev.active_jobs = ['job-1']
    running.step = 'jev_called'
    running.heartbeat_at = new Date(Date.now() - 2000).toISOString()
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.locator('.bd-model-jev.bd-model-working').waitFor()
    assert.equal(await page.locator('.bd-model-qwen.bd-model-working').count(), 0)

    running.state = 'completed'
    live.jobs = []
    live.kill_switch = true
    live.schedule = { state: 'waiting', start_at: new Date(Date.now() + 3600000).toISOString(), next_run_at: new Date(Date.now() + 3600000).toISOString(), stop_at: new Date(Date.now() + 172800000).toISOString(), timezone: 'America/Chicago', display_only: true }
    live.health.qwen = { status: 'stale', detail: 'Last successful observation is old', observed_at: new Date(Date.now() - 10800000).toISOString(), last_status: 'available' }
    live.health.jev = { status: 'data_unavailable', detail: 'Provider returned an error', observed_at: new Date(Date.now() - 60000).toISOString(), last_status: 'data_unavailable' }
    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), live)
    await page.getByRole('status', { name: 'Conductor status' }).getByText('Stopped by you', { exact: true }).waitFor()
    await page.getByRole('button', { name: 'Details' }).click()
    await page.getByRole('region', { name: 'Research schedule', exact: true }).getByText('Planned first run', { exact: true }).waitFor()
    await page.getByRole('button', { name: 'Close' }).click()

    await page.evaluate(data => window.bdSource.snapshot({ data: JSON.stringify(data) }), current)
    actionFail = true
    await page.getByRole('button', { name: 'Pause Comcast · Minneapolis permits' }).click()
    await page.getByRole('alert').filter({ hasText: 'fixture_action_unavailable' }).waitFor()
    unavailable = true
    await page.reload()
    await page.getByRole('alert').filter({ hasText: 'fixture_data_unavailable' }).waitFor()
    assert.equal(await page.getByRole('heading', { name: 'Bots, live' }).count(), 0)
    await page.screenshot({ path: join(output, `bd-unavailable-${width}.png`), fullPage: true })
    assert.deepEqual(errors, [])
    console.log(`BD ${width}px: prototype layout, steering undo, leads inbox, details, kill switch, unavailable states and overflow passed`)
    await context.close()
  }
} finally { await browser.close() }
