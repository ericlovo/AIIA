import { openStudioView } from "./studio-navigation.mjs"
// Studio resilience: one bad record must not take the navigation with it, Space
// must still activate focused controls when voice is configured, and Today and
// Overview must report the same "needs attention" count.
//
// Every API response and the Studio WebSocket are intercepted with synthetic
// data; nothing is written to a real Brain or Command Center.
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const studioDist = process.env.STUDIO_DIST_DIR
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = new Date().toISOString().slice(0, 10)

const agents = [
  { id: 'a0', name: 'Signal Scout', mission: 'Find signal.', status: 'idle', skills: [], tools: [], repo_id: '', runs: [], loop_enabled: false, loop_interval_minutes: 60, loop_max_runs_per_day: 4 },
  { id: 'a1', name: 'Repo Warden', mission: 'Watch the repo.', status: 'idle', skills: [], tools: [], repo_id: '', runs: [], loop_enabled: false, loop_interval_minutes: 60, loop_max_runs_per_day: 4 },
]
const work = (id, overrides = {}) => ({
  id, agent_id: 'a0', title: `Work ${id}`, objective: 'Verify.', status: 'completed', result: 'Evidence.', error: '', priority: 'normal',
  context: '', success_criteria: '', source_handoff_id: '', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z`, ...overrides,
})
// Two to review, one failed, one accepted (not counted), one dismissed failure (not counted).
const assignments = [
  work('review-1'), work('review-2', { agent_id: 'a1' }), work('failed-1', { status: 'failed', result: '', error: 'timeout' }),
  work('accepted-1', { review_status: 'accepted' }), work('dismissed-1', { status: 'failed', result: '', dismissed_at: `${date}T13:00:00Z` }),
]
// One pending workspace and one pending write: two approvals.
const workspaces = [{ id: 'ws-1', assignment_id: 'review-1', agent_id: 'a0', repo_id: 'aiia', status: 'pending', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` }]
const writes = [{ id: 'w-1', workspace_id: 'ws-1', assignment_id: 'review-1', op: 'commit', status: 'pending', result: {}, created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` }]
const EXPECTED_ATTENTION = 5 // 2 review + 1 failed + 2 approvals

async function open({ handoffs = [], voiceConfigured = false, mintVoice = false, failedSources = new Set() } = {}) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  const page = await context.newPage()
  page.setDefaultTimeout(12000)
  const pageErrors = []
  const voiceSessions = []
  const voiceConnections = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
  await page.routeWebSocket('wss://api.x.ai/**', ws => { voiceConnections.push('connect'); ws.onMessage(() => {}) })
  if (studioDist) {
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(studioDist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(studioDist, asset)) })
    })
  }
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (failedSources.has(path)) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'synthetic_unavailable' }) })
    const bodies = {
      '/api/agents': { agents },
      '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
      '/api/agents/models': { default: 'synthetic-model:1b', models: [] },
      '/api/assignments': { assignments },
      '/api/handoffs': { handoffs },
      '/api/git-workspaces': { workspaces },
      '/api/git-writes': { writes },
      '/api/studio/activity': { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] },
      '/api/memory-inbox/review-health': { window_days: 14, since: `${date}T00:00:00+00:00`, filed: 0, reviewed: 0, totals: { open: 0, needs_work: 0, already_fixed: 0, declined: 0, external_failure: 0, unclassified: 0 }, by_source: [], by_project: [] },
      '/api/tasks': [],
      '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
      '/api/monitor': { services: {} },
      '/api/voice/status': voiceConfigured
        ? { status: 'connected', configured: true, provider: 'synthetic', model: '', voice: '', realtime_url: '', reason: '', tools: [] }
        : { status: 'not_configured', configured: false, reason: 'missing_xai_api_key', tools: [] },
    }
    if (path === '/api/voice/session') {
      voiceSessions.push(path)
      // Held open briefly so the conductor sits in "connecting" and re-renders mid-hold.
      await new Promise(resolve => setTimeout(resolve, 1500))
      if (mintVoice) return route.fulfill({ contentType: 'application/json', body: JSON.stringify({ token: 'synthetic', realtime_url: 'wss://api.x.ai/v1/realtime', session: {} }) })
      return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Synthetic: no voice in tests' }) })
    }
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify(bodies[path] ?? {}) })
  })
  await page.goto(studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/')
  await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Today', exact: true }).waitFor()
  return { context, page, pageErrors, voiceSessions, voiceConnections }
}

try {
  // 1. Today and Overview agree on the attention count.
  {
    const { context, page, pageErrors } = await open()
    await page.getByRole('heading', { name: `Needs attention (${EXPECTED_ATTENTION})` }).waitFor()
    await page.getByRole('link', { name: 'Open 2 pending approvals' }).waitFor()
    assert.equal(await page.getByRole('link', { name: /Work review-1 Approve workspace/ }).getAttribute('href'), '#/assignments/review-1')
    assert.equal(await page.getByRole('link', { name: /Work review-1 Review commit/ }).getAttribute('href'), '#/assignments/review-1')
    await openStudioView(page, "Overview")
    const metric = page.getByText('Needs attention', { exact: true }).locator('..')
    await metric.getByText(String(EXPECTED_ATTENTION), { exact: true }).waitFor()
    await metric.getByText('2 to review · 1 failed · 2 approvals').waitFor()
    await page.getByText('Runs today (UTC)').waitFor()
    assert.deepEqual(pageErrors, [])
    await context.close()
  }

  // 2. A handoff missing its text fields renders instead of crashing.
  {
    const partial = { id: 'h-partial', source_assignment_id: 'review-1', target_assignment_id: '', from_agent_id: 'a0', to_agent_id: 'a1', artifact_type: 'summary', status: 'queued', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` }
    const { context, page, pageErrors } = await open({ handoffs: [partial] })
    await openStudioView(page, "Overview")
    await page.getByText('Signal Scout to Repo Warden').waitFor()
    assert.equal(await page.getByText(/crashed/i).count(), 0)
    assert.deepEqual(pageErrors, [])
    await context.close()
  }

  // 3. A view that does crash keeps the tabs, and the operator can leave it.
  {
    const { context, page } = await open({ handoffs: [null] })
    await openStudioView(page, "Overview")
    await page.getByRole('alert').getByText('OVERVIEW CRASHED').waitFor()
    await openStudioView(page, "Today")
    await page.getByRole('heading', { name: `Needs attention (${EXPECTED_ATTENTION})` }).waitFor()
    await context.close()
  }

  // 4. With voice configured, Space reaches a focused control instead of the mic,
  //    and still starts push-to-talk when focus is on the page itself.
  {
    const { context, page, voiceSessions } = await open({ voiceConfigured: true })
    // Voice is opt-in, including its global shortcut.
    await page.evaluate(() => (document.activeElement instanceof HTMLElement) && document.activeElement.blur())
    await page.keyboard.press('Space')
    assert.equal(voiceSessions.length, 0)
    await page.getByRole('button', { name: 'Voice', exact: true }).click()
    await page.getByTitle('Hold to talk').waitFor()
    await openStudioView(page, 'Activity history')
    // Space activates a focused button (the activity work-queue metric)...
    await page.getByRole('button', { name: /Work queue/ }).focus()
    await page.keyboard.press('Space')
    await page.getByRole('heading', { name: 'Assignment queue' }).waitFor()
    // ...and is left alone on a focused link, which Enter activates.
    await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Today', exact: true }).focus()
    await page.keyboard.press('Space')
    await page.keyboard.press('Enter')
    await page.getByRole('heading', { name: 'Today', exact: true }).waitFor()
    assert.equal(voiceSessions.length, 0, 'Space on a focused control must not start the mic')
    await page.evaluate(() => (document.activeElement instanceof HTMLElement) && document.activeElement.blur())
    await page.keyboard.down('Space')
    await page.waitForTimeout(300)
    assert.equal(voiceSessions.length, 1, 'Space with nothing focused is push-to-talk')
    // The conductor re-rendered into "connecting" while Space was held. Releasing
    // it must still be handled as the end of the hold, or the mic stays open.
    const releaseHandled = await page.evaluate(() => {
      const release = new KeyboardEvent('keyup', { code: 'Space', key: ' ', bubbles: true, cancelable: true })
      document.body.dispatchEvent(release)
      return release.defaultPrevented
    })
    assert.equal(releaseHandled, true, 'releasing Space mid-connect must end the hold')
    await context.close()
  }

  // Failed review reads are incomplete, not an empty all-clear; explicit retry recovers.
  {
    const failedSources = new Set(['/api/assignments', '/api/git-workspaces', '/api/git-writes'])
    const { context, page, pageErrors } = await open({ failedSources })
    await page.getByRole('alert').getByText('Some review sources are unavailable. These counts may be incomplete.').waitFor()
    assert.equal(await page.getByText('No work reports, failures, or approvals waiting.').count(), 0)
    failedSources.clear()
    await page.getByRole('button', { name: 'Refresh today', exact: true }).click()
    await page.getByRole('heading', { name: `Needs attention (${EXPECTED_ATTENTION})` }).waitFor()
    assert.equal(await page.getByRole('alert').count(), 0)
    assert.deepEqual(pageErrors, [])
    await context.close()
  }

  // Closing while a session token is pending cannot start a hidden connection.
  {
    const { context, page, voiceSessions, voiceConnections, pageErrors } = await open({ voiceConfigured: true, mintVoice: true })
    await page.getByRole('button', { name: 'Voice', exact: true }).click()
    await page.getByTitle('Hold to talk').waitFor()
    await page.evaluate(() => (document.activeElement instanceof HTMLElement) && document.activeElement.blur())
    await page.keyboard.down('Space')
    await page.waitForTimeout(200)
    assert.equal(voiceSessions.length, 1)
    await page.getByRole('button', { name: 'Voice', exact: true }).click()
    await page.keyboard.up('Space')
    await page.waitForTimeout(1700)
    assert.equal(voiceConnections.length, 0)
    assert.equal(await page.getByTitle('Hold to talk').count(), 0)
    assert.deepEqual(pageErrors, [])
    await context.close()
  }
  console.log('studio resilience: all checks passed')
} finally {
  await browser.close()
}
