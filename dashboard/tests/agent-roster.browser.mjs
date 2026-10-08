import { openStudioView } from './studio-navigation.mjs'
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-agent-roster')
const studioDist = process.env.STUDIO_DIST_DIR
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-10-07'

const agent = (id, name, overrides = {}) => ({
  id, name, mission: `${name} mission.`, persona: 'Direct.', skills: ['Analysis'], tools: ['Local memory'],
  repo_id: '', temperature: 0.35, max_tokens: 1200, loop_enabled: false, loop_interval_minutes: 60,
  loop_task: '', loop_max_runs_per_day: 4, loop_runs_today: 0, loop_day: '', status: 'idle',
  last_run_at: null, last_result: '', last_error: '', runs: [], created_at: `${date}T12:00:00Z`,
  updated_at: `${date}T12:00:00Z`, output_channel: 'studio_inbox', value: { window_days: 14, runs: 1, last_run_at: `${date}T12:00:00Z`, reviewed: 1, unreviewed: 0 },
  ...overrides,
})

const agents = [
  agent('ci', 'CI & Failure Fixer', { kind: 'coding', kind_derived: false, use_when: 'Unblock red checks.', use_when_derived: false, tools: ['Repository read', 'GitHub read'], repo_id: 'aiia', handles: ['ci'] }),
  agent('impl', 'Repo Brief & Implementer', { kind: 'coding', kind_derived: false, use_when: 'Turn a brief into a proposed patch.', use_when_derived: false, tools: ['Repository read', 'Git workspace'], repo_id: 'aiia', suite: 'aiia' }),
  agent('product', 'AIIA Product Lead', { kind: 'product', kind_derived: false, use_when: 'Decide what ships next.', use_when_derived: false, suite: 'aiia' }),
  agent('ops', 'Inbox Triage Clerk', { kind: 'ops', kind_derived: false, use_when: 'Triage inbound mail.', use_when_derived: false, handles: ['inbox'] }),
  agent('retired', 'Dependency Diplomat', { kind: 'coding', kind_derived: false, use_when: 'Do not pick this agent.', use_when_derived: false, retired: true, tools: ['Repository read'], repo_id: 'aiia' }),
  agent('paused', 'Code Reviewer', { kind: 'coding', kind_derived: false, use_when: 'Review a pull request.', use_when_derived: false, tools: ['Repository read'], repo_id: 'aiia', loop_enabled: true, loop_skip_reason: 'awaiting_review', value: { window_days: 14, runs: 4, last_run_at: `${date}T12:00:00Z`, reviewed: 0, unreviewed: 3 } }),
]

const ideas = [{
  id: 'idea-one-00000000', text: 'Capture a follow-up for the Mini.', source: 'slack', project: 'mindmoor',
  workspace_id: 'T_TEST', channel_id: 'C_ONE', author_id: 'U_AUTHOR', created_at: `${date}T12:00:00Z`,
  status: 'unreviewed', memory_id: '', memory_category: '', review_note: '', reviewed_at: '',
  acknowledgement_status: null, acknowledgement_error: null, acknowledgement_ts: null,
  promotion_status: null, promotion_error: null, promotion_ts: null, assignment_id: '',
}]
const leads = [{ id: 'signal-0', company: 'Example Co', text: 'Public expansion signal', created_at: `${date}T12:00:00Z`, inbox_status: 'unreviewed', assignment_id: '', decision: 'unreviewed', review: null }]

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
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
      const url = new URL(route.request().url())
      const path = url.pathname
      const bodies = {
        '/api/agents': { agents },
        '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
        '/api/agents/models': { default: 'qwen3:8b', models: [] },
        '/api/assignments': { assignments: [{ id: 'work-1', agent_id: 'ci', title: 'Unblock CI', objective: 'Fix the red check.', status: 'completed', result: 'The check is green.', error: '', priority: 'normal', context: '', success_criteria: '', source_handoff_id: '', created_at: `${date}T12:00:00Z`, updated_at: `${date}T12:00:00Z` }] },
        '/api/handoffs': { handoffs: [] },
        '/api/git-workspaces': { workspaces: [] },
        '/api/git-writes': { writes: [] },
        '/api/public-signals/leads': { leads, total: 1, offset: 0, limit: 25 },
        '/api/signal-jobs': { jobs: [], ready: false, configured: false, retrieval_enabled: false, screening_enabled: false },
        '/api/memory-inbox': { ideas, total: 1, offset: 0, counts: { unreviewed: 1, promoted: 0, dismissed: 0 } },
        '/api/memory-inbox/review-health': { window_days: 14, since: `${date}T00:00:00+00:00`, filed: 0, reviewed: 0, totals: { open: 0, needs_work: 0, already_fixed: 0, declined: 0, external_failure: 0, unclassified: 0 }, by_source: [], by_project: [] },
        '/api/integrations/slack/status': { configured: false, channel_ids: [], acknowledgements_configured: false },
        '/api/agent-world/layout': { layout: { version: 1, revision: 0, positions: {}, updated_at: null } },
        '/api/tasks': [],
        '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
        '/api/monitor': { services: {} },
        '/api/voice/status': { available: false },
        '/api/tokens/today': { date, total_tokens: 0, total_requests: 0, total_cost: 0, by_provider: {}, by_purpose: {} },
        '/api/tokens/recent': { days: [] },
        '/api/studio/activity': { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] },
      }
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify(bodies[path] ?? {}) })
    })
    await page.goto(studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/')

    await openStudioView(page, 'Agents')
    const coding = page.getByRole('region', { name: 'Coding', exact: true })
    const product = page.getByRole('region', { name: 'Product', exact: true })
    const ops = page.getByRole('region', { name: 'Ops', exact: true })
    await coding.waitFor()
    await product.waitFor()
    await ops.waitFor()
    assert.ok((await coding.innerText()).includes('CI & Failure Fixer'))
    assert.ok((await coding.innerText()).includes('Unblock red checks.'))
    assert.ok((await coding.innerText()).includes('Read-only'))
    assert.ok((await coding.innerText()).includes('Proposes git (approval)'))
    assert.ok((await coding.innerText()).includes('Paused: 3 awaiting review'))
    assert.ok(!(await coding.innerText()).includes('Dependency Diplomat'))
    assert.ok((await product.innerText()).includes('AIIA Product Lead'))
    assert.ok((await product.innerText()).includes('Decide what ships next.'))
    assert.ok((await ops.innerText()).includes('Inbox Triage Clerk'))
    assert.equal(await page.getByRole('button', { name: 'Show retired (1)' }).getAttribute('aria-pressed'), 'false')
    await page.getByRole('button', { name: 'Show retired (1)' }).click()
    assert.ok((await coding.innerText()).includes('Dependency Diplomat'))
    assert.ok((await coding.innerText()).includes('Retired'))
    await coding.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `agents-roster-${width}.png`) })
    await page.getByRole('button', { name: 'Show retired (1)' }).click()
    assert.ok(!(await coding.innerText()).includes('Dependency Diplomat'))

    const optionMeta = async (label, placeholder) => page.getByRole('combobox', { name: label, exact: true }).evaluate((select, empty) => {
      const groups = [...select.querySelectorAll('optgroup')].map(group => ({
        label: group.label,
        options: [...group.querySelectorAll('option')].map(option => ({ value: option.value, text: option.textContent, title: option.title })),
      }))
      const values = [...select.options].map(option => option.value).filter(value => value && value !== empty)
      return { groups, values, placeholder: select.options[0]?.textContent }
    }, placeholder)

    await openStudioView(page, 'Work')
    await page.getByRole('button', { name: 'New work', exact: true }).click()
    const assigned = await optionMeta('Assigned agent', '')
    assert.equal(assigned.placeholder, 'Choose an agent')
    assert.deepEqual(assigned.groups.map(group => group.label), ['Coding', 'Product', 'Ops'])
    assert.ok(assigned.groups[0].options.some(option => option.text.includes('Unblock red checks.')))
    assert.ok(assigned.groups[0].options.some(option => option.title === 'Review a pull request.'))
    assert.ok(!assigned.values.includes('retired'))
    await page.screenshot({ path: join(output, `assign-picker-${width}.png`) })

    await openStudioView(page, 'Handoffs')
    const target = await optionMeta('Target agent', '')
    assert.equal(target.placeholder, 'Choose the next specialist')
    assert.deepEqual(target.groups.map(group => group.label), ['Coding', 'Product', 'Ops'])
    assert.ok(!target.values.includes('retired'))

    await openStudioView(page, 'Memory')
    const capturePicker = page.getByLabel('Agent for capture idea-one')
    await capturePicker.waitFor()
    const captureGroups = await capturePicker.evaluate(select => [...select.querySelectorAll('optgroup')].map(group => group.label))
    assert.deepEqual(captureGroups, ['Coding', 'Product', 'Ops'])
    assert.equal(await capturePicker.locator('option[value="retired"]').count(), 0)

    await openStudioView(page, 'Signals')
    const research = page.getByLabel('Research agent for signal-0')
    await research.waitFor()
    const researchGroups = await research.evaluate(select => [...select.querySelectorAll('optgroup')].map(group => ({
      label: group.label,
      names: [...group.querySelectorAll('option')].map(option => option.textContent),
    })))
    assert.deepEqual(researchGroups.map(group => group.label), ['Coding', 'Product', 'Ops'])
    assert.ok(researchGroups.some(group => group.names.some(name => name.includes('CI & Failure Fixer'))))
    assert.ok(researchGroups.some(group => group.names.some(name => name.includes('AIIA Product Lead'))))
    assert.equal(await research.locator('option[value="retired"]').count(), 0)
    await page.screenshot({ path: join(output, `research-picker-${width}.png`) })

    assert.deepEqual(errors, [])
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
