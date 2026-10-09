import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const studioDist = process.env.STUDIO_DIST_DIR
const base = studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/'
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-morning-note')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })

const agents = [
  { id: 'ci', name: 'CI & Failure Fixer', mission: 'Fix red checks.', kind: 'coding', handles: ['ci', 'failure'], use_when: 'When checks are red.', status: 'idle', skills: [], tools: ['Repository read'], repo_id: 'sanction', runs: [], loop_enabled: false, retired: false },
  { id: 'lead', name: 'AIIA Product Lead', mission: 'Steer AIIA.', kind: 'product', handles: ['aiia'], use_when: 'When the ask is about AIIA.', status: 'idle', skills: [], tools: [], repo_id: '', runs: [], loop_enabled: false, retired: false },
]
const decisions = [
  { id: 'd1', kind: 'inbox', product: 'AIIA', title: 'Merge PR #104: digest redesign', why: 'Checks pass.', more: 'Ships the shorter morning note.', expected_version: '' },
  { id: 'd2', kind: 'inbox', product: 'Mindmoor', title: 'Promote Mindmoor alumni', why: 'No conflicts found.', more: 'Mostly small fixes.', expected_version: '' },
  { id: 'd3', kind: 'review', product: 'Mindmoor', title: 'Code review from Mindmoor Delivery Lead', why: '2 small suggestions.', more: 'Rename one setting.', expected_version: 'v1' },
  { id: 'd4', kind: 'inbox', product: 'Alumni Nations', title: 'Send the Phase 1 kickoff agenda', why: 'Drafted for Oct 15.', more: 'One page.', expected_version: '' },
  { id: 'd5', kind: 'inbox', product: 'Sanction', title: 'Approve the new approval-prompt wording', why: 'Shorter for first-time users.', more: 'Wording only.', expected_version: '' },
  { id: 'd6', kind: 'inbox', product: 'Morrow', title: 'Turn on the editorial look', why: 'Ready in preview.', more: 'Beta testers only.', expected_version: '' },
  { id: 'd7', kind: 'inbox', product: 'TRS', title: 'Change the weekly TRS summary to Mondays', why: 'Client asked.', more: 'Moves Friday to Monday.', expected_version: '' },
]
const note = {
  date: '2026-10-08',
  products: [
    { id: 'aiia', name: 'AIIA', kind: 'product', state: 'waiting', note: 'Digest redesign is finished and checked.' },
    { id: 'mindmoor', name: 'Mindmoor', kind: 'product', state: 'waiting', note: 'Alumni version is 12 changes behind main.' },
    { id: 'sanction', name: 'Sanction', kind: 'product', state: 'blocked', note: 'Automated checks are red on the latest change.' },
    { id: 'mia', name: 'MIA', kind: 'product', state: 'shipped', note: "Last week's update is out." },
    { id: 'morrow', name: 'Morrow', kind: 'product', state: 'shipped', note: 'Conversation board is out to beta testers.' },
  ],
  customers: [
    { id: 'trs', name: "That's Right Sweetie", kind: 'customer', state: 'shipped', note: 'Daily client prep is arriving each morning.' },
    { id: 'alumni-nations', name: 'Alumni Nations', kind: 'customer', state: 'countdown', note: 'Phase 1 kicks off Thursday, Oct 15.', target: '2026-10-15T09:00:00' },
    { id: 'smart-medical', name: 'Smart Medical', kind: 'customer', state: 'unmapped', note: 'Not mapped yet. Nothing set up.' },
  ],
  decisions,
  needs_you_total: 7,
  failure: null,
}

async function open(width, height) {
  const context = await browser.newContext({ viewport: { width, height } })
  const page = await context.newPage()
  page.setDefaultTimeout(12000)
  const pageErrors = []
  const writes = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
  if (studioDist) {
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(studioDist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({
        contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript',
        body: await readFile(join(studioDist, asset)),
      })
    })
  }
  await page.route('**/api/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() !== 'GET') writes.push({ method: request.method(), path, body: request.postDataJSON() })
    if (path === '/api/studio/morning-note') return route.fulfill({ json: note })
    if (path === '/api/agents') return route.fulfill({ json: { agents } })
    if (path === '/api/assignments' && request.method() === 'GET') return route.fulfill({ json: { assignments: [] } })
    if (path === '/api/assignments' && request.method() === 'POST') {
      return route.fulfill({ json: { assignment: { id: 'asg-ask', title: request.postDataJSON().title, agent_id: request.postDataJSON().agent_id, status: 'queued' } } })
    }
    if (path.endsWith('/promote') || path.endsWith('/dismiss') || path.endsWith('/review')) {
      return route.fulfill({ json: { idea: { id: 'ok' }, assignment: { id: 'ok' } } })
    }
    const bodies = {
      '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
      '/api/agents/models': { default: 'synthetic', models: [] },
      '/api/handoffs': { handoffs: [] },
      '/api/git-workspaces': { workspaces: [] },
      '/api/git-writes': { writes: [] },
      '/api/memory-inbox': { ideas: [], total: 0, offset: 0, counts: { unreviewed: 0, promoted: 0, dismissed: 0 } },
      '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
      '/api/monitor': { services: {} },
      '/api/voice/status': { status: 'not_configured', configured: false, reason: 'missing_xai_api_key', tools: [] },
      '/api/tasks': [],
    }
    await route.fulfill({ json: bodies[path] ?? {} })
  })
  await page.goto(`${base}#/note`)
  return { context, page, pageErrors, writes }
}

const forbidden = /\b(runs?|loops?|inbox|sources?)\b/i

try {
  for (const [width, height, label] of [[1280, 800, 'desktop'], [390, 844, 'phone']]) {
    const { context, page, pageErrors, writes } = await open(width, height)
    await page.getByRole('heading', { level: 1, name: /Eric\.$/ }).waitFor()
    await page.getByRole('heading', { name: 'Where things stand', exact: true }).waitFor()
    await page.getByRole('heading', { name: /Needs you/ }).waitFor()
    await page.getByRole('heading', { name: 'Ask for anything', exact: true }).waitFor()
    assert.equal(await page.getByRole('navigation', { name: 'Studio' }).count(), 0)
    const cards = page.locator('.card')
    assert.equal(await cards.count(), 5)
    await page.getByText('Plus 2 more waiting', { exact: true }).waitFor()
    await page.getByText('Alumni Nations', { exact: true }).first().waitFor()
    const copy = await page.locator('main.morning-note').innerText()
    assert.equal(forbidden.test(copy), false, `forbidden word on ${label}: ${copy.match(forbidden)}`)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)
    assert.equal(overflow, false, `${label} overflowed horizontally`)

    await page.screenshot({ path: join(output, `morning-note-${label}.png`), fullPage: true, animations: 'disabled' })

    await page.getByRole('button', { name: 'Approve', exact: true }).first().click()
    await page.waitForFunction(() => document.querySelectorAll('.card').length === 5)
    assert.equal(writes.some(item => item.path.includes('/promote') || item.path.includes('/review')), true)
    await page.getByRole('button', { name: 'Not now', exact: true }).first().click()
    await page.getByRole('button', { name: 'Open', exact: true }).first().click()
    await page.getByText('Ships the shorter morning note.').or(page.getByText('Mostly small fixes.')).or(page.getByText('Rename one setting.')).waitFor()

    await page.getByLabel('What do you need done?').fill('have someone look at why Sanction CI is red')
    await page.getByRole('button', { name: 'Ask', exact: true }).click()
    await page.getByText('Taken by CI & Failure Fixer').waitFor()
    const created = writes.find(item => item.method === 'POST' && item.path === '/api/assignments')
    assert.equal(created?.body.agent_id, 'ci')

    await page.getByRole('button', { name: 'Details', exact: true }).first().click()
    await page.getByRole('heading', { name: 'Today', exact: true }).waitFor()
    assert.equal(await page.evaluate(() => window.location.hash), '#/today')
    assert.deepEqual(pageErrors, [])
    console.log(`${width}x${height}: morning note, decisions, ask, and Details passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
