import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { chromium } from 'playwright'

const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 844 } })
    const page = await context.newPage()
    page.on('pageerror', error => console.error('Backlog page error:', error.message))
    const rows = Array.from({ length: 405 }, (_, i) => ({
      id: `old-${i}`, title: `Decision ${i}`, objective: 'Inspect the evidence',
      agent_id: 'reviewer', priority: 'normal', context: 'Full scope', success_criteria: 'Supported',
      source_kind: 'manual', source_handoff_id: '', status: 'completed',
      result: `Full evidence ${i}`, error: '', review_status: 'unreviewed', review_note: '',
      review_version: `v${i}`, dismissed_at: null, created_at: '2026-10-01', updated_at: '2026-10-01',
    }))
    rows[0].title = 'Eval A-1 fixture'
    rows[1].review_status = 'accepted'
    const detailReads = [], reviews = [], forbidden = []
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url()), path = url.pathname
      let body = {}, status = 200
      if (path === '/api/assignments') {
        assert.equal(url.searchParams.get('summary'), 'true')
        body = { assignments: rows.map(row => ({ ...row, summary: true, has_result: true,
          group_scope: 'same-scope', result: '', context: '', success_criteria: '' })) }
      } else if (/^\/api\/assignments\/[^/]+\/review$/.test(path)) {
        const row = rows.find(row => row.id === path.split('/')[3])
        const data = route.request().postDataJSON()
        assert.equal(data.expected_version, row.review_version)
        reviews.push(row.id)
        Object.assign(row, { review_status: data.decision, review_note: data.note, review_version: `${row.review_version}-next` })
        body = { assignment: row }
      } else if (route.request().method() !== 'GET') {
        forbidden.push(path); status = 403
      } else if (/^\/api\/assignments\/[^/]+$/.test(path)) {
        const id = path.split('/')[3]; detailReads.push(id)
        body = { assignment: rows.find(row => row.id === id) }
      } else if (path.endsWith('/history')) body = { runs: [], total: 0 }
      else if (path === '/api/agents') body = { agents: [{ id: 'reviewer', name: 'Reviewer', skills: [], tools: [], status: 'idle', runs: [] }] }
      else if (path === '/api/git-workspaces') body = { workspaces: [{ id: 'pending', assignment_id: 'old-1', agent_id: 'reviewer', status: 'pending', events: [] }] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/agents/resources') body = { repos: [] }
      else if (path === '/api/tasks') body = []
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto(`${process.env.STUDIO_URL}#/assignments`)
    try {
      await page.getByRole('button', { name: /^Needs you\s*\(\s*404\s*\)$/ }).waitFor()
    } catch (error) {
      const output = process.env.SCREENSHOT_DIR || '/tmp/full-backlog-qa'
      await mkdir(output, { recursive: true })
      await page.screenshot({ path: join(output, `backlog-failure-${width}.png`) })
      console.error(await page.locator('body').innerText())
      throw error
    }
    assert.deepEqual(detailReads, [])
    assert.equal(await page.getByRole('button', { name: /Show more groups/ }).count(), 1)
    while (await page.getByRole('button', { name: /Show more groups/ }).count()) {
      await page.getByRole('button', { name: /Show more groups/ }).click()
    }
    await page.getByRole('button', { name: /Decision 404 Reviewer/ }).click()
    await page.getByText('Full evidence 404', { exact: true }).waitFor()
    assert.deepEqual([...new Set(detailReads)], ['old-404'])
    await page.getByLabel('Review note', { exact: true }).fill('Synthetic quality decision')
    await page.getByRole('button', { name: 'Accept output', exact: true }).click()
    await page.getByRole('button', { name: /^History\s*\(\s*2\s*\)$/ }).waitFor()
    assert.deepEqual(reviews, ['old-404'])
    assert.equal(rows[403].review_status, 'unreviewed')
    await page.getByRole('button', { name: /^History\s*\(\s*2\s*\)$/ }).click()
    await page.getByRole('button', { name: /Eval A-1 fixture Reviewer/ }).waitFor()
    await page.getByRole('button', { name: /^Needs you\s*\(\s*403\s*\)$/ }).click()
    await page.getByRole('button', { name: /Approval needed.*Decision 1 Reviewer/ }).click()
    await page.getByRole('button', { name: 'Approve & create worktree', exact: true }).waitFor()
    assert.deepEqual(forbidden, [])
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await context.close()
  }
} finally { await browser.close() }
