import assert from 'node:assert/strict'
import { chromium } from 'playwright'

const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 844 } })
    const page = await context.newPage()
    const rows = Array.from({ length: 3 }, (_, i) => ({ id: `qa-${i}`, title: i === 2 ? 'Protected proposal' : 'Repeated report', agent_id: 'qa', objective: 'Inspect CI 123', context: '', success_criteria: '', priority: 'normal', status: 'completed', result: 'CI 123 passed', error: '', review_status: 'unreviewed', review_version: `v${i}`, attention_key: i === 2 ? 'protected' : 'same', attention_seen: false, created_at: '2026-10-27', updated_at: '2026-10-27' }))
    const forbidden = []
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      if (path.endsWith('/attention') && route.request().method() === 'POST') {
        const input = route.request().postDataJSON()
        assert.equal(input.key, 'same')
        rows.filter(row => row.attention_key === 'same').forEach(row => { row.attention_seen = input.active })
        body = { assignment: rows.find(row => row.id === path.split('/')[3]) }
      } else if (route.request().method() !== 'GET') forbidden.push(path)
      else if (path === '/api/assignments') body = { assignments: rows.map(row => ({ ...row, summary: true, has_result: true, result: '' })) }
      else if (/^\/api\/assignments\/[^/]+$/.test(path)) body = { assignment: rows.find(row => row.id === path.split('/')[3]) }
      else if (path === '/api/attention/fixture-preview') body = { candidates: Array.from({ length: 230 }, (_, i) => ({ id: `fixture-${i}`, title: 'Inspect mock', result: 'GREEN', reason: 'Mock signature', fingerprint: `f${i}` })), blocked: [], warning: 'Preview only; no records changed.' }
      else if (path === '/api/agents') body = { agents: [{ id: 'qa', name: 'QA Watch', skills: [], tools: [], status: 'idle', runs: [] }] }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/git-workspaces') body = { workspaces: [{ id: 'pending', assignment_id: 'qa-2', agent_id: 'qa', status: 'pending', events: [] }] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/tasks') body = []
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
    })
    await page.goto(`${process.env.STUDIO_URL}#/assignments`)
    await page.getByRole('button', { name: 'Preview suspected test records' }).click()
    await page.getByText(/230 suspected test records/).waitFor()
    await page.getByRole('button', { name: /Repeated report QA Watch/ }).click()
    await page.getByText('CI 123 passed', { exact: true }).waitFor()
    await page.getByRole('button', { name: 'Acknowledge identical reports', exact: true }).click()
    await page.getByRole('button', { name: /^History\s*\(\s*2\s*\)$/ }).waitFor()
    assert.ok(rows.every(row => row.review_status === 'unreviewed'))
    await page.getByRole('button', { name: 'Restore identical reports to attention', exact: true }).click()
    await page.getByRole('button', { name: /^Needs you\s*\(\s*3\s*\)$/ }).waitFor()
    await page.getByRole('button', { name: /Approval needed.*Protected proposal/ }).click()
    assert.equal(await page.getByRole('button', { name: 'Acknowledge identical reports', exact: true }).isEnabled(), false)
    await page.getByRole('button', { name: 'Approve & create worktree', exact: true }).waitFor()
    assert.deepEqual(forbidden, [])
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await context.close()
  }
} finally { await browser.close() }
