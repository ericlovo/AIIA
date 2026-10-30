import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true })
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 844 } })
    const writes = [], errors = []
    let unavailable = false
    page.on('pageerror', e => errors.push(e.message))
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url()), path = url.pathname
      if (req.method() !== 'GET') { writes.push(path); return route.abort() }
      if (url.origin !== 'http://studio.test') return route.abort()
      if (path === '/') return route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) })
      if (path.startsWith('/assets/')) return route.fulfill({ contentType: path.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, path.slice(1))) })
      let body = {}
      if (path === '/api/memory-inbox/attention-preview') {
        if (unavailable) return route.fulfill({ status: 503, json: {} })
        const offset = Number(url.searchParams.get('offset') || 0)
        body = { total: 174, offset, preview_only: true, counts: { verification: 150, activity: 23, linked_work: 1 }, ideas: Array.from({ length: Math.min(50, 174 - offset) }, (_, i) => ({ id: `original-${offset+i}`, text: `Evidence ${offset+i}`, source: 'code_review', status: 'unreviewed', proposed_lane: 'verification', reason: 'Verify current evidence', created_at: '2026-10-30', assignment_id: offset+i === 0 ? 'protected' : '' })) }
      } else if (path === '/api/memory-inbox') body = { ideas: [], total: 174, attention_counts: { needs_you: 174, verification: 2, activity: 1 }, counts: { unreviewed: 177 } }
      else if (path === '/api/assignments') body = { assignments: [] }
      else if (path === '/api/agents') body = { agents: [] }
      else if (path === '/api/git-workspaces') body = { workspaces: [] }
      else if (path === '/api/git-writes') body = { writes: [] }
      else if (path === '/api/tasks') body = []
      await route.fulfill({ json: body })
    })
    await page.goto('http://studio.test/#/assignments')
    await page.getByRole('button', { name: 'Preview all classifications', exact: true }).click()
    await page.getByText('Needs verification: 150', { exact: false }).waitFor()
    await page.getByText('Evidence 0 · code_review', { exact: true }).click()
    await page.getByRole('link', { name: 'Open linked decision and evidence' }).waitFor()
    assert.equal(await page.getByRole('link', { name: 'Open linked decision and evidence' }).getAttribute('href'), '#/assignments/protected')
    for (let i = 0; i < 3; i++) await page.getByRole('button', { name: 'Next records' }).click()
    await page.getByText('Evidence 173 · code_review', { exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: 'Next records' }).isEnabled(), false)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(writes, [])
    assert.deepEqual(errors, [])
    unavailable = true
    await page.reload()
    await page.getByRole('button', { name: 'Preview all classifications', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'Records unavailable' }).waitFor()
    await page.close()
  }
} finally { await browser.close() }
