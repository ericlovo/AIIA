// Synthetic evidence only. Every API and websocket is intercepted; non-local
// requests are blocked. No GitHub, model, assignment creation or production I/O.
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-delivery')
const dist = process.env.STUDIO_DIST_DIR || join(process.cwd(), 'dist')
const studioUrl = new URL('http://studio.test/')
studioUrl.hash = '/projects'
await mkdir(output, { recursive: true })

const checkedAt = '2026-10-02T12:00:00Z'
const localSha = 'a'.repeat(40)
const remoteSha = 'b'.repeat(40)
const projects = ['aiia', 'mindmoor'].map(id => ({
  id, name: id === 'aiia' ? 'AIIA' : 'Mindmoor', path: `/fixture/checkouts/${id}`,
  mounted: true, checkout_status: 'available',
  branch: 'feat/local-checkout-with-a-long-mobile-branch-name', head_sha: localSha,
  detached: false, github_repo: `example/${id}`, html_url: `https://github.com/example/${id}`,
  checked_at: checkedAt, errors: [], deployment: { status: 'unknown', connection: 'not_connected' },
}))
const agents = projects.map(project => ({
  id: `${project.id}-agent`, name: `${project.name} engineer`, repo_id: project.id,
  mission: 'Inspect supplied evidence.', status: 'idle', skills: ['Coding'],
  tools: ['Repository read', 'GitHub read'], loop_enabled: false, runs: [],
}))

function evidence(project) {
  const states = [['Pending CI', 'queued', null], ['Failed CI', 'completed', 'failure'],
    ['Older successful CI', 'completed', 'success'], ['Unrecorded outcome', 'completed', null]]
  return {
    project, status: 'available', provider: 'github_actions', scope: 'repository_recent_runs',
    source_url: `${project.html_url}/actions`,
    api_url: `https://api.github.com/repos/${project.github_repo}/actions/runs?per_page=20&page=1`,
    attempted_at: checkedAt, fetched_at: checkedAt, limit: 20, total_count: 45, has_more: true,
    runs: states.map(([name, status, conclusion], index) => ({
      id: index + 1, name, status, conclusion,
      html_url: `${project.html_url}/actions/runs/${index + 1}`,
      head_sha: remoteSha, head_branch: 'feat/remote-source-branch',
      created_at: checkedAt, updated_at: checkedAt, fetched_at: checkedAt,
    })),
    errors: [],
  }
}

const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, serviceWorkers: 'block' })
    const page = await context.newPage()
    page.setDefaultTimeout(15_000)
    const errors = []
    const requests = []
    const blocked = []
    let mode = 'runs'
    let saveFails = false
    let savedBody
    page.on('pageerror', error => errors.push(error.message))
    await context.routeWebSocket('**/*', socket => {
      if (new URL(socket.url()).hostname !== studioUrl.hostname) blocked.push(socket.url())
      socket.onMessage(() => {})
    })
    await context.route('**/*', async route => {
      const request = route.request()
      const url = new URL(request.url())
      if (url.origin !== studioUrl.origin) {
        blocked.push(request.url())
        return route.abort('blockedbyclient')
      }
      if (url.pathname === '/') return route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) })
      if (/^\/assets\/[A-Za-z0-9_.-]+\.(css|js)$/.test(url.pathname)) return route.fulfill({
        contentType: url.pathname.endsWith('.css') ? 'text/css' : 'text/javascript',
        body: await readFile(join(dist, url.pathname.slice(1))),
      })
      if (!url.pathname.startsWith('/api/')) {
        blocked.push(url.pathname)
        return route.abort('blockedbyclient')
      }
      requests.push({ path: url.pathname, method: request.method() })
      if (request.method() === 'PUT' && url.pathname === '/api/projects/mindmoor/features/f1/links') {
        savedBody = request.postDataJSON()
        return route.fulfill({ status: saveFails ? 503 : 200, contentType: 'application/json', body: JSON.stringify(saveFails ? { detail: 'fixture_save_failure' } : { feature: {} }) })
      }
      if (request.method() !== 'GET') {
        blocked.push(`${request.method()} ${url.pathname}`)
        return route.fulfill({ status: 405, body: '{}' })
      }
      let body
      if (url.pathname === '/api/projects') body = { projects }
      else if (/^\/api\/projects\/(aiia|mindmoor)\/ci$/.test(url.pathname)) {
        const project = projects.find(item => url.pathname.includes(`/${item.id}/`))
        body = evidence(project)
        if (mode === 'http_failure') return route.fulfill({
          status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'fixture_unavailable' }),
        })
        if (mode === 'github_failure') body = {
          ...body, status: 'unavailable', fetched_at: null, total_count: null, has_more: null, runs: [],
          errors: [{ code: 'github_api_unavailable', message: 'Check GitHub CLI authentication, Actions read access and rate limits, then refresh.' }],
        }
        if (mode === 'empty') body = { ...body, runs: [], total_count: 0, has_more: false }
      }
      else if (url.pathname === '/api/projects/mindmoor/delivery') {
        if (mode === 'delivery_failure') return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'fixture_delivery_failure' }) })
        const missing = mode === 'missing_remote'
        const assignment = { id: 'a1', title: 'Implement feature', status: 'completed', agent_id: 'mindmoor-agent', agent_name: 'Mindmoor engineer', review_status: 'unreviewed' }
        body = { repo_id: 'mindmoor', checked_at: checkedAt, status: missing ? 'partial' : 'available', assignment_candidates: [assignment], errors: missing ? [{ code: 'missing', message: 'GitHub remote unavailable' }] : [], features: [
          { id: 'f1', updated_at: checkedAt, title: 'Member workspace', status: 'active', priority: 'P1', assignment_ids: ['a1'], pr_numbers: [12], assignments: [assignment], pull_requests: [{ number: 12, status: missing ? 'unavailable' : 'available', title: 'Member workspace changes', state: 'open', merged: false, draft: false, mergeable: null, head_sha: remoteSha, html_url: missing ? undefined : 'https://github.com/example/mindmoor/pull/12', fetched_at: checkedAt, errors: [], ci: { status: missing ? 'unavailable' : 'available', fetched_at: missing ? null : checkedAt, scope: 'pull_request_head', head_sha: remoteSha, errors: [], runs: [{ id: 9, workflow_id: 7, name: 'Head check', html_url: 'https://github.com/example/mindmoor/actions/runs/9', head_sha: remoteSha, status: 'completed', conclusion: 'failure', updated_at: checkedAt }, { id: 10, workflow_id: 7, name: 'Old successful check', html_url: 'https://github.com/example/mindmoor/actions/runs/10', head_sha: localSha, status: 'completed', conclusion: 'success', updated_at: checkedAt }] } }] },
          { id: 'f2', title: 'Unlinked feature', status: 'backlog', priority: 'P2', assignment_ids: [], pr_numbers: [], assignments: [], pull_requests: [] },
        ], sources: { roadmap: { status: 'available', fetched_at: checkedAt }, assignments: { status: 'available', fetched_at: checkedAt } },
          deployments: { checked_at: checkedAt, status: missing ? 'unavailable' : 'available', provider: 'github_deployments', scope: 'recent_reported_deployments', coverage: 'bounded_recent_history', has_more: false, truncated: false,
            notice: 'Last provider-reported status per exact environment; no live probe.',
            errors: missing ? [{ code: 'deployment_unavailable', message: 'Deployment provider unavailable' }] : [],
            environments: missing ? [] : [
              { environment: 'Preview', sha: remoteSha, deployment_id: 31, state: 'success', created_at: checkedAt, updated_at: checkedAt, source_url: 'https://github.com/example/mindmoor/deployments/31', environment_url: 'https://preview.example.test' },
              { environment: 'Production', sha: localSha, deployment_id: 29, state: 'failure', created_at: checkedAt, updated_at: checkedAt, source_url: 'https://github.com/example/mindmoor/deployments/29', environment_url: null },
            ],
          },
        }
      }
      else if (url.pathname === '/api/agents') body = { agents }
      else if (url.pathname === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (url.pathname === '/api/agents/models') body = { default: '', models: [] }
      else if (url.pathname === '/api/integrations/typesafe/status') body = { ready: false, enabled: false, configured: false }
      else if (url.pathname === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (url.pathname === '/api/monitor') body = { services: {} }
      else if (url.pathname === '/api/assignments') body = { assignments: [] }
      else if (url.pathname === '/api/handoffs') body = { handoffs: [] }
      else if (url.pathname === '/api/git-workspaces') body = { workspaces: [] }
      else if (url.pathname === '/api/git-writes') body = { writes: [] }
      else {
        blocked.push(url.pathname)
        return route.fulfill({ status: 503, body: '{}' })
      }
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
    })

    await page.goto(studioUrl.href)
    await page.getByLabel('Repository', { exact: true }).selectOption('mindmoor')
    const delivery = page.getByRole('region', { name: 'Feature delivery' })
    const feature = delivery.getByRole('article', { name: 'Member workspace', exact: true })
    await feature.getByRole('link', { name: 'Head check #9' }).waitFor()
    assert.equal(await delivery.getByText('Old successful check #10').count(), 0)
    assert.ok((await feature.innerText()).includes('Needs review'))
    await delivery.getByRole('button', { name: 'Blocked (1)', exact: true }).click()
    assert.equal(await feature.count(), 1)
    assert.equal(await delivery.getByRole('article').count(), 1)
    await feature.getByText('Failed workflow observed at a linked PR head. Required-check readiness is unverified.').waitFor()
    await delivery.getByRole('button', { name: 'All (2)' }).click()
    const deployments = delivery.locator('[aria-label="Deployment evidence"]')
    const preview = deployments.locator('div').filter({ has: page.getByText('Preview / success', { exact: true }) })
    const production = deployments.locator('div').filter({ has: page.getByText('Production / failure', { exact: true }) })
    assert.ok((await preview.innerText()).includes(remoteSha))
    assert.ok((await production.innerText()).includes(localSha))
    assert.equal(await preview.getByRole('link').getAttribute('href'), 'https://github.com/example/mindmoor/deployments/31')
    assert.equal(await production.getByRole('link').getAttribute('href'), 'https://github.com/example/mindmoor/deployments/29')
    assert.ok((await deployments.innerText()).includes('not a live health probe'))
    assert.ok((await feature.innerText()).includes('Feature release: Unverified'))
    await deployments.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `delivery-deployments-${width}.png`) })
    assert.ok((await feature.innerText()).includes('completed / failure'))
    assert.equal(await feature.getByRole('link', { name: 'Implement feature' }).getAttribute('href'), '#/assignments/a1')
    assert.equal(await feature.getByRole('link', { name: 'Mindmoor engineer' }).getAttribute('href'), '#/agents/mindmoor-agent')
    await delivery.getByRole('button', { name: 'Unlinked (1)' }).click()
    assert.equal(await feature.count(), 0)
    await delivery.getByRole('button', { name: 'All (2)' }).click()
    await feature.getByRole('button', { name: 'Edit links' }).click()
    await feature.getByLabel('PR numbers (comma separated)').fill('12, 34')
    saveFails = true
    await feature.getByRole('button', { name: 'Save links' }).click()
    await feature.getByText(/Links were not saved: fixture_save_failure/).waitFor()
    assert.equal(await feature.getByLabel('PR numbers (comma separated)').inputValue(), '12, 34')
    await feature.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `delivery-save-failure-${width}.png`) })
    saveFails = false
    await feature.getByRole('button', { name: 'Save links' }).click()
    await feature.getByRole('button', { name: 'Save links' }).waitFor({ state: 'detached' })
    assert.deepEqual(savedBody, { assignment_ids: ['a1'], pr_numbers: [12, 34], expected_updated_at: checkedAt })
    mode = 'missing_remote'
    await delivery.getByRole('button', { name: 'Refresh delivery' }).click()
    await delivery.getByText('GitHub remote unavailable').waitFor()
    await deployments.getByText('Deployment provider unavailable').waitFor()
    assert.equal(await deployments.getByText('Preview / success', { exact: true }).count(), 0)
    assert.equal(await deployments.getByText('Production / failure', { exact: true }).count(), 0)
    await delivery.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `delivery-missing-remote-${width}.png`) })
    mode = 'delivery_failure'
    await delivery.getByRole('button', { name: 'Refresh delivery' }).click()
    await delivery.getByText(/Delivery unavailable: fixture_delivery_failure/).waitFor()
    assert.equal(await delivery.getByRole('article').count(), 0)
    await page.getByLabel('Repository', { exact: true }).selectOption('aiia')
    assert.equal(await delivery.count(), 0, 'Switching scope removes Mindmoor evidence and editor')
    mode = 'runs'
    await page.getByLabel('Repository', { exact: true }).selectOption('mindmoor')
    await delivery.getByRole('button', { name: 'Refresh delivery' }).click()
    await feature.getByRole('link', { name: 'Head check #9' }).waitFor()
    await delivery.scrollIntoViewIfNeeded()
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.screenshot({ path: join(output, `delivery-${width}.png`) })
    await feature.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `delivery-feature-${width}.png`) })
    assert.deepEqual(errors, [])
    assert.deepEqual(blocked, [])
    console.log(`${width}px: scoped delivery, review semantics, head SHA checks, explicit link saves and failure recovery, unavailable remote/request, scope switch passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
