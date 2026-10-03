// Synthetic evidence only. Every API and websocket is intercepted; non-local
// requests are blocked. No GitHub, model, assignment creation or production I/O.
import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-projects')
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
    await page.getByRole('heading', { name: 'Projects', exact: true }).waitFor()
    const checkout = page.getByRole('region', { name: 'Project checkout' })
    const ci = page.getByRole('region', { name: 'GitHub Actions evidence' })
    await ci.getByRole('link', { name: 'Failed CI #2', exact: true }).waitFor()
    assert.equal(requests.filter(request => request.path === '/api/projects/mindmoor/ci').length, 0, 'Only selected repo CI is fetched')
    assert.ok((await checkout.innerText()).includes(projects[0].branch))
    assert.ok((await checkout.innerText()).includes(localSha))
    assert.ok(!(await checkout.innerText()).includes(remoteSha), 'Remote SHA must not replace checkout SHA')
    assert.equal(await checkout.getByRole('link', { name: localSha }).getAttribute('href'), `${projects[0].html_url}/commit/${localSha}`)
    assert.equal(await checkout.getByText('Unknown / not connected', { exact: true }).count(), 1)
    assert.equal(await ci.getByText('Different from checkout commit', { exact: true }).count(), 4)
    const pending = ci.getByRole('listitem').filter({ hasText: 'Pending CI' })
    assert.ok((await pending.innerText()).includes('queued'))
    assert.ok(!(await pending.innerText()).includes('success'))
    const failed = ci.getByRole('listitem').filter({ hasText: 'Failed CI' })
    assert.ok((await failed.innerText()).includes('completed / failure'))
    assert.ok((await failed.innerText()).includes('Source branch: feat/remote-source-branch'))
    assert.ok((await failed.innerText()).includes(remoteSha))
    assert.equal(await failed.getByRole('link').getAttribute('href'), 'https://github.com/example/aiia/actions/runs/2')
    assert.equal(await ci.getByText('completed / conclusion unavailable', { exact: true }).count(), 1)
    assert.equal(await ci.locator('time').count(), 9, 'Fetch and per-run dates remain visible')
    assert.equal(await ci.getByRole('link', { name: 'GitHub Actions', exact: true }).getAttribute('href'), 'https://github.com/example/aiia/actions')

    const main = page.getByRole('main')
    assert.equal(await main.evaluate(element => getComputedStyle(element).overflowY), 'auto')
    for (const target of [page.getByRole('button', { name: 'Refresh projects and CI' }), page.getByLabel('Repository', { exact: true }), failed.getByRole('link')]) {
      assert.ok((await target.boundingBox()).height >= 44, 'Touch target must be at least 44px')
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    await page.screenshot({ path: join(output, `projects-${width}.png`) })
    await failed.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `projects-ci-${width}.png`) })

    mode = 'github_failure'
    await page.getByRole('button', { name: 'Refresh projects and CI' }).click()
    await ci.getByRole('alert').waitFor()
    assert.ok((await ci.innerText()).includes('Check GitHub CLI authentication'))
    assert.equal(await ci.getByRole('listitem').count(), 0, 'Unavailable response clears prior evidence')
    assert.equal(await ci.getByText('No workflow runs reported by GitHub.', { exact: true }).count(), 0)
    await ci.scrollIntoViewIfNeeded()
    await page.screenshot({ path: join(output, `projects-unavailable-${width}.png`) })

    mode = 'http_failure'
    await page.getByRole('button', { name: 'Refresh projects and CI' }).click()
    await ci.getByText(/CI evidence unavailable: fixture_unavailable/).waitFor()
    assert.equal(await ci.getByRole('listitem').count(), 0)
    mode = 'empty'
    await page.getByRole('button', { name: 'Refresh projects and CI' }).click()
    await ci.getByText('No workflow runs reported by GitHub.', { exact: true }).waitFor()

    mode = 'runs'
    await page.getByLabel('Repository', { exact: true }).selectOption('mindmoor')
    await ci.getByRole('link', { name: 'Failed CI #2' }).waitFor()
    assert.equal(await ci.getByRole('link', { name: 'Failed CI #2' }).getAttribute('href'), 'https://github.com/example/mindmoor/actions/runs/2')
    assert.equal(await page.getByLabel('Repository agent').locator('option').count(), 1)
    const createWork = page.getByRole('link', { name: 'Create work', exact: true })
    assert.equal(await createWork.getAttribute('href'), '#/assignments?agent=mindmoor-agent')
    await createWork.click()
    await page.getByRole('complementary', { name: 'New work', exact: true }).waitFor()
    assert.equal(await page.getByLabel(/^Assigned agent/).inputValue(), 'mindmoor-agent')
    assert.ok(requests.every(request => request.method === 'GET'), 'Opening work must not run an agent or create work')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    assert.deepEqual(blocked, [], 'No unexpected or live requests allowed')
    console.log(`${width}px: checkout/run SHA separation, failed and pending CI, source links/dates, explicit GitHub/HTTP failures, refresh recovery, empty state, scoped fetch/work route, 44px controls and mobile scrolling passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
