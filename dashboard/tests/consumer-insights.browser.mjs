import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const dist = process.env.STUDIO_DIST_DIR
const base = dist ? 'http://studio.test/' : process.env.STUDIO_URL
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const asOf = '2026-11-18T16:00:00Z'
const segments = [{ id: 'healthcare', label: 'Healthcare' }, { id: 'client-bd', label: 'Client research' }]

function prospect(id, organization, segment, reviewStatus = 'human_review_required') {
  const evidence = {
    id: `job-${id}:0`, job_id: `job-${id}`, index: 0, bot_id: segment,
    created_at: asOf, source_url: `https://example.org/evidence/${id}`,
    source_quote: `${organization} announced a new public research program.`,
    signals: ['Public program announcement'], insights: ['A hypothesis requiring further research'],
    review_status: reviewStatus, candidate_status: 'supported', provenance: {},
  }
  return {
    id, organization, segment, tenant_id: `tenant-${segment}`, packet_count: 1,
    pending_packet_count: reviewStatus === 'human_review_required' ? 1 : 0,
    latest_at: asOf, review_status: reviewStatus, source_url: evidence.source_url,
    source_quote: evidence.source_quote, signals: evidence.signals, insights: evidence.insights,
    insight_status: 'unverified_hypothesis', job_id: evidence.job_id, index: 0,
    bot_id: segment, href: `#/insights?segment=${segment}&prospect=${id}`, sources: [evidence],
  }
}

const prospects = [
  prospect('orchard', 'Orchard Clinic', 'healthcare'),
  prospect('cedar', 'Cedar Health', 'healthcare', 'watch'),
  prospect('summit', 'Summit Research', 'client-bd'),
]

function research(segment) {
  const rows = prospects.filter(row => segment === 'all' || row.segment === segment)
  return {
    as_of: asOf, segment, scope: 'all_retained_live_jobs', trend_timezone: 'UTC',
    available_segments: segments,
    totals: { research_jobs: 12, completed_jobs: 10, failed_jobs: 2, packets: rows.length,
      prospects: rows.length, insights: rows.length, sources: rows.length,
      awaiting_review: rows.filter(row => row.review_status === 'human_review_required').length },
    segments: segments.filter(row => segment === 'all' || row.id === segment).map(row => ({
      ...row, packets: rows.filter(p => p.segment === row.id).length,
      prospects: rows.filter(p => p.segment === row.id).length,
      insights: rows.filter(p => p.segment === row.id).length,
    })),
    review: ['human_review_required', 'watch'].map(status => ({ status, count: rows.filter(row => row.review_status === status).length })),
    evidence: { supported: rows.length, inconclusive: 0, rejected: 0, unknown: 0 },
    job_outcomes: { supported: 3, inconclusive: 7, rejected: 0, unknown: 0 },
    source_coverage: { with_quote: rows.length, with_url: rows.length, domains: 1 },
    trend: [{ date: '2026-11-17', packets: 2, prospects: 2 }, { date: '2026-11-18', packets: 1, prospects: 1 }],
    excluded_replay_jobs: 2, excluded_synthetic_packets: 1, prospects: rows,
  }
}

function demographics(state, status = 'ok') {
  const unavailable = status === 'unavailable'
  return {
    status, geography: { id: state, name: state === '27' ? 'Minnesota' : 'United States' },
    geographies: [{ id: 'us', name: 'United States' }, { id: '27', name: 'Minnesota' }],
    year: 2024, period: '2020–2024',
    source: { name: 'U.S. Census Bureau · ACS via Census Reporter', provider: 'Census Reporter',
      url: 'https://www.census.gov/programs-surveys/acs', api_url: 'https://api.censusreporter.org/1.0/data/show/acs2024_5yr' },
    fetched_at: unavailable ? null : asOf,
    metrics: unavailable ? { population: null, households: null, median_household_income: null, median_age: null }
      : { population: state === '27' ? 5700000 : 330000000, households: 130000000, median_household_income: 80000, median_age: 39 },
    age_bands: [{ label: 'Under 18', value: unavailable ? null : 66000000, share: unavailable ? null : 20 },
      { label: '18–64', value: unavailable ? null : 198000000, share: unavailable ? null : 60 },
      { label: '65+', value: unavailable ? null : 66000000, share: unavailable ? null : 20 }],
    income_bands: [{ label: 'Under $50k', value: unavailable ? null : 52000000, share: unavailable ? null : 40 },
      { label: '$50k and above', value: unavailable ? null : 78000000, share: unavailable ? null : 60 }],
    error: unavailable ? 'Population data is temporarily unavailable.' : null,
  }
}

async function open(width, hash = '#/insights') {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' })
  const page = await context.newPage()
  page.setDefaultTimeout(12000)
  const errors = []
  const requests = []
  const state = { researchUnavailable: false, demographicsStatus: 'ok' }
  page.on('pageerror', error => errors.push(error.message))
  await page.routeWebSocket('**/ws', socket => socket.onMessage(() => {}))
  if (dist) {
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(dist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: asset.endsWith('.webp') ? 'image/webp' : asset.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(dist, asset)) })
    })
  }
  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    requests.push({ method: request.method(), path: url.pathname, query: url.search })
    if (url.pathname === '/api/bd/insights') {
      return route.fulfill(state.researchUnavailable
        ? { status: 503, json: { detail: 'consumer_insights_unavailable' } }
        : { json: research(url.searchParams.get('segment') || 'all') })
    }
    if (url.pathname === '/api/consumer-insights/demographics') {
      return route.fulfill({ json: demographics(url.searchParams.get('state') || 'us', state.demographicsStatus) })
    }
    const bodies = {
      '/api/agents': { agents: [] },
      '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
      '/api/agents/models': { default: 'synthetic', models: [] },
      '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
      '/api/monitor': { services: {} },
      '/api/voice/status': { status: 'not_configured', configured: false, tools: [] },
      '/api/studio/morning-note': { date: '2026-11-18', products: [], customers: [], decisions: [], needs_you_total: 0, failure: null },
      '/api/assignments': { assignments: [] },
      '/api/memory-inbox': { ideas: [], total: 0, offset: 0, counts: { unreviewed: 0, promoted: 0, dismissed: 0 } },
      '/api/tasks': [],
    }
    return route.fulfill({ json: bodies[url.pathname] || {} })
  })
  await page.goto(`${base}${hash}`)
  return { context, page, errors, requests, state }
}

try {
  for (const width of [1440, 390, 320]) {
    const { context, page, errors, requests } = await open(width, '#/note')
    await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Insights', exact: true }).click()
    await page.getByRole('heading', { name: 'Consumer Insights', exact: true }).waitFor()
    await page.getByRole('link', { name: /Orchard Clinic/ }).waitFor()
    assert.equal(await page.getByRole('navigation', { name: 'Studio' }).getByRole('link', { name: 'Insights', exact: true }).getAttribute('aria-current'), 'page')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    const navTop = (await page.getByRole('navigation', { name: 'Studio' }).boundingBox()).y
    await page.locator('.ci-metric-primary').click()
    assert.equal((await page.getByRole('navigation', { name: 'Studio' }).boundingBox()).y, navTop)
    await page.locator('.studio-surface').evaluate(element => { element.scrollTop = 0 })
    const screenshotDir = process.env.SCREENSHOT_DIR
    if (screenshotDir) {
      await mkdir(screenshotDir, { recursive: true })
      await page.screenshot({ path: join(screenshotDir, `insights-${width}.png`), animations: 'disabled' })
    }
    await page.getByRole('link', { name: /Orchard Clinic/ }).click()
    await page.getByRole('dialog').getByText('Orchard Clinic announced a new public research program.', { exact: true }).waitFor()
    assert.ok((await page.evaluate(() => location.hash)).includes('prospect=orchard'))
    await page.reload()
    await page.getByRole('dialog').getByText('Orchard Clinic announced a new public research program.', { exact: true }).waitFor()
    assert.equal(await page.getByRole('dialog').getByRole('link', { name: 'example.org' }).getAttribute('href'), 'https://example.org/evidence/orchard')
    await page.goBack()
    await page.getByRole('heading', { name: 'Consumer Insights', exact: true }).waitFor()
    assert.equal(await page.evaluate(() => location.hash.includes('prospect=')), false)
    assert.deepEqual(errors, [])
    assert.ok(requests.every(request => request.method === 'GET'))
    await context.close()
  }
  const filtered = await open(1440, '#/insights?segment=healthcare&state=27')
  await filtered.page.getByRole('link', { name: /Orchard Clinic/ }).waitFor()
  assert.equal(await filtered.page.getByRole('link', { name: /Summit Research/ }).count(), 0)
  assert.ok(filtered.requests.some(request => request.path.endsWith('/demographics') && request.query.includes('state=27')))
  await filtered.page.getByLabel('Search prospects').fill('Cedar')
  await filtered.page.getByRole('button', { name: 'Search', exact: true }).click()
  await filtered.page.getByRole('link', { name: /Orchard Clinic/ }).waitFor({ state: 'detached' })
  assert.equal(await filtered.page.getByRole('link', { name: /Orchard Clinic/ }).count(), 0)
  await filtered.page.getByRole('link', { name: /Cedar Health/ }).waitFor()
  await filtered.page.reload()
  assert.equal(await filtered.page.getByLabel('Search prospects').inputValue(), 'Cedar')
  assert.equal(await filtered.page.getByLabel('Census geography').inputValue(), '27')
  await filtered.page.getByRole('button', { name: 'Clear filters' }).click()
  await filtered.page.getByRole('button', { name: 'Clear filters' }).waitFor({ state: 'detached' })
  await filtered.page.getByLabel('Prospect review status').selectOption('human_review_required')
  await filtered.page.getByRole('link', { name: /Orchard Clinic/ }).waitFor()
  await filtered.page.getByRole('link', { name: /Cedar Health/ }).waitFor({ state: 'detached' })
  assert.equal(await filtered.page.getByRole('link', { name: /Cedar Health/ }).count(), 0)
  await filtered.page.getByLabel('Research segment').selectOption('client-bd')
  await filtered.page.getByRole('link', { name: /Summit Research/ }).waitFor()
  assert.equal(await filtered.page.getByRole('link', { name: /Orchard Clinic/ }).count(), 0)
  assert.deepEqual(filtered.errors, [])
  await filtered.context.close()

  const failure = await open(390)
  await failure.page.getByRole('link', { name: /Orchard Clinic/ }).waitFor()
  failure.state.demographicsStatus = 'unavailable'
  await failure.page.reload()
  await failure.page.getByRole('link', { name: /Orchard Clinic/ }).waitFor()
  await failure.page.getByText(/temporarily unavailable/i).first().waitFor()
  failure.state.demographicsStatus = 'ok'
  failure.state.researchUnavailable = true
  await failure.page.reload()
  await failure.page.getByRole('heading', { name: 'Consumer Insights', exact: true }).waitFor()
  await failure.page.getByText(/unavailable/i).first().waitFor()
  assert.deepEqual(failure.errors, [])
  await failure.context.close()
  console.log('Consumer Insights browser checks passed: navigation, phone layouts, source details, deep links, filters and independent data failures.')
} finally {
  await browser.close()
}
