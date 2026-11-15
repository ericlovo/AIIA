// Opt-in integration test against a running local Studio and its seeded dev database.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { chromium } from 'playwright'

const { DESK_NEON_STUDIO_URL, DESK_NEON_RUN_ID, STUDIO_DESK_DEV_TENANT_ID, SCREENSHOT_DIR } = process.env
if (!DESK_NEON_STUDIO_URL && !DESK_NEON_RUN_ID && process.env.STUDIO_URL) {
  console.log('Skipped Neon dev integration smoke in the offline browser suite.')
  process.exit(0)
}
assert.ok(DESK_NEON_STUDIO_URL && DESK_NEON_RUN_ID, 'Set DESK_NEON_STUDIO_URL and DESK_NEON_RUN_ID from the dev fixture.')
const origin = new URL(DESK_NEON_STUDIO_URL).origin
assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(new URL(origin).hostname), 'Use a loopback Studio URL.')
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
assert.match(DESK_NEON_RUN_ID, uuid)
const org = 'Synthetic Orchard'
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  page.setDefaultTimeout(15000)
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const nav = page.getByRole('navigation', { name: 'Studio', exact: true })
  const evidence = page.getByRole('region', { name: 'Run evidence' })
  async function responseTo(path, action, method = 'GET') {
    const [response] = await Promise.all([
      page.waitForResponse(r => r.url() === origin + path && r.request().method() === method),
      action(),
    ])
    assert.ok(response.ok(), method + ' ' + path + ' returned ' + response.status())
    return response.json()
  }
  async function screenshot(name) {
    if (SCREENSHOT_DIR) {
      await mkdir(SCREENSHOT_DIR, { recursive: true })
      await page.screenshot({ path: join(SCREENSHOT_DIR, name + '.png'), animations: 'disabled' })
    }
  }
  function assertFixture(row) {
    assert.equal(row.job_id, DESK_NEON_RUN_ID)
    assert.equal(row.organization, org)
    assert.equal(row.mode, 'replay')
    assert.match(row.tenant_id, uuid)
    if (STUDIO_DESK_DEV_TENANT_ID) assert.equal(row.tenant_id, STUDIO_DESK_DEV_TENANT_ID)
    assert.match(row.bot_id, uuid)
  }
  const queue = await responseTo('/api/bd/desk', () => page.goto(origin + '/#/desk'))
  const pending = queue.packets.find(p => p.job_id === DESK_NEON_RUN_ID)
  let receipt, link
  if (pending) {
    assertFixture(pending)
    assert.equal(pending.index, 0)
    assert.equal(pending.review_status, 'human_review_required')
    const packet = page.getByRole('region', { name: 'BD Research packet' })
      .filter({ has: page.getByText(DESK_NEON_RUN_ID, { exact: true }) })
    await packet.getByRole('heading', { name: org, exact: true }).waitFor()
    await packet.getByText('Synthetic replay', { exact: true }).waitFor()
    await packet.getByText(pending.tenant_id, { exact: true }).waitFor()
    await packet.getByText(pending.bot_id, { exact: true }).waitFor()
    await screenshot('neon-desk-pending')
    const review = await responseTo(
      '/api/bd/desk/' + pending.bot_id + '/' + pending.job_id + '/0/review',
      () => packet.getByRole('button', { name: 'Watch', exact: true }).click(), 'POST',
    )
    receipt = review.receipt
    assert.equal(receipt.tenant_id, pending.tenant_id)
    assert.equal(receipt.bot_id, pending.bot_id)
    link = page.getByRole('link', { name: 'View receipt in Record' })
  } else {
    const record = await responseTo('/api/bd/record', () => nav.getByRole('link', { name: 'Record', exact: true }).click())
    const matches = record.receipts.filter(r => r.job_id === DESK_NEON_RUN_ID && r.packet_index === 0)
    assert.equal(matches.length, 1, 'Expected one existing decision for the seeded run.')
    receipt = matches[0]
    await page.getByRole('searchbox', { name: 'Find a receipt' }).fill(DESK_NEON_RUN_ID)
    link = page.getByRole('main').getByRole('link').filter({ hasText: DESK_NEON_RUN_ID })
  }
  assertFixture(receipt)
  assert.equal(receipt.status, 'watch')
  assert.equal(receipt.packet_index, 0)
  assert.match(receipt.id, uuid)
  await link.waitFor()
  const hash = '#/record/' + receipt.id
  assert.equal(await link.getAttribute('href'), hash)
  const path = '/api/bd/record/' + receipt.id
  const detail = await responseTo(path, () => link.click())
  assert.deepEqual(detail.receipt, receipt)
  assertFixture(detail.packet)
  assert.equal(detail.packet.index, 0)
  assert.equal(detail.packet.review_status, 'watch')
  await page.getByRole('heading', { name: 'Decision receipt' }).waitFor()
  await page.getByRole('main').getByText('Watch', { exact: true }).waitFor()
  await evidence.getByText(DESK_NEON_RUN_ID, { exact: true }).waitFor()
  await screenshot('neon-desk-receipt')
  const run = page.getByRole('link', { name: 'Inspect run ' + DESK_NEON_RUN_ID, exact: true })
  assert.equal(await run.getAttribute('href'), hash + '?show=run')
  await run.click()
  await page.getByRole('heading', { name: 'Run record' }).waitFor()
  const reopened = await responseTo(path, () => page.reload())
  assert.deepEqual(reopened, detail)
  assert.equal(new URL(page.url()).hash, hash + '?show=run')
  await evidence.getByText(DESK_NEON_RUN_ID, { exact: true }).waitFor()
  await evidence.getByText('job completed', { exact: true }).waitFor()
  await evidence.getByText('packet reviewed', { exact: true }).waitFor()
  await screenshot('neon-desk-run')
  await page.getByRole('link', { name: /Back to receipt/ }).click()
  await page.getByRole('heading', { name: 'Decision receipt' }).waitFor()
  await page.getByRole('main').getByText('Watch', { exact: true }).waitFor()
  assert.equal(new URL(page.url()).hash, hash)
  const finalQueue = await responseTo('/api/bd/desk', () => nav.getByRole('link', { name: 'Desk', exact: true }).click())
  assert.equal(finalQueue.packets.some(p => p.job_id === DESK_NEON_RUN_ID && p.index === 0), false)
  await page.getByRole('heading', { name: 'No packets awaiting review' }).waitFor()
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({
    action: pending ? 'saved_watch' : 'reopened_watch', tenant_id: receipt.tenant_id,
    run_id: receipt.job_id, packet_index: receipt.packet_index, receipt_id: receipt.id,
    bot_id: receipt.bot_id, decision: receipt.status, recorded_at: receipt.created_at,
    receipt_url: origin + '/' + hash, run_url: origin + '/' + hash + '?show=run',
  }, null, 2))
} finally {
  await browser.close()
}
