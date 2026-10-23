// Run only against the explicitly isolated synthetic preview, never production.
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
const base = process.env.BD_PREVIEW_URL
if (base !== 'http://127.0.0.1:8231') throw new Error('Set BD_PREVIEW_URL=http://127.0.0.1:8231 for isolated synthetic preview')
const output = process.env.SCREENSHOT_DIR || '/private/tmp/aiia-bd-live'
await mkdir(output, { recursive: true })
const initial = await (await fetch(`${base}/api/bd/snapshot`)).json()
assert.equal(initial.mode, 'replay')
assert.equal(initial.schedule_enabled, false)
assert(initial.bots.every(bot => bot.tenant_id.startsWith('synthetic-')))
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  const errors = [], streamSnapshots = []
  page.on('pageerror', error => errors.push(error.message))
  await page.exposeFunction('recordSnapshot', value => streamSnapshots.push(value))
  await page.addInitScript(() => {
    const NativeSource = window.EventSource
    window.EventSource = class extends NativeSource {
      constructor(...args) { super(...args); this.addEventListener('snapshot', event => window.recordSnapshot(JSON.parse(event.data))) }
    }
  })
  await page.goto(`${base}/#/bd`)
  await page.getByText('Shadow · replay only', { exact: true }).waitFor()
  await page.getByText('Live', { exact: true }).waitFor()
  await page.getByRole('button', { name: 'Run shadow batch' }).click()
  await page.getByText('Shadow batch accepted.', { exact: false }).waitFor()
  await page.waitForFunction(() => document.querySelectorAll('.bd-jobs li').length > 0, null, { timeout: 10000 }).catch(() => {})
  await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
  await page.screenshot({ path: `${output}/desktop-running.png`, fullPage: true })
  await page.locator('.bd-activity-board').screenshot({ path: `${output}/desktop-action-running.png` })
  await page.waitForFunction(() => document.querySelectorAll('.bd-packet').length >= 4, null, { timeout: 60000 })
  await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
  await page.screenshot({ path: `${output}/desktop-complete.png`, fullPage: true })
  await page.locator('.bd-activity-board').screenshot({ path: `${output}/desktop-action-complete.png` })
  await page.locator('.bd-packet summary').first().click()
  await page.screenshot({ path: `${output}/desktop-evidence.png`, fullPage: true })
  const final = await (await fetch(`${base}/api/bd/snapshot`)).json()
  assert(final.jobs.filter(job => job.state === 'completed').length >= 4)
  assert(final.packets.length >= 4)
  assert(final.model_activity.qwen.replayed >= 4)
  assert(final.model_activity.jev.replayed >= 4)
  assert.equal(final.model_activity.qwen.calls, 0)
  assert.equal(final.model_activity.jev.calls, 0)
  assert(streamSnapshots.some(snapshot => snapshot.jobs.some(job => job.state === 'completed')))
  assert.equal(await page.locator('.bd-page').evaluate(el => el.scrollWidth > el.clientWidth), false)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.locator('.bd-page').evaluate(el => { el.scrollTop = 0 })
  await page.screenshot({ path: `${output}/mobile-complete.png`, fullPage: true })
  await page.locator('.bd-delivered-panel').evaluate(el => el.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: `${output}/mobile-delivered.png` })
  await page.locator('.bd-model-stack').evaluate(el => el.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: `${output}/mobile-models.png` })
  await page.locator('.bd-queue-panel').evaluate(el => el.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: `${output}/mobile-queue.png` })
  assert.equal(await page.locator('.bd-page').evaluate(el => el.scrollWidth > el.clientWidth), false)
  await page.locator('.bd-packet').first().getByRole('button', { name: 'Research next', exact: true }).click()
  await page.getByText('Local research triage saved.', { exact: true }).waitFor()
  await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('research_next')
  await page.locator('.bd-packet').first().waitFor()
  const triaged = await (await fetch(`${base}/api/bd/snapshot`)).json()
  assert(triaged.packets.some(packet => packet.review_status === 'research_next'))
  await page.locator('.bd-packet').first().getByRole('button', { name: 'Dismiss', exact: true }).click()
  await page.getByText('No packets in this view yet.', { exact: false }).waitFor()
  await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('dismissed')
  await page.locator('.bd-packet').first().getByRole('button', { name: 'Restore', exact: true }).click()
  await page.getByRole('combobox', { name: 'Packet review status' }).selectOption('human_review_required')
  await page.locator('.bd-packet').first().waitFor()
  if (!(await page.locator('.bd-packet details').first().getAttribute('open'))) {
    await page.locator('.bd-packet details').first().evaluate(el => { el.open = true })
  }
  await page.screenshot({ path: `${output}/mobile-evidence.png`, fullPage: true })
  assert.deepEqual(errors, [])
  await writeFile(`${output}/verification.json`, JSON.stringify({ completed: final.jobs.filter(job => job.state === 'completed').length, packets: final.packets.length, snapshots: streamSnapshots.length, modelActivity: final.model_activity, triageVerified: ['research_next', 'dismissed', 'human_review_required'], observedRunning: streamSnapshots.some(snapshot => snapshot.jobs.some(job => job.state === 'running')), errors }, null, 2))
  console.log(await (await import('node:fs/promises')).readFile(`${output}/verification.json`, 'utf8'))
} finally { await browser.close() }
