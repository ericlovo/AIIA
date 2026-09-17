// The Map's agent inspector shows the agent's real configuration and edits it
// one field at a time, and runs it. Every response is synthetic; no Command Center, Brain or
// model is reached.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const output = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-agent-inspector')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })
const date = '2026-09-17'

function seedAgents() {
  const base = {
    persona: 'Focused, pragmatic, and direct.', skills: [], tools: [], repo_id: '',
    temperature: 0.35, max_tokens: 1200, model: '', suite: '', memory_namespace: '',
    loop_enabled: false, loop_interval_minutes: 60, loop_task: '', loop_max_runs_per_day: 4,
    loop_runs_today: 0, loop_day: date, status: 'idle', last_run_at: null, last_result: '',
    last_error: '', runs: [], created_at: `${date}T09:00:00Z`, updated_at: `${date}T09:00:00Z`,
  }
  return [
    {
      ...base, id: 'agent-1', name: 'Release Gate Reviewer', mission: 'Gate release candidates.',
      persona: 'Skeptical reviewer who cites evidence.', skills: ['Analysis', 'Research'],
      tools: ['Repository read', 'Local memory'], repo_id: 'aiia', temperature: 0.2, max_tokens: 1600,
      suite: 'release', loop_enabled: true, loop_interval_minutes: 120, loop_task: 'Review the latest candidate.',
      loop_max_runs_per_day: 6, loop_runs_today: 2,
      last_result: `GO verdict. ${'Evidence line. '.repeat(60)}TAIL_MARKER`,
      last_error: 'local_model_unavailable',
    },
    { ...base, id: 'agent-2', name: 'Docs Scout', mission: 'Find stale docs.', model: 'llama3.1:8b' },
    { ...base, id: 'agent-3', name: 'Busy Builder', mission: 'Build in the background.', status: 'running' },
  ]
}

const models = {
  default: 'qwen3:8b',
  models: [
    { id: 'qwen3:8b', label: 'qwen3:8b', family: 'qwen3', parameter_size: '8.2B', size_gb: 5.2, default: true },
    { id: 'llama3.1:8b', label: 'llama3.1:8b', family: 'llama', parameter_size: '8.0B', size_gb: 4.9, default: false },
    { id: 'ghost:1b', label: 'ghost:1b', family: 'ghost', parameter_size: '1B', size_gb: 0.8, default: false },
  ],
}
// ghost:1b was listed, then removed from Ollama before the PATCH validated it.
const installed = new Set(['qwen3:8b', 'llama3.1:8b'])

async function until(check, message) {
  for (let i = 0; i < 200; i++) {
    if (check()) return
    await new Promise(resolve => setTimeout(resolve, 25))
  }
  assert.fail(message)
}

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } })
    const page = await context.newPage()
    page.setDefaultTimeout(12000)
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    const agents = seedAgents()
    const patchCalls = []
    let modelsDown = false
    // Holds the next PATCH response open so the optimistic value can be observed first.
    let holdPatch = null
    const runCalls = []
    let holdRun = null
    let miniBusy = false
    await page.routeWebSocket('**/ws', ws => ws.onMessage(() => {}))
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname
      let body = {}
      let status = 200
      if (path === '/api/agents') body = { agents }
      else if (path === '/api/agents/models') {
        status = modelsDown ? 503 : 200
        body = modelsDown ? { detail: 'models_unavailable' } : models
      } else if (/^\/api\/agents\/[^/]+$/.test(path) && route.request().method() === 'PATCH') {
        const id = path.split('/')[3]
        const fields = route.request().postDataJSON()
        patchCalls.push({ id, fields })
        if (holdPatch) await holdPatch
        const agent = agents.find(item => item.id === id)
        // Mirror contract C1 and C2 on the merged result.
        const merged = { ...agent, ...fields }
        if (!agent) { status = 404; body = { detail: 'agent_not_found' } }
        else if (merged.loop_enabled && !merged.loop_task.trim()) { status = 422; body = { detail: 'loop_task_required' } }
        else if ('model' in fields && fields.model && modelsDown) { status = 503; body = { detail: 'models_unavailable' } }
        else if ('model' in fields && fields.model && !installed.has(fields.model)) { status = 422; body = { detail: 'unknown_model' } }
        else { Object.assign(agent, fields, { updated_at: `${date}T10:00:00Z` }); body = { agent } }
      }
      else if (/^\/api\/agents\/[^/]+\/run$/.test(path)) {
        const id = path.split('/')[3]
        runCalls.push({ id, body: route.request().postDataJSON() })
        if (holdRun) await holdRun
        const agent = agents.find(item => item.id === id)
        if (miniBusy) { status = 409; body = { detail: 'mini_busy' } }
        else {
          Object.assign(agent, { last_result: 'Synthetic run output.', last_run_at: `${date}T11:00:00Z` })
          body = { agent, model: 'llama3.1:8b', latency_ms: 2340 }
        }
      }
      else if (path === '/api/agents/resources') body = { repos: [], github: { status: 'disconnected' } }
      else if (path === '/api/assignments') body = { assignments: [] }
      else if (path === '/api/handoffs') body = { handoffs: [] }
      else if (path === '/api/agent-world/layout') body = { layout: { version: 1, revision: 0, positions: {}, updated_at: null } }
      else if (path === '/api/tasks') body = []
      else if (path === '/api/health') body = { aiia: { status: 'online' }, ollama: { status: 'online' } }
      else if (path === '/api/monitor') body = { services: {} }
      else if (path === '/api/voice/status') body = { available: false }
      else if (path === '/api/tokens/today') body = { date, total_tokens: 0, total_requests: 0, total_cost: 0, by_provider: {}, by_purpose: {} }
      else if (path === '/api/tokens/recent') body = { days: [] }
      else if (path === '/api/studio/activity') body = { today: date, start: date, days: [], agent_days: [], runs: [], total: 0, matching: 0, imported: 0, usage_by_agent: [] }
      else if (path === '/api/memory-inbox') body = { ideas: [], total: 0, offset: 0, counts: { unreviewed: 0, promoted: 0, dismissed: 0 } }
      await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
    })

    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5191/')
    await page.getByRole('tab', { name: 'Map', exact: true }).click()
    await page.waitForFunction(() => document.querySelectorAll('[data-graph-node]').length === 3)
    const inspector = page.getByRole('complementary', { name: 'Node controls' })
    const config = inspector.getByRole('definition')
    const row = async label => {
      const terms = await inspector.locator('dt').allInnerTexts()
      const index = terms.indexOf(label)
      assert.ok(index >= 0, `missing config row ${label}`)
      return (await config.nth(index).innerText()).trim()
    }

    // A3: the real configuration, not just name and mission.
    await page.locator('[data-agent-target="agent-1"]').press('Enter')
    await inspector.waitFor()
    await inspector.getByText('Release Gate Reviewer', { exact: true }).waitFor()
    const settings = inspector.getByRole('region', { name: 'Agent settings' })
    const modelPicker = settings.getByLabel('Model', { exact: true })
    const selectedModel = () => modelPicker.evaluate(select => select.selectedOptions[0].textContent)
    await page.waitForFunction(() => document.querySelector('[aria-label="Node controls"] select')?.selectedOptions[0]?.textContent === 'Task default: qwen3:8b')
    assert.equal(await settings.getByLabel('Temperature', { exact: true }).inputValue(), '0.2')
    assert.equal(await settings.getByLabel('Max tokens', { exact: true }).inputValue(), '1600')
    assert.equal(await settings.getByLabel('Suite', { exact: true }).inputValue(), 'release')
    assert.equal(await settings.getByLabel('Loop', { exact: true }).isChecked(), true)
    assert.equal(await settings.getByLabel('Loop interval (min)', { exact: true }).inputValue(), '120')
    assert.equal(await settings.getByLabel('Loop runs per day', { exact: true }).inputValue(), '6')
    assert.ok((await settings.innerText()).includes('On · every 2h · 2 of 6 runs today'))
    assert.equal(await row('Tools'), 'Repository read, Local memory')
    assert.equal(await row('Skills'), 'Analysis, Research')
    assert.equal(await row('Repo'), 'aiia')
    assert.equal(await row('Persona'), 'Skeptical reviewer who cites evidence.')
    const text = await inspector.innerText()
    assert.ok(text.includes('GO verdict.'))
    assert.ok(!text.includes('TAIL_MARKER'), 'last result must be truncated')
    assert.ok(text.includes('local_model_unavailable'))
    const rect = await inspector.boundingBox()
    assert.ok(rect.x >= 0 && rect.x + rect.width <= width)
    await page.screenshot({ path: join(output, `inspector-config-${width}.png`) })

    // A4: a successful edit is optimistic and sends only that field.
    const temperature = settings.getByLabel('Temperature', { exact: true })
    let release
    holdPatch = new Promise(resolve => { release = resolve })
    await temperature.fill('0.55')
    await temperature.press('Enter')
    await until(() => patchCalls.length === 1, 'temperature PATCH not sent')
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-1', fields: { temperature: 0.55 } })
    assert.equal(agents[0].temperature, 0.2, 'the synthetic server has not applied it yet')
    assert.equal(await temperature.inputValue(), '0.55')
    holdPatch = null
    release()
    await until(() => agents[0].temperature === 0.55, 'temperature PATCH not applied')
    await page.waitForTimeout(100)
    assert.equal(await temperature.inputValue(), '0.55')
    assert.equal(await inspector.getByRole('alert').count(), 0)

    // Out-of-range input is caught locally and never sent.
    await temperature.fill('1.5')
    await temperature.press('Enter')
    await settings.getByText('Temperature must be between 0 and 1.', { exact: true }).waitFor()
    assert.equal(patchCalls.length, 1)
    await temperature.press('Escape')
    assert.equal(await temperature.inputValue(), '0.55')

    // The model picker lists installed models and pins one with a single-field PATCH.
    const options = await modelPicker.locator('option').allTextContents()
    assert.deepEqual(options, ['Task default: qwen3:8b', 'qwen3:8b · 8.2B', 'llama3.1:8b · 8.0B', 'ghost:1b · 1B'])
    await modelPicker.selectOption('llama3.1:8b')
    await until(() => agents[0].model === 'llama3.1:8b', 'model PATCH not applied')
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-1', fields: { model: 'llama3.1:8b' } })
    assert.equal(await selectedModel(), 'llama3.1:8b · 8.0B')

    // Suite, loop interval and daily maximum commit on blur or Enter, each alone.
    await settings.getByLabel('Suite', { exact: true }).fill('ops')
    await settings.getByLabel('Loop interval (min)', { exact: true }).focus()
    await until(() => agents[0].suite === 'ops', 'suite PATCH not applied')
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-1', fields: { suite: 'ops' } })
    await settings.getByLabel('Loop interval (min)', { exact: true }).fill('45')
    await settings.getByLabel('Loop runs per day', { exact: true }).focus()
    await until(() => agents[0].loop_interval_minutes === 45, 'interval PATCH not applied')
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-1', fields: { loop_interval_minutes: 45 } })
    await settings.getByLabel('Loop runs per day', { exact: true }).fill('8')
    await settings.getByLabel('Loop runs per day', { exact: true }).press('Enter')
    await until(() => agents[0].loop_max_runs_per_day === 8, 'daily maximum PATCH not applied')
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-1', fields: { loop_max_runs_per_day: 8 } })
    await settings.getByText('On · every 45m · 2 of 8 runs today', { exact: true }).waitFor()
    assert.equal(patchCalls.length, 5)
    await page.screenshot({ path: join(output, `inspector-edited-${width}.png`) })

    // A rejected edit shows optimistically, then rolls back with a readable message.
    // The loop toggle is operated from the keyboard.
    await inspector.getByRole('button', { name: 'Close node controls' }).click()
    await page.locator('[data-agent-target="agent-2"]').press('Enter')
    await inspector.getByText('Docs Scout', { exact: true }).waitFor()
    assert.equal(await selectedModel(), 'llama3.1:8b · 8.0B')
    assert.equal(await row('Tools'), 'None')
    assert.ok(!(await inspector.innerText()).includes('Last error'))
    const loop = settings.getByLabel('Loop', { exact: true })
    assert.equal(await loop.isChecked(), false)
    holdPatch = new Promise(resolve => { release = resolve })
    await loop.focus()
    await page.keyboard.press('Space')
    await until(() => patchCalls.length === 6, 'loop PATCH not sent')
    assert.equal(await loop.isChecked(), true)
    holdPatch = null
    release()
    const alert = inspector.getByRole('alert')
    await alert.waitFor()
    assert.match(await alert.innerText(), /^Loop not saved: the loop needs a loop task first/)
    assert.equal(await loop.isChecked(), false)
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-2', fields: { loop_enabled: true } })
    assert.equal(agents[1].loop_enabled, false)
    await page.screenshot({ path: join(output, `inspector-rollback-${width}.png`) })

    // Unknown and unverifiable models roll back with their own messages.
    await modelPicker.selectOption('ghost:1b')
    await alert.getByText('Model not saved: that model is not installed on the Mini.', { exact: true }).waitFor()
    assert.equal(await modelPicker.inputValue(), 'llama3.1:8b')
    modelsDown = true
    await modelPicker.selectOption('qwen3:8b')
    await alert.getByText(/^Model not saved: Ollama is not reachable/).waitFor()
    assert.equal(await modelPicker.inputValue(), 'llama3.1:8b')
    assert.equal(agents[1].model, 'llama3.1:8b')
    modelsDown = false

    // A5: Run now sends the task, disables itself while pending, and reports the busy Mini.
    await inspector.getByRole('button', { name: 'Close node controls' }).click()
    await page.locator('[data-agent-target="agent-1"]').press('Enter')
    await inspector.getByText('Release Gate Reviewer', { exact: true }).waitFor()
    const runForm = inspector.getByRole('form', { name: 'Run agent' })
    const runTask = runForm.getByLabel('Task for this run', { exact: true })
    const runButton = runForm.getByRole('button', { name: 'Run now', exact: true })
    assert.equal(await runButton.isDisabled(), true, 'an empty task cannot run')
    await runTask.fill('Review candidate 42.')
    holdRun = new Promise(resolve => { release = resolve })
    await runButton.click()
    await until(() => runCalls.length === 1, 'run request not sent')
    assert.deepEqual(runCalls[0], { id: 'agent-1', body: { task: 'Review candidate 42.' } })
    const working = runForm.getByRole('button', { name: 'Mini working', exact: true })
    assert.equal(await working.isDisabled(), true)
    // Reopening the inspector mid-run keeps the control disabled.
    await inspector.getByRole('button', { name: 'Close node controls' }).click()
    await page.locator('[data-agent-target="agent-1"]').press('Enter')
    assert.equal(await inspector.getByRole('form', { name: 'Run agent' }).getByRole('button', { name: 'Mini working', exact: true }).isDisabled(), true)
    await page.screenshot({ path: join(output, `inspector-run-pending-${width}.png`) })
    holdRun = null
    release()
    await runForm.getByRole('status').filter({ hasText: 'Run finished on llama3.1:8b in 2.3s.' }).waitFor()
    await inspector.getByText('Synthetic run output.', { exact: true }).waitFor()
    assert.equal(runCalls.length, 1)

    miniBusy = true
    await runTask.fill('Review candidate 43.')
    await runButton.click()
    await runForm.getByRole('alert').filter({ hasText: 'Mini busy — wait for the active run to finish.' }).waitFor()
    assert.equal(runCalls.length, 2)
    assert.equal(await runTask.inputValue(), 'Review candidate 43.', 'a refused run keeps the task for a retry')
    assert.equal(await runButton.isEnabled(), true)
    miniBusy = false
    await page.screenshot({ path: join(output, `inspector-run-busy-${width}.png`) })

    // A running agent keeps its controls usable and says the change waits for the next run.
    await inspector.getByRole('button', { name: 'Close node controls' }).click()
    await page.locator('[data-agent-target="agent-3"]').press('Enter')
    await inspector.getByText('Busy Builder', { exact: true }).waitFor()
    await settings.getByText('Running now. Changes apply to the next run.', { exact: true }).waitFor()
    assert.equal(await settings.getByLabel('Max tokens', { exact: true }).isEnabled(), true)
    await settings.getByLabel('Max tokens', { exact: true }).fill('900')
    await settings.getByLabel('Max tokens', { exact: true }).press('Enter')
    await until(() => agents[2].max_tokens === 900, 'running agent PATCH not applied')
    assert.deepEqual(patchCalls.at(-1), { id: 'agent-3', fields: { max_tokens: 900 } })

    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
    assert.deepEqual(errors, [])
    console.log(`${width}px: inspector configuration, optimistic PATCH, local validation, model picker, rollback messages, running note, run now and busy run passed`)
    await context.close()
  }
} finally {
  await browser.close()
}
console.log(`Screenshots: ${output}`)
