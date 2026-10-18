// Exercise the built Console with fully synthetic voice I/O. Handshakes and
// permission grants advance only when the test releases their explicit gates.
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const studioDist = process.env.STUDIO_DIST_DIR
const base = studioDist ? 'http://studio.test/' : process.env.STUDIO_URL || 'http://127.0.0.1:5184/'
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_PATH })

function installVoiceMocks() {
  const sockets = []
  const permissions = []
  const tracks = []
  const processors = []
  const contexts = []
  let sourceCount = 0

  class MockWebSocket extends EventTarget {
    static CONNECTING = 0
    static OPEN = 1
    static CLOSING = 2
    static CLOSED = 3
    readyState = MockWebSocket.CONNECTING
    sent = []
    constructor(url) {
      super()
      this.url = String(url)
      if (new URL(this.url).pathname === '/ws') queueMicrotask(() => this.open())
      else sockets.push(this)
    }
    open() {
      if (this.readyState !== MockWebSocket.CONNECTING) return false
      this.readyState = MockWebSocket.OPEN
      const event = new Event('open')
      this.onopen?.(event)
      this.dispatchEvent(event)
      return true
    }
    send(raw) {
      if (this.readyState !== MockWebSocket.OPEN) throw new Error('Send on closed synthetic socket')
      this.sent.push(JSON.parse(raw))
    }
    close() {
      if (this.readyState === MockWebSocket.CLOSED) return
      this.readyState = MockWebSocket.CLOSED
      queueMicrotask(() => {
        const event = new Event('close')
        this.onclose?.(event)
        this.dispatchEvent(event)
      })
    }
  }

  class MockAudioContext {
    state = 'running'
    destination = {}
    constructor() { contexts.push(this) }
    async resume() { this.state = 'running' }
    async close() { this.state = 'closed' }
    createMediaStreamSource() {
      sourceCount++
      return { connect() {}, disconnect() {} }
    }
    createScriptProcessor() {
      const processor = {
        connected: false,
        onaudioprocess: null,
        connect() { this.connected = true },
        disconnect() { this.connected = false },
      }
      processors.push(processor)
      return processor
    }
  }

  const getUserMedia = () => new Promise(resolve => permissions.push({ resolve, resolved: false }))
  Object.defineProperty(window, 'WebSocket', { configurable: true, value: MockWebSocket })
  Object.defineProperty(window, 'AudioContext', { configurable: true, value: MockAudioContext })
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } })
  window.__voiceMock = {
    openSocket: index => sockets[index].open(),
    grantPermission(index) {
      const pending = permissions[index]
      if (pending.resolved) throw new Error('Synthetic permission already resolved')
      pending.resolved = true
      const track = { readyState: 'live', stop() { this.readyState = 'ended' } }
      tracks.push(track)
      pending.resolve({ getTracks: () => [track] })
    },
    emitAudio() {
      for (const processor of processors) {
        if (processor.connected) processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array(128) } })
      }
    },
    snapshot: () => ({
      sockets: sockets.map(socket => ({ state: socket.readyState, sent: socket.sent.map(event => event.type) })),
      permissionCount: permissions.length,
      tracks: tracks.map(track => track.readyState),
      sourceCount,
      connectedProcessors: processors.filter(processor => processor.connected).length,
      contexts: contexts.map(context => context.state),
    }),
  }
}

async function open(width) {
  const context = await browser.newContext({ viewport: { width, height: 900 } })
  await context.addInitScript(installVoiceMocks)
  const page = await context.newPage()
  page.setDefaultTimeout(12000)
  const errors = []
  const unexpected = []
  const mints = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/*', route => {
    if (new URL(route.request().url()).origin === new URL(base).origin) return route.continue()
    unexpected.push(route.request().url())
    return route.abort()
  })
  if (studioDist) {
    await page.route('http://studio.test/', async route => route.fulfill({ contentType: 'text/html', body: await readFile(join(studioDist, 'index.html')) }))
    await page.route('http://studio.test/assets/**', async route => {
      const asset = new URL(route.request().url()).pathname.slice(1)
      await route.fulfill({ contentType: asset.endsWith('.css') ? 'text/css' : 'text/javascript', body: await readFile(join(studioDist, asset)) })
    })
  }
  await page.route('**/api/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (path === '/api/voice/session' && request.method() === 'POST') {
      mints.push(path)
      return route.fulfill({ json: { token: 'synthetic-only', realtime_url: 'wss://voice.test/realtime', session: {} } })
    }
    if (request.method() !== 'GET') {
      unexpected.push(`${request.method()} ${path}`)
      return route.fulfill({ status: 500, json: { detail: 'Unexpected write in voice lifecycle fixture' } })
    }
    const bodies = {
      '/api/agents': { agents: [] },
      '/api/agents/resources': { repos: [], github: { status: 'disconnected' } },
      '/api/assignments': { assignments: [] },
      '/api/handoffs': { handoffs: [] },
      '/api/git-workspaces': { workspaces: [] },
      '/api/git-writes': { writes: [] },
      '/api/tasks': [],
      '/api/health': { aiia: { status: 'online' }, ollama: { status: 'online' } },
      '/api/monitor': { services: {} },
      '/api/voice/status': { status: 'connected', configured: true, provider: 'synthetic', reason: '', tools: [] },
    }
    await route.fulfill({ json: bodies[path] ?? {} })
  })
  await page.goto(`${base}#/today`)
  await page.getByRole('heading', { name: 'Today', exact: true }).waitFor()
  assert.equal(await page.getByTitle('Hold to talk', { exact: true }).count(), 0)
  assert.equal(mints.length, 0)
  await page.getByRole('button', { name: 'Voice', exact: true }).click()
  await page.getByTitle('Hold to talk', { exact: true }).waitFor()
  const panel = page.locator('section').filter({ has: page.getByText('Voice Conductor', { exact: true }) })
  return { context, page, panel, errors, unexpected, mints }
}

const snapshot = page => page.evaluate(() => window.__voiceMock.snapshot())
const openSocket = (page, index = 0) => page.evaluate(index => window.__voiceMock.openSocket(index), index)
const grantPermission = (page, index = 0) => page.evaluate(index => window.__voiceMock.grantPermission(index), index)
const waitSockets = (page, count) => page.waitForFunction(count => window.__voiceMock.snapshot().sockets.length === count, count)
const waitPermissions = (page, count) => page.waitForFunction(count => window.__voiceMock.snapshot().permissionCount === count, count)

async function hold(page) {
  await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur() })
  await page.keyboard.down('Space')
}

async function idle(panel) {
  await panel.getByText('ready', { exact: true }).waitFor()
  assert.equal(await panel.getByTitle('Hold to talk', { exact: true }).getAttribute('aria-pressed'), 'false')
}

async function nextHoldWorks(page, panel, permissionIndex) {
  await hold(page)
  await waitPermissions(page, permissionIndex + 1)
  await grantPermission(page, permissionIndex)
  await panel.getByText('listening', { exact: true }).waitFor()
  assert.equal(await panel.getByTitle('Hold to talk', { exact: true }).getAttribute('aria-pressed'), 'true')
  await page.evaluate(() => window.__voiceMock.emitAudio())
  assert.equal((await snapshot(page)).sockets[0].sent.filter(type => type === 'input_audio_buffer.append').length, 1)
  await page.keyboard.up('Space')
  await idle(panel)
  const stopped = await snapshot(page)
  assert.equal(stopped.connectedProcessors, 0)
  assert.ok(stopped.tracks.every(state => state === 'ended'))
  assert.ok(stopped.sockets[0].sent.includes('input_audio_buffer.commit'))
  assert.ok(stopped.sockets[0].sent.includes('response.create'))
}

try {
  for (const width of [1440, 390]) {
    for (const stage of ['handshake', 'permission']) {
      // Release ends only the hold; the open panel must remain usable.
      const { context, page, panel, errors, unexpected, mints } = await open(width)
      try {
        await hold(page)
        await waitSockets(page, 1)
        await panel.getByText('connecting', { exact: true }).waitFor()
        if (stage === 'permission') {
          assert.equal(await openSocket(page), true)
          await waitPermissions(page, 1)
        }
        await page.keyboard.up('Space')
        if (stage === 'handshake') assert.equal(await openSocket(page), true)
        else await grantPermission(page)
        await idle(panel)
        const released = await snapshot(page)
        assert.equal(released.permissionCount, stage === 'permission' ? 1 : 0)
        assert.equal(released.connectedProcessors, 0)
        assert.ok(released.tracks.every(state => state === 'ended'))
        await page.evaluate(() => window.__voiceMock.emitAudio())
        assert.equal((await snapshot(page)).sockets[0].sent.includes('input_audio_buffer.append'), false)
        await nextHoldWorks(page, panel, stage === 'permission' ? 1 : 0)
        assert.equal(mints.length, 1, 'the next hold reuses the connected session')
        assert.deepEqual(errors, [])
        assert.deepEqual(unexpected, [])
        console.log(`${width}px: release during ${stage} returns idle; next hold records only synthetic audio`)
      } finally {
        await context.close()
      }
    }

    for (const stage of ['handshake', 'permission']) {
      // Closing cancels the session, including a late permission grant.
      const { context, page, panel, errors, unexpected, mints } = await open(width)
      try {
        await hold(page)
        await waitSockets(page, 1)
        await panel.getByText('connecting', { exact: true }).waitFor()
        if (stage === 'permission') {
          assert.equal(await openSocket(page), true)
          await waitPermissions(page, 1)
        }
        await page.getByRole('button', { name: 'Voice', exact: true }).click()
        await panel.waitFor({ state: 'detached' })
        await page.keyboard.up('Space')
        await page.waitForFunction(() => window.__voiceMock.snapshot().sockets[0].state === 3)
        if (stage === 'handshake') assert.equal(await openSocket(page), false, 'a cancelled socket cannot finish connecting')
        else {
          await grantPermission(page)
          await page.waitForFunction(() => window.__voiceMock.snapshot().tracks[0] === 'ended')
        }
        await page.evaluate(() => window.__voiceMock.emitAudio())
        const closed = await snapshot(page)
        assert.equal(closed.permissionCount, stage === 'permission' ? 1 : 0)
        assert.equal(closed.sourceCount, 0, 'late microphone grants must never attach an audio source')
        assert.equal(closed.connectedProcessors, 0)
        assert.ok(closed.contexts.every(state => state === 'closed'))
        assert.equal(closed.sockets[0].sent.includes('input_audio_buffer.append'), false)
        assert.equal(await page.getByTitle('Hold to talk', { exact: true }).count(), 0)
        await page.keyboard.press('Space')
        assert.equal(mints.length, 1, 'closed voice does not respond to the global shortcut')
        assert.deepEqual(errors, [])
        assert.deepEqual(unexpected, [])
        console.log(`${width}px: close during ${stage} closes socket and prevents late microphone capture`)
      } finally {
        await context.close()
      }
    }
  }
} finally {
  await browser.close()
}
