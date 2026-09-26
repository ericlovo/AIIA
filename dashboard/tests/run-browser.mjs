// Serves the built Studio and runs every *.browser.mjs suite against it, one at
// a time. Each suite mocks every API call itself, so no Brain, Command Center
// or Ollama is needed. Run `npm run build` first.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { readdir } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { preview } from 'vite'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dist = join(root, 'dist')
if (!existsSync(join(dist, 'index.html'))) {
  console.error('dist/ is missing. Run `npm run build` first.')
  process.exit(1)
}

const only = process.argv.slice(2)
const suites = (await readdir(join(root, 'tests')))
  .filter(name => name.endsWith('.browser.mjs'))
  .filter(name => !only.length || only.some(filter => name.includes(filter)))
  .sort()

const server = await preview({ root, logLevel: 'warn', preview: { port: 0, host: '127.0.0.1', strictPort: false } })
const address = server.httpServer.address()
const studioUrl = `http://127.0.0.1:${address.port}/`

function run(suite) {
  return new Promise(done => {
    const started = Date.now()
    const child = spawn(process.execPath, [join('tests', suite)], {
      cwd: root,
      stdio: 'inherit',
      env: {
        ...process.env,
        STUDIO_URL: studioUrl,
        STUDIO_DIST_DIR: dist,
        // One folder per suite, so screenshot names from different suites cannot collide.
        ...(process.env.SCREENSHOT_DIR ? { SCREENSHOT_DIR: join(process.env.SCREENSHOT_DIR, suite.replace('.browser.mjs', '')) } : {}),
      },
    })
    child.on('exit', code => done({ suite, ok: code === 0, seconds: ((Date.now() - started) / 1000).toFixed(1) }))
  })
}

const results = []
try {
  for (const suite of suites) {
    console.log(`\n▶ ${suite}`)
    results.push(await run(suite))
  }
} finally {
  await new Promise(done => server.httpServer.close(done))
}

console.log('\nBrowser suites:')
for (const result of results) console.log(`  ${result.ok ? 'pass' : 'FAIL'}  ${result.suite}  (${result.seconds}s)`)
const failed = results.filter(result => !result.ok)
if (!results.length) {
  console.error('No browser suites matched.')
  process.exit(1)
}
process.exit(failed.length ? 1 : 0)
