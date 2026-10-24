import test from 'node:test'
import assert from 'node:assert/strict'
import { ConnectionError, studioFetch, studioJSON } from '../src/lib/connection.ts'

test('connection errors distinguish access, network, and non-JSON responses', async () => {
  const original = globalThis.fetch
  try {
    let options: RequestInit | undefined
    globalThis.fetch = async (_input, init) => { options = init; return new Response('', { status: 401 }) }
    await assert.rejects(studioFetch('/api/projects'), (error: unknown) => error instanceof ConnectionError && error.kind === 'session')
    assert.equal(options?.redirect, 'manual')
    assert.equal(options?.credentials, 'same-origin')
    globalThis.fetch = async () => { throw new TypeError('Failed to fetch') }
    await assert.rejects(studioFetch('/api/projects'), (error: unknown) => error instanceof ConnectionError && error.kind === 'network')
    await assert.rejects(studioJSON(new Response('<html>Login</html>', { headers: { 'content-type': 'text/html' } })), (error: unknown) => error instanceof ConnectionError && error.kind === 'response')
    await assert.rejects(studioJSON(new Response('failure', { status: 500 })), (error: unknown) => error instanceof Error && !(error instanceof ConnectionError))
    globalThis.fetch = async () => new Response('', { status: 302, headers: { location: 'https://untrusted.test/' } })
    await assert.rejects(studioFetch('/api/projects'), ConnectionError)
    await assert.rejects(studioFetch('https://untrusted.test/'), /same-origin/)
    await assert.rejects(studioFetch('/\\evil.example'), /same-origin/)
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'EGRESS_DENIED' }), { status: 403, headers: { 'content-type': 'application/json' } })
    await assert.rejects(studioJSON(await studioFetch('/api/voice')), (error: unknown) => error instanceof Error && !(error instanceof ConnectionError) && error.message === 'EGRESS_DENIED')
    for (const path of ['/\t/evil.example', '/\n/evil.example']) await assert.rejects(studioFetch(path), /same-origin/)
    globalThis.fetch = async () => new Response('{}', { headers: { 'content-type': 'application/json' } })
    assert.equal((await studioFetch('/api/agents/QA paused watch/loop')).status, 200)
    const abort = new DOMException('Aborted', 'AbortError')
    globalThis.fetch = async () => { throw abort }
    await assert.rejects(studioFetch('/api/projects'), error => error === abort)
  } finally { globalThis.fetch = original }
})