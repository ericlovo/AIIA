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
    const refusal = (reason: string) => new Response(JSON.stringify({ detail: 'authentication_required', reason, auth: 'cloudflare_access' }), { status: 401, headers: { 'content-type': 'application/json' } })
    const expectSession = (reason: string | undefined, text: string) => (error: unknown) => error instanceof ConnectionError && error.kind === 'session' && error.reason === reason && error.message === text
    await assert.rejects(studioFetch('/api/projects'), expectSession(undefined, 'Studio access is blocked or your session has expired.'))
    globalThis.fetch = async () => refusal('token_expired')
    await assert.rejects(studioFetch('/api/projects'), expectSession('token_expired', 'Your Studio session has expired.'))
    globalThis.fetch = async () => refusal('missing_token')
    await assert.rejects(studioFetch('/api/projects'), expectSession('missing_token', 'Studio access is blocked or your session has expired.'))
    globalThis.fetch = async () => refusal('access_not_configured')
    await assert.rejects(studioFetch('/api/projects'), expectSession('access_not_configured', 'Studio is refusing remote access because its access check is not configured.'))
    // A 401 that is not the Command Center's own refusal carries no reason to trust.
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'invalid_slack_signature', reason: 'token_expired' }), { status: 401, headers: { 'content-type': 'application/json' } })
    await assert.rejects(studioFetch('/api/projects'), expectSession(undefined, 'Studio access is blocked or your session has expired.'))
    // No authentication evidence: a failed fetch and a 503 from the access check stay generic.
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'access_keys_unavailable' }), { status: 503, headers: { 'content-type': 'application/json' } })
    await assert.rejects(studioJSON(await studioFetch('/api/projects')), (error: unknown) => error instanceof Error && !(error instanceof ConnectionError) && error.message === 'access_keys_unavailable')
    globalThis.fetch = async () => { throw new TypeError('Failed to fetch') }
    await assert.rejects(studioFetch('/api/projects'), (error: unknown) => error instanceof ConnectionError && error.kind === 'network' && error.reason === undefined && error.message === 'This browser could not reach Studio.')
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