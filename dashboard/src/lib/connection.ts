export type ConnectionKind = 'session' | 'network' | 'response'

// "session" is claimed only on authentication evidence: the Command Center's own
// 401 ({ detail: 'authentication_required', reason }), or Cloudflare Access
// answering at the edge (a login redirect, a bare 401, a non-JSON 403). A fetch
// that throws ("Failed to fetch") proves nothing about the session, so it stays
// the generic "network" fallback.
const AUTH_DETAIL = 'authentication_required'
function message(kind: ConnectionKind, reason?: string) {
  if (kind === 'network') return 'This browser could not reach Studio.'
  if (kind === 'response') return 'Studio returned a page instead of data.'
  if (reason === 'token_expired') return 'Your Studio session has expired.'
  if (reason === 'access_not_configured') return 'Studio is refusing remote access because its access check is not configured.'
  return 'Studio access is blocked or your session has expired.'
}

export class ConnectionError extends Error {
  kind: ConnectionKind
  reason?: string
  constructor(kind: ConnectionKind, reason?: string) {
    super(message(kind, reason))
    this.name = 'ConnectionError'
    this.kind = kind
    this.reason = reason
  }
}

export async function studioFetch(path: string, init?: RequestInit): Promise<Response> {
  if (!path.startsWith('/') || path.startsWith('//') || path.includes('\\') || Array.from(path).some(character => character.charCodeAt(0) < 32)) throw new Error('Studio requests must use a same-origin path.')
  try {
    const response = await fetch(path, { ...init, credentials: 'same-origin', redirect: 'manual' })
    if (response.status === 401) {
      const payload = await response.clone().json().catch(() => null)
      throw new ConnectionError('session', payload?.detail === AUTH_DETAIL && typeof payload.reason === 'string' ? payload.reason : undefined)
    }
    if (response.type === 'opaqueredirect' || response.redirected || [301, 302, 303, 307, 308].includes(response.status) || (response.status === 403 && !response.headers.get('content-type')?.includes('application/json'))) throw new ConnectionError('session')
    return response
  } catch (error) {
    if (error instanceof ConnectionError || (error instanceof Error && error.name === 'AbortError')) throw error
    throw new ConnectionError('network')
  }
}

export async function studioJSON<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = typeof payload?.detail === 'string' ? payload.detail : `${response.status} ${response.statusText}`
    throw new Error(detail === 'mini_busy' ? 'Mini busy — wait for the active run to finish.' : detail)
  }
  if (!response.headers.get('content-type')?.includes('application/json')) throw new ConnectionError('response')
  try { return await response.json() as T } catch { throw new ConnectionError('response') }
}
