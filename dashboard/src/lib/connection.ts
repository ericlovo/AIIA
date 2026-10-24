export type ConnectionKind = 'session' | 'network' | 'response'
export class ConnectionError extends Error {
  kind: ConnectionKind
  constructor(kind: ConnectionKind) {
    super(kind === 'session' ? 'Studio access is blocked or your session has expired.' : kind === 'network' ? 'This browser could not reach Studio.' : 'Studio returned a page instead of data.')
    this.name = 'ConnectionError'
    this.kind = kind
  }
}

export async function studioFetch(path: string, init?: RequestInit): Promise<Response> {
  if (!path.startsWith('/') || path.startsWith('//') || path.includes('\\') || Array.from(path).some(character => character.charCodeAt(0) < 32)) throw new Error('Studio requests must use a same-origin path.')
  try {
    const response = await fetch(path, { ...init, credentials: 'same-origin', redirect: 'manual' })
    if (response.type === 'opaqueredirect' || response.redirected || [301, 302, 303, 307, 308, 401].includes(response.status) || (response.status === 403 && !response.headers.get('content-type')?.includes('application/json'))) throw new ConnectionError('session')
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
