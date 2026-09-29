import type { ReviewBucket } from '../lib/api'

export type StudioView = 'switchboard' | 'activity' | 'agents' | 'assignments' | 'handoffs' | 'memory' | 'world' | 'signals'

export const VIEWS: { id: StudioView; label: string }[] = [
  { id: 'switchboard', label: 'Today' },
  { id: 'activity', label: 'Overview' },
  { id: 'agents', label: 'Agents' },
  { id: 'signals', label: 'Signals' },
  { id: 'assignments', label: 'Assignments' },
  { id: 'handoffs', label: 'Handoffs' },
  { id: 'memory', label: 'Memory' },
  { id: 'world', label: 'Map' },
]

/**
 * The Studio's location, parsed from the URL hash. Every view and every
 * cross-view jump (open this assignment, review this bucket) is a route, so it
 * can be linked to, reloaded, and reached with the back button.
 */
export type StudioRoute =
  | { view: 'switchboard'; taskId?: string }
  | { view: 'activity' }
  | { view: 'agents'; agentId?: string }
  | { view: 'assignments'; assignmentId?: string; agentId?: string }
  | { view: 'handoffs'; from?: string; to?: string }
  | { view: 'memory'; review?: ReviewBucket | 'all' }
  | { view: 'world' }
  | { view: 'signals' }

export const DEFAULT_ROUTE: StudioRoute = { view: 'switchboard' }

const PATHS: Record<StudioView, string> = {
  switchboard: 'today', activity: 'overview', agents: 'agents', assignments: 'assignments',
  handoffs: 'handoffs', memory: 'memory', world: 'map', signals: 'signals',
}
const VIEW_BY_PATH = Object.fromEntries(Object.entries(PATHS).map(([view, path]) => [path, view])) as Record<string, StudioView>
const REVIEW_BUCKETS: readonly string[] = ['open', 'needs_work', 'already_fixed', 'declined', 'external_failure', 'unclassified', 'all']

/** `#/assignments/abc?agent=x` → route. Anything unrecognised is null, never a guess. */
export function parseRoute(hash: string): StudioRoute | null {
  const match = /^#\/([a-z]+)(?:\/([^/?]+))?(?:\?(.*))?$/.exec(hash)
  if (!match) return null
  const [, path, rawId, rawQuery] = match
  const view = VIEW_BY_PATH[path]
  if (!view) return null
  let id: string | undefined
  try {
    id = rawId ? decodeURIComponent(rawId) : undefined
  } catch {
    return null
  }
  const query = new URLSearchParams(rawQuery ?? '')
  const param = (name: string) => query.get(name) || undefined
  switch (view) {
    case 'switchboard': return id ? null : { view, taskId: param('task') }
    case 'agents': return { view, agentId: id }
    case 'assignments': return { view, assignmentId: id, agentId: id ? undefined : param('agent') }
    case 'handoffs': return id ? null : { view, from: param('from'), to: param('to') }
    case 'memory': {
      if (id) return null
      const review = param('review')
      if (review && !REVIEW_BUCKETS.includes(review)) return null
      return { view, review: review as ReviewBucket | 'all' | undefined }
    }
    default: return id ? null : { view }
  }
}

export function formatRoute(route: StudioRoute): string {
  const path = `#/${PATHS[route.view]}`
  const query = new URLSearchParams()
  let id = ''
  switch (route.view) {
    case 'switchboard': if (route.taskId) query.set('task', route.taskId); break
    case 'agents': id = route.agentId ?? ''; break
    case 'assignments':
      id = route.assignmentId ?? ''
      if (!id && route.agentId) query.set('agent', route.agentId)
      break
    case 'handoffs':
      if (route.from) query.set('from', route.from)
      if (route.to) query.set('to', route.to)
      break
    case 'memory': if (route.review) query.set('review', route.review); break
  }
  const search = query.toString()
  return `${path}${id ? `/${encodeURIComponent(id)}` : ''}${search ? `?${search}` : ''}`
}
