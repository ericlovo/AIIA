import { useEffect, useSyncExternalStore } from 'react'
import { DEFAULT_ROUTE, formatRoute, parseRoute, type StudioRoute } from './studioRoute'

// The hash is the source of truth. `visit` counts navigations, so asking for the
// route you are already on (clicking the same metric twice) still re-enters it.
let visit = 0
const listeners = new Set<() => void>()
const emit = () => { for (const listener of listeners) listener() }

function subscribe(listener: () => void) {
  if (listeners.size === 0) window.addEventListener('hashchange', emit)
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0) window.removeEventListener('hashchange', emit)
  }
}

const snapshot = () => `${window.location.hash}\n${visit}`

/** A new history entry: moving between views, or jumping into a record from elsewhere. */
export function navigate(route: StudioRoute) {
  const target = formatRoute(route)
  if (window.location.hash !== target) {
    window.location.hash = target
    return
  }
  // Same address: only the counter changes, so the view re-enters. Bumping it
  // for a new address too would re-key the current view before the hash event.
  visit += 1
  emit()
}

/**
 * Rewrite the URL without a history entry or a re-render: selection inside a
 * view, so the address stays copyable without remounting what is on screen.
 */
export function replaceRoute(route: StudioRoute) {
  const target = formatRoute(route)
  if (window.location.hash !== target) window.history.replaceState(window.history.state, '', target)
}

export function useStudioRoute(): { route: StudioRoute; key: string } {
  const key = useSyncExternalStore(subscribe, snapshot)
  const parsed = parseRoute(window.location.hash)
  // An empty or unknown address lands on the morning note, without a back-button trap.
  const invalid = !parsed
  useEffect(() => { if (invalid) replaceRoute(DEFAULT_ROUTE) }, [invalid, key])
  return { route: parsed ?? DEFAULT_ROUTE, key }
}
