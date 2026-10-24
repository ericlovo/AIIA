import { useCallback, useSyncExternalStore } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ConnectionError } from '../lib/connection'

export function ConnectionRecovery() {
  const client = useQueryClient()
  const cache = client.getQueryCache()
  const subscribe = useCallback((notify: () => void) => cache.subscribe(notify), [cache])
  const snapshot = useCallback(() => {
    const failures = cache.getAll().filter(query => query.getObserversCount() > 0 && query.state.status === 'error' && query.state.error instanceof ConnectionError)
    return (failures.find(query => (query.state.error as ConnectionError).kind === 'session') ?? failures[0])?.state.error as ConnectionError | undefined
  }, [cache])
  const error = useSyncExternalStore(subscribe, snapshot)
  if (!error) return null
  return <section aria-label="Studio connection" role="alert" className="shrink-0 border-b border-amber-400/20 bg-neutral-900 px-3 py-2 text-sm text-neutral-200 sm:px-6">
    <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1">
      <div><strong>{error.message}</strong><p className="text-xs text-neutral-400">Current service status is unverified. {error.kind === 'session' ? 'Reconnect to check access; sign in if prompted.' : 'Retry, or reconnect to reload this page and check access.'}</p></div>
      <div className="flex gap-2">
        <button type="button" className="min-h-11 rounded border border-neutral-600 px-3" onClick={() => void client.refetchQueries({ type: 'active' })}>Retry connection</button>
        <button type="button" className="min-h-11 rounded bg-emerald-300 px-3 text-neutral-950" onClick={() => window.location.reload()}>Reconnect Studio</button>
      </div>
    </div>
  </section>
}
