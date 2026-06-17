import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'

type FileHit = {
  path: string
  name: string
  score: number
  excerpt: string
  first: boolean
}

export function Files() {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')

  const { data: status } = useQuery({
    queryKey: ['files-status'],
    queryFn: () => fetch('/api/v1/files/status').then(r => r.json()),
    refetchInterval: 5000,
  })

  const { data, isFetching } = useQuery({
    queryKey: ['file-search', submitted],
    queryFn: () =>
      submitted
        ? fetch(`/api/v1/files/search?q=${encodeURIComponent(submitted)}`).then(r => r.json())
        : Promise.resolve({ results: [] }),
    enabled: !!submitted,
  })

  const indexMutation = useMutation({
    mutationFn: () => fetch('/api/v1/files/index', { method: 'POST' }).then(r => r.json()),
  })

  const hits: FileHit[] = data?.results ?? []

  function search() {
    const q = query.trim()
    if (q) setSubmitted(q)
  }

  return (
    <section className="bg-neutral-950 flex flex-col min-h-0">
      <div className="px-5 pt-5 pb-3 shrink-0 border-b border-neutral-900">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-[11px] font-semibold tracking-[0.25em] text-neutral-500 uppercase">
              Files
            </h2>
            <p className="text-[11px] text-neutral-600">Semantic search across local files</p>
          </div>
          <div className="flex items-center gap-2">
            {status && (
              <span className="text-[10px] text-neutral-600">
                {status.running
                  ? 'indexing…'
                  : status.last_result?.files
                  ? `${status.last_result.files} files indexed`
                  : 'not indexed'}
              </span>
            )}
            <button
              onClick={() => indexMutation.mutate()}
              disabled={status?.running || indexMutation.isPending}
              className="text-[10px] px-2 py-1 rounded bg-neutral-800 text-neutral-400 hover:text-white hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-default cursor-pointer"
            >
              {status?.running || indexMutation.isPending ? 'Indexing…' : 'Re-index'}
            </button>
          </div>
        </div>
      </div>

      <div className="px-5 pt-4 pb-3 shrink-0">
        <div className="flex gap-2">
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && search()}
            placeholder="Search your files…"
            className="flex-1 bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm text-neutral-300 outline-none focus:border-purple-500/40 placeholder:text-neutral-700"
          />
          <button
            onClick={search}
            disabled={!query.trim() || isFetching}
            className="px-4 py-2 rounded-lg bg-purple-600 text-white text-sm hover:bg-purple-700 disabled:opacity-40 disabled:cursor-default cursor-pointer"
          >
            {isFetching ? '…' : 'Search'}
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-5 pb-5 space-y-2">
        {!submitted && !status?.last_result?.files && (
          <div className="mt-8 text-center text-neutral-700 text-sm">
            <p>Click Re-index to scan your local files,</p>
            <p>then search by topic, keyword, or concept.</p>
          </div>
        )}
        {submitted && hits.length === 0 && !isFetching && (
          <p className="mt-8 text-center text-neutral-600 text-sm">No results for "{submitted}"</p>
        )}
        {hits.filter(h => h.first).map(hit => (
          <div key={hit.path} className="bg-neutral-900 rounded-lg p-3 border border-neutral-800/50">
            <div className="flex items-start justify-between gap-2 mb-1">
              <span className="text-xs font-medium text-purple-400 truncate">{hit.name}</span>
              <span className="text-[10px] text-neutral-600 shrink-0">{(hit.score * 100).toFixed(0)}%</span>
            </div>
            <p className="text-[10px] text-neutral-500 truncate mb-1.5">{hit.path}</p>
            <p className="text-xs text-neutral-400 leading-relaxed line-clamp-3">{hit.excerpt}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
