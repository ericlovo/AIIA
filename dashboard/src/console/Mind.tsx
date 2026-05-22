import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

const CATEGORIES = ['decisions', 'patterns', 'lessons', 'project', 'meta', 'team', 'agents', 'sessions']

const CATEGORY_COLORS: Record<string, string> = {
  decisions: 'text-purple-400 bg-purple-500/10',
  patterns: 'text-blue-400 bg-blue-500/10',
  lessons: 'text-amber-400 bg-amber-500/10',
  project: 'text-emerald-400 bg-emerald-500/10',
  meta: 'text-pink-400 bg-pink-500/10',
  team: 'text-cyan-400 bg-cyan-500/10',
  agents: 'text-indigo-400 bg-indigo-500/10',
  sessions: 'text-neutral-500 bg-neutral-800',
}

function timeAgo(iso: string): string {
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h`
  return `${Math.floor(h / 24)}d`
}

export function Mind() {
  const [query, setQuery] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [teaching, setTeaching] = useState(false)

  const { data, isFetching } = useQuery({
    queryKey: ['mind-memories', query, categoryFilter],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (query.trim()) params.set('search', query.trim())
      if (categoryFilter) params.set('category', categoryFilter)
      const res = await fetch(`/api/memories?${params}`)
      return res.ok ? res.json() : { memories: [] }
    },
    refetchInterval: 60_000,
  })

  const memories: Array<{ id: string; fact: string; category: string; created_at: string }> =
    data?.memories ?? []

  return (
    <aside className="bg-neutral-950 flex flex-col min-h-0">
      {/* Header */}
      <div className="px-5 pt-5 pb-3 shrink-0">
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="text-[11px] font-semibold tracking-[0.25em] text-neutral-500 uppercase">Mind</h2>
          <button
            onClick={() => setTeaching(true)}
            className="text-[10px] text-purple-400 hover:text-purple-300 cursor-pointer"
          >
            + teach
          </button>
        </div>
        <p className="text-[11px] text-neutral-600">Memory, knowledge, what I've learned</p>
      </div>

      {/* Search */}
      <div className="px-5 pb-3 shrink-0">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search memory..."
          className="w-full bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-xs text-neutral-300 outline-none focus:border-purple-500/40 placeholder:text-neutral-700"
        />
        <div className="flex gap-1 mt-2 flex-wrap">
          <button
            onClick={() => setCategoryFilter('')}
            className={`text-[10px] px-2 py-0.5 rounded-full ${!categoryFilter ? 'bg-purple-500/20 text-purple-300' : 'text-neutral-600 hover:text-neutral-500'}`}
          >
            all
          </button>
          {CATEGORIES.map(c => (
            <button
              key={c}
              onClick={() => setCategoryFilter(categoryFilter === c ? '' : c)}
              className={`text-[10px] px-2 py-0.5 rounded-full ${categoryFilter === c ? CATEGORY_COLORS[c] : 'text-neutral-600 hover:text-neutral-500'}`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* Memory stream */}
      <div className="flex-1 min-h-0 overflow-y-auto px-5 pb-5">
        {memories.length === 0 && !isFetching && (
          <div className="mt-8 text-center">
            <div className="text-xs text-neutral-600 italic mb-3">
              {query
                ? 'Nothing matches your search.'
                : categoryFilter
                  ? `No ${categoryFilter} memories yet.`
                  : "I don't remember anything yet."}
            </div>
            {(query || categoryFilter) && (
              <button
                onClick={() => { setQuery(''); setCategoryFilter('') }}
                className="text-[11px] text-purple-400 hover:text-purple-300 cursor-pointer"
              >
                clear filters
              </button>
            )}
          </div>
        )}
        <div className="space-y-2">
          {memories.slice(0, 50).map(m => (
            <MemoryCard key={m.id} memory={m} />
          ))}
        </div>
      </div>

      {teaching && <TeachModal onClose={() => setTeaching(false)} />}
    </aside>
  )
}

function MemoryCard({ memory }: { memory: { id: string; fact: string; category: string; created_at: string } }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = memory.fact.length > 140
  const show = !isLong || expanded ? memory.fact : memory.fact.slice(0, 140) + '…'
  return (
    <div
      className="bg-neutral-900 border border-neutral-900 rounded-lg p-3 hover:border-neutral-800 transition-colors cursor-pointer"
      onClick={() => isLong && setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className={`text-[9px] px-1.5 py-0.5 rounded uppercase tracking-wider ${CATEGORY_COLORS[memory.category] ?? 'text-neutral-500 bg-neutral-900'}`}>
          {memory.category}
        </span>
        <span className="text-[10px] text-neutral-700">{timeAgo(memory.created_at)}</span>
      </div>
      <p className="text-[12px] text-neutral-400 leading-relaxed whitespace-pre-wrap">{show}</p>
    </div>
  )
}

function TeachModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [fact, setFact] = useState('')
  const [category, setCategory] = useState('decisions')
  const save = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fact: fact.trim(), category }),
      })
      if (!res.ok) throw new Error('Save failed')
      return res.json()
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['mind-memories'] })
      onClose()
    },
  })

  return (
    <div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-30"
      onClick={onClose}
    >
      <div
        className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 w-[420px] shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="text-[10px] text-purple-400 tracking-[0.3em] font-semibold mb-2">TEACH AIIA</div>
        <h3 className="text-lg text-white font-light mb-4">What should I remember?</h3>
        <select
          value={category}
          onChange={e => setCategory(e.target.value)}
          className="text-xs bg-neutral-900 border border-neutral-800 text-neutral-300 rounded-md px-2 py-1.5 mb-3 w-full cursor-pointer outline-none"
        >
          {CATEGORIES.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <textarea
          value={fact}
          onChange={e => setFact(e.target.value)}
          autoFocus
          placeholder="A fact, decision, pattern, or lesson..."
          rows={4}
          className="w-full bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm text-neutral-300 outline-none focus:border-purple-500/40 placeholder:text-neutral-700 resize-none mb-3"
        />
        <div className="flex gap-2">
          <button
            onClick={() => save.mutate()}
            disabled={!fact.trim() || save.isPending}
            className="flex-1 py-2 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700 disabled:opacity-50"
          >
            {save.isPending ? 'Saving…' : 'Remember'}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm text-neutral-500 hover:text-white cursor-pointer"
          >
            Cancel
          </button>
        </div>
        {save.isError && (
          <div className="mt-2 text-xs text-red-400">{String((save.error as Error)?.message ?? save.error)}</div>
        )}
      </div>
    </div>
  )
}
