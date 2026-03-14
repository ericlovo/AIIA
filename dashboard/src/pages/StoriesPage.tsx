import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { Story } from '../lib/api'

const PRIORITY_COLORS: Record<string, string> = {
  P0: 'bg-red-500/20 text-red-400 border-red-500/30',
  P1: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  P2: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  P3: 'bg-[#2a2a2a] text-[#888] border-[#333]',
}

const STATUS_COLORS: Record<string, string> = {
  backlog: 'text-[#888]',
  active: 'text-blue-400',
  in_progress: 'text-purple-400',
  blocked: 'text-red-400',
  shipped: 'text-green-400',
  cancelled: 'text-[#555]',
}

const KANBAN_COLUMNS = [
  { status: 'backlog', label: 'Backlog', accent: '#666' },
  { status: 'active', label: 'Active', accent: '#3b82f6' },
  { status: 'in_progress', label: 'In Progress', accent: '#a855f7' },
  { status: 'blocked', label: 'Blocked', accent: '#ef4444' },
  { status: 'shipped', label: 'Shipped', accent: '#22c55e' },
]

type ViewMode = 'list' | 'kanban'

export function StoriesPage() {
  const [view, setView] = useState<ViewMode>('list')
  const [filterProduct, setFilterProduct] = useState<string>('')
  const [filterPriority, setFilterPriority] = useState<string>('')
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [selected, setSelected] = useState<Story | null>(null)

  const { data, isLoading } = useQuery({ queryKey: ['stories'], queryFn: () => api.stories() })
  const { data: summary } = useQuery({ queryKey: ['storySummary'], queryFn: api.storySummary })
  const { data: prioritized, isFetching: isPrioritizing, refetch: runPrioritize } = useQuery({
    queryKey: ['prioritizeAll'],
    queryFn: () => api.prioritize(20),
    enabled: false,
  })

  const stories = data?.stories ?? []
  const products = [...new Set(stories.map(s => s.product))].sort()

  const filtered = stories.filter(s => {
    if (filterProduct && s.product !== filterProduct) return false
    if (filterPriority && s.priority !== filterPriority) return false
    if (filterStatus && s.status !== filterStatus) return false
    return true
  })

  // Merge priority scores from prioritized data
  const scoredMap = new Map((prioritized?.stories ?? []).map(s => [s.id, s]))
  const enriched = filtered.map(s => ({ ...s, ...(scoredMap.get(s.id) ?? {}) }))

  if (isLoading) return <div className="text-[#666] text-sm">Loading...</div>

  return (
    <div className="max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-baseline gap-3">
          <h1 className="text-xl font-semibold text-white">Stories</h1>
          <span className="text-sm text-[#666]">{summary?.total ?? stories.length} total</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => runPrioritize()}
            disabled={isPrioritizing}
            className="text-xs px-3 py-1.5 rounded-md bg-purple-600/20 text-purple-400 border border-purple-500/30 hover:bg-purple-600/30 cursor-pointer disabled:opacity-50"
          >
            {isPrioritizing ? 'Scoring...' : 'Prioritize'}
          </button>
          <div className="flex rounded-md border border-[#2a2a2a] overflow-hidden">
            <button
              onClick={() => setView('list')}
              className={`text-xs px-3 py-1.5 cursor-pointer ${view === 'list' ? 'bg-[#222] text-white' : 'text-[#666] hover:text-[#999]'}`}
            >List</button>
            <button
              onClick={() => setView('kanban')}
              className={`text-xs px-3 py-1.5 cursor-pointer ${view === 'kanban' ? 'bg-[#222] text-white' : 'text-[#666] hover:text-[#999]'}`}
            >Board</button>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-5">
        <FilterSelect value={filterProduct} onChange={setFilterProduct} options={products} placeholder="Product" />
        <FilterSelect value={filterPriority} onChange={setFilterPriority} options={['P0', 'P1', 'P2', 'P3']} placeholder="Priority" />
        <FilterSelect value={filterStatus} onChange={setFilterStatus} options={KANBAN_COLUMNS.map(c => c.status).concat('cancelled')} placeholder="Status" />
        {(filterProduct || filterPriority || filterStatus) && (
          <button onClick={() => { setFilterProduct(''); setFilterPriority(''); setFilterStatus('') }} className="text-xs text-[#666] hover:text-[#999] cursor-pointer">Clear</button>
        )}
      </div>

      {/* Views */}
      <div className="flex gap-6">
        <div className="flex-1 min-w-0">
          {view === 'list' ? (
            <ListView stories={enriched} onSelect={setSelected} selected={selected} />
          ) : (
            <KanbanView stories={enriched} onSelect={setSelected} />
          )}
        </div>

        {/* Detail drawer */}
        {selected && (
          <StoryDrawer story={selected} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  )
}

function FilterSelect({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void; options: string[]; placeholder: string
}) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="text-xs bg-[#141414] border border-[#2a2a2a] text-[#888] rounded-md px-2 py-1.5 cursor-pointer focus:border-[#444] outline-none"
    >
      <option value="">{placeholder}</option>
      {options.map(o => <option key={o} value={o}>{o}</option>)}
    </select>
  )
}

function ListView({ stories, onSelect, selected }: { stories: Story[]; onSelect: (s: Story) => void; selected: Story | null }) {
  // Sort by composite score desc (falls back to priority_score), then priority
  const sorted = [...stories].sort((a, b) => {
    const aScore = a.composite_score ?? a.priority_score ?? 0
    const bScore = b.composite_score ?? b.priority_score ?? 0
    if (bScore !== aScore) return bScore - aScore
    const pOrder = ['P0', 'P1', 'P2', 'P3']
    return pOrder.indexOf(a.priority) - pOrder.indexOf(b.priority)
  })

  return (
    <div className="bg-[#141414] border border-[#222] rounded-xl overflow-hidden">
      <div className="grid grid-cols-[1fr_100px_80px_60px_80px] gap-2 px-4 py-2 text-xs text-[#555] uppercase tracking-wide border-b border-[#1e1e1e]">
        <span>Title</span>
        <span>Product</span>
        <span>Status</span>
        <span>Priority</span>
        <span className="text-right">Score</span>
      </div>
      {sorted.map(s => (
        <div
          key={s.id}
          onClick={() => onSelect(s)}
          className={`grid grid-cols-[1fr_100px_80px_60px_80px] gap-2 px-4 py-2.5 text-sm cursor-pointer border-b border-[#1a1a1a] last:border-0 transition-colors ${
            selected?.id === s.id ? 'bg-[#1a1a2a]' : 'hover:bg-[#181818]'
          }`}
        >
          <span className="text-[#ddd] truncate">{s.title}</span>
          <span className="text-xs text-[#666]">{s.product}</span>
          <span className={`text-xs ${STATUS_COLORS[s.status] ?? 'text-[#666]'}`}>{s.status}</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium inline-flex items-center justify-center ${PRIORITY_COLORS[s.priority]}`}>
            {s.priority}
          </span>
          <span className="text-xs text-[#888] font-mono text-right">{s.composite_score != null ? s.composite_score : s.priority_score ?? '—'}</span>
        </div>
      ))}
    </div>
  )
}

function KanbanView({ stories, onSelect }: { stories: Story[]; onSelect: (s: Story) => void }) {
  const qc = useQueryClient()
  const updateMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => api.updateStory(id, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['stories'] }),
  })

  return (
    <div className="flex gap-4 overflow-x-auto pb-4">
      {KANBAN_COLUMNS.map(col => {
        const colStories = stories.filter(s => s.status === col.status)
        return (
          <div
            key={col.status}
            className="min-w-[220px] flex-1 bg-[#111] rounded-xl border border-[#1e1e1e]"
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              const id = e.dataTransfer.getData('storyId')
              if (id) updateMutation.mutate({ id, status: col.status })
            }}
          >
            <div className="flex items-center gap-2 px-3 py-2.5 border-b border-[#1e1e1e]">
              <span className="w-2 h-2 rounded-full" style={{ background: col.accent }} />
              <span className="text-xs text-[#888] font-medium">{col.label}</span>
              <span className="text-xs text-[#555] ml-auto">{colStories.length}</span>
            </div>
            <div className="p-2 space-y-2 min-h-[100px]">
              {colStories.map(s => (
                <div
                  key={s.id}
                  draggable
                  onDragStart={e => e.dataTransfer.setData('storyId', s.id)}
                  onClick={() => onSelect(s)}
                  className="bg-[#181818] border border-[#222] rounded-lg p-3 cursor-pointer hover:border-[#333] transition-colors"
                >
                  <div className="text-sm text-[#ddd] mb-1.5">{s.title}</div>
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] px-1 py-0.5 rounded border ${PRIORITY_COLORS[s.priority]}`}>{s.priority}</span>
                    <span className="text-[10px] text-[#555]">{s.product}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function StoryDrawer({ story, onClose }: { story: Story; onClose: () => void }) {
  return (
    <div className="w-[360px] shrink-0 bg-[#141414] border border-[#222] rounded-xl p-5 sticky top-0 max-h-[calc(100vh-120px)] overflow-y-auto">
      <div className="flex items-start justify-between mb-4">
        <span className={`text-xs px-2 py-1 rounded border font-medium ${PRIORITY_COLORS[story.priority]}`}>{story.priority}</span>
        <button onClick={onClose} className="text-[#555] hover:text-[#999] cursor-pointer text-lg leading-none">&times;</button>
      </div>

      <h2 className="text-base font-semibold text-white mb-2">{story.title}</h2>

      <div className="space-y-3 text-sm">
        <div className="flex justify-between">
          <span className="text-[#666]">Product</span>
          <span className="text-[#ccc]">{story.product}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#666]">Status</span>
          <span className={STATUS_COLORS[story.status]}>{story.status}</span>
        </div>
        {story.composite_score != null && (
          <div className="flex justify-between">
            <span className="text-[#666]">Composite</span>
            <span className="text-[#ccc] font-mono">{story.composite_score}/100</span>
          </div>
        )}
        {story.priority_score != null && (
          <div className="flex justify-between">
            <span className="text-[#666]">Additive</span>
            <span className="text-[#888] font-mono text-xs">{story.priority_score}/150</span>
          </div>
        )}
        {story.geometric && (
          <div className="flex justify-between">
            <span className="text-[#666]">Alignment</span>
            <span className="text-[#888] font-mono text-xs">{(story.geometric.alignment * 100).toFixed(0)}%</span>
          </div>
        )}
        {story.source_type && (
          <div className="flex justify-between">
            <span className="text-[#666]">Source</span>
            <span className="text-[#ccc]">{story.source_type}</span>
          </div>
        )}

        {story.description && (
          <div className="pt-2 border-t border-[#1e1e1e]">
            <div className="text-xs text-[#666] mb-1">Description</div>
            <p className="text-[#aaa] text-sm leading-relaxed">{story.description}</p>
          </div>
        )}

        {story.priority_reasoning && (
          <div className="pt-2 border-t border-[#1e1e1e]">
            <div className="text-xs text-[#666] mb-1">Priority reasoning</div>
            <p className="text-[#aaa] text-sm">{story.priority_reasoning}</p>
          </div>
        )}

        {story.filter_scores && Object.keys(story.filter_scores).length > 0 && (
          <div className="pt-2 border-t border-[#1e1e1e]">
            <div className="text-xs text-[#666] mb-2">Impact breakdown</div>
            {Object.entries(story.filter_scores).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 mb-1">
                <span className="text-xs text-[#888] w-28">{k.replace('_', ' ')}</span>
                <div className="flex-1 bg-[#1a1a1a] rounded-full h-1.5">
                  <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${(v as number) * 10}%` }} />
                </div>
                <span className="text-xs text-[#666] w-6 text-right font-mono">{v as number}</span>
              </div>
            ))}
          </div>
        )}

        {story.tags && story.tags.length > 0 && (
          <div className="pt-2 border-t border-[#1e1e1e]">
            <div className="text-xs text-[#666] mb-1.5">Tags</div>
            <div className="flex flex-wrap gap-1">
              {story.tags.map(t => (
                <span key={t} className="text-[10px] px-2 py-0.5 rounded-full bg-[#1e1e1e] text-[#888] border border-[#2a2a2a]">{t}</span>
              ))}
            </div>
          </div>
        )}

        <div className="text-xs text-[#555] pt-2">
          Created {new Date(story.created_at).toLocaleDateString()}
        </div>
      </div>
    </div>
  )
}
