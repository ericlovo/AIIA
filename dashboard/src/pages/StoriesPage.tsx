import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { Story } from '../lib/api'

const PRIORITIES = ['P0', 'P1', 'P2', 'P3']
const STATUSES = ['backlog', 'active', 'in_progress', 'blocked', 'shipped', 'cancelled']

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
  const qc = useQueryClient()
  const [view, setView] = useState<ViewMode>('list')
  const [filterProduct, setFilterProduct] = useState<string>('')
  const [filterPriority, setFilterPriority] = useState<string>('')
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [selected, setSelected] = useState<Story | null>(null)
  const [creating, setCreating] = useState(false)

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

  const scoredMap = new Map((prioritized?.stories ?? []).map(s => [s.id, s]))
  const enriched = filtered.map(s => ({ ...s, ...(scoredMap.get(s.id) ?? {}) }))

  function handleSaved() {
    qc.invalidateQueries({ queryKey: ['stories'] })
    qc.invalidateQueries({ queryKey: ['storySummary'] })
  }

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
            onClick={() => setCreating(true)}
            className="text-xs px-3 py-1.5 rounded-md bg-purple-600 text-white hover:bg-purple-700 cursor-pointer"
          >
            + New Story
          </button>
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
        <FilterSelect value={filterPriority} onChange={setFilterPriority} options={PRIORITIES} placeholder="Priority" />
        <FilterSelect value={filterStatus} onChange={setFilterStatus} options={STATUSES} placeholder="Status" />
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
            <KanbanView stories={enriched} onSelect={setSelected} onSaved={handleSaved} />
          )}
        </div>

        {selected && (
          <StoryDrawer
            story={selected}
            products={products}
            onClose={() => setSelected(null)}
            onSaved={handleSaved}
            onDeleted={() => { setSelected(null); handleSaved() }}
          />
        )}
      </div>

      {creating && (
        <CreateStoryModal
          products={products}
          onClose={() => setCreating(false)}
          onCreated={() => { setCreating(false); handleSaved() }}
        />
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// Shared components
// ─────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────
// List view
// ─────────────────────────────────────────────────────────────

function ListView({ stories, onSelect, selected }: { stories: Story[]; onSelect: (s: Story) => void; selected: Story | null }) {
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
          <span className="text-xs text-[#888] font-mono text-right">{s.composite_score != null ? s.composite_score : s.priority_score ?? '--'}</span>
        </div>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// Kanban view
// ─────────────────────────────────────────────────────────────

function KanbanView({ stories, onSelect, onSaved }: { stories: Story[]; onSelect: (s: Story) => void; onSaved: () => void }) {
  const updateMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => api.updateStory(id, { status }),
    onSuccess: onSaved,
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

// ─────────────────────────────────────────────────────────────
// Story drawer (editable)
// ─────────────────────────────────────────────────────────────

function StoryDrawer({ story, products, onClose, onSaved, onDeleted }: {
  story: Story
  products: string[]
  onClose: () => void
  onSaved: () => void
  onDeleted: () => void
}) {
  const [editTitle, setEditTitle] = useState(story.title)
  const [editStatus, setEditStatus] = useState(story.status)
  const [editPriority, setEditPriority] = useState(story.priority)
  const [editProduct, setEditProduct] = useState(story.product)
  const [editDescription, setEditDescription] = useState(story.description)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Detect if anything changed
  const dirty =
    editTitle !== story.title ||
    editStatus !== story.status ||
    editPriority !== story.priority ||
    editProduct !== story.product ||
    editDescription !== story.description

  async function save() {
    if (!dirty || saving) return
    setSaving(true)
    setError(null)
    try {
      await api.updateStory(story.id, {
        title: editTitle,
        status: editStatus,
        priority: editPriority,
        product: editProduct,
        description: editDescription,
      })
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete() {
    if (!confirm(`Delete "${story.title}"?`)) return
    setDeleting(true)
    try {
      await api.deleteStory(story.id)
      onDeleted()
    } catch (e) {
      setError((e as Error).message)
      setDeleting(false)
    }
  }

  return (
    <div className="w-[380px] shrink-0 bg-[#141414] border border-[#222] rounded-xl p-5 sticky top-0 max-h-[calc(100vh-120px)] overflow-y-auto">
      <div className="flex items-start justify-between mb-4">
        <span className={`text-xs px-2 py-1 rounded border font-medium ${PRIORITY_COLORS[editPriority]}`}>{editPriority}</span>
        <button onClick={onClose} className="text-[#555] hover:text-[#999] cursor-pointer text-lg leading-none">&times;</button>
      </div>

      {/* Editable title */}
      <input
        value={editTitle}
        onChange={e => setEditTitle(e.target.value)}
        className="w-full text-base font-semibold text-white bg-transparent border-b border-[#2a2a2a] focus:border-purple-500/50 outline-none pb-1 mb-3"
      />

      <div className="space-y-3 text-sm">
        {/* Product */}
        <div className="flex items-center justify-between">
          <span className="text-[#666]">Product</span>
          <select value={editProduct} onChange={e => setEditProduct(e.target.value)} className="text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#ccc] rounded-md px-2 py-1 cursor-pointer outline-none">
            {products.map(p => <option key={p} value={p}>{p}</option>)}
            <option value="">--</option>
          </select>
        </div>

        {/* Status */}
        <div className="flex items-center justify-between">
          <span className="text-[#666]">Status</span>
          <select value={editStatus} onChange={e => setEditStatus(e.target.value)} className={`text-xs bg-[#1a1a1a] border border-[#2a2a2a] rounded-md px-2 py-1 cursor-pointer outline-none ${STATUS_COLORS[editStatus] ?? 'text-[#ccc]'}`}>
            {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        {/* Priority */}
        <div className="flex items-center justify-between">
          <span className="text-[#666]">Priority</span>
          <select value={editPriority} onChange={e => setEditPriority(e.target.value)} className="text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#ccc] rounded-md px-2 py-1 cursor-pointer outline-none">
            {PRIORITIES.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        {/* Scores (read-only) */}
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
        {story.source_type && (
          <div className="flex justify-between">
            <span className="text-[#666]">Source</span>
            <span className="text-[#ccc]">{story.source_type}</span>
          </div>
        )}

        {/* Description */}
        <div className="pt-2 border-t border-[#1e1e1e]">
          <div className="text-xs text-[#666] mb-1">Description</div>
          <textarea
            value={editDescription}
            onChange={e => setEditDescription(e.target.value)}
            rows={4}
            className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm text-[#aaa] outline-none focus:border-purple-500/50 resize-y placeholder:text-[#555]"
            placeholder="What does this story accomplish?"
          />
        </div>

        {/* Priority reasoning (read-only, from prioritizer) */}
        {story.priority_reasoning && (
          <div className="pt-2 border-t border-[#1e1e1e]">
            <div className="text-xs text-[#666] mb-1">Priority reasoning</div>
            <p className="text-[#aaa] text-sm">{story.priority_reasoning}</p>
          </div>
        )}

        {/* Impact breakdown (read-only) */}
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

        {/* Tags (read-only for now) */}
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

        {/* Error banner */}
        {error && (
          <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-2 pt-3 border-t border-[#1e1e1e]">
          <button
            onClick={save}
            disabled={!dirty || saving}
            className="flex-1 py-2 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700 disabled:opacity-40 disabled:cursor-default"
          >
            {saving ? 'Saving...' : 'Save changes'}
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="px-3 py-2 rounded-lg bg-red-500/10 text-red-400 text-sm cursor-pointer hover:bg-red-500/20 border border-red-500/30 disabled:opacity-40"
          >
            {deleting ? '...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// Create story modal
// ─────────────────────────────────────────────────────────────

function CreateStoryModal({ products, onClose, onCreated }: {
  products: string[]
  onClose: () => void
  onCreated: () => void
}) {
  const [title, setTitle] = useState('')
  const [product, setProduct] = useState(products[0] ?? '')
  const [priority, setPriority] = useState('P2')
  const [description, setDescription] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function create() {
    if (!title.trim() || saving) return
    setSaving(true)
    setError(null)
    try {
      await api.createStory({
        title: title.trim(),
        product,
        priority,
        description: description.trim(),
        source_type: 'manual',
      })
      onCreated()
    } catch (e) {
      setError((e as Error).message)
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-30" onClick={onClose}>
      <div className="bg-[#141414] border border-[#222] rounded-xl p-6 w-[480px] shadow-2xl" onClick={e => e.stopPropagation()}>
        <h2 className="text-lg font-semibold text-white mb-4">New Story</h2>

        <div className="space-y-3">
          <div>
            <label className="text-xs text-[#666] mb-1 block">Title</label>
            <input
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="What needs to happen?"
              autoFocus
              className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm text-[#ddd] outline-none focus:border-purple-500/50 placeholder:text-[#555]"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-[#666] mb-1 block">Product</label>
              <select value={product} onChange={e => setProduct(e.target.value)} className="w-full text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#ccc] rounded-md px-2 py-2 cursor-pointer outline-none">
                {products.map(p => <option key={p} value={p}>{p}</option>)}
                <option value="">Other</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-[#666] mb-1 block">Priority</label>
              <select value={priority} onChange={e => setPriority(e.target.value)} className="w-full text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#ccc] rounded-md px-2 py-2 cursor-pointer outline-none">
                {PRIORITIES.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
          </div>

          <div>
            <label className="text-xs text-[#666] mb-1 block">Description</label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Context, scope, acceptance criteria..."
              rows={4}
              className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm text-[#ddd] outline-none focus:border-purple-500/50 placeholder:text-[#555] resize-y"
            />
          </div>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <div className="flex gap-2 pt-2">
            <button
              onClick={create}
              disabled={!title.trim() || saving}
              className="flex-1 py-2.5 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700 disabled:opacity-50"
            >
              {saving ? 'Creating...' : 'Create Story'}
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2.5 rounded-lg bg-[#1e1e1e] text-[#888] text-sm cursor-pointer hover:text-white border border-[#2a2a2a]"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
