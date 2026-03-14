import { useState, useRef, useEffect } from 'react'

type Tab = 'ask' | 'search' | 'remember'

const CATEGORIES = ['decisions', 'patterns', 'lessons', 'project', 'meta', 'team', 'agents', 'sessions']

export function BrainOverlay({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [tab, setTab] = useState<Tab>('ask')

  if (!open) return null

  return (
    <div className="w-[400px] shrink-0 bg-[#111] border-l border-[#222] flex flex-col h-screen">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#222]">
        <div className="flex gap-1">
          {(['ask', 'search', 'remember'] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-xs px-3 py-1.5 rounded-md cursor-pointer capitalize ${
                tab === t ? 'bg-purple-600/20 text-purple-400' : 'text-[#666] hover:text-[#999]'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <button onClick={onClose} className="text-[#555] hover:text-[#999] cursor-pointer text-lg">&times;</button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {tab === 'ask' && <AskTab />}
        {tab === 'search' && <SearchTab />}
        {tab === 'remember' && <RememberTab />}
      </div>
    </div>
  )
}

function AskTab() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function send() {
    if (!input.trim() || streaming) return
    const q = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setStreaming(true)

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q, mode: 'text' }),
      })

      const reader = res.body?.getReader()
      if (!reader) return

      let answer = ''
      setMessages(prev => [...prev, { role: 'assistant', content: '' }])

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(line.slice(6))
            if (evt.type === 'chunk') {
              answer += evt.content
              setMessages(prev => {
                const copy = [...prev]
                copy[copy.length - 1] = { role: 'assistant', content: answer }
                return copy
              })
            }
          } catch { /* skip */ }
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err}` }])
    }
    setStreaming(false)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-sm text-[#555] italic">Ask AIIA anything. Searches knowledge + memory + LLM reasoning.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`text-sm ${m.role === 'user' ? 'text-blue-400' : 'text-[#ccc]'}`}>
            <span className="text-[10px] text-[#555] uppercase block mb-0.5">{m.role === 'user' ? 'You' : 'AIIA'}</span>
            <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="p-3 border-t border-[#222]">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
            placeholder="Ask AIIA..."
            className="flex-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm text-[#ddd] outline-none focus:border-purple-500/50 placeholder:text-[#555]"
          />
          <button
            onClick={send}
            disabled={streaming || !input.trim()}
            className="px-3 py-2 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

function SearchTab() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<{ fact: string; category: string; created_at: string }[]>([])
  const [categoryFilter, setCategoryFilter] = useState('')
  const [loading, setLoading] = useState(false)

  async function search() {
    if (!query.trim()) return
    setLoading(true)
    try {
      const res = await fetch(`/api/memories?search=${encodeURIComponent(query.trim())}${categoryFilter ? `&category=${categoryFilter}` : ''}`)
      const data = await res.json()
      setResults(data.memories ?? [])
    } catch { setResults([]) }
    setLoading(false)
  }

  return (
    <div className="p-4">
      <div className="flex gap-2 mb-3">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && search()}
          placeholder="Search memories..."
          className="flex-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm text-[#ddd] outline-none focus:border-purple-500/50 placeholder:text-[#555]"
        />
        <button onClick={search} disabled={loading} className="px-3 py-2 rounded-lg bg-[#1e1e1e] text-[#888] text-sm cursor-pointer hover:text-white border border-[#2a2a2a] disabled:opacity-50">
          Search
        </button>
      </div>
      <select
        value={categoryFilter}
        onChange={e => setCategoryFilter(e.target.value)}
        className="text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#888] rounded-md px-2 py-1.5 mb-3 cursor-pointer outline-none"
      >
        <option value="">All categories</option>
        {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
      </select>

      <div className="space-y-2">
        {results.map((r, i) => (
          <div key={i} className="bg-[#1a1a1a] border border-[#222] rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400">{r.category}</span>
              <span className="text-[10px] text-[#555]">{new Date(r.created_at).toLocaleDateString()}</span>
            </div>
            <p className="text-sm text-[#ccc] leading-relaxed">{r.fact}</p>
          </div>
        ))}
        {results.length === 0 && query && !loading && (
          <p className="text-sm text-[#555] italic">No results.</p>
        )}
      </div>
    </div>
  )
}

function RememberTab() {
  const [fact, setFact] = useState('')
  const [category, setCategory] = useState('decisions')
  const [saved, setSaved] = useState(false)

  async function save() {
    if (!fact.trim()) return
    try {
      await fetch('/api/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fact: fact.trim(), category }),
      })
      setSaved(true)
      setFact('')
      setTimeout(() => setSaved(false), 2000)
    } catch { /* */ }
  }

  return (
    <div className="p-4">
      <p className="text-sm text-[#666] mb-3">Save a fact to AIIA's memory.</p>
      <select
        value={category}
        onChange={e => setCategory(e.target.value)}
        className="text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#888] rounded-md px-2 py-1.5 mb-3 w-full cursor-pointer outline-none"
      >
        {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
      <textarea
        value={fact}
        onChange={e => setFact(e.target.value)}
        placeholder="What should AIIA remember?"
        rows={4}
        className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm text-[#ddd] outline-none focus:border-purple-500/50 placeholder:text-[#555] resize-none mb-3"
      />
      <button
        onClick={save}
        disabled={!fact.trim()}
        className="w-full py-2 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700 disabled:opacity-50"
      >
        {saved ? 'Saved!' : 'Remember'}
      </button>
    </div>
  )
}
