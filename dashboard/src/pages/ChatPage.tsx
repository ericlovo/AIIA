import { useEffect, useMemo, useRef, useState } from 'react'
import {
  deriveThreadTitle,
  loadChatState,
  newThreadId,
  saveChatState,
  TEAM_USERS,
  type ChatMessage,
  type ChatThread,
  type TeamUser,
} from '../lib/chatStore'

// ─────────────────────────────────────────────────────────────
// Helpers (module-level so they aren't recreated per render)
// ─────────────────────────────────────────────────────────────

function nowIso(): string {
  return new Date().toISOString()
}

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime()
  const m = Math.floor(ms / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

interface StreamOptions {
  history: ChatMessage[]
  user: TeamUser
  onChunk: (content: string) => void
  onThinking: (isThinking: boolean) => void
  signal: AbortSignal
}

// Fast path: talk to Ollama directly with gemma4:e4b (~28 tok/s on M4).
// Bypasses AIIA's RAG pipeline, which uses gemma4:26b at 32K context and
// has multi-minute first-token latency on this hardware. Knowledge and
// memory integration can come back as an opt-in "deep" mode later.
const OLLAMA_MODEL = 'gemma4:e4b'

function buildSystemPrompt(user: TeamUser): string {
  return [
    'You are AIIA, an AI teammate running on the Mac Mini at Aplora AI.',
    `You are talking to ${user}, a member of the Aplora team.`,
    'Be concise and direct. Match the tone of a trusted colleague, not a chatbot.',
    'If you need more context before answering, ask one focused question.',
  ].join(' ')
}

interface OllamaChatFrame {
  message?: { role: string; content: string }
  done?: boolean
}

async function streamChat({ history, user, onChunk, onThinking, signal }: StreamOptions): Promise<void> {
  // Ollama expects a messages array (system + alternating user/assistant).
  // Filter out the empty assistant placeholder we appended for the pending reply.
  const messages = [
    { role: 'system', content: buildSystemPrompt(user) },
    ...history
      .filter(m => m.content.length > 0 || m.role === 'user')
      .map(m => ({ role: m.role, content: m.content })),
  ]

  const res = await fetch('/ollama/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages,
      stream: true,
      options: {
        temperature: 0.7,
        num_ctx: 8192,
        num_predict: 2048,
      },
      keep_alive: '24h',
    }),
    signal,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 200)}` : ''}`)
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''
  let accumulated = ''
  let seenContent = false

  // Ollama streams newline-delimited JSON, one frame per line, not SSE.
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        const frame = JSON.parse(trimmed) as OllamaChatFrame
        const thinking = (frame.message as Record<string, unknown> | undefined)?.thinking
        const content = frame.message?.content

        // Gemma 4 E4B emits thinking tokens first (content=''), then
        // real content tokens. Signal the UI to show a "thinking" state
        // while only thinking tokens flow.
        if (typeof thinking === 'string' && thinking.length > 0 && !seenContent) {
          onThinking(true)
        }
        if (typeof content === 'string' && content.length > 0) {
          if (!seenContent) {
            seenContent = true
            onThinking(false)
          }
          accumulated += content
          onChunk(accumulated)
        }
      } catch {
        // Partial line, wait for more data
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Identity modal (first-load)
// ─────────────────────────────────────────────────────────────

function IdentityModal({ onSelect }: { onSelect: (u: TeamUser) => void }) {
  return (
    <div className="absolute inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-20">
      <div className="bg-[#111] border border-[#222] rounded-xl p-6 w-[320px] shadow-2xl">
        <div className="text-xs font-bold text-purple-400 tracking-wider mb-1">AIIA</div>
        <h2 className="text-lg font-semibold text-white mb-1">Who are you?</h2>
        <p className="text-sm text-[#666] mb-4">
          Pick your name so AIIA knows who it's talking to. You can change this later.
        </p>
        <div className="space-y-2">
          {TEAM_USERS.map(u => (
            <button
              key={u}
              onClick={() => onSelect(u)}
              className="w-full py-2.5 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] text-[#ddd] text-sm cursor-pointer hover:bg-[#1e1e1e] hover:border-purple-500/50 transition-colors"
            >
              {u}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// Thread list
// ─────────────────────────────────────────────────────────────

function ThreadList({
  threads,
  activeId,
  onSelect,
  onNew,
  onDelete,
}: {
  threads: ChatThread[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
}) {
  return (
    <aside className="w-64 shrink-0 border-r border-[#222] flex flex-col bg-[#0d0d0d]">
      <div className="p-3 border-b border-[#222]">
        <button
          onClick={onNew}
          className="w-full py-2 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700"
        >
          + New chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {threads.length === 0 && (
          <p className="p-4 text-xs text-[#555] italic">No conversations yet.</p>
        )}
        {threads.map(t => (
          <div
            key={t.id}
            className={`group relative border-b border-[#181818] ${
              activeId === t.id ? 'bg-[#181818]' : 'hover:bg-[#141414]'
            }`}
          >
            <button
              onClick={() => onSelect(t.id)}
              className="w-full text-left px-3 py-2.5 cursor-pointer"
            >
              <div className="text-sm text-[#ddd] truncate pr-6">{t.title}</div>
              <div className="text-[10px] text-[#555] mt-0.5">{timeAgo(t.createdAt)}</div>
            </button>
            <button
              onClick={() => onDelete(t.id)}
              className="absolute top-2 right-2 text-[#444] hover:text-red-400 opacity-0 group-hover:opacity-100 cursor-pointer text-sm"
              aria-label="Delete conversation"
              title="Delete conversation"
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </aside>
  )
}

// ─────────────────────────────────────────────────────────────
// Message bubble
// ─────────────────────────────────────────────────────────────

function MessageBubble({ msg, isThinking }: { msg: ChatMessage; isThinking?: boolean }) {
  const isUser = msg.role === 'user'
  const showThinking = !isUser && !msg.content && isThinking
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[720px] rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-purple-600/20 border border-purple-500/30 text-white'
            : 'bg-[#1a1a1a] border border-[#222] text-[#ddd]'
        }`}
      >
        {msg.content || (showThinking ? (
          <span className="text-purple-400 italic animate-pulse">Thinking...</span>
        ) : !isUser ? (
          <span className="text-[#555] italic">…</span>
        ) : null)}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────

export function ChatPage() {
  const initial = useMemo(() => loadChatState(), [])
  const [user, setUser] = useState<TeamUser | null>(initial.user)
  const [threads, setThreads] = useState<ChatThread[]>(initial.threads)
  const [activeThreadId, setActiveThreadId] = useState<string | null>(initial.activeThreadId)
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Derive active thread during render (no effect needed)
  const activeThread = threads.find(t => t.id === activeThreadId) ?? null

  // Persist to localStorage whenever state changes
  useEffect(() => {
    saveChatState({ version: 1, user, threads, activeThreadId })
  }, [user, threads, activeThreadId])

  // Scroll to bottom on new messages in active thread
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeThread?.messages.length, streaming])

  // Cancel any in-flight stream when the page unmounts
  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  function createThread(): string {
    const id = newThreadId()
    const thread: ChatThread = {
      id,
      title: 'New chat',
      createdAt: nowIso(),
      messages: [],
    }
    setThreads(prev => [thread, ...prev])
    setActiveThreadId(id)
    return id
  }

  function deleteThread(id: string) {
    setThreads(prev => prev.filter(t => t.id !== id))
    setActiveThreadId(curr => {
      if (curr !== id) return curr
      // Pick the next thread if available, else null
      const remaining = threads.filter(t => t.id !== id)
      return remaining[0]?.id ?? null
    })
  }

  async function send() {
    if (!user || !input.trim() || streaming) return
    const content = input.trim()
    setInput('')
    setError(null)

    // Ensure we have an active thread; create one on demand
    let threadId = activeThreadId
    if (!threadId) {
      threadId = createThread()
    }

    // Snapshot prior messages for the LLM request — this is the conversation
    // as the user saw it BEFORE sending, plus the new user turn. We derive it
    // here so streamChat doesn't need to read state that hasn't flushed yet.
    const priorMessages = threads.find(t => t.id === threadId)?.messages ?? []
    const userMsg: ChatMessage = { role: 'user', content, createdAt: nowIso() }
    const historyForRequest: ChatMessage[] = [...priorMessages, userMsg]

    // Append user message + empty assistant placeholder; derive title if new
    setThreads(prev =>
      prev.map(t => {
        if (t.id !== threadId) return t
        const assistantMsg: ChatMessage = { role: 'assistant', content: '', createdAt: nowIso() }
        const nextTitle = t.title === 'New chat' ? deriveThreadTitle(content) : t.title
        return { ...t, title: nextTitle, messages: [...t.messages, userMsg, assistantMsg] }
      })
    )

    setStreaming(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChat({
        history: historyForRequest,
        user,
        signal: controller.signal,
        onThinking: setThinking,
        onChunk: accumulated => {
          setThreads(prev =>
            prev.map(t => {
              if (t.id !== threadId) return t
              const msgs = [...t.messages]
              const lastIdx = msgs.length - 1
              if (lastIdx < 0 || msgs[lastIdx].role !== 'assistant') return t
              msgs[lastIdx] = { ...msgs[lastIdx], content: accumulated }
              return { ...t, messages: msgs }
            })
          )
        },
      })
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError((err as Error).message ?? String(err))
      }
    } finally {
      setStreaming(false)
      setThinking(false)
      abortRef.current = null
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  // Gate the page behind identity selection
  if (!user) {
    return (
      <div className="relative h-full">
        <IdentityModal onSelect={setUser} />
      </div>
    )
  }

  return (
    <div className="flex h-full">
      <ThreadList
        threads={threads}
        activeId={activeThreadId}
        onSelect={setActiveThreadId}
        onNew={createThread}
        onDelete={deleteThread}
      />

      {/* Main chat area */}
      <section className="flex-1 flex flex-col min-w-0">
        {/* Header: identity + active thread title */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-[#222]">
          <div className="min-w-0">
            <h1 className="text-sm font-semibold text-white truncate">
              {activeThread?.title ?? 'Chat'}
            </h1>
            <p className="text-[11px] text-[#555]">
              Talking to AIIA on the Mac Mini brain
            </p>
          </div>
          <button
            onClick={() => setUser(null)}
            className="text-xs text-[#666] hover:text-purple-400 cursor-pointer px-2 py-1 rounded border border-[#222] hover:border-purple-500/30"
            title="Switch user"
          >
            {user}
          </button>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
          {!activeThread || activeThread.messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="text-xs font-bold text-purple-400 tracking-wider mb-2">AIIA</div>
                <p className="text-sm text-[#888]">
                  Ask AIIA anything. Queries route through the brain's knowledge store,
                  memory, and production data tools.
                </p>
              </div>
            </div>
          ) : (
            activeThread.messages.map((m, i) => (
              <MessageBubble
                key={i}
                msg={m}
                isThinking={thinking && i === activeThread.messages.length - 1}
              />
            ))
          )}
          {error && (
            <div className="text-sm text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
              Error: {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="px-6 py-4 border-t border-[#222]">
          <div className="flex gap-2">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder={streaming ? 'AIIA is thinking…' : 'Ask AIIA…'}
              disabled={streaming}
              className="flex-1 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-4 py-2.5 text-sm text-[#ddd] outline-none focus:border-purple-500/50 placeholder:text-[#555] disabled:opacity-60"
            />
            {streaming ? (
              <button
                onClick={stop}
                className="px-4 py-2.5 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] text-[#888] text-sm cursor-pointer hover:text-white"
              >
                Stop
              </button>
            ) : (
              <button
                onClick={send}
                disabled={!input.trim()}
                className="px-4 py-2.5 rounded-lg bg-purple-600 text-white text-sm cursor-pointer hover:bg-purple-700 disabled:opacity-50"
              >
                Send
              </button>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
