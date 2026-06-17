import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { TeamUser } from '../lib/chatStore'

type IntentKind = 'question' | 'task' | 'fact'

type LogEntry =
  | { id: string; kind: 'me'; text: string; ts: string }
  | { id: string; kind: 'aiia'; text: string; ts: string; streaming?: boolean }
  | { id: string; kind: 'action'; intent: IntentKind; text: string; outcome: string; ts: string }
  | { id: string; kind: 'thinking'; ts: string }

function nowIso() {
  return new Date().toISOString()
}

function newId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

export function Direct({ user }: { user: TeamUser }) {
  const qc = useQueryClient()
  const [input, setInput] = useState('')
  const [log, setLog] = useState<LogEntry[]>([])
  const [busy, setBusy] = useState(false)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  // Transient recording state — refs, not state, to avoid spurious re-renders
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log])

  useEffect(() => () => abortRef.current?.abort(), [])

  const startRecording = useCallback(async () => {
    if (recording || transcribing) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunksRef.current = []
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        chunksRef.current = []
        if (blob.size < 1000) return
        setTranscribing(true)
        try {
          const fd = new FormData()
          fd.append('file', blob, 'audio.webm')
          const res = await fetch('/api/v1/voice/transcribe', { method: 'POST', body: fd })
          if (res.ok) {
            const { text } = await res.json()
            if (text?.trim()) setInput(prev => (prev ? prev + ' ' + text.trim() : text.trim()))
          }
        } finally {
          setTranscribing(false)
        }
      }
      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch {
      // mic permission denied — silently ignore
    }
  }, [recording, transcribing])

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop()
    recorderRef.current = null
    setRecording(false)
  }, [])

  async function submit() {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)

    const meEntry: LogEntry = { id: newId(), kind: 'me', text, ts: nowIso() }
    const thinkingId = newId()
    setLog(prev => [...prev, meEntry, { id: thinkingId, kind: 'thinking', ts: nowIso() }])

    try {
      const intent = classifyIntent(text)
      await route(intent, text, user, thinkingId, setLog, qc, abortRef)
    } catch (err) {
      setLog(prev =>
        prev.map(e =>
          e.id === thinkingId
            ? {
                id: e.id,
                kind: 'aiia',
                text: `Something went wrong: ${(err as Error).message}`,
                ts: nowIso(),
              }
            : e
        )
      )
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  return (
    <section className="bg-neutral-950 flex flex-col min-h-0">
      <div className="px-5 pt-5 pb-3 shrink-0 border-b border-neutral-900">
        <h2 className="text-[11px] font-semibold tracking-[0.25em] text-neutral-500 uppercase">Direct</h2>
        <p className="text-[11px] text-neutral-600">Ask, tell, teach — I'll figure out which</p>
      </div>

      {/* Log */}
      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-3">
        {log.length === 0 && <EmptyState user={user} />}
        {log.map(entry => (
          <LogBubble key={entry.id} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Unified input */}
      <div className="px-5 pb-5 pt-3 border-t border-neutral-900 shrink-0">
        <div className="relative">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={busy ? 'Thinking…' : "What do you need AIIA to do?"}
            rows={3}
            disabled={busy}
            className="w-full bg-neutral-900 border border-neutral-800 rounded-xl px-3 py-3 pr-24 text-sm text-neutral-300 outline-none focus:border-purple-500/40 placeholder:text-neutral-700 resize-none disabled:opacity-70"
          />
          <div className="absolute right-2 bottom-2 flex gap-1">
            {/* Mic button */}
            <button
              onClick={recording ? stopRecording : startRecording}
              disabled={busy || transcribing}
              className={`w-10 h-10 rounded-lg text-sm cursor-pointer transition-colors ${
                recording
                  ? 'bg-red-600 text-white animate-pulse'
                  : transcribing
                  ? 'bg-neutral-700 text-neutral-400'
                  : 'bg-neutral-800 text-neutral-400 hover:text-white hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-default'
              }`}
              title={recording ? 'Stop recording' : transcribing ? 'Transcribing…' : 'Voice input'}
            >
              {transcribing ? '…' : recording ? '◉' : '🎤'}
            </button>
            {/* Send/Stop button */}
            <button
              onClick={busy ? stop : submit}
              disabled={!busy && !input.trim()}
              className={`w-10 h-10 rounded-lg text-sm cursor-pointer transition-colors ${
                busy
                  ? 'bg-neutral-800 text-neutral-500 hover:text-white'
                  : 'bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-40 disabled:cursor-default'
              }`}
              title={busy ? 'Stop' : 'Send (Enter)'}
            >
              {busy ? '■' : '↑'}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-3 mt-2 text-[10px] text-neutral-700">
          <span>Enter to send · Shift+Enter for newline · 🎤 for voice</span>
        </div>
      </div>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────
// Intent classifier — regex-based, fast, good enough as a first pass.
// Refine later with a local LLM call if precision becomes an issue.
// ─────────────────────────────────────────────────────────────

function classifyIntent(text: string): IntentKind {
  const t = text.toLowerCase().trim()

  // Fact patterns — explicit teaching directives
  if (/^(remember|note|fact|lesson|decision)[:\s]/i.test(text)) return 'fact'
  if (/^(i (just )?(learned|decided)|we (just )?(decided|learned))/i.test(t)) return 'fact'

  // Question patterns — interrogatives
  if (t.endsWith('?')) return 'question'
  if (/^(what|how|why|when|where|who|which|does|do|did|is |are |can |should |could |would )/i.test(t)) {
    return 'question'
  }

  // Task patterns — imperative verbs AIIA can execute
  if (/^(add|build|create|fix|refactor|implement|write|update|remove|delete|deploy|make|set up|write tests|investigate)\b/i.test(t)) {
    return 'task'
  }

  // Default to question — safer than assuming task
  return 'question'
}

// ─────────────────────────────────────────────────────────────
// Router — dispatches classified intent to the right backend
// ─────────────────────────────────────────────────────────────

async function route(
  intent: IntentKind,
  text: string,
  user: TeamUser,
  thinkingId: string,
  setLog: React.Dispatch<React.SetStateAction<LogEntry[]>>,
  qc: ReturnType<typeof useQueryClient>,
  abortRef: React.MutableRefObject<AbortController | null>
): Promise<void> {
  if (intent === 'fact') {
    // Strip a leading "remember:" / "note:" prefix before saving
    const fact = text.replace(/^(remember|note|fact|lesson|decision)[:\s]+/i, '').trim()
    const res = await fetch('/api/memories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, category: 'decisions' }),
    })
    if (!res.ok) throw new Error(`Memory save failed: ${res.status}`)
    qc.invalidateQueries({ queryKey: ['mind-memories'] })
    setLog(prev =>
      prev.map(e =>
        e.id === thinkingId
          ? {
              id: e.id,
              kind: 'action',
              intent: 'fact',
              text,
              outcome: 'Got it. Saved to memory under decisions.',
              ts: nowIso(),
            }
          : e
      )
    )
    return
  }

  if (intent === 'task') {
    // Create a P1 story tagged with the author — this is how team attribution
    // flows: Paul's stories get @paul, Tony's get @tony, etc. The "Pick up"
    // button adds @<picker> so the card shows both author and owner.
    const res = await fetch('/api/roadmap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: text.length <= 80 ? text : text.slice(0, 77) + '…',
        priority: 'P1',
        status: 'backlog',
        description: text,
        source_type: `direct:${user.toLowerCase()}`,
        tags: [`from:${user.toLowerCase()}`],
      }),
    })
    if (!res.ok) throw new Error(`Story create failed: ${res.status}`)
    const body = await res.json().catch(() => ({}))
    const storyId = body?.story?.id ? ` (${body.story.id})` : ''
    qc.invalidateQueries({ queryKey: ['rn-stories'] })
    setLog(prev =>
      prev.map(e =>
        e.id === thinkingId
          ? {
              id: e.id,
              kind: 'action',
              intent: 'task',
              text,
              outcome: `Filed as a P1 story${storyId}. Proactive executor will decompose it off-hours.`,
              ts: nowIso(),
            }
          : e
      )
    )
    return
  }

  // Question — stream chat response via Ollama
  const controller = new AbortController()
  abortRef.current = controller
  const res = await fetch('/ollama/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gemma4:e4b',
      messages: [
        {
          role: 'system',
          content: `You are AIIA, an AI teammate at Aplora AI. You are talking to ${user}. Be concise and direct. Match the tone of a trusted colleague, not a chatbot.`,
        },
        { role: 'user', content: text },
      ],
      stream: true,
      options: { temperature: 0.7, num_ctx: 8192, num_predict: 2048 },
      keep_alive: '24h',
    }),
    signal: controller.signal,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)

  // Swap the "thinking" entry for a streaming AIIA entry
  setLog(prev =>
    prev.map(e =>
      e.id === thinkingId ? { id: e.id, kind: 'aiia', text: '', ts: nowIso(), streaming: true } : e
    )
  )

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')
  const decoder = new TextDecoder()
  let buffer = ''
  let accumulated = ''

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
        const frame = JSON.parse(trimmed) as { message?: { content?: string } }
        const content = frame.message?.content
        if (typeof content === 'string' && content.length > 0) {
          accumulated += content
          setLog(prev =>
            prev.map(e =>
              e.id === thinkingId ? { ...e, text: accumulated, streaming: true } : e
            )
          )
        }
      } catch {
        // partial frame
      }
    }
  }
  setLog(prev =>
    prev.map(e => (e.id === thinkingId ? { ...e, streaming: false } : e))
  )
}

// ─────────────────────────────────────────────────────────────
// UI bits
// ─────────────────────────────────────────────────────────────

function EmptyState({ user }: { user: TeamUser }) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center">
      <div className="text-[10px] text-purple-400 tracking-[0.3em] font-semibold mb-3">
        HI {user.toUpperCase()}
      </div>
      <p className="text-sm text-neutral-500 max-w-[280px] leading-relaxed">
        Ask me a question, tell me to build something, or teach me a fact. I'll figure out
        which is which.
      </p>
      <div className="mt-6 space-y-2 text-[11px] text-neutral-600">
        <div>"What's the status of the latest deployment?"</div>
        <div>"Add input validation to the workspace API"</div>
        <div>"Remember: the Tuesday meeting moved to 10am"</div>
      </div>
    </div>
  )
}

function LogBubble({ entry }: { entry: LogEntry }) {
  if (entry.kind === 'me') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[300px] rounded-xl rounded-br-sm px-3 py-2 text-sm bg-purple-600/15 border border-purple-500/30 text-white">
          {entry.text}
        </div>
      </div>
    )
  }
  if (entry.kind === 'thinking') {
    return (
      <div className="flex items-center gap-2 text-xs text-purple-400 italic animate-pulse px-1">
        <span className="w-1 h-1 rounded-full bg-purple-400" />
        thinking…
      </div>
    )
  }
  if (entry.kind === 'action') {
    const icon = entry.intent === 'fact' ? '✚' : entry.intent === 'task' ? '▸' : '?'
    const color =
      entry.intent === 'fact'
        ? 'text-amber-400 border-amber-500/30 bg-amber-500/5'
        : 'text-blue-400 border-blue-500/30 bg-blue-500/5'
    return (
      <div className={`rounded-lg border p-3 ${color}`}>
        <div className="flex items-center gap-2 mb-1 text-[10px] tracking-[0.2em] uppercase opacity-80">
          <span>{icon}</span>
          <span>{entry.intent}</span>
        </div>
        <div className="text-xs text-neutral-300">{entry.outcome}</div>
      </div>
    )
  }
  // aiia
  return (
    <div className="flex justify-start">
      <div className="max-w-[320px] rounded-xl rounded-bl-sm px-3 py-2 text-sm bg-neutral-900 border border-neutral-800 text-neutral-300 whitespace-pre-wrap">
        {entry.text || <span className="text-neutral-600 italic">…</span>}
        {entry.streaming && <span className="inline-block w-1 h-3 bg-purple-400 ml-0.5 animate-pulse" />}
      </div>
    </div>
  )
}
