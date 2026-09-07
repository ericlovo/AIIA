import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type VoiceSessionResponse, type VoiceStatusResponse, type VoiceToolResult } from '../lib/api'

type Phase = 'idle' | 'connecting' | 'listening' | 'speaking' | 'tool'
type TranscriptLine = { role: 'user' | 'assistant'; text: string }
type ToolChip = { id: string; name: string; state: 'running' | 'ok' | 'error'; detail: string }

const SAMPLE_RATE = 24_000

export function VoiceConductor() {
  const qc = useQueryClient()
  const { data, isLoading, isError } = useQuery({
    queryKey: ['voice-status'],
    queryFn: api.voiceStatus,
    refetchInterval: 20_000,
    retry: false,
  })
  const [open, setOpen] = useState(false)
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState('')
  const [transcript, setTranscript] = useState<TranscriptLine[]>([])
  const [chips, setChips] = useState<ToolChip[]>([])
  const sessionRef = useRef<LiveSession | null>(null)

  const configured = data?.status === 'connected'
  const reason = data?.reason ?? (isError ? 'status_unavailable' : '')

  const stopTalking = useCallback(async () => {
    const session = sessionRef.current
    if (!session) return
    await session.commit()
    setPhase(current => (current === 'listening' ? 'idle' : current))
  }, [])

  const startTalking = useCallback(async () => {
    if (!configured || phase === 'connecting') return
    setError('')
    try {
      if (!sessionRef.current) {
        setPhase('connecting')
        const minted = await api.voiceSession()
        sessionRef.current = await LiveSession.connect(minted, {
          onPhase: setPhase,
          onTranscript: line => setTranscript(prev => mergeTranscript(prev, line)),
          onTool: chip => {
            setChips(prev => upsertChip(prev, chip))
            if (chip.state !== 'running') {
              qc.invalidateQueries({ queryKey: ['agents'] })
              qc.invalidateQueries({ queryKey: ['assignments'] })
              qc.invalidateQueries({ queryKey: ['handoffs'] })
            }
          },
          onError: message => setError(message),
          onClose: () => {
            sessionRef.current = null
            setPhase('idle')
          },
        })
      }
      await sessionRef.current.startMic()
      setPhase('listening')
    } catch (err) {
      sessionRef.current = null
      setPhase('idle')
      setError(err instanceof Error ? err.message : 'voice_session_failed')
    }
  }, [configured, phase, qc])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || event.repeat) return
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (!configured) return
      event.preventDefault()
      if (event.type === 'keydown') void startTalking()
      else void stopTalking()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKey)
    }
  }, [configured, startTalking, stopTalking])

  useEffect(() => () => {
    sessionRef.current?.close()
    sessionRef.current = null
  }, [])

  return (
    <section className="z-20 shrink-0 border-t border-neutral-900 bg-neutral-950">
      <div className="flex flex-wrap items-center gap-3 px-4 py-3 sm:px-6">
        <button
          type="button"
          onPointerDown={event => {
            if (event.button !== 0) return
            event.preventDefault()
            void startTalking()
          }}
          onPointerUp={() => void stopTalking()}
          onPointerCancel={() => void stopTalking()}
          onPointerLeave={event => {
            if (event.buttons) void stopTalking()
          }}
          disabled={!configured}
          aria-pressed={phase === 'listening'}
          className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full border text-lg ${micClass(phase, configured)}`}
          title={configured ? 'Hold to talk' : notConfiguredLabel(data, reason)}
        >
          {phase === 'connecting' ? '…' : phase === 'listening' ? '●' : '🎤'}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] font-semibold tracking-[0.22em] uppercase text-cyan-400">Voice Conductor</span>
            <StatusDot phase={phase} configured={configured} loading={isLoading} />
          </div>
          <p className="mt-1 text-xs text-neutral-500">
            {isLoading
              ? 'Checking Grok Voice…'
              : configured
                ? holdHint(phase)
                : notConfiguredLabel(data, reason)}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setOpen(value => !value)}
          className="border border-neutral-800 px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-neutral-400 hover:border-neutral-600 hover:text-neutral-200"
        >
          {open ? 'Hide' : 'Transcript'}
        </button>
      </div>
      {chips.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-4 pb-3 sm:px-6">
          {chips.slice(-6).map(chip => (
            <span
              key={chip.id}
              className={`border px-2 py-1 text-[10px] uppercase tracking-[0.12em] ${chipClass(chip.state)}`}
            >
              {chip.name} · {chip.state}{chip.detail ? ` · ${chip.detail}` : ''}
            </span>
          ))}
        </div>
      )}
      {error && <p className="px-4 pb-3 text-xs text-amber-300/90 sm:px-6">{error}</p>}
      {open && (
        <div className="max-h-48 overflow-y-auto border-t border-neutral-900 px-4 py-3 text-xs leading-relaxed text-neutral-300 sm:px-6">
          {transcript.length === 0 && (
            <p className="text-neutral-600">Hold the mic (or spacebar) to talk. Assignments and Mini status appear as tool chips.</p>
          )}
          {transcript.map((line, index) => (
            <p key={`${line.role}-${index}`} className={line.role === 'user' ? 'text-cyan-200/90' : 'mt-2 text-neutral-300'}>
              <span className="mr-2 text-[10px] uppercase tracking-[0.14em] text-neutral-600">{line.role}</span>
              {line.text}
            </p>
          ))}
        </div>
      )}
    </section>
  )
}

function holdHint(phase: Phase): string {
  if (phase === 'connecting') return 'Minting a short-lived Grok token…'
  if (phase === 'listening') return 'Listening — release to send.'
  if (phase === 'speaking') return 'Speaking…'
  if (phase === 'tool') return 'Calling Agent Studio…'
  return 'Hold to talk · spacebar works too. Orchestrates assignments only.'
}

function notConfiguredLabel(data: VoiceStatusResponse | undefined, reason: string): string {
  if (reason === 'airgap') return 'Voice blocked under AIIA_AIRGAP.'
  if (reason === 'missing_xai_api_key' || data?.status === 'not_configured') {
    return 'Not configured — set XAI_API_KEY on the Mini (~/.aiia/keys.json or .env).'
  }
  return 'Voice status unavailable.'
}

function micClass(phase: Phase, configured: boolean): string {
  if (!configured) return 'cursor-not-allowed border-neutral-800 text-neutral-700'
  if (phase === 'listening') return 'border-cyan-400 bg-cyan-500/20 text-cyan-200 shadow-[0_0_24px_rgba(34,211,238,0.25)]'
  if (phase === 'speaking') return 'border-purple-400/70 bg-purple-500/10 text-purple-200'
  if (phase === 'tool') return 'border-amber-400/70 bg-amber-500/10 text-amber-200'
  return 'border-neutral-700 text-neutral-200 hover:border-cyan-500/60'
}

function chipClass(state: ToolChip['state']): string {
  if (state === 'ok') return 'border-emerald-500/40 text-emerald-300'
  if (state === 'error') return 'border-red-500/40 text-red-300'
  return 'border-amber-500/40 text-amber-200'
}

function StatusDot({ phase, configured, loading }: { phase: Phase; configured: boolean; loading: boolean }) {
  const color = loading
    ? 'bg-neutral-600'
    : !configured
      ? 'bg-neutral-600'
      : phase === 'listening'
        ? 'bg-cyan-400'
        : phase === 'speaking'
          ? 'bg-purple-400'
          : phase === 'tool'
            ? 'bg-amber-400'
            : 'bg-green-500'
  const label = loading ? 'checking' : !configured ? 'not configured' : phase === 'idle' ? 'ready' : phase
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-neutral-500">
      <i className={`h-1.5 w-1.5 rounded-full ${color}`} />
      {label}
    </span>
  )
}

function mergeTranscript(lines: TranscriptLine[], incoming: TranscriptLine): TranscriptLine[] {
  const last = lines[lines.length - 1]
  if (last && last.role === incoming.role) {
    return [...lines.slice(0, -1), { role: incoming.role, text: incoming.text }]
  }
  return [...lines, incoming]
}

function upsertChip(chips: ToolChip[], incoming: ToolChip): ToolChip[] {
  const index = chips.findIndex(chip => chip.id === incoming.id)
  if (index === -1) return [...chips, incoming]
  const next = chips.slice()
  next[index] = incoming
  return next
}

type SessionHandlers = {
  onPhase: (phase: Phase) => void
  onTranscript: (line: TranscriptLine) => void
  onTool: (chip: ToolChip) => void
  onError: (message: string) => void
  onClose: () => void
}

class LiveSession {
  private ws: WebSocket
  private audioCtx: AudioContext | null = null
  private media: MediaStream | null = null
  private processor: ScriptProcessorNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private playTime = 0
  private pendingTools = 0
  private closed = false
  private handlers: SessionHandlers

  private constructor(ws: WebSocket, handlers: SessionHandlers) {
    this.ws = ws
    this.handlers = handlers
  }

  static connect(session: VoiceSessionResponse, handlers: SessionHandlers): Promise<LiveSession> {
    return new Promise((resolve, reject) => {
      const url = session.realtime_url || 'wss://api.x.ai/v1/realtime?model=grok-voice-latest'
      const ws = new WebSocket(url, [`xai-client-secret.${session.token}`])
      const live = new LiveSession(ws, handlers)
      const timer = window.setTimeout(() => {
        ws.close()
        reject(new Error('grok_voice_timeout'))
      }, 12_000)
      ws.onopen = () => {
        window.clearTimeout(timer)
        ws.send(JSON.stringify({ type: 'session.update', session: session.session }))
        resolve(live)
      }
      ws.onerror = () => {
        window.clearTimeout(timer)
        reject(new Error('grok_voice_unavailable'))
      }
      ws.onclose = () => {
        live.closed = true
        live.stopMic()
        handlers.onClose()
      }
      ws.onmessage = event => {
        void live.onMessage(event.data)
      }
    })
  }

  async startMic() {
    if (this.media) return
    this.audioCtx = this.audioCtx ?? new AudioContext({ sampleRate: SAMPLE_RATE })
    if (this.audioCtx.state === 'suspended') await this.audioCtx.resume()
    this.media = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    })
    this.source = this.audioCtx.createMediaStreamSource(this.media)
    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1)
    this.processor.onaudioprocess = event => {
      if (this.ws.readyState !== WebSocket.OPEN) return
      const input = event.inputBuffer.getChannelData(0)
      const pcm = floatToPcm16(input)
      this.ws.send(JSON.stringify({
        type: 'input_audio_buffer.append',
        audio: bytesToBase64(pcm),
      }))
    }
    this.source.connect(this.processor)
    this.processor.connect(this.audioCtx.destination)
  }

  async commit() {
    this.stopMic()
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }))
      this.ws.send(JSON.stringify({ type: 'response.create' }))
    }
  }

  stopMic() {
    this.processor?.disconnect()
    this.source?.disconnect()
    this.processor = null
    this.source = null
    this.media?.getTracks().forEach(track => track.stop())
    this.media = null
  }

  close() {
    this.stopMic()
    if (!this.closed && this.ws.readyState === WebSocket.OPEN) this.ws.close()
    void this.audioCtx?.close()
    this.audioCtx = null
  }

  private async onMessage(raw: unknown) {
    const event = typeof raw === 'string' ? JSON.parse(raw) : JSON.parse(String(raw))
    const type = String(event.type || '')
    if (type === 'response.output_audio.delta' && event.delta) {
      this.handlers.onPhase('speaking')
      this.enqueuePlayback(String(event.delta))
    } else if (type === 'response.output_audio_transcript.delta' || type === 'response.audio_transcript.delta') {
      const delta = String(event.delta || event.text || '')
      if (delta) this.handlers.onTranscript({ role: 'assistant', text: delta })
    } else if (type === 'response.output_audio_transcript.done' || type === 'response.audio_transcript.done') {
      const text = String(event.transcript || event.text || '')
      if (text) this.handlers.onTranscript({ role: 'assistant', text })
    } else if (type.includes('input_audio_transcription')) {
      const text = String(event.transcript || event.text || '')
      if (text) this.handlers.onTranscript({ role: 'user', text })
    } else if (type === 'response.function_call_arguments.done') {
      await this.handleTool(event)
    } else if (type === 'response.done') {
      if (this.pendingTools === 0) this.handlers.onPhase('idle')
    } else if (type === 'error') {
      this.handlers.onError(String(event.error?.message || event.message || 'grok_voice_error'))
    }
  }

  private async handleTool(event: { name?: string; call_id?: string; arguments?: string }) {
    const name = event.name || 'tool'
    const callId = event.call_id || name
    let args: Record<string, unknown> = {}
    try {
      args = event.arguments ? JSON.parse(event.arguments) as Record<string, unknown> : {}
    } catch {
      args = {}
    }
    this.pendingTools += 1
    this.handlers.onPhase('tool')
    this.handlers.onTool({ id: callId, name, state: 'running', detail: '' })
    let output: Record<string, unknown> | VoiceToolResult
    try {
      const result = await api.voiceTool(name, args)
      output = result
      const detail = summarizeTool(name, result.result)
      this.handlers.onTool({ id: callId, name, state: 'ok', detail })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'tool_failed'
      output = { ok: false, error: message }
      this.handlers.onTool({ id: callId, name, state: 'error', detail: message })
    }
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'conversation.item.create',
        item: {
          type: 'function_call_output',
          call_id: callId,
          output: JSON.stringify(output),
        },
      }))
    }
    this.pendingTools = Math.max(0, this.pendingTools - 1)
    if (this.pendingTools === 0 && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'response.create' }))
    }
  }

  private enqueuePlayback(b64: string) {
    if (!this.audioCtx) this.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE })
    const ctx = this.audioCtx
    const bytes = base64ToBytes(b64)
    const floats = pcm16ToFloat(bytes)
    if (!floats.length) return
    const buffer = ctx.createBuffer(1, floats.length, SAMPLE_RATE)
    buffer.getChannelData(0).set(floats)
    const node = ctx.createBufferSource()
    node.buffer = buffer
    node.connect(ctx.destination)
    const startAt = Math.max(this.playTime, ctx.currentTime)
    node.start(startAt)
    this.playTime = startAt + buffer.duration
  }
}

function summarizeTool(name: string, result: Record<string, unknown> | undefined): string {
  if (!result) return ''
  if (name === 'mini_status') return String(result.status ?? '')
  if (name === 'list_agents') return `${result.count ?? 0} agents`
  if (name === 'list_assignments') return `${result.count ?? 0} assignments`
  if (name === 'list_handoffs') return `${result.count ?? 0} handoffs`
  if (name === 'create_assignment') {
    const assignment = result.assignment as { id?: string } | undefined
    return assignment?.id ? `created ${assignment.id}` : 'created'
  }
  if (name === 'run_assignment') {
    const assignment = result.assignment as { status?: string } | undefined
    return assignment?.status ?? 'ran'
  }
  return ''
}

function floatToPcm16(input: Float32Array): Uint8Array {
  const out = new ArrayBuffer(input.length * 2)
  const view = new DataView(out)
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]))
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  return new Uint8Array(out)
}

function pcm16ToFloat(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const floats = new Float32Array(Math.floor(bytes.byteLength / 2))
  for (let i = 0; i < floats.length; i += 1) {
    floats[i] = view.getInt16(i * 2, true) / 0x8000
  }
  return floats
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return bytes
}
