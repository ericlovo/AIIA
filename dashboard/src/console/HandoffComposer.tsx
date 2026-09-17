import { useState } from 'react'
import type { Agent, Assignment, Handoff } from '../lib/api'
import { HANDOFF_INSTRUCTIONS_MAX, defaultHandoffInstructions, handoffErrorText, handoffInstructionsError } from './mapRelationships'

interface HandoffComposerProps {
  source: Assignment
  fromAgent: Agent | null
  toAgent: Agent
  onCreate: (instructions: string) => Promise<Handoff>
  onCreated: (handoff: Handoff) => void
  onCancel: () => void
  onOpenForm: () => void
}

export function HandoffComposer({ source, fromAgent, toAgent, onCreate, onCreated, onCancel, onOpenForm }: HandoffComposerProps) {
  const [instructions, setInstructions] = useState(() => defaultHandoffInstructions(source))
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const invalid = handoffInstructionsError(instructions)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (invalid || pending) return
    setPending(true)
    setError('')
    try {
      onCreated(await onCreate(instructions.trim()))
    } catch (failure) {
      setError(handoffErrorText(failure instanceof Error ? failure.message : ''))
      setPending(false)
    }
  }

  return (
    <aside aria-label="Confirm handoff" onKeyDown={event => { if (event.key === 'Escape') onCancel() }} className="pointer-events-auto absolute right-4 top-4 max-h-[calc(100%-2rem)] w-[min(320px,calc(100%-2rem))] overflow-y-auto border border-fuchsia-400/50 bg-[#0d0910] p-4 text-left shadow-2xl sm:right-6">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[9px] font-semibold tracking-[0.18em] uppercase text-fuchsia-300/80">New handoff · brief</div>
          <div className="mt-1 break-words text-sm font-medium text-white">{source.title}</div>
          <div className="mt-1 break-words text-xs text-white/50">{fromAgent?.name ?? 'Removed agent'} → {toAgent.name}</div>
        </div>
        <button type="button" onClick={onCancel} className="text-lg leading-none text-white/30 hover:text-white" aria-label="Cancel handoff">×</button>
      </div>
      <form onSubmit={submit} className="mt-3">
        <label className="block">
          <span className="mb-1.5 block text-[9px] font-semibold tracking-[0.16em] uppercase text-white/40">Instructions for {toAgent.name}</span>
          <textarea
            autoFocus
            value={instructions}
            maxLength={HANDOFF_INSTRUCTIONS_MAX}
            rows={5}
            onChange={event => setInstructions(event.target.value)}
            className="w-full border border-white/15 bg-black/40 px-2.5 py-2 text-xs leading-relaxed text-white outline-none focus:border-fuchsia-300/70"
          />
        </label>
        {invalid && <p className="mt-1.5 text-[11px] text-amber-200/80">{invalid}</p>}
        {error && <p role="alert" className="mt-2 border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-red-300">{error}</p>}
        <div className="mt-3 flex flex-wrap gap-2">
          <button type="submit" disabled={Boolean(invalid) || pending} className="bg-fuchsia-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950 disabled:cursor-not-allowed disabled:opacity-50">{pending ? 'Creating handoff' : 'Create handoff'}</button>
          <button type="button" onClick={onCancel} className="border border-white/15 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-white/70 hover:border-white/40">Cancel</button>
          <button type="button" onClick={onOpenForm} className="px-1 py-2 text-[10px] text-white/40 underline hover:text-white">Open full form</button>
        </div>
      </form>
    </aside>
  )
}
