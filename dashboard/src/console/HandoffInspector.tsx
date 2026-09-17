import { useRef, useState } from 'react'
import type { Agent, Assignment, Handoff } from '../lib/api'
import { formatHandoffTime, handoffErrorText } from './mapRelationships'

interface HandoffInspectorProps {
  handoff: Handoff
  agents: Agent[]
  assignments: Assignment[]
  onClose: () => void
  onOpenAssignment: (assignmentId: string) => void
  onDelete: (handoffId: string) => Promise<void>
}

export function HandoffInspector({ handoff, agents, assignments, onClose, onOpenAssignment, onDelete }: HandoffInspectorProps) {
  const [confirming, setConfirming] = useState(false)
  const [pending, setPending] = useState(false)
  const inFlight = useRef(false)
  const [error, setError] = useState('')
  const agentName = (id: string) => agents.find(agent => agent.id === id)?.name ?? 'Removed agent'
  const assignmentTitle = (id: string) => assignments.find(item => item.id === id)?.title ?? 'Assignment not loaded'
  const running = handoff.status === 'running'
  const target = assignments.find(item => item.id === handoff.target_assignment_id)

  async function remove() {
    if (inFlight.current) return
    inFlight.current = true
    setPending(true)
    setError('')
    try {
      await onDelete(handoff.id)
    } catch (failure) {
      setError(handoffErrorText(failure instanceof Error ? failure.message : ''))
      inFlight.current = false
      setPending(false)
      setConfirming(false)
    }
  }

  return (
    <aside aria-label="Handoff controls" onKeyDown={event => { if (event.key === 'Escape') onClose() }} className="pointer-events-auto absolute right-4 top-4 max-h-[calc(100%-2rem)] w-[min(300px,calc(100%-2rem))] overflow-y-auto border border-fuchsia-400/40 bg-[#0d0910] p-4 text-left shadow-2xl sm:right-6">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[9px] font-semibold tracking-[0.18em] uppercase text-fuchsia-300/80">Handoff · {handoff.artifact_type}</div>
          <div className="mt-1 break-words text-sm font-medium text-white">{agentName(handoff.from_agent_id)} → {agentName(handoff.to_agent_id)}</div>
        </div>
        <button type="button" onClick={onClose} className="text-lg leading-none text-white/30 hover:text-white" aria-label="Close handoff controls">×</button>
      </div>
      <dl className="mt-3 grid grid-cols-[4.5rem_minmax(0,1fr)] gap-x-2 gap-y-1.5 text-xs">
        <dt className="text-white/35">Source</dt>
        <dd className="break-words text-white/75">{agentName(handoff.from_agent_id)} · {assignmentTitle(handoff.source_assignment_id)}</dd>
        <dt className="text-white/35">Target</dt>
        <dd className="break-words text-white/75">{agentName(handoff.to_agent_id)} · {assignmentTitle(handoff.target_assignment_id)}</dd>
        <dt className="text-white/35">Status</dt>
        <dd className={handoff.status === 'failed' ? 'text-red-300' : handoff.status === 'running' ? 'text-amber-300' : handoff.status === 'completed' ? 'text-cyan-300' : 'text-white/75'}>{handoff.status}</dd>
        <dt className="text-white/35">Created</dt>
        <dd className="text-white/75">{formatHandoffTime(handoff.created_at)}</dd>
      </dl>
      {handoff.instructions && <p className="mt-3 line-clamp-4 break-words text-xs leading-relaxed text-white/45">{handoff.instructions}</p>}
      {error && <p role="alert" className="mt-3 border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-red-300">{error}</p>}
      {confirming ? (
        <div className="mt-4 border border-red-900/50 bg-red-950/20 p-3">
          <p className="text-xs text-red-100/80">Remove this handoff? The target assignment stays in the queue.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" disabled={pending} onClick={() => { void remove() }} className="bg-red-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950 disabled:cursor-wait disabled:opacity-50">{pending ? 'Removing' : 'Confirm remove'}</button>
            <button type="button" disabled={pending} onClick={() => setConfirming(false)} className="border border-white/15 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-white/70 hover:border-white/40">Keep handoff</button>
          </div>
        </div>
      ) : (
        <div className="mt-4 flex flex-wrap gap-2">
          {target && <button type="button" onClick={() => onOpenAssignment(target.id)} className="bg-cyan-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950">Open target work</button>}
          <button type="button" disabled={running} title={running ? 'A running handoff cannot be removed' : undefined} onClick={() => setConfirming(true)} className="border border-red-400/40 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-red-200 hover:border-red-200 disabled:cursor-not-allowed disabled:opacity-40">Remove handoff</button>
        </div>
      )}
    </aside>
  )
}
