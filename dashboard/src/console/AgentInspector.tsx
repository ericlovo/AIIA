import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type Agent } from '../lib/api'
import { listSummary, loopSummary, modelSummary, truncateText } from './agentConfig'

interface AgentInspectorProps {
  agent: Agent
  onClose: () => void
  onManageAgent: (agentId: string) => void
  onAssignAgent: (agentId: string) => void
}

export function AgentInspector({ agent, onClose, onManageAgent, onAssignAgent }: AgentInspectorProps) {
  const models = useQuery({ queryKey: ['agent-models'], queryFn: api.agentModels, staleTime: 60_000, retry: false })
  const lastResult = truncateText(agent.last_result)
  const lastError = truncateText(agent.last_error)
  return (
    <aside aria-label="Node controls" className="pointer-events-auto absolute right-4 top-4 max-h-[calc(100%-2rem)] w-[min(320px,calc(100%-2rem))] overflow-y-auto border border-white/15 bg-[#090c10] p-4 text-left shadow-2xl sm:right-6">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[9px] font-semibold tracking-[0.18em] uppercase text-cyan-300/70">agent controls</div>
          <div className="mt-1 break-words text-sm font-medium text-white">{agent.name}</div>
        </div>
        <button type="button" onClick={onClose} className="text-lg leading-none text-white/30 hover:text-white" aria-label="Close node controls">×</button>
      </div>
      <p className="mt-3 line-clamp-4 text-xs leading-relaxed text-white/45">{agent.mission}</p>

      <dl aria-label="Agent configuration" className="mt-4 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[11px]">
        <ConfigRow label="Model">{modelSummary(agent.model, models.data?.default)}</ConfigRow>
        <ConfigRow label="Temperature">{agent.temperature ?? '—'}</ConfigRow>
        <ConfigRow label="Max tokens">{agent.max_tokens ?? '—'}</ConfigRow>
        <ConfigRow label="Tools">{listSummary(agent.tools)}</ConfigRow>
        <ConfigRow label="Skills">{listSummary(agent.skills)}</ConfigRow>
        <ConfigRow label="Repo">{agent.repo_id || 'None'}</ConfigRow>
        <ConfigRow label="Suite">{agent.suite || 'None'}</ConfigRow>
        <ConfigRow label="Loop">{loopSummary(agent)}</ConfigRow>
        <ConfigRow label="Persona">{truncateText(agent.persona, 160) || 'None'}</ConfigRow>
      </dl>

      <div className="mt-4 text-[9px] font-semibold tracking-[0.16em] uppercase text-white/35">Last result</div>
      <p title={agent.last_result || undefined} className="mt-1 break-words text-xs leading-relaxed text-white/60">{lastResult || 'No result yet'}</p>
      {lastError && (
        <>
          <div className="mt-3 text-[9px] font-semibold tracking-[0.16em] uppercase text-red-300/70">Last error</div>
          <p title={agent.last_error} className="mt-1 break-words text-xs leading-relaxed text-red-300">{lastError}</p>
        </>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <button type="button" onClick={() => onAssignAgent(agent.id)} className="bg-cyan-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950">Assign work</button>
        <button type="button" onClick={() => onManageAgent(agent.id)} className="border border-white/15 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-white/70 hover:border-white/40">Edit agent</button>
      </div>
    </aside>
  )
}

function ConfigRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-white/35">{label}</dt>
      <dd className="min-w-0 break-words text-white/75">{children}</dd>
    </>
  )
}
