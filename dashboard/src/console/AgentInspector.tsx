import { useId, useRef, useState, type ReactNode } from 'react'
import { useMutation, useMutationState, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type Agent } from '../lib/api'
import {
  NUMBER_LIMITS,
  RUN_TASK_MAX_LENGTH,
  SUITE_MAX_LENGTH,
  listSummary,
  loopSummary,
  modelChoices,
  parseBoundedNumber,
  patchFailureMessage,
  beginPatch,
  isNewestEdit,
  pendingValues,
  readableError,
  runFailureMessage,
  runSuccessMessage,
  settlePatch,
  truncateText,
  withAgentFields,
  withAgentRecord,
  type AgentPatch,
  type PatchLedger,
} from './agentConfig'

interface AgentInspectorProps {
  agent: Agent
  onClose: () => void
  onManageAgent: (agentId: string) => void
  onAssignAgent: (agentId: string) => void
}

type AgentsData = { agents: Agent[] }

const CONTROL = 'w-full min-w-0 border border-white/15 bg-[#0b0e12] px-2 py-1.5 text-[11px] text-white/85 focus:border-cyan-300 focus:outline-none aria-[invalid=true]:border-red-400'

export function AgentInspector({ agent, onClose, onManageAgent, onAssignAgent }: AgentInspectorProps) {
  const queryClient = useQueryClient()
  const models = useQuery({ queryKey: ['agent-models'], queryFn: api.agentModels, staleTime: 60_000, retry: false })
  // In-flight values win over the cache so a background refetch cannot flicker an edit back.
  const ledger = useRef<PatchLedger>({})
  const seq = useRef(0)
  const [pending, setPending] = useState<AgentPatch>({})
  const [failure, setFailure] = useState('')
  const view = { ...agent, ...pending }

  function settle(fields: AgentPatch, id: number, record: Agent | null): AgentPatch {
    const result = settlePatch(ledger.current, fields, id, record)
    ledger.current = result.ledger
    setPending(pendingValues(result.ledger))
    if (Object.keys(result.settled).length > 0) queryClient.setQueryData<AgentsData>(['agents'], data => withAgentFields(data, agent.id, result.settled))
    // The record may be older than an edit still in flight, so fetch the whole agent once all are settled.
    if (Object.keys(result.ledger).length === 0) void queryClient.invalidateQueries({ queryKey: ['agents'] })
    return result.settled
  }

  const patch = useMutation({
    mutationFn: ({ fields }: { fields: AgentPatch; id: number }) => api.patchAgent(agent.id, fields),
    onMutate: async ({ fields, id }) => {
      setFailure('')
      const cached = queryClient.getQueryData<AgentsData>(['agents'])?.agents.find(item => item.id === agent.id)
      ledger.current = beginPatch(ledger.current, cached ?? agent, fields, id)
      setPending(pendingValues(ledger.current))
      await queryClient.cancelQueries({ queryKey: ['agents'] })
      queryClient.setQueryData<AgentsData>(['agents'], data => withAgentFields(data, agent.id, fields))
    },
    onSuccess: ({ agent: record }, { fields, id }) => {
      settle(fields, id, record)
    },
    onError: (error, { fields, id }) => {
      // A failure for an edit that a newer edit of the same field replaced is moot, so it says nothing.
      const newest = isNewestEdit(ledger.current, fields, id)
      settle(fields, id, null)
      if (newest) setFailure(patchFailureMessage(fields, error.message))
    },
  })

  function change(fields: AgentPatch) {
    seq.current += 1
    patch.mutate({ fields, id: seq.current })
  }

  const [task, setTask] = useState('')
  const runKey = ['agent-run', agent.id]
  const run = useMutation({
    mutationKey: runKey,
    mutationFn: (runTask: string) => api.runAgent(agent.id, runTask),
    onSuccess: ({ agent: record }) => {
      queryClient.setQueryData<AgentsData>(['agents'], data => withAgentRecord(data, record))
      setTask('')
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['agents'] })
      void queryClient.invalidateQueries({ queryKey: ['studio-activity'] })
    },
  })
  // Read from the mutation cache, not component state, so closing and reopening the
  // inspector during a long run keeps the control disabled and still shows the outcome.
  const latestRun = useMutationState({ filters: { mutationKey: runKey }, select: mutation => mutation.state }).at(-1)
  const running = run.isPending || latestRun?.status === 'pending'
  const runData = latestRun?.data as Awaited<ReturnType<typeof api.runAgent>> | undefined
  const runNotice = latestRun?.status === 'success'
    ? { tone: 'ok', text: runSuccessMessage(runData?.model, runData?.latency_ms) }
    : latestRun?.status === 'error'
      ? { tone: 'error', text: runFailureMessage(latestRun.error?.message ?? '') }
      : null
  const taskId = useId()

  const lastResult = truncateText(agent.last_result)
  const lastError = truncateText(agent.last_error)
  const choices = modelChoices(view.model, models.data)
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

      <section aria-label="Agent settings" className="mt-4">
        {agent.status === 'running' && <p role="status" className="mb-3 border border-amber-300/30 bg-amber-950/30 px-2.5 py-1.5 text-[11px] text-amber-200">Running now. Changes apply to the next run.</p>}
        {failure && <p role="alert" className="mb-3 border border-red-900/60 bg-red-950/40 px-2.5 py-1.5 text-[11px] text-red-300">{failure}</p>}
        <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-2 text-[11px]">
          <Field label="Model">
            {id => (
              <select id={id} value={view.model ?? ''} onChange={event => change({ model: event.target.value })} className={CONTROL}>
                {choices.map(choice => <option key={choice.id} value={choice.id}>{choice.label}</option>)}
              </select>
            )}
          </Field>
          {models.isError && <p className="col-span-2 -mt-1 text-[10px] text-amber-200/80">Model list unavailable: {readableError(models.error.message)}</p>}
          <Field label="Temperature">
            {id => <NumberInput id={id} label="Temperature" value={view.temperature} step={0.05} limits={NUMBER_LIMITS.temperature} onCommit={value => change({ temperature: value })} />}
          </Field>
          <Field label="Max tokens">
            {id => <NumberInput id={id} label="Max tokens" value={view.max_tokens} step={50} limits={NUMBER_LIMITS.max_tokens} onCommit={value => change({ max_tokens: value })} />}
          </Field>
          <Field label="Suite">
            {id => <TextInput id={id} value={view.suite ?? ''} maxLength={SUITE_MAX_LENGTH} placeholder="None" onCommit={value => change({ suite: value })} />}
          </Field>
          <Field label="Loop">
            {id => (
              <div className="flex items-center gap-2 text-white/75">
                <input id={id} type="checkbox" checked={Boolean(view.loop_enabled)} aria-describedby={`${id}-summary`} onChange={event => change({ loop_enabled: event.target.checked })} className="accent-cyan-300" />
                <span id={`${id}-summary`}>{loopSummary(view)}</span>
              </div>
            )}
          </Field>
          <Field label="Loop interval (min)">
            {id => <NumberInput id={id} label="Loop interval" value={view.loop_interval_minutes} step={15} limits={NUMBER_LIMITS.loop_interval_minutes} onCommit={value => change({ loop_interval_minutes: value })} />}
          </Field>
          <Field label="Loop runs per day">
            {id => <NumberInput id={id} label="Loop daily maximum" value={view.loop_max_runs_per_day} step={1} limits={NUMBER_LIMITS.loop_max_runs_per_day} onCommit={value => change({ loop_max_runs_per_day: value })} />}
          </Field>
        </div>
      </section>

      <form
        aria-label="Run agent"
        className="mt-4 border-t border-white/10 pt-3"
        onSubmit={event => {
          event.preventDefault()
          if (running || !task.trim()) return
          run.mutate(task.trim())
        }}
      >
        <label htmlFor={taskId} className="text-[9px] font-semibold tracking-[0.16em] uppercase text-white/35">Task for this run</label>
        <textarea
          id={taskId}
          rows={2}
          value={task}
          maxLength={RUN_TASK_MAX_LENGTH}
          disabled={running}
          placeholder="What should this agent do now?"
          onChange={event => setTask(event.target.value)}
          className={`${CONTROL} mt-1 resize-y disabled:opacity-50`}
        />
        {runNotice && <p role={runNotice.tone === 'error' ? 'alert' : 'status'} className={`mt-2 text-[11px] ${runNotice.tone === 'error' ? 'text-red-300' : 'text-emerald-300'}`}>{runNotice.text}</p>}
        <button type="submit" disabled={running || !task.trim()} className="mt-2 bg-amber-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950 disabled:cursor-not-allowed disabled:opacity-50">{running ? 'Mini working' : 'Run now'}</button>
      </form>

      <dl aria-label="Agent configuration" className="mt-4 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[11px]">
        <ConfigRow label="Tools">{listSummary(agent.tools)}</ConfigRow>
        <ConfigRow label="Skills">{listSummary(agent.skills)}</ConfigRow>
        <ConfigRow label="Repo">{agent.repo_id || 'None'}</ConfigRow>
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

function Field({ label, children }: { label: string; children: (id: string) => ReactNode }) {
  const id = useId()
  return (
    <>
      <label htmlFor={id} className="text-white/35">{label}</label>
      <div className="min-w-0">{children(id)}</div>
    </>
  )
}

interface NumberInputProps {
  id: string
  label: string
  value: number | undefined
  step: number
  limits: { min: number; max: number; integer: boolean }
  onCommit: (value: number) => void
}

// Edits stay local until Enter or blur, so arrow keys and typing send one PATCH, not one per keystroke.
function NumberInput({ id, label, value, step, limits, onCommit }: NumberInputProps) {
  const [draft, setDraft] = useState<string | null>(null)
  const [error, setError] = useState('')
  const errorId = `${id}-error`

  function commit() {
    if (draft === null) return
    const parsed = parseBoundedNumber(draft, limits)
    if ('error' in parsed) {
      setError(`${label} ${parsed.error}.`)
      return
    }
    setDraft(null)
    setError('')
    if (parsed.value !== value) onCommit(parsed.value)
  }

  return (
    <>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={limits.min}
        max={limits.max}
        step={step}
        value={draft ?? (value ?? '')}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? errorId : undefined}
        onChange={event => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={event => {
          if (event.key === 'Enter') { event.preventDefault(); commit() }
          if (event.key === 'Escape') { setDraft(null); setError('') }
        }}
        className={CONTROL}
      />
      {error && <p id={errorId} className="mt-1 text-[10px] text-red-300">{error}</p>}
    </>
  )
}

function TextInput({ id, value, maxLength, placeholder, onCommit }: { id: string; value: string; maxLength: number; placeholder: string; onCommit: (value: string) => void }) {
  const [draft, setDraft] = useState<string | null>(null)

  function commit() {
    if (draft === null) return
    const next = draft.trim()
    setDraft(null)
    if (next !== value) onCommit(next)
  }

  return (
    <input
      id={id}
      type="text"
      value={draft ?? value}
      maxLength={maxLength}
      placeholder={placeholder}
      onChange={event => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={event => {
        if (event.key === 'Enter') { event.preventDefault(); commit() }
        if (event.key === 'Escape') setDraft(null)
      }}
      className={CONTROL}
    />
  )
}
