import { useRef, useState } from 'react'
import { SuitePatchRejected, type Agent, type AgentSuitePatch } from '../lib/api'
import type { SuiteGroup } from './graphLayout'
import { EMPTY_SUITE_FORM, buildSuitePatch, describeSuiteSettings, suiteDetailText, type SuiteForm } from './suiteModulation'

interface SuitePanelProps {
  group: SuiteGroup
  members: Agent[]
  onApply: (patch: AgentSuitePatch) => Promise<{ count: number }>
  onClose: () => void
}

interface Rejection {
  agentName: string
  text: string
}

const inputClass = 'w-full border border-white/15 bg-black/40 px-2 py-1.5 text-xs text-white outline-none placeholder:text-white/25 focus:border-cyan-300/70'

export function SuitePanel({ group, members, onApply, onClose }: SuitePanelProps) {
  const [form, setForm] = useState<SuiteForm>(EMPTY_SUITE_FORM)
  const [pending, setPending] = useState(false)
  // State lands on the next render; a second submit in the same task must see the first.
  const inFlight = useRef(false)
  const [error, setError] = useState('')
  const [rejections, setRejections] = useState<Rejection[]>([])
  const [notice, setNotice] = useState('')
  const { patch, errors } = buildSuitePatch(form)
  const current = describeSuiteSettings(members)
  const running = members.some(member => member.status === 'running')
  const touched = JSON.stringify(form) !== JSON.stringify(EMPTY_SUITE_FORM)

  function update(next: Partial<SuiteForm>) {
    setForm(previous => ({ ...previous, ...next }))
    setNotice('')
  }

  async function apply(event: React.FormEvent) {
    event.preventDefault()
    if (errors.length || inFlight.current) return
    inFlight.current = true
    setPending(true)
    setError('')
    setRejections([])
    setNotice('')
    try {
      const result = await onApply(patch)
      setForm(EMPTY_SUITE_FORM)
      setNotice(`Updated ${result.count} ${result.count === 1 ? 'agent' : 'agents'} in ${group.slug}.`)
    } catch (failure) {
      if (failure instanceof SuitePatchRejected) {
        setError('Nothing changed. Every member must accept the update.')
        setRejections(failure.failures.map(item => ({
          agentName: members.find(member => member.id === item.agent_id)?.name ?? item.agent_id,
          text: suiteDetailText(item.detail),
        })))
      } else {
        setError(suiteDetailText(failure instanceof Error ? failure.message : ''))
      }
    } finally {
      inFlight.current = false
      setPending(false)
    }
  }

  return (
    <aside aria-label={`Tune ${group.slug} suite`} onKeyDown={event => { if (event.key === 'Escape') onClose() }} className="pointer-events-auto absolute left-4 top-4 z-30 max-h-[calc(100%-2rem)] w-[min(320px,calc(100%-2rem))] overflow-y-auto border border-white/15 bg-[#090c10] p-4 text-left shadow-2xl sm:left-6">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-[9px] font-semibold tracking-[0.18em] uppercase" style={{ color: group.color }}>
            <i aria-hidden="true" className="h-2 w-2 rounded-full" style={{ background: group.color }} />Suite controls
          </div>
          <div className="mt-1 break-words text-sm font-medium text-white">{group.slug}</div>
          <p className="mt-1 break-words text-xs text-white/45">{members.map(member => member.name).join(', ')}</p>
        </div>
        <button type="button" onClick={onClose} className="text-lg leading-none text-white/30 hover:text-white" aria-label="Close suite controls">×</button>
      </div>
      <p className="mt-2 text-[11px] text-white/40">Blank fields keep each agent's own value. One request updates all {members.length} {members.length === 1 ? 'member' : 'members'}, or none.{running ? ' Running agents use the change on their next run.' : ''}</p>
      <form onSubmit={apply} className="mt-3 space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <label className="block text-[10px] text-white/50">Model
            <select aria-label="Model" value={form.model} onChange={event => update({ model: event.target.value as SuiteForm['model'] })} className={`${inputClass} mt-1`}>
              <option value="keep">Keep current</option>
              <option value="default">Task default</option>
              <option value="custom">Specific model</option>
            </select>
          </label>
          {form.model === 'custom' && (
            <label className="block text-[10px] text-white/50">Model id
              <input aria-label="Model id" value={form.modelId} onChange={event => update({ modelId: event.target.value })} placeholder="qwen3:8b" className={`${inputClass} mt-1`} />
            </label>
          )}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="block text-[10px] text-white/50">Temperature
            <input aria-label="Temperature" type="number" min="0" max="1" step="0.05" inputMode="decimal" value={form.temperature} onChange={event => update({ temperature: event.target.value })} placeholder={current.temperature} className={`${inputClass} mt-1`} />
          </label>
          <label className="block text-[10px] text-white/50">Max tokens
            <input aria-label="Max tokens" type="number" min="128" max="2000" step="1" inputMode="numeric" value={form.maxTokens} onChange={event => update({ maxTokens: event.target.value })} placeholder={current.maxTokens} className={`${inputClass} mt-1`} />
          </label>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <label className="block text-[10px] text-white/50">Loop
            <select aria-label="Loop" value={form.loop} onChange={event => update({ loop: event.target.value as SuiteForm['loop'] })} className={`${inputClass} mt-1`}>
              <option value="keep">Keep</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label className="block text-[10px] text-white/50">Every min
            <input aria-label="Loop interval minutes" type="number" min="15" max="1440" inputMode="numeric" value={form.loopInterval} onChange={event => update({ loopInterval: event.target.value })} placeholder={current.loopInterval} className={`${inputClass} mt-1`} />
          </label>
          <label className="block text-[10px] text-white/50">Runs / day
            <input aria-label="Loop runs per day" type="number" min="1" max="48" inputMode="numeric" value={form.loopMaxRuns} onChange={event => update({ loopMaxRuns: event.target.value })} placeholder={current.loopMaxRuns} className={`${inputClass} mt-1`} />
          </label>
        </div>
        <p data-suite-summary className="text-[10px] text-white/35">Now: temperature {current.temperature} · max tokens {current.maxTokens} · loop {current.loop}</p>
        {touched && errors.map(message => <p key={message} className="text-[11px] text-amber-200/80">{message}</p>)}
        {error && (
          <div role="alert" className="border border-red-900/60 bg-red-950/40 px-3 py-2 text-xs text-red-300">
            <p>{error}</p>
            {rejections.length > 0 && (
              <ul className="mt-2 space-y-1">
                {rejections.map(item => <li key={`${item.agentName}:${item.text}`} className="break-words"><b className="font-medium text-red-100">{item.agentName}</b>: {item.text}</li>)}
              </ul>
            )}
          </div>
        )}
        {notice && <p role="status" className="text-xs text-emerald-300">{notice}</p>}
        <button type="submit" disabled={errors.length > 0 || pending} className="bg-cyan-300 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-950 disabled:cursor-not-allowed disabled:opacity-50">{pending ? 'Applying' : `Apply to ${members.length} ${members.length === 1 ? 'agent' : 'agents'}`}</button>
      </form>
    </aside>
  )
}
