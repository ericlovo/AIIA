import type { AgentSuitePatch } from '../lib/api'

export interface SuiteForm {
  model: 'keep' | 'default' | 'custom'
  modelId: string
  temperature: string
  maxTokens: string
  loop: 'keep' | 'on' | 'off'
  loopInterval: string
  loopMaxRuns: string
}

export const EMPTY_SUITE_FORM: SuiteForm = {
  model: 'keep',
  modelId: '',
  temperature: '',
  maxTokens: '',
  loop: 'keep',
  loopInterval: '',
  loopMaxRuns: '',
}

interface SuiteMemberSettings {
  temperature: number
  max_tokens: number
  loop_enabled: boolean
  loop_interval_minutes: number
  loop_max_runs_per_day: number
}

function numberInRange(raw: string, label: string, minimum: number, maximum: number, integer: boolean, errors: string[]) {
  const text = raw.trim()
  if (!text) return undefined
  const value = Number(text)
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value)) || value < minimum || value > maximum) {
    errors.push(`${label} must be ${integer ? 'a whole number ' : ''}from ${minimum} to ${maximum}.`)
    return undefined
  }
  return value
}

// Blank fields mean "leave each agent as it is", so the request carries only what
// the operator chose to change. Limits mirror AgentCreateRequest.
export function buildSuitePatch(form: SuiteForm): { patch: AgentSuitePatch; errors: string[] } {
  const errors: string[] = []
  const patch: AgentSuitePatch = {}
  if (form.model === 'default') patch.model = ''
  if (form.model === 'custom') {
    const id = form.modelId.trim()
    if (!id) errors.push('Enter a model id or choose task default.')
    else patch.model = id
  }
  const temperature = numberInRange(form.temperature, 'Temperature', 0, 1, false, errors)
  if (temperature !== undefined) patch.temperature = temperature
  const maxTokens = numberInRange(form.maxTokens, 'Max tokens', 128, 2000, true, errors)
  if (maxTokens !== undefined) patch.max_tokens = maxTokens
  if (form.loop !== 'keep') patch.loop_enabled = form.loop === 'on'
  const interval = numberInRange(form.loopInterval, 'Loop interval', 15, 1440, true, errors)
  if (interval !== undefined) patch.loop_interval_minutes = interval
  const maxRuns = numberInRange(form.loopMaxRuns, 'Runs per day', 1, 48, true, errors)
  if (maxRuns !== undefined) patch.loop_max_runs_per_day = maxRuns
  if (!errors.length && Object.keys(patch).length === 0) errors.push('Choose at least one setting to change.')
  return { patch, errors }
}

const DETAILS: Record<string, string> = {
  loop_task_required: 'Needs a loop task before its loop can be enabled.',
  unknown_model: 'That model is not installed on the Mini.',
  models_unavailable: 'The model list is unavailable because Ollama is unreachable.',
  suite_not_found: 'No agent is tagged with this suite any more.',
  agent_not_found: 'This agent no longer exists.',
  empty_patch: 'Choose at least one setting to change.',
}

export function suiteDetailText(detail: string) {
  if (DETAILS[detail]) return DETAILS[detail]
  // FastAPI field validation arrives as a list, which Error stringifies to [object Object].
  if (!detail || detail.startsWith('[object')) return 'The suite update was refused.'
  return detail
}

function span(values: number[]) {
  if (!values.length) return '—'
  const low = Math.min(...values)
  const high = Math.max(...values)
  return low === high ? String(low) : `${low}–${high}`
}

export function describeSuiteSettings(members: SuiteMemberSettings[]) {
  const looping = members.filter(member => member.loop_enabled).length
  return {
    temperature: span(members.map(member => member.temperature)),
    maxTokens: span(members.map(member => member.max_tokens)),
    loop: looping === 0 ? 'off' : looping === members.length ? 'on' : `on for ${looping} of ${members.length}`,
    loopInterval: span(members.map(member => member.loop_interval_minutes)),
    loopMaxRuns: span(members.map(member => member.loop_max_runs_per_day)),
  }
}

export function withUpdatedAgents<T extends { id: string }>(agents: T[], updated: T[]) {
  const byId = new Map(updated.map(agent => [agent.id, agent]))
  return agents.map(agent => byId.get(agent.id) ?? agent)
}
