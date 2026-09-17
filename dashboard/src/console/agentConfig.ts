import type { Agent, AgentDefinition, AgentModel } from '../lib/api'

export const RESULT_PREVIEW_CHARS = 240

/** Collapses whitespace and cuts long text so the inspector stays a glance, not a log. */
export function truncateText(text: string | null | undefined, max = RESULT_PREVIEW_CHARS): string {
  const clean = (text ?? '').replace(/\s+/g, ' ').trim()
  if (clean.length <= max) return clean
  return `${clean.slice(0, max - 1).trimEnd()}…`
}

/** An empty model means the Brain picks its task-role default at run time. */
export function modelSummary(model: string | undefined, defaultModel: string | undefined): string {
  if (model) return model
  return defaultModel ? `Task default: ${defaultModel}` : 'Task default'
}

/** Loop state, with the interval and today's usage of the daily cap when the loop is on. */
export function loopSummary(agent: Pick<Agent, 'loop_enabled' | 'loop_interval_minutes' | 'loop_runs_today' | 'loop_max_runs_per_day'>): string {
  if (!agent.loop_enabled) return 'Off'
  const interval = agent.loop_interval_minutes ?? 0
  const every = interval && interval % 60 === 0 ? `${interval / 60}h` : `${interval}m`
  return `On · every ${every} · ${agent.loop_runs_today ?? 0} of ${agent.loop_max_runs_per_day ?? 0} runs today`
}

export function listSummary(items: string[] | undefined): string {
  return items?.length ? items.join(', ') : 'None'
}

export type AgentPatch = Partial<AgentDefinition>
export type EditableField = 'model' | 'temperature' | 'max_tokens' | 'suite' | 'loop_enabled' | 'loop_interval_minutes' | 'loop_max_runs_per_day'

export const FIELD_LABELS: Record<EditableField, string> = {
  model: 'Model',
  temperature: 'Temperature',
  max_tokens: 'Max tokens',
  suite: 'Suite',
  loop_enabled: 'Loop',
  loop_interval_minutes: 'Loop interval',
  loop_max_runs_per_day: 'Loop daily maximum',
}

/** Mirrors AgentCreateRequest so an out-of-range value is caught before a round trip. */
export const NUMBER_LIMITS = {
  temperature: { min: 0, max: 1, integer: false },
  max_tokens: { min: 128, max: 2_000, integer: true },
  loop_interval_minutes: { min: 15, max: 1_440, integer: true },
  loop_max_runs_per_day: { min: 1, max: 48, integer: true },
} as const

export const SUITE_MAX_LENGTH = 64

export function parseBoundedNumber(raw: string, limits: { min: number; max: number; integer: boolean }): { value: number } | { error: string } {
  const text = raw.trim()
  const value = Number(text)
  if (!text || !Number.isFinite(value)) return { error: 'must be a number' }
  if (limits.integer && !Number.isInteger(value)) return { error: 'must be a whole number' }
  if (value < limits.min || value > limits.max) return { error: `must be between ${limits.min} and ${limits.max}` }
  return { value }
}

const PATCH_ERRORS: Record<string, string> = {
  loop_task_required: 'the loop needs a loop task first. Add one in Edit agent, then turn the loop on.',
  unknown_model: 'that model is not installed on the Mini.',
  models_unavailable: 'Ollama is not reachable, so the model cannot be checked. Try again when it is running.',
  agent_not_found: 'this agent no longer exists.',
  empty_patch: 'there was nothing to change.',
}

/** Turns API detail codes into a sentence. FastAPI field errors arrive as an array and stringify badly. */
export function readableError(message: string): string {
  if (PATCH_ERRORS[message]) return PATCH_ERRORS[message]
  if (!message || message.includes('[object Object]') || message.startsWith('422')) return 'the Mini rejected that value.'
  return message
}

export function patchFailureMessage(fields: AgentPatch, message: string): string {
  const field = Object.keys(fields)[0] as EditableField | undefined
  const label = field ? FIELD_LABELS[field] ?? field : 'Change'
  return `${label} not saved: ${readableError(message)}`
}

/** Applies fields to one agent in the ['agents'] query data without touching the others. */
export function withAgentFields<T extends { agents: Agent[] }>(data: T | undefined, agentId: string, fields: AgentPatch): T | undefined {
  if (!data) return data
  return { ...data, agents: data.agents.map(agent => agent.id === agentId ? { ...agent, ...fields } : agent) }
}

export function withAgentRecord<T extends { agents: Agent[] }>(data: T | undefined, record: Agent): T | undefined {
  if (!data) return data
  return { ...data, agents: data.agents.map(agent => agent.id === record.id ? record : agent) }
}

/** The values a patch overwrote, so a failure restores exactly those fields. */
export function previousFields(agent: Agent | undefined, fields: AgentPatch): AgentPatch {
  if (!agent) return {}
  return Object.fromEntries(Object.keys(fields).map(key => [key, agent[key as keyof AgentPatch]])) as AgentPatch
}

/** Drops settled fields from the pending overlay unless a newer edit to the same field replaced them. */
export function settlePending(pending: AgentPatch, fields: AgentPatch): AgentPatch {
  const next: AgentPatch = { ...pending }
  for (const [key, value] of Object.entries(fields)) {
    if (next[key as keyof AgentPatch] === value) delete next[key as keyof AgentPatch]
  }
  return next
}

export interface ModelChoice {
  id: string
  label: string
}

/** Options for the model picker. A pinned model the list no longer has stays visible and is marked. */
export function modelChoices(current: string | undefined, catalog: { default?: string; models?: AgentModel[] } | undefined): ModelChoice[] {
  const choices: ModelChoice[] = [{ id: '', label: modelSummary('', catalog?.default) }]
  for (const model of catalog?.models ?? []) {
    choices.push({ id: model.id, label: model.parameter_size ? `${model.label} · ${model.parameter_size}` : model.label })
  }
  if (current && !choices.some(choice => choice.id === current)) {
    choices.push({ id: current, label: catalog?.models ? `${current} (not installed)` : current })
  }
  return choices
}
