import type { Agent } from '../lib/api'

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
