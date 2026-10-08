import type { Agent, AgentValue, OutputChannel } from '../lib/api'

export const VALUE_WINDOW_DAYS = 14

export function emptyValue(agent?: Pick<Agent, 'last_run_at'>): AgentValue {
  return { window_days: VALUE_WINDOW_DAYS, runs: 0, last_run_at: agent?.last_run_at ?? null, reviewed: 0, unreviewed: 0 }
}

export function agentValue(agent: Agent): AgentValue {
  return agent.value ?? emptyValue(agent)
}

export function channelLabel(channel: OutputChannel | string | undefined): string {
  if (channel === 'slack') return 'Slack'
  if (channel === 'memory_inbox') return 'Memory inbox'
  return 'Studio work'
}

export function hasSlackGap(agent: Pick<Agent, 'output_channel' | 'output_channel_note'>): boolean {
  return agent.output_channel === 'slack' && Boolean(agent.output_channel_note)
}

export function noReviewedOutput(agent: Agent): boolean {
  return agentValue(agent).reviewed === 0
}

export function valueSummary(agent: Agent): string {
  const value = agentValue(agent)
  const last = value.last_run_at ? new Date(value.last_run_at).toLocaleString() : 'never'
  return `${value.runs} runs / ${value.window_days}d · ${value.reviewed} reviewed · ${value.unreviewed} waiting · last ${last}`
}

export function valueGlance(agent: Agent): string {
  const value = agentValue(agent)
  if (value.reviewed === 0) return `0 reviewed / ${value.window_days}d`
  return `${value.reviewed} reviewed · ${value.unreviewed} waiting`
}

export type ValueSort = 'name' | 'runs' | 'reviewed' | 'last_run'

export function sortAgents(agents: Agent[], sort: ValueSort): Agent[] {
  const copy = [...agents]
  copy.sort((a, b) => {
    const av = agentValue(a)
    const bv = agentValue(b)
    if (sort === 'runs') return bv.runs - av.runs || a.name.localeCompare(b.name)
    if (sort === 'reviewed') return av.reviewed - bv.reviewed || bv.unreviewed - av.unreviewed || a.name.localeCompare(b.name)
    if (sort === 'last_run') return (bv.last_run_at || '').localeCompare(av.last_run_at || '') || a.name.localeCompare(b.name)
    return a.name.localeCompare(b.name)
  })
  return copy
}

export function filterAgents(agents: Agent[], query: string, onlyQuiet: boolean): Agent[] {
  const needle = query.trim().toLowerCase()
  return agents.filter(agent => {
    if (onlyQuiet && !noReviewedOutput(agent)) return false
    if (!needle) return true
    return `${agent.name} ${agent.one_liner || ''} ${agent.use_when || ''} ${agent.mission} ${agent.repo_id} ${agent.kind || ''} ${(agent.handles ?? []).join(' ')} ${agent.suite || ''}`.toLowerCase().includes(needle)
  })
}
