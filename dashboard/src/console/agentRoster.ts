import type { Agent, AgentKind } from '../lib/api'

export const KIND_ORDER: Array<AgentKind | ''> = ['coding', 'product', 'ops', '']

export const KIND_LABELS: Record<AgentKind | '', string> = {
  coding: 'Coding',
  product: 'Product',
  ops: 'Ops',
  '': 'Unsorted',
}

const CODING_TOOLS = new Set(['Repository read', 'GitHub read', 'Git workspace'])

export type AgentGroup = { id: AgentKind | ''; label: string; agents: Agent[] }

export function deriveKind(agent: Pick<Agent, 'tools' | 'repo_id'>): AgentKind | '' {
  const tools = new Set(agent.tools ?? [])
  for (const tool of CODING_TOOLS) if (tools.has(tool)) return 'coding'
  return agent.repo_id?.trim() ? 'coding' : ''
}

export function resolveKind(agent: Pick<Agent, 'kind' | 'tools' | 'repo_id'>): AgentKind | '' {
  if (agent.kind === 'coding' || agent.kind === 'product' || agent.kind === 'ops') return agent.kind
  return deriveKind(agent)
}

export function resolveUseWhen(agent: Pick<Agent, 'use_when' | 'one_liner' | 'mission'>): string {
  return (agent.use_when || agent.one_liner || '').trim()
}

export function isRetired(agent: Pick<Agent, 'retired'>): boolean {
  return Boolean(agent.retired)
}

export function activeAgents(agents: Agent[]): Agent[] {
  return agents.filter(agent => !isRetired(agent))
}

export function proposesGit(agent: Pick<Agent, 'tools'>): boolean {
  return (agent.tools ?? []).includes('Git workspace')
}

export function gitStanceLabel(agent: Pick<Agent, 'tools'>): string {
  return proposesGit(agent) ? 'Proposes git (approval)' : 'Read-only'
}

export function productRepoLabel(agent: Pick<Agent, 'repo_id' | 'suite'>): string {
  const product = agent.suite?.trim() ?? ''
  const repo = agent.repo_id?.trim() ?? ''
  if (product && repo && product !== repo) return `${product} · ${repo}`
  return repo || product || 'No repo'
}

export function pausedReviewLabel(agent: Pick<Agent, 'loop_skip_reason' | 'value'>): string | null {
  if (agent.loop_skip_reason !== 'awaiting_review') return null
  const n = agent.value?.unreviewed ?? 0
  return `Paused: ${n} awaiting review`
}

export function pickerOptionLabel(agent: Pick<Agent, 'name' | 'use_when' | 'one_liner' | 'mission'>): string {
  const line = resolveUseWhen(agent)
  if (!line || line === agent.name) return agent.name
  const clipped = line.length > 72 ? `${line.slice(0, 71)}…` : line
  return `${agent.name} — ${clipped}`
}

export function groupAgentsByKind(agents: Agent[]): AgentGroup[] {
  const buckets = new Map<AgentKind | '', Agent[]>()
  for (const id of KIND_ORDER) buckets.set(id, [])
  for (const agent of agents) {
    const kind = resolveKind(agent)
    buckets.get(kind)?.push(agent)
  }
  return KIND_ORDER
    .map(id => ({ id, label: KIND_LABELS[id], agents: buckets.get(id) ?? [] }))
    .filter(group => group.agents.length > 0)
}

export function pickerGroups(agents: Agent[], excludeIds: string[] = []): AgentGroup[] {
  const blocked = new Set(excludeIds)
  const visible = activeAgents(agents)
    .filter(agent => !blocked.has(agent.id))
    .sort((a, b) => a.name.localeCompare(b.name))
  return groupAgentsByKind(visible)
}
