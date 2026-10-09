import type { Agent } from '../lib/api'
import { activeAgents, resolveKind, resolveUseWhen } from './agentRoster.ts'

export type MorningState = 'blocked' | 'waiting' | 'shipped' | 'clean' | 'unmapped' | 'unmounted'

export interface MorningLine {
  id: string
  name: string
  kind: 'product' | 'customer' | string
  state: MorningState | string
  note: string
  waiting?: string
  extra?: string
  target?: string
}

export interface MorningDecision {
  id: string
  kind: 'inbox' | 'review' | string
  product: string
  title: string
  why: string
  more: string
  expected_version?: string
}

export interface MorningNotePayload {
  date: string
  products: MorningLine[]
  customers: MorningLine[]
  decisions: MorningDecision[]
  needs_you_total: number
  failure: string | null
}

export const STATE_LABEL: Record<string, string> = {
  shipped: 'Shipped',
  clean: 'Shipped',
  blocked: 'Blocked',
  waiting: 'Waiting on you',
  countdown: 'Coming up',
  unmapped: 'Not mapped',
  unmounted: 'Not mapped',
}

const COUNT_WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

export const FORBIDDEN_HOME_WORDS = /\b(runs?|loops?|inbox|sources?)\b/i

export function greeting(d: Date): string {
  const hour = d.getHours()
  if (hour < 5) return 'Good evening'
  if (hour < 12) return 'Good morning'
  if (hour < 17) return 'Good afternoon'
  return 'Good evening'
}

export function formatLongDate(d: Date): string {
  return d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
}

export function formatClock(d: Date): string {
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}

export function countdown(targetIso: string, now: Date): { text: string; calDays: number; ms: number } {
  const target = new Date(targetIso)
  const ms = target.getTime() - now.getTime()
  const dayStart = (value: Date) => new Date(value.getFullYear(), value.getMonth(), value.getDate())
  const calDays = Math.round((dayStart(target).getTime() - dayStart(now).getTime()) / 86_400_000)
  if (ms <= 0) return { text: calDays === 0 ? 'Kicking off today' : 'Phase 1 is underway', calDays, ms }
  const days = Math.floor(ms / 86_400_000)
  const hours = Math.floor((ms % 86_400_000) / 3_600_000)
  const mins = Math.floor((ms % 3_600_000) / 60_000)
  if (calDays === 0) return { text: `Today, in ${hours ? `${hours} hr ` : ''}${mins} min`, calDays, ms }
  if (calDays === 1) return { text: `Tomorrow · ${days ? `${days} day, ` : ''}${hours} hr to go`, calDays, ms }
  return { text: `${days} days, ${hours} hr to go`, calDays, ms }
}

export function kickoffPhrase(targetIso: string, now: Date): string | null {
  const { calDays, ms } = countdown(targetIso, now)
  if (ms <= 0) return null
  if (calDays === 0) return 'today'
  if (calDays === 1) return 'tomorrow'
  const target = new Date(targetIso)
  return target.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })
}

function names(list: MorningLine[]): string {
  return list.map(item => item.name).join(' and ')
}

export function morningLede(products: MorningLine[], customers: MorningLine[], needCount: number, now = new Date()): string {
  const blocked = products.filter(item => item.state === 'blocked')
  const well = products.filter(item => item.state === 'shipped' || item.state === 'clean')
  const parts: string[] = []
  if (well.length) parts.push(`${names(well)} ${well.length === 1 ? 'is' : 'are'} in good shape.`)
  if (blocked.length) parts.push(`${names(blocked)} ${blocked.length === 1 ? 'is' : 'are'} stuck, and someone is already on it.`)
  const word = COUNT_WORDS[needCount] ?? String(needCount)
  parts.push(needCount ? `${word} ${needCount === 1 ? 'thing needs' : 'things need'} a quick answer from you.` : 'Nothing needs you right now.')
  const alumni = customers.find(item => item.id === 'alumni-nations' && item.target)
  const kick = alumni?.target ? kickoffPhrase(alumni.target, now) : null
  if (kick) parts.push(`Alumni Nations kicks off ${kick}.`)
  return parts.join(' ')
}

function escapeRe(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function wordMatch(text: string, needle: string): boolean {
  const token = needle.trim()
  if (!token) return false
  return new RegExp(`\\b${escapeRe(token)}\\b`, 'i').test(text)
}

export function pickAgentForAsk(text: string, agents: Agent[]): { agent: Agent; why: string } | null {
  const live = activeAgents(agents)
  if (!live.length) return null
  const hay = text.trim()
  if (!hay) return null

  for (const agent of live) {
    for (const handle of agent.handles ?? []) {
      if (wordMatch(hay, handle)) return { agent, why: `it mentions ${handle}` }
    }
  }

  if (/\b(ci|checks?|red|failing|fails?|failed|failure|broken|tests?|errors?|crash\w*|flaky)\b/i.test(hay)) {
    const coding = live.find(agent => resolveKind(agent) === 'coding' && (agent.handles ?? []).some(tag => /ci|fail|test/i.test(tag)))
      || live.find(agent => resolveKind(agent) === 'coding')
    if (coding) return { agent: coding, why: 'it sounds like something is failing' }
  }

  if (/\b(review|pr|pull request|look over|second pair|diff)\b/i.test(hay)) {
    const reviewer = live.find(agent => (agent.handles ?? []).some(tag => /review/i.test(tag)) || /review/i.test(resolveUseWhen(agent)))
    if (reviewer) return { agent: reviewer, why: 'it asks for a review' }
  }

  const stop = new Set(['when', 'something', 'should', 'about', 'needed', 'this', 'that', 'with', 'from', 'need', 'asks'])
  for (const agent of live) {
    const when = resolveUseWhen(agent)
    const tokens = when.toLowerCase().split(/[^a-z0-9]+/).filter(token => token.length > 4 && !stop.has(token))
    if (tokens.some(token => wordMatch(hay, token))) return { agent, why: when || 'the roster said to pick this one' }
  }

  for (const agent of live) {
    if (wordMatch(hay, agent.name) || (agent.suite && wordMatch(hay, agent.suite))) {
      return { agent, why: `it's about ${agent.name}` }
    }
  }

  const product = live.find(agent => resolveKind(agent) === 'product')
  return { agent: product ?? live[0], why: 'the roster picked this one' }
}

export function assignmentTitle(text: string): string {
  const line = text.trim().replace(/\s+/g, ' ')
  return line.length <= 120 ? line : `${line.slice(0, 119).trim()}…`
}

export function homeCopyHasForbiddenWords(text: string): boolean {
  return FORBIDDEN_HOME_WORDS.test(text)
}
