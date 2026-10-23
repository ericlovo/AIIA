export type Budget = Record<'tokens' | 'calls' | 'seconds', { used: number | null; limit: number | null }>
export interface BDBot {
  id: string; name: string; segment: string; tenant_id: string; budgets: Budget
  counts: Record<'queued' | 'running' | 'completed' | 'failed', number>
}
export interface BDJob {
  id: string; bot_id: string; tenant_id: string; segment: string
  state: 'queued' | 'leased' | 'running' | 'completed' | 'failed' | 'cancelled'
  step: string; created_at: string; started_at: string | null; heartbeat_at: string | null
}
export interface BDEvent { id: string | number; kind: string; message: string; created_at: string; bot_id: string; job_id: string }
export interface BDPacket {
  id: string; index: number; contact_policy?: string; disposition?: string; source_quote?: string; job_id: string; bot_id: string; tenant_id: string; segment: string; created_at: string
  mode: string; review_status: string; organization: string; source_url: string
  emails: string[]; phones: string[]; signals: string[]; insights: string[]; summary: string
}
export interface BDSnapshot {
  mode: string; schedule_enabled: boolean; kill_switch: boolean; bots: BDBot[]; jobs: BDJob[]
  events: BDEvent[]; packets: BDPacket[]; budget: Budget
  health: Record<'qwen' | 'jev' | 'managed_leads', { status: 'available' | 'data_unavailable' | 'replay'; detail: string }>
  jev: { calls_per_minute: number | null; verdict_mix: Record<'Choice' | 'Score' | 'Noul', number> | null; latency_ms: number | null; errors: number | null }
}
export function ageLabel(at: string | null, now: number): string {
  if (!at) return 'unavailable'
  const elapsed = now - Date.parse(at)
  if (!Number.isFinite(elapsed)) return 'unavailable'
  const seconds = Math.max(0, Math.floor(elapsed / 1000))
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`
}
export function publicSource(url: string): string | null {
  try {
    const parsed = new URL(url)
    return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : null
  } catch { return null }
}
export function heartbeatFresh(at: string | null, now: number): boolean {
  if (!at) return false
  const age = now - Date.parse(at)
  return Number.isFinite(age) && age >= 0 && age < 30_000
}
