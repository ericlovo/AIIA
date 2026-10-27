export type ResearchOutcome = 'supported' | 'rejected' | 'inconclusive'
export type Budget = Record<'tokens' | 'calls' | 'seconds', { used: number | null; limit: number | null }>
export interface BDBot {
  id: string; name: string; segment: string; tenant_id: string; budgets: Budget
  counts: Record<'queued' | 'running' | 'completed' | 'failed', number>
}
export interface BDJob {
  id: string; bot_id: string; tenant_id: string; segment: string
  state: 'queued' | 'leased' | 'running' | 'completed' | 'failed' | 'cancelled'
  mode?: string; research_outcome?: ResearchOutcome; result_reason?: string
  next_query?: string | null
  finished_at?: string | null; error?: string | null; attempts?: number
  step: string; created_at: string; started_at: string | null; heartbeat_at: string | null
}
export interface BDEvent { provider?: string; model?: string; latency_ms?: number | null; mode?: string; primitive?: string; id: string | number; kind: string; message: string; created_at: string; bot_id: string; job_id: string }
export interface BDPacket {
  candidate_status?: ResearchOutcome; candidate_reason?: string; next_query?: string | null
  id: string; index: number; contact_policy?: string; disposition?: string; source_quote?: string; job_id: string; bot_id: string; tenant_id: string; segment: string; created_at: string
  mode: string; review_status: string; organization: string; source_url: string
  emails: string[]; phones: string[]; signals: string[]; insights: string[]; summary: string
}
export interface ModelActivity {
  status: string; active_jobs: string[]; mode?: string
  calls: number | null; attempts?: number | null; completed: number | null
  replayed: number | null; errors: number | null; last_event_at: string | null; latency_ms: number | null
  metrics_scope?: string
}
export interface BDSchedule {
  state: 'waiting' | 'scheduled' | 'paused' | 'disabled' | 'expired'
  next_run_at: string | null; start_at: string | null; stop_at: string | null
  timezone: string; display_only?: boolean
}
export interface ProviderHealth {
  status: 'available' | 'data_unavailable' | 'replay' | 'stale'
  detail: string; observed_at?: string | null; last_status?: string
}
export interface StatusBanner {
  state: 'paused_kill_switch' | 'paused_schedule_off' | 'running'
  label: 'Paused: kill switch on' | 'Paused: schedule off' | 'Running'
}
export interface DailySummary {
  day: string; timezone: string
  candidates_proposed: number; inconclusive: number; failed: number; awaiting_human_review: number
}
export interface BDSnapshot {
  schedule?: BDSchedule
  status_banner?: StatusBanner
  daily_summary?: DailySummary
  model_activity?: Partial<Record<'qwen' | 'jev', ModelActivity>>
  job_counts?: Record<BDJob['state'], number>
  job_window?: { limit: number; total: number; returned: number; truncated: boolean; queue_total: number; queue_returned: number }
  mode: string; schedule_enabled: boolean; kill_switch: boolean; bots: BDBot[]; jobs: BDJob[]
  events: BDEvent[]; packets: BDPacket[]; budget: Budget
  health: Record<'qwen' | 'jev' | 'managed_leads', ProviderHealth>
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

export function modelWorkingJobs(snapshot: BDSnapshot, provider: 'qwen' | 'jev', now: number): BDJob[] {
  if (snapshot.kill_switch || snapshot.mode === 'replay') return []
  const activity = snapshot.model_activity?.[provider]
  return snapshot.jobs.filter(job => job.state === 'running' && heartbeatFresh(job.heartbeat_at, now)
    && (activity ? activity.active_jobs.includes(job.id) : job.step === `${provider}_called`))
}
export function queueJobs(jobs: BDJob[]): BDJob[] {
  return jobs.filter(job => job.state === 'queued' || job.state === 'leased')
    .sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at))
}
export function latestPackets(packets: BDPacket[]): BDPacket[] {
  return [...packets].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
}

export function plannedWaiting(schedule: BDSchedule | undefined, now: number): boolean {
  return schedule?.state === 'waiting' && schedule.start_at != null
    && Number.isFinite(Date.parse(schedule.start_at)) && Date.parse(schedule.start_at) > now
}
export function scheduleTime(at: string | null | undefined, timezone = 'America/Chicago'): string {
  if (!at || !Number.isFinite(Date.parse(at))) return 'Not scheduled'
  try {
    return new Intl.DateTimeFormat('en-US', {
      timeZone: timezone, weekday: 'short', month: 'short', day: 'numeric',
      hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
    }).format(new Date(at))
  } catch { return 'Time unavailable' }
}
export function providerStatus(health: ProviderHealth | undefined): string {
  if (health?.status === 'stale') return 'Unverified now'
  if (health?.status === 'data_unavailable') return health.observed_at ? 'Last call failed' : 'Unverified'
  if (health?.status === 'replay') return 'Recorded replay'
  return health?.status === 'available' ? 'Idle' : 'Unverified'
}
export function statusBanner(data: Pick<BDSnapshot, 'kill_switch' | 'schedule_enabled' | 'status_banner'>): StatusBanner {
  if (data.kill_switch) return { state: 'paused_kill_switch', label: 'Paused: kill switch on' }
  if (!data.schedule_enabled) return { state: 'paused_schedule_off', label: 'Paused: schedule off' }
  return { state: 'running', label: 'Running' }
}
export function scheduleIdleText(data: BDSnapshot, now: number): string {
  if (plannedWaiting(data.schedule, now)) return 'Waiting for planned start'
  if (data.schedule?.state === 'expired') return 'Scheduled window ended'
  if (data.kill_switch || data.schedule?.state === 'paused') return 'Conductor paused'
  if (data.schedule?.state === 'scheduled') return 'Waiting for next run'
  return 'Waiting for work'
}
export function scheduleQueueText(data: BDSnapshot, now: number): string {
  if (plannedWaiting(data.schedule, now)) return `Planned start: ${scheduleTime(data.schedule?.start_at, data.schedule?.timezone)}.${data.kill_switch ? ' Currently paused.' : ''}`
  if (data.schedule?.state === 'expired') return 'The configured research window has ended.'
  if (data.kill_switch || data.schedule?.state === 'paused') return 'The conductor is paused. Release the kill switch before starting work.'
  if (data.schedule?.next_run_at) return `Next planned run: ${scheduleTime(data.schedule.next_run_at, data.schedule.timezone)}.`
  if (data.schedule_enabled) return 'Waiting for the next scheduled cycle.'
  return data.mode === 'replay' ? 'The schedule is off. Start a shadow batch to see the research flow.' : 'No jobs are queued. The schedule is disabled.'
}

export function packetOutcome(packet: BDPacket): ResearchOutcome {
  if (packet.candidate_status === 'supported' && packet.organization.trim() && publicSource(packet.source_url)) return 'supported'
  return packet.candidate_status === 'rejected' ? 'rejected' : 'inconclusive'
}
export function outcomeLabel(outcome: ResearchOutcome): string {
  return outcome === 'supported' ? 'Supported candidate' : outcome === 'rejected' ? 'Rejected research' : 'Inconclusive research'
}
export function researchReason(reason: string | undefined): string {
  const reasons: Record<string, string> = {
    source_supported_candidate: 'Source evidence and segment checks support further research; human review is still required.',
    segment_rejected: 'The research judgment rejected this result for the segment.',
    segment_evidence_unverified: 'Segment evidence checks have not verified this result.',
    source_evidence_missing: 'The retained result is missing source evidence.',
    no_supported_prospect: 'The evidence did not support a prospect for this segment.',
    no_named_organization: 'No named organization was supported by the source.',
    no_new_evidence: 'No new evidence was found in this run.',
    segment_entity_or_event_unconfirmed: 'The organization or market event was not confirmed for this segment.',
    healthcare_entity_or_event_unconfirmed: 'The healthcare organization or market event was not confirmed.',
    entity_not_grounded: 'The organization could not be grounded in the source.',
    segment_role_mismatch: 'The organization does not match the segment role.',
    disposition_reject: 'The research judgment rejected this result.',
    disposition_watch: 'The research judgment requested watching for stronger evidence.',
    disposition_uncertain: 'The research judgment remained uncertain.',
  }
  return reason ? reasons[reason] ?? reason.replaceAll('_', ' ') : 'Evidence checks have not established a supported candidate.'
}
