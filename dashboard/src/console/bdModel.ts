export type ResearchOutcome = 'supported' | 'rejected' | 'inconclusive'
export type Budget = Record<'tokens' | 'calls' | 'seconds', { used: number | null; limit: number | null }>
export interface BDBot {
  id: string; name: string; segment: string; tenant_id: string; budgets: Budget
  counts: Record<'queued' | 'running' | 'completed' | 'failed', number>
  kind?: string; research_segment?: string | null
  paused?: boolean; around_the_clock?: boolean; focused?: boolean
}
export interface BDJob {
  id: string; bot_id: string; tenant_id: string; segment: string
  state: 'queued' | 'leased' | 'running' | 'completed' | 'failed' | 'cancelled'
  mode?: string; research_outcome?: ResearchOutcome; result_reason?: string
  next_query?: string | null
  finished_at?: string | null; error?: string | null; attempts?: number
  step: string; created_at: string; started_at: string | null; heartbeat_at: string | null
  priority?: number; stage?: 'queued' | 'extract' | 'judge' | 'awaiting_human' | string
  operator_query?: string | null; operator_target?: string | null
}
export interface BDEvent { provider?: string; model?: string; latency_ms?: number | null; mode?: string; primitive?: string; id: string | number; kind: string; message: string; created_at: string; bot_id: string; job_id: string }
export interface BDPacket {
  candidate_status?: ResearchOutcome; candidate_reason?: string; next_query?: string | null
  id: string; index: number; contact_policy?: string; disposition?: string; source_quote?: string; job_id: string; bot_id: string; tenant_id: string; segment: string; created_at: string
  mode: string; review_status: string; organization: string; source_url: string
  emails: string[]; phones: string[]; signals: string[]; insights: string[]; summary: string
  permit?: { status?: string; number?: string | null; address?: string | null; project_title?: string | null; permit_status?: string | null } | null
  verdict?: string | null; evidence_score?: number | null
  extraction_model?: string | null; judgment_model?: string | null
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
export interface AssignmentLane {
  assigned: string[]; queue: string[]; recent: Array<{ id: number | string; kind: string; job_id?: string; bot_id?: string; created_at: string; latency_ms?: number | null }>
  status: string; calls: number | null; errors: number | null; latency_ms: number | null
  verdict_mix?: Record<'Choice' | 'Score' | 'Noul', number> | null
}
export interface BDAssignments {
  qwen: AssignmentLane; jev: AssignmentLane
  pipeline: Array<{ id: string; stage: 'extract' | 'judge' | 'awaiting_human' | string }>
}
export interface SteerAudit {
  id: number; action: string; created_at: string; bot_id?: string; job_id?: string
  kind?: string; focus_id?: string; query?: string; target?: string; enabled?: boolean
  overnight_enabled?: boolean; around_the_clock?: boolean; priority?: number
}
export interface BDSteering {
  paused_bots: string[]; focus: { kind: 'bot' | 'segment'; id: string } | null
  overnight_enabled: boolean; around_the_clock: Record<string, boolean>
  audit: SteerAudit[]
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
  steering?: BDSteering
  assignments?: BDAssignments
  cadence?: { timezone: string; weekday?: { interval_seconds: number }; overnight?: { enabled: boolean; interval_seconds: number } }
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
export function unavailableCount(value: number | null | undefined): string {
  return value == null ? 'data_unavailable' : value.toLocaleString()
}
export function jobStage(job: Pick<BDJob, 'state' | 'step' | 'stage'>): string {
  if (job.stage) return job.stage
  const step = job.step || ''
  if (job.state === 'queued' || job.state === 'leased') return 'queued'
  if (job.state === 'running') return step.includes('jev') ? 'judge' : 'extract'
  if (job.state === 'completed') return 'awaiting_human'
  return job.state
}
export function statusBanner(data: Pick<BDSnapshot, 'kill_switch' | 'schedule_enabled' | 'status_banner'>): StatusBanner {
  if (data.kill_switch) return { state: 'paused_kill_switch', label: 'Paused: kill switch on' }
  if (!data.schedule_enabled) return { state: 'paused_schedule_off', label: 'Paused: schedule off' }
  return { state: 'running', label: 'Running' }
}

export const LANE_ORDER = ['comcast', 'srg', 'healthcare', 'pe-diligence', 'alumni-associations'] as const
export const CADENCE_OPTIONS = [
  { id: '15', seconds: 900, label: 'Every 15 min', phrase: 'every 15 min' },
  { id: '60', seconds: 3600, label: 'Hourly', phrase: 'hourly' },
  { id: '360', seconds: 21600, label: '4× daily', phrase: '4× daily' },
] as const
export const LANE_PROFILE: Record<string, { title: string; kind: string; isNew?: boolean; targets: string[] }> = {
  comcast: { title: 'Comcast · Minneapolis permits', kind: 'Client', isNew: true, targets: ['New business fit-outs', 'Office tenant improvements'] },
  srg: { title: 'SRG', kind: 'Client', targets: ['Mid-size operators expanding'] },
  healthcare: { title: 'Healthcare', kind: 'Segment', targets: ['Clinics modernizing IT'] },
  'pe-diligence': { title: 'PE technical diligence', kind: 'Segment', targets: ['Software add-on deals'] },
  'alumni-associations': { title: 'Alumni & associations', kind: 'Segment', targets: ['Associations refreshing member tools'] },
}

export type LeadBanner = { mode: 'running' | 'paused' | 'stopped'; label: string; sub: string }
export function orderedBots(bots: BDBot[]): BDBot[] {
  return [...bots].sort((a, b) => {
    const left = LANE_ORDER.indexOf(a.id as typeof LANE_ORDER[number])
    const right = LANE_ORDER.indexOf(b.id as typeof LANE_ORDER[number])
    return (left < 0 ? 99 : left) - (right < 0 ? 99 : right)
  })
}
export function laneProfile(bot: Pick<BDBot, 'id' | 'name' | 'kind'>): { title: string; kind: string; isNew: boolean; targets: string[] } {
  const profile = LANE_PROFILE[bot.id]
  return {
    title: profile?.title ?? bot.name,
    kind: profile?.kind ?? (bot.kind === 'client' ? 'Client' : 'Segment'),
    isNew: Boolean(profile?.isNew),
    targets: profile?.targets ?? [],
  }
}
export function cadencePhrase(seconds: number | null | undefined): string {
  if (seconds == null) return 'on the standing cadence'
  const exact = CADENCE_OPTIONS.find(option => option.seconds === seconds)
  if (exact) return exact.phrase
  if (seconds < 60) return `every ${seconds}s`
  if (seconds % 3600 === 0) return seconds === 3600 ? 'hourly' : `every ${seconds / 3600} hours`
  if (seconds % 60 === 0) return `every ${seconds / 60} min`
  return `every ${seconds}s`
}
export function selectedCadence(seconds: number | null | undefined): string | null {
  return CADENCE_OPTIONS.find(option => option.seconds === seconds)?.id ?? null
}
export function weekdayInterval(data: Pick<BDSnapshot, 'cadence'>): number | null {
  const seconds = data.cadence?.weekday?.interval_seconds
  return typeof seconds === 'number' ? seconds : null
}
export function leadBanner(data: Pick<BDSnapshot, 'kill_switch' | 'schedule_enabled' | 'steering' | 'cadence' | 'events'>, now: number): LeadBanner {
  const interval = weekdayInterval(data)
  const last = data.events?.[0]?.created_at
  const lastText = last && Number.isFinite(Date.parse(last)) ? `Last check ${ageLabel(last, now)} ago` : 'No checks in this snapshot'
  const cadence = cadencePhrase(interval)
  const overnight = Boolean(data.steering?.overnight_enabled ?? data.cadence?.overnight?.enabled)
  if (data.kill_switch) return { mode: 'stopped', label: 'Stopped by you', sub: 'All bots are off. Nothing happens until you resume.' }
  if (!data.schedule_enabled) return { mode: 'paused', label: 'Paused by you', sub: 'Nothing new is being checked. Leads already here stay.' }
  return {
    mode: 'running',
    label: overnight ? 'Running 24/7' : 'Running',
    sub: `${lastText} · checks ${cadence}`,
  }
}
export function jobLabel(job: Pick<BDJob, 'operator_query' | 'operator_target' | 'segment' | 'id'>, packets: BDPacket[]): string {
  const target = job.operator_target?.trim()
  if (target) return target
  const query = job.operator_query?.trim()
  if (query) return query
  const packet = packets.find(item => item.job_id === job.id)
  const permit = packet?.permit?.project_title?.trim()
  if (packet?.organization?.trim()) return packet.organization.trim()
  if (permit) return permit
  return (job.segment || 'research').replaceAll('-', ' ')
}
export function leadContact(packet: BDPacket): string {
  if (packet.emails[0]) return packet.emails[0]
  if (packet.phones[0]) return packet.phones[0]
  return 'Held · no outreach'
}
export function operatorTargets(data: BDSnapshot, botId: string): string[] {
  const seen = new Set<string>()
  const targets: string[] = []
  for (const job of data.jobs) {
    if (job.bot_id !== botId || job.state === 'cancelled') continue
    const text = (job.operator_target || job.operator_query || '').trim()
    if (text && !seen.has(text)) { seen.add(text); targets.push(text) }
  }
  return targets
}
export function medianLatencySeconds(events: BDEvent[], kinds: string[]): number | null {
  const values = events.filter(event => kinds.includes(event.kind) && event.latency_ms != null).map(event => event.latency_ms as number).sort((a, b) => a - b)
  if (!values.length) return null
  return values[Math.floor(values.length / 2)] / 1000
}
export function formatSeconds(value: number | null | undefined): string {
  return value == null ? 'data_unavailable' : `${value.toFixed(1)} s`
}
export function auditLine(item: SteerAudit): string {
  const target = item.query || item.target
  if (item.action === 'pause') return 'You paused every bot.'
  if (item.action === 'resume') return 'You resumed every bot.'
  if (item.action === 'pause_bot') return `You paused ${item.bot_id}.`
  if (item.action === 'resume_bot') return `You resumed ${item.bot_id}.`
  if (item.action === 'focus') return `You focused on ${item.bot_id || item.focus_id || item.kind}.`
  if (item.action === 'clear_focus') return 'You cleared focus.'
  if (item.action === 'bump') return `You moved a job to the front.`
  if (item.action === 'queue') return target ? `You added a target: "${target}".` : 'You added a target.'
  if (item.action === 'cancel') return 'You cancelled a queued target.'
  if (item.action === 'set_cadence' && item.overnight_enabled != null) return item.overnight_enabled ? 'You turned overnight 24/7 on.' : 'You turned overnight 24/7 off.'
  if (item.action === 'set_cadence') return 'You changed how often bots check.'
  if (item.action === 'set_budget') return 'You set the daily budget.'
  if (item.action === 'kill_switch') return item.enabled ? 'You stopped every bot.' : 'You turned everything back on.'
  return item.action.replaceAll('_', ' ')
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
    permit_fields_not_in_source: 'The source did not include a permit number, address, project title, and status.',
    generic_permit_reference: 'A generic permit theme is not a named commercial prospect.',
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
