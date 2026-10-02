import type { Agent, AgentDefinition, Assignment, AssignmentDefinition, RepositoryResource } from '../lib/api'

export const JOB_RECIPES = [
  { id: 'change-review', name: 'Repository change review', output: 'Change risk report', task: 'Inspect the visible recent commits and working-tree summary. Identify up to three regression risks and one focused next review.' },
  { id: 'delivery-brief', name: 'Repository delivery brief', output: 'Delivery brief', task: 'Summarize visible recent commits, working-tree changes, and unresolved release questions. Separate observed changes from unverified CI and deployment status.' },
] as const

export interface JobDraft {
  recipeId: string
  name: string
  repoId: string
  interval: string
  cap: string
}

export function buildJob(draft: JobDraft, repos: RepositoryResource[]): AgentDefinition {
  const recipe = JOB_RECIPES.find(item => item.id === draft.recipeId)
  if (!recipe) throw new Error('Choose a supported recipe.')
  if (!draft.name.trim() || draft.name.trim().length > 80) throw new Error('Use a job name between 1 and 80 characters.')
  if (!repos.some(repo => repo.id === draft.repoId)) throw new Error('Choose an available project repository.')
  const interval = Number(draft.interval)
  const cap = Number(draft.cap)
  if (!draft.interval.trim() || !Number.isInteger(interval) || interval < 15 || interval > 1440) throw new Error('Interval must be a whole number from 15 to 1440 minutes.')
  if (!draft.cap.trim() || !Number.isInteger(cap) || cap < 1 || cap > 48) throw new Error('Daily cap must be a whole number from 1 to 48 runs.')
  return {
    name: draft.name.trim(), mission: recipe.task,
    persona: 'Evidence first. Separate observed facts, unknowns, and recommendations.',
    skills: ['Analysis'], tools: ['Repository read'], repo_id: draft.repoId,
    temperature: 0.2, max_tokens: 1600, loop_enabled: false,
    loop_interval_minutes: interval, loop_max_runs_per_day: cap,
    loop_task: `${recipe.task}\nUse only the supplied repository snapshot. Cite visible paths and commit IDs. Missing source, prior snapshots, CI results, and deployment evidence are unknown, not passing. Do not invent changes since a previous run. Return Evidence, Findings, and Next action. Read-only analysis: do not execute commands, modify files, publish, or send messages.`,
  }
}

// Persist the tested configuration with the assignment, not in browser storage.
function testContext(agent: Agent): string {
  return `Studio Jobs test v1\n${JSON.stringify({ repo_id: agent.repo_id, tools: agent.tools, mission: agent.mission, persona: agent.persona, skills: agent.skills, model: agent.model ?? '', temperature: agent.temperature, max_tokens: agent.max_tokens, loop_task: agent.loop_task, suite: agent.suite ?? '', memory_namespace: agent.memory_namespace ?? '' })}`
}

export function jobTestDefinition(agent: Agent): AssignmentDefinition {
  return {
    title: `Test: ${agent.name}`, agent_id: agent.id, objective: agent.loop_task,
    priority: 'normal', context: testContext(agent),
    success_criteria: 'Return source-backed evidence, explicit unknowns, and one next action. Do not claim tests or deployments succeeded without evidence. No repository writes or publication.',
  }
}

export function latestJobTest(agent: Agent, assignments: Assignment[]): Assignment | undefined {
  return assignments.filter(item => item.agent_id === agent.id && item.context === testContext(agent) && item.objective === agent.loop_task)
    .sort((a, b) => b.created_at.localeCompare(a.created_at))[0]
}

export function jobTestPassed(assignment?: Assignment): boolean {
  return !!assignment && assignment.status === 'completed' && !!assignment.result.trim() && !assignment.error && assignment.review_status !== 'rejected' && !assignment.dismissed_at && !assignment.recovery_pending
}

export function canResumeJob(agent: Agent, assignments: Assignment[]): boolean {
  const hasScheduledHistory = !!agent.loop_checked_at || agent.runs.some(run => run.trigger === 'interval')
  const hasJobsTest = assignments.some(item => item.agent_id === agent.id && item.context.startsWith('Studio Jobs test v1\n'))
  return hasScheduledHistory && (!hasJobsTest || jobTestPassed(latestJobTest(agent, assignments)))
}

export function runsToday(agent: Agent, now = Date.now()): number {
  return agent.loop_day === new Date(now).toISOString().slice(0, 10) ? agent.loop_runs_today : 0
}

export function jobState(agent: Agent, now = Date.now()): { label: string; detail: string; tone: 'neutral' | 'good' | 'warning' } {
  if (!agent.loop_enabled) return { label: 'Paused', detail: agent.status === 'running' ? 'Current run continues. Future checks are paused.' : 'No future checks.', tone: 'neutral' }
  if (agent.status === 'running') return { label: 'Running', detail: 'Executing on the Mini.', tone: 'good' }
  if (runsToday(agent, now) >= agent.loop_max_runs_per_day) return { label: 'Blocked', detail: 'Daily cap reached. Resets at 00:00 UTC.', tone: 'warning' }
  if (agent.loop_backoff_until && Date.parse(agent.loop_backoff_until) > now) return { label: 'Blocked', detail: `Retry backoff until ${jobTime(agent.loop_backoff_until)}.`, tone: 'warning' }
  const reasons: Record<string, string> = {
    awaiting_review: 'Review limit reached. Review saved work to continue.',
    check_incomplete: 'Source check incomplete. Open work for missing evidence.',
    assignment_capacity_reached: 'Work capacity reached.',
  }
  if (agent.loop_skip_reason && reasons[agent.loop_skip_reason]) return { label: 'Blocked', detail: reasons[agent.loop_skip_reason], tone: 'warning' }
  if (agent.status === 'error' || agent.last_error) return { label: 'Failed', detail: agent.last_error || 'Last execution failed.', tone: 'warning' }
  if (agent.loop_skip_reason === 'unchanged_repository_input') return { label: 'No change', detail: 'Last source check found unchanged inputs.', tone: 'good' }
  if (agent.loop_skip_reason) return { label: 'Blocked', detail: `Last check: ${agent.loop_skip_reason.replaceAll('_', ' ')}.`, tone: 'warning' }
  return { label: 'Enabled', detail: 'Interval checks enabled.', tone: 'good' }
}

export function nextJobCheck(agent: Agent, now = Date.now()): string {
  if (!agent.loop_enabled) return 'Paused'
  if (agent.status === 'running') return 'After current run'
  if (jobState(agent, now).label === 'Blocked') return 'Waiting on blocker'
  const last = Date.parse(agent.loop_checked_at || agent.last_run_at || '')
  if (!Number.isFinite(last)) return 'Eligible now'
  const next = last + agent.loop_interval_minutes * 60_000
  return next <= now ? 'Eligible now' : jobTime(new Date(next).toISOString())
}

export function jobTime(value?: string | null): string {
  if (!value) return 'Not reported'
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return 'Unknown time'
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })
}

export function jobError(error: unknown): string {
  return error instanceof Error ? error.message : 'Request failed. Refresh to confirm the saved state.'
}
