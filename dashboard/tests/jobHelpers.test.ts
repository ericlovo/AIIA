import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Agent, Assignment, RepositoryResource } from '../src/lib/api.ts'
import { buildJob, canResumeJob, jobState, jobTestDefinition, jobTestPassed, jobTime, latestJobTest, nextJobCheck, runsToday, type JobDraft } from '../src/console/jobHelpers.ts'

const now = Date.parse('2026-10-02T12:00:00Z')
const repos = [{ id: 'qa-project', name: 'QA project', branch: 'qa', dirty: false }] as RepositoryResource[]
const draft: JobDraft = { recipeId: 'change-review', name: 'QA watch', repoId: 'qa-project', interval: '240', cap: '3' }
function agent(overrides: Partial<Agent> = {}): Agent {
  return { ...buildJob(draft, repos), id: 'qa-job', status: 'idle', last_run_at: null, last_result: '', last_error: '', runs: [], loop_runs_today: 0, loop_day: '2026-10-02', created_at: '', updated_at: '', ...overrides }
}

test('recipes create paused, bounded repository-only jobs', () => {
  const job = buildJob(draft, repos)
  assert.equal(job.loop_enabled, false)
  assert.deepEqual(job.tools, ['Repository read'])
  assert.equal(job.repo_id, 'qa-project')
  assert.match(job.loop_task, /unknown, not passing/)
  assert.match(job.loop_task, /do not execute commands, modify files, publish, or send messages/)
  assert.equal(buildJob({ ...draft, interval: '15', cap: '1' }, repos).loop_interval_minutes, 15)
  assert.equal(buildJob({ ...draft, interval: '1440', cap: '48' }, repos).loop_max_runs_per_day, 48)
})

test('invalid bounds, unknown recipes and missing repositories fail before a write', () => {
  for (const interval of ['', ' ', '14', '1441', '15.5', 'Infinity', 'no']) assert.throws(() => buildJob({ ...draft, interval }, repos), /Interval/)
  for (const cap of ['', '0', '49', '1.2', 'NaN']) assert.throws(() => buildJob({ ...draft, cap }, repos), /Daily cap/)
  assert.throws(() => buildJob({ ...draft, recipeId: 'generic-executor' }, repos), /recipe/)
  assert.throws(() => buildJob({ ...draft, repoId: 'missing' }, repos), /repository/)
  assert.throws(() => buildJob({ ...draft, name: ' ' }, repos), /name/)
  assert.throws(() => buildJob({ ...draft, name: 'x'.repeat(81) }, repos), /name/)
})

test('test assignments persist the exact config and latest attempt survives navigation', () => {
  const job = agent()
  const definition = jobTestDefinition(job)
  const old = { ...definition, id: 'old', status: 'completed', result: 'Evidence', created_at: '2026-10-01T00:00:00Z' } as Assignment
  const latest = { ...old, id: 'latest', status: 'failed', created_at: '2026-10-02T00:00:00Z' } as Assignment
  assert.equal(definition.objective, job.loop_task)
  assert.equal(latestJobTest(job, [old, latest])?.id, 'latest')
  assert.equal(latestJobTest({ ...job, repo_id: 'other' }, [latest]), undefined)
  assert.equal(latestJobTest({ ...job, loop_task: 'Changed task' }, [latest]), undefined)
  assert.equal(latestJobTest({ ...job, tools: ['Git workspace'] }, [latest]), undefined)
  assert.equal(latestJobTest({ ...job, model: 'other' }, [latest]), undefined)
  assert.equal(latestJobTest({ ...job, loop_enabled: true, name: 'Renamed' }, [latest])?.id, 'latest')
})

test('paused running jobs explicitly preserve their current run', () => {
  assert.deepEqual(jobState(agent({ status: 'running' }), now), { label: 'Paused', detail: 'Current run continues. Future checks are paused.', tone: 'neutral' })
  assert.equal(nextJobCheck(agent({ status: 'running' }), now), 'Paused')
  assert.equal(jobState(agent({ loop_enabled: true, status: 'running' }), now).label, 'Running')
})

test('failed, rejected, dismissed and changed tests cannot qualify for enable or resume', () => {
  const job = agent({ loop_checked_at: '2026-10-02T00:00:00Z' })
  const passed = { ...jobTestDefinition(job), id: 'test', created_at: '2026-10-02T00:00:00Z', status: 'completed', result: 'Evidence', error: '', review_status: 'unreviewed' } as Assignment
  assert.equal(jobTestPassed(passed), true)
  assert.equal(canResumeJob(job, [passed]), true)
  for (const overrides of [{ status: 'failed' as const }, { error: 'Failed' }, { review_status: 'rejected' as const }, { dismissed_at: '2026-10-02' }, { recovery_pending: true }, { result: ' ' }]) {
    const invalid = { ...passed, ...overrides }
    assert.equal(jobTestPassed(invalid), false)
    assert.equal(canResumeJob(job, [invalid]), false)
  }
  assert.equal(canResumeJob({ ...job, repo_id: 'changed' }, [passed]), false)
  assert.equal(canResumeJob({ ...job, memory_namespace: 'changed' }, [passed]), false)
  assert.equal(canResumeJob({ ...job, loop_checked_at: null }, [passed]), false)
  assert.equal(canResumeJob(job, []), true)
})

test('daily usage resets by UTC date without hiding current-day caps', () => {
  const capped = agent({ loop_enabled: true, loop_runs_today: 3 })
  assert.equal(runsToday(capped, now), 3)
  assert.equal(jobState(capped, now).label, 'Blocked')
  assert.equal(nextJobCheck(capped, now), 'Waiting on blocker')
  assert.equal(runsToday({ ...capped, loop_day: '2026-10-01' }, now), 0)
  assert.equal(jobState({ ...capped, loop_day: '2026-10-01' }, now).label, 'Enabled')
})

test('review, source, capacity and retry blockers remain distinct from no change', () => {
  for (const reason of ['awaiting_review', 'check_incomplete', 'assignment_capacity_reached', 'unknown_blocker']) assert.equal(jobState(agent({ loop_enabled: true, loop_skip_reason: reason }), now).label, 'Blocked')
  assert.equal(jobState(agent({ loop_enabled: true, loop_backoff_until: '2026-10-02T13:00:00Z' }), now).label, 'Blocked')
  assert.equal(jobState(agent({ loop_enabled: true, loop_skip_reason: 'unchanged_repository_input' }), now).label, 'No change')
  assert.equal(jobState(agent({ loop_enabled: true, status: 'error', last_error: 'Model unavailable' }), now).label, 'Failed')
})

test('interval eligibility uses the scheduler fallback without inventing a reservation', () => {
  const job = agent({ loop_enabled: true })
  assert.equal(nextJobCheck(job, now), 'Eligible now')
  assert.equal(nextJobCheck({ ...job, last_run_at: '2026-10-02T07:00:00Z' }, now), 'Eligible now')
  assert.equal(nextJobCheck({ ...job, loop_checked_at: '2026-10-02T11:00:00Z' }, now), jobTime('2026-10-02T15:00:00Z'))
  assert.equal(jobTime('invalid'), 'Unknown time')
  assert.equal(jobTime(null), 'Not reported')
})
