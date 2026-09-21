import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { ReviewHealth } from '../src/lib/api.ts'
import { activeSources, formatRate, reviewMetrics, reviewSummary } from '../src/console/reviewHealth.ts'

const health = (overrides: Partial<ReviewHealth> = {}): ReviewHealth => ({
  window_days: 14,
  since: '2026-09-07T00:00:00+00:00',
  filed: 10,
  reviewed: 8,
  totals: { open: 2, needs_work: 4, already_fixed: 2, declined: 1, external_failure: 1, unclassified: 0 },
  by_source: [{ source: 'code_review', open: 2, needs_work: 3, already_fixed: 2, declined: 1, external_failure: 1, unclassified: 0 }],
  by_project: [{ project: 'mindmoor', open: 2, needs_work: 4, already_fixed: 2, declined: 1, external_failure: 1, unclassified: 0 }],
  ...overrides,
})

test('rates are shares of reviewed proposals, so an untouched backlog cannot flatter a loop', () => {
  const metrics = reviewMetrics(health())
  const byBucket = Object.fromEntries(metrics.map(metric => [metric.bucket, metric]))

  assert.equal(byBucket.needs_work.count, 4)
  assert.equal(formatRate(byBucket.needs_work.rate), '50%')
  assert.equal(formatRate(byBucket.already_fixed.rate), '25%')
  // Open has no verdict to be a share of.
  assert.equal(byBucket.open.count, 2)
  assert.equal(byBucket.open.rate, null)
  assert.equal(formatRate(byBucket.open.rate), '—')
})

test('every bucket is reported, including the ones that are zero', () => {
  const metrics = reviewMetrics(health())

  assert.deepEqual(metrics.map(metric => metric.bucket), [
    'open', 'needs_work', 'already_fixed', 'declined', 'external_failure', 'unclassified',
  ])
})

test('unclassified is named, never folded into declined', () => {
  const legacy = health({
    filed: 4, reviewed: 4,
    totals: { open: 0, needs_work: 0, already_fixed: 0, declined: 1, external_failure: 0, unclassified: 3 },
  })

  const byBucket = Object.fromEntries(reviewMetrics(legacy).map(metric => [metric.bucket, metric]))

  assert.equal(byBucket.unclassified.count, 3)
  assert.equal(formatRate(byBucket.unclassified.rate), '75%')
  assert.equal(byBucket.declined.count, 1)
  assert.equal(formatRate(byBucket.declined.rate), '25%')
})

test('an empty window reports zeroes and no rates, not a division by zero', () => {
  const empty = health({
    filed: 0, reviewed: 0,
    totals: { open: 0, needs_work: 0, already_fixed: 0, declined: 0, external_failure: 0, unclassified: 0 },
    by_source: [], by_project: [],
  })

  const metrics = reviewMetrics(empty)

  assert.equal(metrics.length, 6)
  assert.ok(metrics.every(metric => metric.count === 0 && metric.rate === null))
  assert.equal(reviewSummary(empty), 'No local proposals filed in the last 14 days.')
  assert.deepEqual(activeSources(empty), [])
})

test('missing data renders as zeroes rather than crashing the switchboard', () => {
  const metrics = reviewMetrics(undefined)

  assert.equal(metrics.length, 6)
  assert.ok(metrics.every(metric => metric.count === 0 && metric.rate === null))
  assert.equal(reviewSummary(undefined), 'Review health unavailable.')
})

test('the summary leads with what a person should do about it', () => {
  assert.equal(reviewSummary(health()), '10 filed in 14 days · 2 open · 50% of reviewed became work.')
  assert.equal(
    reviewSummary(health({ filed: 3, reviewed: 0, totals: { open: 3, needs_work: 0, already_fixed: 0, declined: 0, external_failure: 0, unclassified: 0 } })),
    '3 filed in 14 days, none reviewed yet.',
  )
})

test('sources are summarised by what they filed and what is still open', () => {
  assert.deepEqual(activeSources(health()), [
    { source: 'code_review', label: 'code review', filed: 9, open: 2 },
  ])
})
