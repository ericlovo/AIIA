# Scheduled-loop review backpressure

## Evidence and decision

Read-only inspection of the Mini's production assignment_data.json on September
23 found 60 completed, unreviewed, non-dismissed scheduled assignments across two
agents (24 and 36). The existing guard covers queued/running assignments only.

This slice caps new scheduled work when an agent has three pending output reviews.
Three is an initial conservative product limit, not a learned quality score.
The pending-review count is derived from assignments; no database migration or
extra counter is needed. No production records were reviewed, dismissed, or deleted.

## Behavior

- Only completed, unreviewed, non-dismissed interval assignments count.
- Accepted/rejected reviews and dismissals release capacity. They remain distinct
  decisions; dismissal does not become a quality verdict.
- Manual runs remain available. Already queued work still runs before the guard.
- Waiting advances loop_checked_at, so other due agents get a turn. It consumes
  no inference or daily run budget and preserves the last executed input hash.
- Switchboard lanes and agent details say Waiting for review.
- Once capacity is available, work resumes on the next normal cadence check,
  still subject to the daily cap, error backoff, and unchanged-input suppression.
- Existing backlog above the limit is preserved, not automatically cleared.

## Verification and release

Backend execution tests cover the cap, accepted/rejected/dismissed release paths,
manual execution, scheduler fairness, persisted waiting state, and input hashes.
Registry tests cover status/agent/trigger scoping and restart persistence.
The synthetic browser gate exercises the waiting lane and inspector at
1440, 653, and 390 pixels without contacting the production API.

Final gates: backend full suite 687 passed, 9 skipped, 6 failed (PDF dependency
below); dashboard 60 tests passed, production build passed, browser gate passed
at all three widths. Python lint, format checks, and git diff whitespace checks
passed. Screenshots are in /tmp/aiia-review-backpressure/.

The six PDF research tests still encounter the existing Homebrew pyexpat linkage
failure (missing _XML_SetAllocTrackerActivationThreshold). Do not describe the
full suite as green. Dashboard lint has its existing ErrorBoundary warning.

Not deployed by this slice. Before release, reconcile with current main, back up
runtime files, deploy through the normal process, then verify a due agent reports
awaiting_review without creating a new assignment. Do not approve or dismiss real
work merely to smoke-test the resume path. This guard does not regulate the
separate local-proposal inbox producers.

## Next human step

Review a small real sample with Eric/Tony before adding quality scoring. Keep
inbox proposal outcomes separate from assignment output reviews. Do not infer
quality from queue length, execution success, or dismissal alone.
