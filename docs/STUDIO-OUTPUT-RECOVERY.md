# Assignment Output Recovery

Implemented in the reconciliation worktree; this slice has not been deployed.

## Contract

An assignment start now durably records a unique `attempt_id` before inference.
The agent's saved run uses that exact ID. Normal completion records
`completed_run_id` in the assignment. No SQLite schema migration is required.

When assignment completion cannot be saved, the result can be recovered from the
durable agent JSON (including its pending outbox) or SQLite ledger. Recovery never
uses unsaved in-memory output, invokes a model, or infers identity from timestamps.
The run ID, assignment ID, agent ID, and assignment trigger must all match.

`POST /api/assignments/{id}/recover` applies the saved result and linked handoff
state in one atomic assignment-registry mutation. Repeated recovery is a no-op
for an already-applied attempt, preserving review notes, versions, and decisions.
Failed and oversized results remain failures rather than becoming successful work.

Startup attempts this reconciliation before starting background workers. Unresolved
records remain visible; startup does not invent missing evidence or rerun work.
The Assignments and Handoffs detail panels expose **Recover saved output**, pending
state, errors, and a recovered confirmation. The assignments API adds
`recovery_pending`; it describes unfinished reconciliation, not proof that output
exists. Recovery is rejected while the Mini execution lock is held.

A fresh assignment retry is blocked when the prior unresolved attempt already has
durable output. The operator must recover it first. If no output was ever saved,
an explicit retry creates a new attempt ID, so late/older evidence cannot overwrite
the new attempt. History-read failures block retry rather than treating history as
empty. A new attempt clears the prior recovered confirmation.

## Boundaries

- Legacy assignments have no attempt ID and are not automatically matched by
  assignment ID alone. They retain their existing manual workflow.
- This is single-process reconciliation, not a distributed lease or a transaction
  spanning the agent and assignment files. It cannot recreate output never saved.
- Recovery does not repair previously truncated artifacts, rerun evaluations,
  grant tools, accept outputs, or approve Git operations.
- Durable JSON evidence can recover even when SQLite is unavailable. If neither
  source is readable or contains the exact run, the error remains explicit.

## Verification

- 18 new recovery cases cover matching, stale attempts, repeat application after
  review, linked handoff updates, storage rollback, missing evidence, legacy work,
  busy rejection, output limits, JSON outbox recovery, and startup retries.
- The existing completion-failure integration test now proves actual run-ID
  propagation and recovery with exactly one mocked model request.
- Full backend suite: **402 passed, 9 skipped, 1 known baseline failure**:
  `test_watcher_content_type_for_m4a` on this Mac returns `audio/mp4a-latm`.
- Ruff passes. Frontend production build and five frontend tests pass.
- Synthetic browser checks at 1440x1000 and 390x844 pass: visible recovery error,
  disabled pending button, success, reload, and no horizontal page overflow.
  Screenshots inspected. Exactly two intercepted recovery requests; no model or
  live backend requests. These are UI checks, not live deployment acceptance.

No production restart, schedule activation, live inference, or deployment was
performed by this slice. Previous deployment evidence describes `f9e7b7e`, not
these new recovery changes.
