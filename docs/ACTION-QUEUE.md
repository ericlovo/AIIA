# Action queue HTTP contract

The action queue is separate from Studio assignments and proposal inbox reviews.
It holds engineering findings awaiting approval. Approval is not merely dismissal:
AUTO-tier work can execute immediately when an execution engine is available.

## Lifecycle

```text
pending -> approved -> executing -> completed | failed
   |           |           
   |           +-> completed (manual completion)
   +-----------+-> rejected (before execution only)
   +-> expired (pending for more than 72 hours)
```

HTTP approval requires pending; rejection accepts pending or approved; completion
accepts approved or executing. Repeats and terminal-state transitions return 409,
not idempotent success. Unknown IDs return 404. Failed actions use the execution
engine's retry path, not the approval endpoint. Internal queue methods remain
permissive for existing producers; these guards are the HTTP boundary contract.

Expiry applies only to pending actions, strictly older than 72 hours from
created_at (UTC; legacy timezone-less timestamps are interpreted as UTC). List,
summary and transition requests apply and persist expiry before proceeding.
There is no promise of an expiry WebSocket event; consumers must reconcile reads.

## Routes

| Route | Success payload | Notes |
| --- | --- | --- |
| GET /api/actions | `{actions: Action[], summary: Summary}` | status, action_type filters; limit defaults to 50; summary is unfiltered |
| POST /api/actions | `{action: Action}` | action_type, severity, title required; invalid/missing required values return 400 |
| GET /api/actions/summary | Summary | total, by_status, pending_by_severity; absent categories have no key |
| POST /api/actions/{id}/approve | Action | May additionally include auto_executed and execution_result |
| POST /api/actions/{id}/reject | Action | Optional JSON reason, default empty string |
| POST /api/actions/{id}/complete | Action | Optional JSON result, default empty string |

Success status is 200, including creation. Transition errors use FastAPI's
`{detail: string}` payload. Create validation uses `{error: string}`. No standalone
GET-by-ID route is defined here. Successful transitions broadcast action_updated.

Creation accepts description, proposed_fix, source_task (defaults to api),
files_affected and the existing auto_approve option. That option is an existing
privileged API behavior, not a recommendation for the UI. Creation deduplicates
pending records by type and title. Storage retains at most 200 records.

## Action fields

Types below describe normal producer output; legacy records are not schema-migrated.
The create endpoint uses an untyped body, so not all optional field types are
validated. A typed input contract is follow-up work.

| Field | Type | Meaning |
| --- | --- | --- |
| id | string | 12-character generated identifier |
| type | string | lint_fix, test_fix, security_fix, ci_fix, review, tech_debt, post_commit_review, verify_lint, verify_test, verify_security, commit |
| severity | string | info, warn, error, critical; urgency, not execution permission |
| title | string | Short finding name |
| description | string | Evidence/problem: primary context for why human attention is needed |
| proposed_fix | string | Suggested remedy; not an authorized command |
| source_task | string | Producer/task identifier; there is no source field |
| status | string | Lifecycle state listed above |
| files_affected | string[] | Suspected scope of the finding |
| created_at | string | ISO timestamp |
| updated_at | string or null | Last lifecycle update |
| rejected_reason | string or null | Human reason for rejection |
| completed_result | string or null | Completion output or failure explanation |
| parent_id | string or null | Parent action for chained work |
| chain_on_complete | string or null | Action type to create on completion |
| execution_log_id | string or null | Execution log reference |
| execution_started_at | string or null | ISO execution start timestamp |
| retry_count | integer | Retry count, initially zero |
| files_changed | string[] | Execution changes, initially empty |

Producers include execution/executor.py, execution/chains.py,
execution/story_executor.py, autonomy/proactive_executor.py,
autonomy/self_healing.py and autonomy/gated_downgrade.py. Display title,
description, source_task, severity and proposed_fix together for approval context.
Do not equate severity with an AUTO/SUPERVISED/GATED safety tier.

## Dashboard comparison (frontend intentionally unchanged)

| dashboard/src/lib/api.ts | Actual contract / required follow-up |
| --- | --- |
| Action's 11 declared fields | Present in normal queue output with matching types |
| Missing updated_at and completed_result | Add nullable strings to display outcome/history |
| Missing parent_id, chain_on_complete | Add nullable strings if displaying chains |
| Missing execution_log_id, execution_started_at | Add nullable strings for execution visibility |
| Missing retry_count, files_changed | Add integer and string[] |
| approveAction expects `{approved: boolean}` | Returns Action, optionally enriched with execution outcome |
| rejectAction expects `{rejected: boolean}` | Returns Action |
| summary is Record<string, unknown> | Shape is total: number, by_status and pending_by_severity: maps of counts |

## Verification and limits

test_action_routes.py isolates persistence in a temporary file and disables
startup/lifespan and execution. It covers shapes, filters, pending deduplication,
all status/operation pairs, missing IDs and persisted expiry. Production data and
the air-gap configuration are untouched.

The read-expiry and HTTP guards do not provide multi-process locking or atomic
disk writes. Existing queue persistence logs write failures rather than failing
the operation; durable acknowledgement is separate follow-up work.
