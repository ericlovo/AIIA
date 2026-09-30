# Agent timelines and lead review

## Daily workflow

1. Open Agents and select an agent. Activity shows recent recorded attempts;
   expand one for its task, output or failure evidence, and assignment link.
   Configuration remains a separate view. This is not a tool-call trace.
2. Open Signals, then Review inbox. Public signals are selected immediately.
3. Expand Lead qualification on a signal. Record a decision and rationale.
   Qualified for follow-up additionally requires a company, HTTPS primary-source
   URL, account fit, and observed change. Sources are human-reviewed, not
   automatically fetched or verified by this form.
4. Use the existing agent selection and Accept as work action to queue research.
   Qualification itself neither sends outreach nor closes the inbox item.

Review history records revisions and timestamps, not named reviewer identity:
the current Studio uses a shared workspace. Concurrent edits return a conflict
instead of overwriting a newer review. The losing draft remains visible until
the user explicitly discards it and loads the latest revision.

## Storage and deployment

`GET/PUT /api/public-signals/{idea_id}/qualification` uses the existing inbox
SQLite database. Two additive tables, `lead_reviews` and `lead_review_history`,
are created on first access. Current review and history are written together in
one transaction; expected_version is required on writes. Only public_signals
records qualify. No new egress, scheduling, outreach, or model calls are added.

Back up the runtime inbox database before deploying this backend slice. Deploy
backend and dashboard together, restart the Command Center in its normal service
manager, and verify a synthetic review before using real leads. Do not overwrite
runtime files with worktree fixtures. Frontend-only mobile PR #91 was deployed
separately without a Brain restart.

## Boundaries

- Agent timelines use the existing activity endpoint's 91-day window, capped at
  200 matching runs. Older history is not exposed by this view.
- Lead decisions are attached to individual signals, not deduplicated company
  records. A company-level pipeline and qualification filters are later work.
- Existing research assignments carry the original captured signal; the review
  stays attached to the inbox record, not silently injected into existing work.
- Jev screens public evidence; a human qualifies an account for follow-up.
