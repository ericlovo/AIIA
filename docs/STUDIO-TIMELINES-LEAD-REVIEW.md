# Agent timelines and lead review

## Daily workflow

1. Open Agents and select an agent. Activity shows recent recorded attempts;
   expand one for its task, output or failure evidence, and assignment link.
   Configuration remains a separate view. This is not a tool-call trace.
2. Open Signals. Lead queue lists public signals with a qualification filter,
   company-name search, and 25-signal pages. Open Review inbox for full triage.
3. Expand Lead qualification on a signal. Record a decision and rationale.
   Qualified for follow-up additionally requires a company, HTTPS primary-source
   URL, account fit, and observed change. Sources are human-reviewed, not
   automatically fetched or verified by this form.
4. Use the existing agent selection and Accept as work action to queue research.
   Public-signal assignments include a snapshot of the latest saved qualification
   read before the capture is claimed, including revision and timestamp. Unsaved
   browser drafts are not included. Later reviews do not rewrite existing work.
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
  records. Queue groups use trimmed, lowercased entered names within the current
  page only; unnamed signals stay separate. These are not verified company
  identities or account-level verdicts. Company-level deduplication is later work.
- The queue includes dismissed and assigned signals with their disposition.
  Dismissed items cannot queue research here; existing assignment links open Work.
  Qualification saves refresh filtered results. Research queues but never starts
  an agent. The saved review, not unsaved form fields, enters the assignment.
- `GET /api/public-signals/leads` supports status, company, offset and limit
  (default 25, maximum 100). Company search is a literal substring, ASCII
  case-insensitive using SQLite lower(); SQL wildcard characters remain literal.
  Counts and rows share a read transaction. Private captures are never included.
  No extra model calls, news retrieval, or outbound messages are triggered.
- New public-signal research assignments carry the original capture plus the
  saved qualification snapshot as untrusted evidence. Research briefs must cover
  ownership, geography and account fit, observed change, dated citations,
  counter-evidence, and open questions. Missing tool access must be reported,
  not treated as source verification. Nothing fetches a URL at assignment time.
- Signals without a review remain assignable for research and are explicitly
  marked unverified. Research/watch/rejected decisions are preserved verbatim;
  assigning research never upgrades them to qualified.
- A missing review table means no reviews have been recorded. A corrupt review
  or unavailable storage refuses assignment before claiming the capture.
  Existing assignments are unchanged. The scope of this policy is the research
  brief, not a new execution permission boundary; tool permissions still apply.
- Jev screens public evidence; a human qualifies an account for follow-up.
