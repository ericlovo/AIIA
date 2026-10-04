# Jobs-first Studio: first release gate

Implementation: Today, Inbox, Jobs, Work and Projects are the everyday navigation.
Agents, Signals, Memory, Map, Handoffs, Overview and Activity history remain
under Studio. Existing record URLs are preserved. This is the first functional
slice of [the product reset](STUDIO-PRODUCT-RESET-2026-10-02.md), not completion
of the weekend roadmap.

## What this slice changes

- Today lists unresolved work and approvals, work awaiting manual start, active
  assignments, enabled recurring jobs and recently accepted output. Counts keep
  the existing review/dismissal semantics. Failed reads are not an all-clear.
- Inbox opens unreviewed Slack captures by default. Today and Inbox show separate
  counts for Slack captures, proposals, and work review. The counts link to their
  own queues; work review is not a count of Slack messages. Existing `#/memory`
  links, review metrics, and public-signal links remain valid.
  Pending proposals exclude findings already accepted as work even when their
  storage status is still unreviewed. Reopening a queue count clears old filters.
- Jobs lists existing agent intervals with last check, next eligible check,
  UTC daily cap, blocker and pause/resume. New repository change-review and
  delivery-brief jobs are created paused, tested explicitly and enabled only
  after evidence is inspected in this flow.
- Work opens its creation form explicitly. Results, failures, recovery and
  review decisions precede optional assignment metadata, history and Git details.
- Projects uses configured repository mounts and read-only GitHub Actions
  evidence. Local branch/SHA and remote workflow branch/SHA remain separate.
  Each run has a source link and observation time. Deployment is unknown.
- Voice and system status open on demand. Closing Voice cancels pending
  connection setup and closes an active session.

## What is not promised

- These are interval jobs, not arbitrary cron/calendar schedules. The daily
  cap resets in UTC. Next eligible check is not a reserved execution time.
- A new Work assignment still requires manual start. This does not introduce
  a durable background queue or machine-wide inference arbitration.
- The two recipes summarize repository snapshots. They do not fetch full
  changed source or CI logs, run tests, write code, deploy, or verify production.
- The Jobs test-before-enable flow is a UX guard, not a new backend permission
  boundary. Existing agent configuration and loop APIs retain their behavior.
- Built-in tasks are read-only here. Their execution status is not CI status.
- Public discovery and Jev screening retain their existing explicit opt-ins.
  Nothing in this release enables paid inference, public retrieval or outreach.
- Jev makes typed relevance/routing decisions over permitted text. Ollama on
  the Mini produces the local repository analyses. GitHub state is retrieved
  deterministically, not inferred by either model.

## Tony and Eric walkthrough

Use a separate QA dataset first, then one explicitly authorized bounded run.
Record time-to-complete, wrong turns and confusing language. The human gate is
still pending until both users complete this without developer coaching.

| Task | Expected result |
| --- | --- |
| Open Today | Find a failed item and its failure evidence in one click |
| Open Inbox | Slack captures open first; select Proposals to see only pending findings, excluding those already accepted as work |
| Follow queue counts | Slack and Proposals show their own pending rows; Work review opens unresolved reports, failures and approvals |
| Review a capture | Log to memory, dismiss, or queue work; counts refresh and nothing runs automatically |
| Inspect an old mention-only capture | It is labelled incomplete; logging, queueing and retrying its save receipt are unavailable; dismissal remains explicit |
| Reload a proposal link | `#/inbox?source=loops` stays on proposals; back returns to the previous queue |
| Lose an inbox source | Its count reads Unavailable, not zero; other queues stay usable |
| Open an approval | Land on the owning assignment with its pending Git decision visible |
| Open Work | No creation form until New work; a queued assignment says Awaiting manual start |
| Create a job | Choose recipe, repository, interval and daily cap; creation alone runs nothing |
| Test the job | One saved assignment and one explicit local run; busy/error states remain visible |
| Enable it | Inspect the saved result first; enabled state survives page refresh |
| Pause it | Future scheduling pauses; any current execution is not described as cancelled |
| Review output | Accept/reject/dismiss remain distinct and survive refresh; original evidence remains |
| Inspect Projects | Name the checkout commit and a workflow's different commit; open the source |
| Lose GitHub access | See evidence unavailable, never a green inferred health result |
| Use a phone | Navigate all five destinations, review captures, create a job and inspect output without horizontal scrolling |
| Close Voice mid-connect | No late microphone or hidden voice connection starts |

## Verification commands

```sh
pytest local_brain/tests/ -q
ruff check local_brain/
ruff format --check local_brain/
python scripts/ruff_ratchet.py
cd dashboard
npm run lint
npm test
npm run build
npm run test:browser
```

Browser suites intercept APIs and sockets with synthetic fixtures. They verify
layout and request/response behavior without mutating live records or invoking
models. Passing them is not evidence of a real Mini run, Tony's authentication,
or live scheduling after restart.

For recipe output quality, use the opt-in local evaluation and human rubric in
[Repository job report quality](STUDIO-REPOSITORY-JOB-QUALITY.md). Passing browser
flows or matching report headings does not establish factual grounding.

## Rollout boundary

Deploy frontend and Command Center together: Projects requires the new
`GET /api/projects` and `GET /api/projects/{repo_id}/ci` routes. Back up runtime
data before the normal deployment. No data migration is required by this slice.
Do not overwrite local environment files or runtime JSON/SQLite stores.
Confirm the merged loop-safety changes are present in the deployed backend
before relying on review backpressure and no-change checks.

Rollback restores the previous application revision and dashboard build while
preserving runtime data. New jobs use the existing agent/assignment schema.

## Inbox capture boundary

Mention-only Slack events are acknowledged at the HTTP transport level but do
not create captures or enqueue a misleading saved receipt. Empty or mention-only
`/aiia-capture` commands return the existing ephemeral usage prompt. This is not
chat: there is no new thread guidance message or surrounding-thread retrieval. Real text
continues through the existing signed, allowlisted, idempotent capture path.
Existing empty captures and already-sent receipts are retained, not rewritten.
No new scopes, outbound integrations, or schema migration are required.

The operational watch cleanup is separate from this code rollout. Pause a watch
before clearing its review backlog, since clearing backpressure can make it
eligible to run again. Preserve distinct unresolved claims as queued, manual
work, and dismiss stale reports with a reason and expected review versions.
Dismissal must not accept model output, delete history, or clear the independent
Slack/proposal queues. Re-enable a watch only after a fresh, bounded test and
human review of its evidence and settings.
