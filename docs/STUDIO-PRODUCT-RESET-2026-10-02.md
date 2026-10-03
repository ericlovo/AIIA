# AIIA Studio: jobs first

Status: proposed product reset, grounded in live reads on October 2, 2026.
Weekend target: October 3-4, America/Chicago.
Primary users: Eric and Tony, working together on Mindmoor and Performance Labs.

## The product promise

AIIA watches selected work, does bounded preparation on the Mini, and brings back
evidence and a clear next action. A person can schedule a useful job, understand
its result, and stop it without learning the agent framework.

The release gate is a human walkthrough: both Tony and Eric complete the core
tasks without developer coaching. Passing automated tests is necessary but does
not establish that the product makes sense.

## What is actually there

Source snapshot: AIIA `origin/main` at `f12b29b`; live checkout `3c2d3b1`.
PR #96 is merged but not present in the live checkout. PR #95 is open with green
checks. This document and the accompanying interaction sketch are proposals,
not a deployment.

Live reads returned:

- Brain and Ollama online; optional default/platform services offline.
- 24 agents; two enabled Studio loops, both waiting for review. Their latest
  execution timestamps are September 24.
- 131 assignments: 128 completed, one queued, two failed. Completion does not
  mean accepted by a person.
- 80 pending scheduled reviews are historical, created September 21-24.
- Twelve built-in tasks expose their own last/next runs, separately from agents.
- Public news/lead retrieval and Jev screening are disabled; credential present;
  both discovery schedules disabled.

The durable assignment/run history, review semantics, repository mounts,
local inference and source-based change checks are useful foundations to retain.
The user experience currently exposes those implementation pieces as separate
places to learn.

## Why another visual polish pass is insufficient

1. There is no single place to create and manage a recurring job. Agent loops,
   built-in tasks and discovery jobs have different controls and schedules.
2. Today combines action queues, quality metrics, usage, a heatmap, fleet lanes,
   recipes and a run ledger. It asks the user to interpret the system before acting.
3. Queuing an assignment does not start a general background worker. A user can
   reasonably expect progress while the item is waiting for a manual start.
4. The default CI Monitor targets AIIA only and can report task success after an
   API retrieval error. A green task execution is not proof that Mindmoor CI passes.
5. The repository snapshot and GitHub summaries lack enough revision-specific
   evidence for reliable CI failure investigation.
6. Studio serialization excludes other callers of the local model. The one-slot
   display is not a machine-wide workload queue.

## Four everyday destinations

| Destination | User question | Primary action | Content kept out of the default view |
| --- | --- | --- | --- |
| Today | What needs me, and what will happen next? | Open a decision or create a job | Fleet configuration, token charts, heatmap, raw transcript |
| Jobs | What recurring work have we set up? | New job; test; enable/pause | Model temperature, personas, raw cron expressions |
| Work | What did it do and is the result useful? | Inspect evidence; accept; request changes | Separate handoff ledger and raw execution metadata |
| Projects | Is Mindmoor healthy and what changed? | Open a failing check or deployment | Global ambiguous health lamps |

Agents, the graph, memory administration, connection setup and usage remain
available under Studio/settings or relevant detail views. A project selector
scopes everyday views consistently. Advanced pages keep working deep links.
The map becomes a debugging view of real jobs and handoffs after the job model
is reliable; it is not the first thing Tony must configure.

Today has three sections: needs your decision, running/upcoming jobs, recent
meaningful changes. Every row says what happened, which project, why it matters,
and one next action. Historical backlog is grouped with its age and an explicit
review action. Grouping does not accept, dismiss or hide records from totals.

## The core journey

1. New job: choose a supported recipe and project.
2. Choose cadence, delivery location and a bounded execution policy.
3. Show a plain summary: what it reads, where computation happens, what it may
   change, when it runs next and what stops it.
4. Test once, inspect the resulting evidence, then enable the schedule.
5. See queued/running/quiet/result-ready/blocked states in one place.
6. Open the result in Work. Accept, request changes or explicitly close it.
7. Pause future scheduling without losing history. If work is already running,
   show that it continues; do not imply pause cancels it.

The first recipes should be deliberately narrow:

| Recipe | Output | Execution |
| --- | --- | --- |
| Watch Mindmoor CI | New failed/recovered check with revision and link | Deterministic GitHub read; optional Mini explanation |
| Morning delivery brief | Source-linked changes, open blockers and next actions | Deterministic evidence assembly + bounded Mini summary |
| Research a repository question | Evidence packet answering one question | Read-only revision-pinned retrieval + Mini draft |
| Review market signals | A short list of evidence-backed candidates | Approved public retrieval + Jev screening + optional Mini summary |

Start with Mindmoor CI and one manual research task. Prove those journeys before
adding more agent types or increasing schedule frequency.

## Scheduling and execution contract

Use one Jobs view over existing runners first, with explicit source capabilities.
Do not add an independent fourth scheduler or silently duplicate an existing job.
Existing system jobs that cannot be paused should show that capability honestly.

Current agent loops support intervals (15 minutes to 24 hours) and a UTC daily
cap (1-48 runs); they do not support arbitrary wall-clock cron schedules. Public
discovery has fixed 12-hour intervals. A weekday 9 AM picker in the design sketch
is a proposed capability, not an available setting.

New schedule work must persist timezone and next due time, produce a readable
preview of the next three occurrences, and define restart/missed-run behavior.
For watches, coalesce missed occurrences into at most one fresh check. Use an
existing maintained scheduling library after checking current compatibility.
Deduplicate by job plus occurrence, including daylight-saving boundaries.

Execution states must separate:

- Draft/paused: no future execution.
- Scheduled: next eligible time and timezone visible.
- Queued: accepted by a durable worker, with reason for waiting.
- Awaiting manual start: existing assignment behavior, clearly labeled.
- Running: current stage, start time and execution location.
- No change: successful evidence check; history only, no new review obligation.
- Ready for review: a result requiring a decision.
- Blocked: missing connection, review cap, daily cap or unavailable executor.
- Failed: unsuccessful attempt; evidence, retry conditions and prior output retained.

Do not relabel manually queued assignments as auto-running. An explicit new
"Queue and run locally" action can authorize supported jobs to enter a durable
worker queue. Existing approval gates still govern writes and publication.

## Division of work

| Owner | Good use | Boundary |
| --- | --- | --- |
| Deterministic code | Fetching, timestamps, Git/CI state, deduplication, schedule math, budgets, tests, permissions | No model needed to determine a known check conclusion |
| Jev | Relevance classification, bounded routing, rubric scores over permitted text | External API; typed decisions, not retrieval, prose generation or coding |
| Mini / Ollama | Private memory/retrieval, bounded log summaries, extraction, first drafts and briefs | Limited context and shared memory/compute; output needs evidence and quality checks |
| Codex / human review | Architecture, ambiguous debugging, implementation, test design, quality judgment | Explicit development work; do not claim Studio has an automatic Codex worker |
| Eric / Tony | Accept findings, choose priorities, approve consequential actions | Clear decisions, not routine unchanged checks |

Existing MCP offload/digest tools explicitly call the local model; they do not
automatically intercept frontier-agent work. Local tokens and model duration are
measurable. Avoided cloud cost is not yet measured, and zero API inference fee
does not imply zero operating cost.

Jev currently supports an opt-in agent-routing suggestion and public headline
screening. The public pipeline reads bounded RSS headline data; it is not yet
primary-source verification, company enrichment or qualified contact research.
Six pending items means unassigned/unreviewed discovery inbox items, not all
outstanding research assignments. The UI must not imply a stronger cap.

For lead discovery, retain source and publication/observation dates, establish
company identity, verify the actual change, then apply a shared fit rubric.
Eric and Tony should label a small fixed evidence set; compare Jev decisions to
those judgments before automating promotion. Uncertainty routes to review.

TypeSafe's current guidance confirms these model roles:
[System One](https://docs.typesafe.ai/concepts/system-one) and
[Jev with coding agents](https://docs.typesafe.ai/introduction/coding-agents).

## Make M4 offloading provable

All model calls should eventually use one resource arbiter with durable queued
state, bounded input/output, priorities, timeout and restart recovery. Cover
Studio, MCP and inference-using background tasks. Keep collection of deterministic
health evidence available while an inference slot is occupied.

Before increasing automation, measure a fixed small set: CI log digest,
revision summary, repository question and memory retrieval. Record input/output
tokens, queue wait, execution duration, model, truncation, retry and Eric/Tony's
accept/reject decision. Record actual hardware utilization where available.
Report observed numbers; do not invent throughput, savings or quality scores.

## Trust prerequisites

- CI state must include repository, branch, SHA, workflow/run ID, source URL,
  conclusion and fetched-at time. Distinguish queued, running, failed, passing
  and unavailable. Expose stale evidence rather than silently showing green.
- Deployment state comes from the deployment provider, separately from CI.
  A merged commit is not proof that Vercel deployed it. Show unknown until connected.
- Pin research to the chosen revision. The mounted Mindmoor checkout is on an
  older branch than remote main; never label checkout context as current production.
- Document exactly which connections send data off the Mini. Current airgap is
  an application policy with exceptions, not physical network isolation.
- Do not use the present PII endpoint as a release gate for sensitive data:
  malformed model JSON currently becomes a negative finding. Make detection
  failure explicit before relying on it for any outbound decision.
- Before expanding automatic writes, make approval persistence failures explicit:
  the legacy action queue currently logs save errors without failing the operation.
- Confirm Tony's real login and shared-workspace visibility. The current Studio
  is a shared workspace, not proof of provisioned per-user tenancy or actor audit.

## Weekend delivery order

| Slice | Concrete outcome | Acceptance | Dependency |
| --- | --- | --- | --- |
| 0. Baseline | Reconcile #95/#96, known runtime, preserve existing data and URLs; prepare an isolated QA dataset | Running revision and connections visible; rollback rehearsed | None |
| 1. Jobs-first shell | Today/Jobs/Work/Projects; compact voice/status; old routes preserved; job rows from existing runners | Both users find next action, pauseable job and result without opening agent config | Baseline |
| 2. Trustworthy Mindmoor delivery | Repo/branch/SHA-specific GitHub evidence, source links, stale/error handling; useful investigation packet | A real failure is distinguishable from a fetch failure; investigation uses the same revision | Evidence adapter can run alongside shell |
| 3. One complete job | Create from CI-watch recipe, test, enable/pause, see next occurrence and result | Saved schedule survives restart; duplicate triggers create one run; no-change produces no review item | Jobs shell + evidence |
| 4. Work and Mini queue | Distinguish manual start from queued execution; review in one detail view; source-to-result trace | Busy Mini does not lose accepted work; review survives refresh; handoff preserves evidence | Job lifecycle |
| 5. QA and measured pilot | Eric/Tony walkthrough, repeat after fixes; bounded overnight watch; lead evidence pilot if ready | Core acceptance sheet passes; useful results and issues recorded | First four slices |

The weekend success target is the CI-watch -> investigation -> review flow, plus
one reliable scheduled local brief if capacity permits. Calendar scheduling,
machine-wide inference arbitration and lead qualification are substantial backend
work; finish their contracts and first bounded implementations before promising
all of them by Sunday. Each slice should leave a working checkpoint.

Suggested parallel work: orchestration owns contracts/integration; UX worker
owns the shell and task flows; evidence worker owns GitHub/project status; runtime
worker owns job scheduling and queue behavior; independent QA worker runs the
acceptance script. Give each worker disjoint files and integrate after contract review.

## Human acceptance sheet

Run in a separate QA workspace with realistic seeded history, failures, queued
work and disabled connections. Then run a bounded real job on the Mini. The
interactive design sketch uses example data and cannot pass these checks.

| Task | Pass condition | Evidence |
| --- | --- | --- |
| First arrival | In 30 seconds, each user identifies what needs attention, what's running and the next scheduled job | Observed separately, no coaching |
| Schedule a watch | In 3 minutes, create/test/enable a Mindmoor watch, identify next run and pause it | Real saved config and scheduler observation |
| Check delivery | In 60 seconds, distinguish CI, deployment and agent status; open original run | Same repo/branch/SHA in app and source |
| Request work | State a task, know whether it needs manual start, run it and reopen it later | Same durable work ID after refresh/restart |
| Review a result | Find evidence, accept or request changes, see the saved decision | Review event and reopened UI agree |
| Understand execution | Each can state what ran on Mini, what used Jev and what data was sent | UI matches actual calls and usage |
| Recover a failure | With unavailable GitHub or Ollama, see the real failure and a useful recovery action | No false success, duplicate work or lost result |
| Share work | Tony opens Eric's work link and sees the same state after his own login | Two real user sessions; local automation is insufficient proof |
| Mobile/keyboard | Complete watch pause and result review at 390px and with keyboard only | No clipped actions, hidden evidence or focus traps |

Record: participant, task, completion time, assistance needed, wrong turns,
misinterpretation, severity and evidence link. Both users must pass the core
watch/investigate/review tasks without coaching before calling the redo accepted.

## Drift to reconcile with the implementation

README still advertises v0.6.0 while the code/package is v0.7.0. The September 26
UX sprint still says PRs 1-3 are in review and describes planned navigation that
does not match current routes. Changelog coverage does not yet explain the lead
queue/timeline flow. Update those surfaces as the corresponding changes ship;
this plan supersedes the earlier sprint's navigation target, not its safety tests.

Defer an open-ended graph builder, more personas, autonomous outreach, generalized
coding swarms and unmeasured high-frequency loops until the simple journeys pass.

## Code evidence

- `dashboard/src/console/StudioNav.tsx`, `Console.tsx`, `Switchboard.tsx`:
  navigation, persistent footer and overloaded overview.
- `dashboard/src/console/WorkBoard.tsx`, `AgentStudio.tsx`: independent create/run
  actions and configuration-first agent creation.
- `local_brain/command_center/server.py`, `agent_registry.py`: Studio execution
  lock, loop intervals, pending-review guard and unchanged-input checks.
- `local_brain/command_center/aiia_tasks.py`: independent built-in task runner and
  AIIA-only CI monitor.
- `local_brain/command_center/repository_tools.py`: snapshot contents and bounded
  GitHub summaries.
- `local_brain/command_center/typesafe_advisor.py`, `public_signals.py`:
  advisory routing and public headline screening.
- `local_brain/mcp_server.py`: explicit offload/digest tools and input truncation.
- `local_brain/local_api.py`, `egress.py`, `command_center/action_queue.py`:
  PII error handling, outbound policy and legacy persistence boundary.
- Mindmoor `origin/main` at `be99f01`: `.github/workflows/ci.yml`, `vercel.json`,
  `scripts/sync.sh`. Live Vercel deployment state was not inspected.
