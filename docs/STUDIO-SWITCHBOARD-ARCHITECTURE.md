# Agent Studio: Switchboard and Execution Architecture

Date: 2026-09-10; integrated 2026-09-11
Branch: `codex/studio-review-integration`
Base: `ef7d3cf` (remote main); incorporates PR #54 at `5592b1b`
Status: implemented in an isolated worktree; not deployed to the Mini service.

## Executive Decision

Build a local execution control plane, not another collection of agent avatars.
The graph should describe work, dependencies, permissions, and evidence. Agents
are reusable executors attached to that graph. The switchboard is the operational
view of the same records, not a separate source of truth.

Start with trustworthy activity and bounded development loops. Then establish a
durable worker and typed artifacts before enabling autonomous graph execution.
An attractive graph without reliable execution records would conceal failures.

### Product Direction Clarification

The original Supermemory and Obsidian approaches were deliberately abandoned in
favor of local memory. The local-facing memory node visualization was conceptual
design, not a functioning memory backend. Treat the missing Studio retrieval path
as unfinished implementation, not a regression in those abandoned integrations.
Do not reintroduce either dependency.

Keep run history, agent memory, and its visual graph distinct. First inventory
existing local storage, then complete one evidence-backed path: approve a finding,
store it locally with scope and provenance, retrieve it in a later assignment,
and expose its source in the graph. Until then, mark agent memory as planned
rather than available. These are next-step decisions, not implemented features.

Prioritize meaningful access to activity: needs attention, active work, and recent
outcomes should lead the next overview iteration; the heatmap supports history.
For each agent, expose what it is doing, why it started, the inputs and tools it
uses, its evidence and outputs, blockers, and the proposed next task or handoff.
The interaction is overview -> select work -> inspect evidence -> guide action.
Every graph node and edge should open actual records. Measure accepted work and
resolved blockers, not activity volume alone. Memory must be inspectable,
correctable, and excludable context rather than an invisible capability claim.

## Scan Scope and Observed State

Reviewed the Studio frontend, graph/layout integration, assignment/handoff
registries, execution routes, interval scheduler, repository/GitHub context,
approved Git operations, system-task runner, local chat path, egress policy,
existing tests, and CI configuration. This is an engineering architecture scan,
not a penetration test or a certification of tenant isolation.

The copied runtime snapshot contains:

- 23 agents and two enabled agent loops.
- 36 assignments: 34 completed and two failed, with none currently queued.
- 105 retained agent attempts imported into the new ledger; nine classify as
  failed. These counts are a snapshot, not a historical completeness claim.
- CI Signal Officer: 30-minute interval, 12/day ceiling.
- Mindmoor Delivery Watch: 60-minute interval, eight/day ceiling.

Earlier conversation references to a queued batch must not be mistaken for the
current queue. Existing runtime records were not changed during this work.

The isolated preview disables application lifespan/background schedulers and
blocks execution endpoints. GitHub and voice are disconnected there. It uses
copied local data, so its service-health indicators are not production health.
Browser-created preview agents exist only in this copy and start paused.

## What Exists and What It Actually Does

| Surface | Current implementation | Boundary |
| --- | --- | --- |
| Agent definitions | Name, mission, persona, skill labels, selected capabilities, repository, model parameters, interval settings | Skill labels are prompt configuration, not executable skill packages |
| Temperature/token cap | Studio sends both to `/v1/chat`; local API passes both to the Ollama client | The displayed model name in the older agent view is hardcoded; execution resolves the configured task-role model |
| Repository read | Bounded status, five commits, diff statistics, first 80 tracked paths, README excerpt | Not arbitrary source-file retrieval, full patch review, or test execution |
| GitHub read | Explicit GET requests through `gh api` | Credential connectivity was not revalidated; not a GitHub App integration |
| Local memory | Studio adds a sentence saying memory is available | The inspected Studio `/v1/chat` path does not retrieve memory. This capability is misleading until retrieval is wired |
| Assignments | Durable objective, context, criteria, priority, result, status | Explicit execution; not an automatically draining durable queue |
| Handoffs | Upstream artifact/context creates a downstream assignment | No general dependency evaluator, fan-out/join scheduler, or versioned graph execution |
| Agent interval loops | Bounded interval and daily allowance, serviced by Command Center | In-process scheduler; no durable claim/lease, backoff, or crash recovery |
| Studio execution lock | Serializes Studio runs through one process-local lock | Does not serialize every background service or a second Command Center process |
| Git workspaces/writes | Proposed workspaces and allowlisted local operations require approval | Does not turn an agent response into unrestricted shell or publication authority |
| System task runner | Separately scheduled maintenance/CI-related routines | Function return can count as success even when the returned report describes failures |
| Agent map | Agent nodes, layout persistence, assignment and handoff navigation | Spatial layout is not an execution graph definition |

Sources: `dashboard/src/console/`, `local_brain/command_center/server.py`,
`agent_registry.py`, `assignment_registry.py`, `repository_tools.py`,
`git_workspace_registry.py`, `git_write_registry.py`, `aiia_tasks.py`,
`studio_layout.py`, `studio_events.py`, and `local_brain/local_api.py`.

## Implemented in This Sprint

### Switchboard

- Default Studio view with 13 weeks of UTC execution activity.
- Click a date to filter the ledger; filter by agent and outcome.
- Agent lanes show 14 days of activity, current state, repository, queue count,
  configured cadence, and daily-cap/due status.
- Summary shows recorded attempts today, Studio occupancy, open assignments,
  and the configured aggregate loop ceiling. A ceiling is not actual capacity.
- Inspect a run's task, output/error, model, duration, trigger, and assignment.
- Inspect an agent, pause/enable its existing loop, edit it, or assign work.
- Footer dots open system-loop details. A recovered latest run no longer stays
  red because of an old lifetime failure ratio.
- Existing Overview, Agents, Assignments, Handoffs, and Map remain available.
- Mobile selections scroll to the inspector; the layout has no page overflow in
  the tested phone viewport.

### Local Run Ledger

`run_ledger.py` writes SQLite beside the agent registry, independent of its
12-entry recent-run cache. New attempts have unique IDs. Imports of retained
legacy attempts are idempotent; malformed legacy records are logged and skipped.
Deleting an agent does not delete its historical evidence.

Metadata-only list responses avoid sending every artifact on each refresh.
Full task/output is fetched only when a run is opened. The database is ignored
by Git and created with owner-only file permissions.

Important limitations:

- Only attempts that reach `finish_run` are persisted. A crash before that point
  is not represented as a durable interrupted attempt yet.
- The calendar means recorded completion activity, not commits, quality scores,
  or proof of useful work. Nonempty output is not an acceptance decision.
- Imported history is incomplete beyond the previously retained cache.
- SQLite is not encrypted; artifacts now persist beyond the 12-run cache.
  Retention, deletion, backups, and per-organization authorization need explicit
  policy before sensitive production use.
- Configuration metadata is recorded at finish time, not from an immutable
  launch snapshot. Versioned run configuration is part of the worker phase.

### Development Loop Recipes

All new recipes start paused and use only `Repository read`. Configuring a recipe
opens an editable agent definition; saving persists it through the existing API.

| Recipe | Cadence / ceiling | Output |
| --- | --- | --- |
| Change Radar | 4h / 3 per day | Evidence-bound change-risk report |
| Regression Planner | 8h / 2 per day | Proposed regression scenarios |
| Release Review | 12h / 1 per day | Release handoff and missing gates |
| Mindmoor Local Planner | 12h / 1 per day | One bounded local-job proposal |

These are useful planning agents within today's snapshot-only tools, not claims
that tests, commits, deployments, or cron migrations have actually happened.
They explicitly mark missing evidence and do not invent prior-run comparisons.
Do not enable every recipe just to increase the activity chart.

### CI

Added a dashboard job: `npm ci`, lint, status behavior tests, production build.
Backend regressions cover ledger persistence/import/filtering, run-detail routes,
loop enable validation, missing IDs, and task-status projection. Browser checks
exercise the isolated UI without local-model inference or production writes.

## Priority Findings

1. **Execution truth before autonomy.** Add durable start/finish/interrupted
   events and artifact acceptance. Current completion status is insufficient for
   unattended engineering work.
2. **Local memory is not implemented in this run path.** Wire bounded retrieval
   with provenance and scope, or remove/rename the capability until it works.
3. **One Mini needs one resource arbiter.** Multiple process-local schedulers can
   compete for the same model/GPU. Add a worker-level lease and shared resource
   budget before increasing recurring workloads.
4. **Local is not automatically private.** `egress.py` explicitly allows
   `xai.realtime` under the air-gap flag. The inspected `gh` subprocess path does
   not call that egress gate. Audit all connectors, subprocesses, browser calls,
   and voice routes before asserting no data leaves the machine.
5. **Shared access is not tenant isolation.** Cloudflare login alone does not
   establish per-organization authorization for agents, artifacts, repositories,
   memory, or write approvals. Carry verified actor identity into every mutation.
6. **Health needs semantic outcomes.** Distinguish task transport completion from
   discovered problems, failed tests, blocked prerequisites, and successful work.
7. **Docs have drifted.** Existing contributor guidance still describes older
   frontend/CI assumptions. Prefer actual package manifests and workflow source.

## Target Architecture

Use these first-class records, sharing IDs across grid, queue, and graph:

| Record | Required properties |
| --- | --- |
| Organization | Members, repository grants, data policy, resource allowance |
| Agent version | Specialty, persona, executable skill versions, allowed tools, model settings |
| Assignment | Objective, criteria, owner, priority, input artifacts, sensitivity |
| Run attempt | Immutable configuration, trigger, queue/start/end times, lease, outcome, resource use |
| Artifact | Type, schema/version, content hash, producer run, scope, provenance, acceptance |
| Workflow version | Typed nodes/edges, iteration limits, completion rules, budgets |
| Approval | Actor, exact action/payload hash, scope, expiry, execution outcome |

Graph node types: trigger, assignment, capability/tool, review gate, branch,
join, and bounded loop. Agents attach to assignment nodes. A handoff transfers
typed artifacts, not another growing anonymous chat transcript.

Reject unbounded cycles. Every loop needs max iterations, wall-clock timeout,
failure policy, unchanged-input suppression, and a stop condition. Every write
needs a scoped execution identity and an auditable approval rule.

## Next Sprint, in Order

1. **Durable local worker.** Persist queued/running/completed/failed/blocked/
   interrupted transitions; atomically claim a run; reconcile expired leases;
   snapshot configuration before inference; count attempts separately from
   completed work. Test process termination and restart without double execution.
2. **Assignment-backed recurrence.** A schedule creates a deduplicated assignment
   with trigger/window identity. Do not emit identical low-signal reports when
   the repository/input hash is unchanged. Manual work has explicit priority;
   schedules have backpressure, daily budgets, and error backoff.
3. **Bounded engineering capabilities.** Add allowlisted file reads and diff
   retrieval; then a sandboxed test adapter with timeout, output cap, exit code,
   and artifact capture. Repo tests are executable untrusted code, even locally.
4. **Reviewable artifact chain.** Change Radar -> Regression Planner -> test
   execution -> reviewer -> human release gate. Reuse assignment/handoff records
   and approved worktrees; do not add autonomous push/deploy as a shortcut.
5. **Organization and privacy policy.** Verified membership and resource grants;
   local-only capability profile that denies remote connectors; scoped memory;
   secret filtering; artifact retention/export rules. GitHub App credentials stay
   in the service, never in agent prompts. Read-only is the default.
6. **Versioned graph editor.** Start with validated DAGs plus an explicit bounded
   loop node. Show why each node is queued, blocked, awaiting review, or complete.
   Add graph execution only after the worker contract is tested.

Acceptance demonstration: a local repository change creates exactly one bounded
assignment, produces an evidence-linked artifact, routes it to a reviewer,
survives worker restart without duplicate execution, and appears consistently in
the switchboard, assignment history, and graph. A denied network/write attempt
must appear as a policy decision, not disappear into an empty response.

## Deployment and Validation Boundary

Production source had existing local edits and runtime data. This implementation
was isolated rather than pulling over those changes or restarting live services.
No new production schedule was enabled and no production assignment was run.

Before rollout: review this branch, back up runtime JSON and any existing SQLite
ledger, reconcile production-local changes, build the dashboard, and restart only
the intended Command Center instance. Confirm one scheduler instance, correct
Cloudflare authorization, and expected history import. Keep the old cache during
rollback; rolling back the UI must not delete the new ledger.

This branch is a visibility foundation. It is deliberately not labeled a full
workflow engine, a CI runner, multi-tenant isolation, or a complete air gap.

### Verification Results

- Eight focused backend tests passed, including mocked parameter forwarding.
- Three frontend status tests passed.
- Production dashboard build passed with the preview banner disabled.
- Ruff passed for all touched Python modules; `git diff --check` passed.
- ESLint had no errors and one pre-existing ErrorBoundary directive warning.
- Desktop 1440x1000 and mobile 390x844 browser checks passed: agent search,
  pause/resume, assignment navigation, date/outcome filters, artifact detail,
  footer drilldown, recipe configuration, paused creation, and mobile inspection.
- No browser page errors or horizontal page overflow occurred in those checks.
- Production assignment-data checksum remained unchanged.
- Full backend suite, real model inference, production deployment, remote user
  access, GitHub authentication, and tenant isolation were not verified here.
- The local test environment warns that `pytest-asyncio` is absent and reports
  existing FastAPI lifecycle deprecations; these focused tests use `asyncio.run`.


## Review follow-up: ledger recovery

Completed runs now enter an atomic JSON outbox (`pending_runs` in the existing
agent registry file) before insertion into SQLite. The outbox stores the same
run ID and finish-time agent metadata. It is separate from the 12-run cache and
survives agent deletion. Startup, subsequent completions and history reads retry
only storage. Successful replay removes the pending entries; repeated replay
uses the ledger's existing ID constraint and does not invoke inference.

While SQLite is unavailable, a successfully saved assignment keeps its completed
output and the agent reloads as idle (or error for a failed model response).
History reads return 503 with an explicit recovery message. The calendar reports
unavailable history instead of zero recorded activity. Repo-less lanes now say
"No repository"; they make no memory-retrieval claim.

If the atomic registry write itself fails, the API reports
`run_output_persistence_failed` (503), retains the in-memory pending result, and
does not promise durability across process loss. Failure before the result is
saved, the separate assignment-write crash window, multi-process coordination,
tenant scoping and power-loss guarantees remain outside this correction. This
is recovery of completed output, not the proposed durable worker.

Back up **both** the complete registry JSON and SQLite ledger before rollback.
Older registry writers discard the new `pending_runs` key on save. Do not run an
older writer against outstanding pending entries: drain recovery first or retain
a separate untouched backup for replay with this version.

Regression coverage includes a successful mocked assignment followed by a
SQLite failure, reload of the same fixture, exactly-once ledger recovery without
another model call, 20 pending runs surviving cache eviction and agent deletion,
SQLite-open failure at restart, cleanup-write failure after successful insertion,
and explicit storage failure responses for manual and assignment execution.
The user/tenant authorization boundary remains unresolved; no shared rollout or
security-architecture change is included here.

## Artifact review and needs attention

Assignment execution status remains queued/running/completed/failed. Completed
outputs separately carry `review_status` (unreviewed/accepted/rejected),
`review_note` (up to 2,000 characters), `reviewed_at`, and `review_version`.
Legacy outputs default to unreviewed. Assignment details show the objective,
success criteria, context and work product before accept/reject/reopen controls.
The switchboard lists failed/interrupted work first, rejected outputs next, then
unreviewed or missing outputs; within each group, priority and oldest creation
break ties. Accepted outputs leave that list. The selected agent scopes the
list; execution date/outcome filters apply only to the run ledger.

`POST /api/assignments/{id}/review` accepts:

```json
{
  "decision": "rejected",
  "expected_version": "the review_version returned with the displayed assignment",
  "note": "The conclusion is not supported by the supplied evidence."
}
```

The response contains the updated assignment. Unknown IDs return 404; malformed
requests return 422. Running, failed or empty outputs return 409, as does an old
version (`review_changed_refresh_required`). Every execution and review changes
the version, including a rerun producing identical text. A new execution clears
the prior decision. Review writes atomically replace the assignment JSON;
failure returns 503 (`review_persistence_failed`) and restores the prior
in-memory decision. Other assignment writes now use atomic replacement too,
but retain their existing best-effort error handling.

Review is a manual quality annotation in the existing local shared workspace.
It does not claim verified reviewer identity, tenant isolation, a full review
audit trail, or cross-process concurrency control. It neither invokes an agent
nor approves a Git write; existing handoff eligibility is unchanged. Historical
run snapshots keep their execution outcome and do not inherit a later review
of the assignment's current output. No new SQL query, cloud call, schedule,
Voice action, or airgap exception is introduced.

Focused checks: 36 backend tests, five frontend tests, TypeScript/Vite build,
Ruff and ESLint (one existing ErrorBoundary warning). Synthetic rendered checks
at 1280x900 and 472x797 cover needs-attention navigation, accepting correct
output, rejecting incorrect output, reopening review, note persistence through
reload of the same JSON fixture, and save-failure feedback. Screenshots were
inspected in the task. Production inference and multi-user authorization were
not exercised. Keep the complete assignment JSON in rollback backups; older
execution code will not invalidate these review fields correctly, so do not
mix old writers with the new review UI.

## Integration verification — 2026-09-11

The combined switchboard, history recovery and artifact review change now applies
on remote main (`ef7d3cf`). Both additive conflicts are resolved: agent suite
defaults remain alongside the ledger imports, and `/api/agent-suites` remains
alongside the Studio routes. The separate, unpublished reliability commit on
local main (`381add1`) is not included. Preserve it for its own integration;
this branch does not claim its retention, handoff or definition-save guarantees.

Validation in a temporary Python 3.11 environment using the declared package and
dev dependencies:

- All 44 focused Studio and suite tests pass.
- Full backend suite: 338 passed, nine skipped, one failed. The failure is
  `test_watcher_content_type_for_m4a`: this host resolves `.m4a` as
  `audio/mp4a-latm`, outside the test's two expected MIME types. The same isolated
  test fails on a pristine copy of remote main. Journal code is unchanged.
- Repository-wide Ruff lint and formatting checks pass (156 Python files).
- Dashboard lint has zero errors and one existing ErrorBoundary warning; all
  five behavior tests and the TypeScript/Vite production build pass.
- Sanitization guard and whitespace checks pass. No runtime data is included.

The frontend source matches the already rendered integrated preview; integration
adds backend formatting and these documentation updates. No new inference or
live mutation was needed. This is a local integration, not a deployment or a
claim that remote CI passed. The existing shared-access/tenant boundary remains
unresolved and must be settled before shared rollout.
