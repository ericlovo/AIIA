# Studio Reconciliation: 2026-09-11

Status: integration candidate validated; **not approved for live replacement**.

Release gate outcome: **NO-GO**. Additional temporary-registry probes confirmed
assignment persistence, recovery durability, linked retention, and downstream
artifact truncation failures. See [release gate](STUDIO-RELEASE-GATE-2026-09-11.md)
for evidence and the requirements to reopen rollout.

## Source and Scope

- Base: `ef7d3cf`, current `origin/main` at inspection.
- Candidate branch: `codex/studio-mini-reconcile-20260911`.
- Transfer: `studio-mini-handoff-2026-09-11.zip`, all five SHA-256 checks passed.
- Ordered patches: integration `70f51c2`, then RUN controls `9c5ca30`.
- Both patches applied cleanly on current main, without overwriting the live
  checkout or its runtime files.
- Original switchboard `5592b1b` is included through the transfer integration,
  not cherry-picked again. Suite fields and their frontend edit preservation
  are present, along with the current-main catalog and tests.
- Owner direction remains local memory, meaningful access to activity, and
  inspectable evidence. Namespace labels do not prove retrieval or isolation.

This slice reconciles source and validates an integration candidate. It does
not install the candidate, activate schedules, invoke models, expand egress,
or certify shared access.

## Installed Build Findings

Both production listeners belong to the original checkout, whose Git HEAD is
`fdbf863` with local egress/voice edits. The processes started on September 8 and
are children of the `com.aiia.brain` launchd job. The dashboard JavaScript and
CSS bytes served on port 8200 match that checkout's `dashboard/dist` files.

This proves the asset-directory mapping, not the exact build revision of those
assets. The processes do not expose an immutable source/build revision. A later
release should embed revision information rather than infer it from cwd.

Read-only health checks returned HTTP 200: Brain online, air-gap enabled, and
Voice connected with the existing `xai.realtime` exception. Current main already
contains the corresponding official allowlist implementation. Preserve it;
do not blindly reapply the older local patch on top.

The configured task model in the installed `.env` is `qwen3:8b`. No inference
was performed, so a per-call effective model remains unverified.

Registry paths are module-relative. The installed agent, assignment, layout,
workspace, and write registries remain in the original checkout. No installed
SQLite run ledger was found during inventory. Moving the service's import path
without transferring runtime state would select a different set of records.

A private rollback archive contains the installed dashboard, those registries,
and the two locally edited policy/voice source files. Archived JSON parsed
successfully. This is a per-file live snapshot, not a cross-file transaction;
take a fresh coordinated snapshot immediately before a future migration.
Machine paths, hashes, and process IDs are in the local inventory, not this repo.

## Fresh Validation

- Isolated Python 3.11 environment with repository-declared dev dependencies.
  No packages installed into the live service environment.
- Full backend suite: **338 passed, 9 skipped, 1 failed**.
- Failure: `test_journal_pipeline.py::test_watcher_content_type_for_m4a` receives
  `audio/mp4a-latm` on this host. The same test fails identically on a pristine
  `ef7d3cf` snapshot in the same environment. Not changed in this slice.
- Ruff: all checks pass; all 156 Python files meet formatting checks.
- Frontend: five tests pass; production build passes; ESLint has zero errors
  and one existing unused directive warning in ErrorBoundary.
- Fresh synthetic browser checks pass at 1280x800 and 390x844: recorded-task
  rerun, duplicate-click guard, disabled pending controls, refreshed output,
  visible HTTP 409 feedback, no mutation retry, and reload against the same mock
  records. Screenshots were inspected; no page errors or horizontal page overflow
  occurred. Two intercepted POSTs were made, with zero backend/model calls.
  This verifies frontend behavior, not live execution or disk persistence.
- `npm ci` reported 11 dependency findings (one low, two moderate, eight high).
  No broad dependency upgrade or audit fix was attempted.

The installed Python 3.12 invocation encountered an Expat symbol import error
while reading plist data. Validation used a separate Python 3.11 environment;
this observation is not a diagnosis that the already-running service is broken.

## Reliability Gaps Found During Reconciliation

The separate reliability commit `381add1` is absent from local objects and is
not supplied by the transfer. Do not claim the frozen evaluation's entry
conditions are satisfied merely because the integration tests pass.

Two temporary-registry probes establish concrete gaps:

1. **Restart recovery:** persist a running loop agent, construct a new registry
   against the same file, and check status/eligibility. It remains `running`
   and is not eligible. No live process was stopped to demonstrate this.
2. **Definition-save failure:** inject `PersistenceError` during an update.
   The caller receives no exception, in-memory configuration changes, and disk
   remains unchanged. Atomic file replacement alone does not prevent a false
   success response or roll back in-memory state.

These were inherited gaps, not introduced by the RUN-button patch. The follow-up
sprint below addresses both in the candidate. Artifact outbox recovery addresses
a different failure mode and remains separately tested.

## Reliability Sprint Follow-Up

- Definition changes now either persist atomically or restore the prior in-memory
  definitions and return HTTP 503. Existing agent object references are preserved.
  Save, remove, and manual-run failures are visible in Agent Studio.
- Run status and scheduled-run accounting persist in one mutation before model
  invocation. A failed start produces neither inference nor a running event.
- Startup changes persisted `running` agents to `error`, records
  `interrupted_agent_run; review before resuming`, and disables their loops.
  Prior results, history, counters, and suite configuration remain intact.
  No synthetic run or automatic inference is created. If recovery cannot persist,
  registry initialization fails instead of starting the scheduler with stale state.
- Operators must review interrupted work before a manual run or explicitly
  re-enabling its loop. Existing interval and daily-cap rules still apply.
- Completed-output outbox retention is intentionally unchanged: it must preserve
  recoverable output rather than use definition-mutation rollback semantics.

Validation after these changes:

- Backend: **354 passed, 9 skipped, 1 failed**. The sole failure remains the
  baseline macOS `.m4a` MIME mismatch above. Sixteen new regression cases cover
  write failures, validation rollback, restart recovery, and API responses.
- Ruff checks pass; all 157 Python files meet formatting checks.
- Frontend: five tests and production build pass; lint has zero errors and the
  same existing ErrorBoundary warning.
- Synthetic browser checks at 1440x1000 and 390x844 verify visible save, delete,
  and run-start storage errors, retained form input, and interrupted-agent state.
  Screenshots were inspected with no page errors or horizontal page overflow.
  All three mutation requests were intercepted; no live API or model was called.

This is a single-service-writer recovery contract, not a machine-wide lease.
Interrupted assignments and a durable started-attempt ledger lifecycle remain
separate work. These fixes do not establish parity with absent commit `381add1`,
tenant isolation, or working local-memory retrieval.

## Remaining Gates

1. Compare retention, assignment recovery, and handoff semantics separately;
   do not assume the missing reliability commit is fully represented here.
2. Identify and back up the live installation again before a deliberate rollout,
   preserving runtime registries and local policy/voice edits.
3. Perform the bounded two-call RUN smoke test after controlled deployment.
4. Only after entry conditions pass, run the frozen six-call behavior baseline.

No merge, remote push, deployment, live inference, or schedule change was
performed during reconciliation or the reliability follow-up.
