# Studio Release Gate: 2026-09-11

Decision: **NO-GO for production replacement or release acceptance.**

Candidate: `codex/studio-mini-reconcile-20260911`, base `ef7d3cf`, with
uncommitted integration and reliability changes. This is not an immutable
release revision. The frozen transfer evaluation's entry conditions require
verified deployed identity and reliability contracts, not just passing tests.

## Passing Evidence

- Fresh focused regression run: **61 passed**, covering agent persistence,
  assignment lifecycle/review, workspace/write boundaries, repository tools,
  ledger recovery, and switchboard behavior.
- Prior full candidate run: 354 passed, 9 skipped, one pre-existing macOS `.m4a`
  MIME failure, reproduced on pristine main. Frontend tests/build and synthetic
  desktop/mobile checks passed as recorded in the reconciliation report.
- Read-only live checks: Brain `/health` on 8100 and Command Center `/api/health`
  on 8200 returned HTTP 200. Python listeners still own both ports.
  Initial sandbox connection failures were retried outside the sandbox; they
  were not treated as a production outage.
- Production remains on `main` at the previously inspected source revision,
  behind its local remote-tracking ref by two commits, with local policy/voice
  changes. Neither that checkout nor its processes were changed by this gate.

## Failed Contracts

All probes used synthetic records in temporary directories and the candidate's
actual AssignmentRegistry. They made no model calls or production writes.

| Gate | Reproduction | Observed failure |
| --- | --- | --- |
| Assignment persistence | Create a saved record, inject PersistenceError into atomic_write_json, create another record | Returns queued; memory has two records while disk bytes remain unchanged |
| Recovery durability | Persist running work; inject the same write failure while reconstructing the registry | Initialization succeeds with failed in memory, but running remains on disk |
| Linked retention | Set MAX_ASSIGNMENTS to 2; complete source, create handoff/target, then create a third assignment | Source disappears after reload while its handoff survives |
| Intact downstream evidence | Finish source with a 25,013-character artifact, then hand it off | Downstream context is truncated to 20,000 characters and lacks the complete artifact |

Relevant implementation: `local_brain/command_center/assignment_registry.py`.
`save()` logs and swallows PersistenceError; creation slices the assignment list
without protecting links; load uses the same best-effort save; downstream context
passes through the generic create_assignment truncation. These are distinct from
the now-fixed agent-definition persistence contract. Assignment restart recovery
exists on the happy path; its storage-failure behavior is the blocker.

## Remediation Status (same day)

All four failed contracts were fixed in the candidate and covered by
`local_brain/tests/test_assignment_reliability.py` (Codex started; Claude Code
verified and closed out). Production was not touched.

| Gate | Fix | Regression test |
| --- | --- | --- |
| Assignment persistence | Every registry mutation runs under `_durable_mutation`: one atomic write per call, exact in-memory rollback (records and list order) on `PersistenceError`, error surfaced to the API as HTTP 503 with no execution and no success event | `test_failed_mutation_restores_both_lists_and_references` (create, start, finish, handoff, delete, delete_handoff, review), `test_api_storage_failure_has_no_execution_or_success_event` |
| Recovery durability | `_recover_interrupted` is a durable mutation; if the recovery write fails, `AssignmentRegistry()` raises and the server does not start, so disk is never behind memory | `test_recovery_failure_blocks_initialization_and_preserves_disk` |
| Linked retention | Capacity eviction only removes completed/failed records that are neither a handoff target nor referenced by any handoff; with no safe candidate, creation raises `assignment_capacity_reached` before mutating. Load never truncates existing records | `test_capacity_preserves_links_and_active_work`, `test_failed_eviction_restores_exact_order`, `test_handoff_capacity_failure_leaves_no_orphans`, `test_load_never_truncates_existing_records` |
| Intact downstream evidence | Handoff creates the link and the downstream assignment in one write; downstream context carries the full artifact (`MAX_CONTEXT_LENGTH` = result cap + framing). Results over `MAX_RESULT_LENGTH` (40,000) are rejected at finish and at handoff, never clipped | `test_full_supported_artifact_survives_handoff_and_reload` (25,013 and 40,000 chars), `test_oversize_result_rejected_without_mutation` |

Verification on the candidate after the fix:

- Focused run: 31 passed (`test_assignment_reliability.py`, `test_assignment_review.py`).
- Full `local_brain` suite: 378 passed, 8 failed. The failures are not candidate
  regressions: the known macOS `.m4a` MIME assertion, plus seven
  `test_streaming_chat.py` cases that call the live production listeners on
  8100/8200 and now receive HTTP 401 because production requires auth. Those
  seven were counted as skipped in the earlier run when the sandbox could not
  reach the ports. They should gate on an opt-in env var rather than hit
  production from a unit run; tracked separately.
- `ruff check` and `ruff format --check` clean on the changed files.

Known limits, not blockers for the four contracts:

- If storage fails while an assignment is `running`, the completion write is
  rolled back and the record stays `running` until the next restart recovers
  it as `interrupted_by_restart`. The model output is still in the run ledger.
- The full handoff context is sent to Brain `/v1/chat` without a `num_ctx`
  override, so a 40,000-character artifact may exceed the task model's context
  window on the Brain side. That is an inference-layer limit and is not
  certified by this gate.

Update 16:30 CDT: items 4 and 5 were completed the same day (frozen `f9e7b7e`, coordinated backup, owner-run deploy, verified build, two-call RUN smoke passed; see `docs/STUDIO-RUN-SMOKE-2026-09-11.md`). The six-call evaluation is still not run. Original text follows.

The decision above stood as NO-GO until items 4 and 5 below were done: the
candidate is still uncommitted, no revision is frozen, and the RUN smoke test
and six-call evaluation have not run.

## Required Before Retesting

1. Make assignment and handoff mutations durable as a unit, rolling back exact
   prior state on failure. Surface errors and prevent inference after a failed
   start. Include failure injection at creation, transition, deletion, and recovery.
2. Protect active/linked records at retention limits; reject capacity when no safe
   eviction exists. Verify source, target, and handoff links after same-file reload.
3. Preserve supported artifacts intact downstream, or reject oversize data before
   mutating state. Never silently truncate work represented as a complete handoff.
4. Freeze a reviewed candidate revision and reconcile production-local changes.
   Take a coordinated registry/ledger backup before migration, not while writers
   are active. Preserve pending outbox data when planning rollback.
5. Verify the deployed build independently, then execute the bounded RUN smoke
   test. Run the frozen six-call evaluation only after entry conditions pass;
   record actual model provenance or explicitly mark it unverified.

The two-call live smoke and six-call behavior baseline are **NOT RUN**. No
deployment, restart, schedule activation, inference, merge, or push was performed.
This gate does not certify tenant isolation, GitHub access, or memory retrieval.
