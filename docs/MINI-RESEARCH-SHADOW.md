# MINI-01: replay research runner

## Scope and verified baseline

Prepared October 22, 2026 against AIIA main `1e1c0a105adbeea8196b332c0d01ad89c2d0ef26`.
This slice prepares synthetic replay assignments and a mocked staging boundary.
It does not activate research, deploy to the Mini, retrieve URLs, call Jev, run
Qwen, train a model, or approve a candidate. No production server startup hook or
background scheduler is added. Existing Studio records and runtime stores are
not reused for these fixtures.

The local Qwen evaluation harness is
`local_brain/scripts/evaluate_job_recipes.py`; its configured `qwen3:8b` is a
source-code default, not evidence of the Mini's installed or running model.
`local_brain/config.py` also has role-specific model defaults and environment
overrides. The Mini's model identifier, digest, hardware, context, loaded model,
running revision and preservation state remain **unverified here**.

Verified with `git fetch origin`, `git rev-parse HEAD`, inspection of the harness,
configuration, `.github/workflows/` and `scripts/mini-deploy.sh`. The checked-in
Mini updater polls main; feature branches do not trigger that script. Unknown
host-local automation is not independently verified. No merge or deploy is part
of this handoff.

## Backend contract

- `research_shadow.py`: versioned synthetic assignment, bounded replay input,
  deterministic evidence checks, replay-only review receipt and mock staging.
- `research_shadow_store.py`: explicit-path SQLite job state, tenant-scoped
  queries, fenced expiring leases, bounded retries and durable checkpoints.
- `research_shadow_routes.py`: opt-in `/api/research-shadow` router factory for
  status/configuration and pause/cancel/kill controls. It is not mounted by
  `server.py`. The caller must inject authenticated operator authorization and
  a server-bound synthetic tenant. A tenant prefix is a replay guard, not auth.

The accepted tenant namespace is `synthetic-*`. Config includes a disabled
schedule, model reference `replay:saved-extraction`, source/geography/taxonomy
scope, versioned policy, budgets and retention metadata. Secret references are
identifiers only; no credential is needed or resolved. The cockpit projects only
safe configuration fields and does not return source text or worker lease tokens.

The staging boundary is a mock. Managed Leads must implement and review its
own tenant-bound staging permission, idempotency contract, revision checks and
RLS tests before any real submission adapter can be enabled. Existing account
read credentials, administrator cookies and database-owner credentials are not
substitutes. Research receipts never confer approval or outreach authority.

Exact evidence spans establish text provenance, not semantic truth. Replayed
judgments are recorded policy outputs, not calibrated probabilities or live
quality measurements. Malformed/unsupported results must abstain or be withheld.
No measured local inference speed, provider cost or useful-lead rate is claimed.
Each job handles exactly one saved page; max_pages is an upper bound, not a
multi-page crawler. Token/spend limits gate recorded usage; real provider resource
reservation and a global Mini inference lock are activation dependencies.
Retention configuration is a contract for future activation; this slice does
not schedule deletion of stores. Use temporary synthetic state and remove it
only through the operator's normal test-artifact procedure.

## Reproduce offline

From the isolated checkout, using its Python environment:

```sh
pytest local_brain/tests/test_research_shadow*.py -q
python -m local_brain.scripts.replay_research_shadow
```

The example creates and removes its own temporary database. It demonstrates a
completed mock receipt, reopening the database and deduplicating the same job.
It has no service credentials, network transport or real inference.

## Local verification

October 22: 49 shadow store/adapter/API tests plus 22 existing public-signal
regressions passed (71 total). The disposable replay demo completed and verified
restart deduplication. Ruff lint/format, the ignored-rule ratchet and the
case-collision guard passed. The ratchet used the installed Ruff executable
through its injectable runner because this Python lacks the Ruff module. These
checks do not establish full-suite CI, UI behavior or live Mini readiness.

## Mini transfer prompt

Paste into the separate Mini Codex session after reviewing the implementation:

> Prepare MINI-01 in an isolated checkout and temporary synthetic state directory.
> Read the actual checkout's AGENTS.md first. Do not update the live checkout,
> restart services, alter auto-deploy, change schedules/models/airgap/egress,
> download models, access memory, or modify runtime snapshots. Inspect the actual
> service manager and process owners read-only; record the deployed SHA and
> service health without printing credentials or private records. Read hardware
> information locally and inspect the installed and loaded Ollama model inventory
> using its confirmed local endpoint. Inspect the selected model's digest,
> quantization, architecture and context metadata, plus runtime context settings;
> do not infer active context from the model's maximum. Do not invoke inference.
> Report all unavailable facts as unverified. Check the prior #114 backup manifest
> and ignored/untracked monitor snapshot status without printing snapshot contents.
> Do not stop production writers for this replay exercise.
>
> In the isolated checkout, run the research-shadow test suite and the documented
> synthetic replay example using a new temporary database. Do not point the store
> at an existing Studio database. Verify duplicate/restart recovery, cancellation,
> budget limits, tenant separation, malformed recorded judgments and injection
> fixtures. Confirm receipts say replay-only and correction-needed and that no
> network submission occurs. Do not mount the new router in the live server.
>
> Return the tested commit, exact commands/results, runtime/model facts with their
> sources, preservation status and dependencies. Stop before live provider
> wiring, production sources, scheduling, deployment or merge. Do not use an
> admin cookie or owner database credentials. Human approval remains separate.

## Activation dependencies and blind spots

1. **VERIFIED:** replay code has no provider transport; results are synthetic.
2. **LANDMINE:** checked-in model defaults do not identify the running Mini model.
3. **LANDMINE:** an ignored file previously tracked by Git can be removed on
   update; keep the existing #114 external backup and recovery procedure.
4. **UNVERIFIED:** actual Mini runtime, backup and deployment health require the
   separate session above; no delivery to that machine is claimed.
5. **DEPENDENCY:** Managed Leads write-service authentication and staging contract.
6. **DECIDE:** approved source scope, egress, measured budgets, model/provider
   integration, retention enforcement and activation belong to the owner.
7. **DEPENDENCY:** JEV-01 policy adapter and regression/holdout evidence. Replay
   records are not an implementation or evaluation of that team's live adapter.
8. **OUT OF SCOPE:** product-request-to-PR automation and all merge/deploy authority.
