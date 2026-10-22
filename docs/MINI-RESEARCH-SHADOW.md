# MINI-01: replay research runner

## Scope and verified baseline

Prepared October 22, 2026 against AIIA main `1e1c0a105adbeea8196b332c0d01ad89c2d0ef26`.
The default backend prepares synthetic replay assignments and an offline staging
mock. A separate explicit operator utility can submit a previously saved shadow
envelope with a private runner credential file. It is not mounted or scheduled.
Neither path retrieves sources, calls Jev, trains a model or approves a candidate. No production server startup hook or
background scheduler is added. Existing Studio records and runtime stores are
not reused for these fixtures.

The local Qwen evaluation harness is
`local_brain/scripts/evaluate_job_recipes.py`; its configured `qwen3:8b` is a
source-code default, not evidence of the Mini's installed or running model.
`local_brain/config.py` also has role-specific model defaults and environment
overrides. A later read-only Mini inspection and separately authorized synthetic local
inference are recorded below. The original preparation did not verify runtime.

Verified with `git fetch origin`, `git rev-parse HEAD`, inspection of the harness,
configuration, `.github/workflows/` and `scripts/mini-deploy.sh`. The checked-in
Mini updater polls main; feature branches do not trigger that script. Unknown
host-local automation is not independently verified. No merge or Mini deployment is part of this handoff.

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

The staging boundary remains an offline mock, now aligned with the local Managed
Leads `RUNNER-STAGING.md` v1 contract: GET assignment, POST submissions and GET
receipt. The server implementation is locally available; deployment, credential
configuration and live activation remain unverified and unauthorized here. Existing account
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

## Managed Leads wire alignment

The mock preserves the fetched `brief` and `assignmentRevision`, checks the bound
workspace, and emits `schemaVersion: 1`, `mode: shadow`, a deterministic UUID
idempotency key and a minimal `source` submission. Candidates use `identityStatus:
unknown`, `decision: pending`, explicit gaps and empty claims/contacts. Exact
source spans and Jev recordings stay in the private replay record, not in
unsupported wire fields. Withheld results do not submit an empty candidate batch.

`runner_response` contains the protocol receipt (`status: shadow_received`), not
an approval, operational delivery or account ID. The outer replay receipt's
`correction-needed` is a local review reminder; no remote correction queue or
review screen is implied. The mock handles exact replay, conflicting keys, stale
scope and runner/tenant receipt isolation. It only validates the minimal emitted
subset; the real service remains authoritative for the full submission schema,
RLS, quotas and authentication.

Authentication fixtures use separate `X-Runner-Client` and an explicitly fake
Bearer value in memory. No environment credentials, cookies, queries, network
transport or production bindings are used. Any live transport requires a separate
review and activation. Generate a new submission key only for genuinely new work;
a retry must preserve the exact original envelope even after scope changes.

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
checks did not establish live Mini readiness. The initial PR subsequently passed
all six CI checks; later contract/relay verification is recorded below.

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
4. **PARTIAL:** local runtime and one synthetic inference were verified below;
   the backup manifest was not inspected and this branch was not deployed.
5. **DEPENDENCY:** Review/deployment of Managed Leads runner staging service,
   separate runner binding and explicit activation; no live transport is added.
6. **DECIDE:** approved source scope, egress, measured budgets, model/provider
   integration, retention enforcement and activation belong to the owner.
7. **DEPENDENCY:** JEV-01 policy adapter and regression/holdout evidence. Replay
   records are not an implementation or evaluation of that team's live adapter.
8. **OUT OF SCOPE:** product-request-to-PR automation and all merge/deploy authority.

## Explicit synthetic relay verification (October 22)

The existing SSH connection reached the Mini. Brain and Command Center processes
used AIIA revision `1e1c0a105adbeea8196b332c0d01ad89c2d0ef26`. The hardware reported
`Mac16,10` and 24 GiB RAM. Ollama had local `qwen3:8b` (8.2B Q4_K_M), digest
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`, installed;
no model was loaded before the test. Model metadata reports a 40,960-token maximum
context; the single test requested 2,048 tokens and capped generation at 180.

Using the existing harness's local-model verification and Ollama request pattern,
one real Qwen call extracted the expected organization/location/signal from an
invented sentence. It reported 61 input tokens, 29 output tokens and 6,756 ms total
duration. This is one cold synthetic observation, not throughput, quality or cost
evaluation. No Jev judgment occurred in this lane.

Topology: Mini loopback Ollama → existing SSH → workstation controller → explicit
Managed Leads shadow intake. Only invented synthetic material crosses that relay.
It does not enable Mini network egress. Airgap remained enabled; `typesafe.signals`
and `news.fetch` remained disabled. Model settings, schedules, runtime stores and
service revisions were not changed. The requested model was unloaded after the
bounded call via request-level keep-alive zero.

The controller accepts only the fixed Managed Leads HTTPS origin, a private
credential file (0600, parent 0700, owned by the operator), and an explicitly
expected workspace. It never reads browser sessions or environment tokens.
Preserve the exact saved envelope and UUID on retry; a changed scope needs a new
research run rather than relabeling the old payload. Network or server errors
must not expose headers, credentials or response bodies. No redirects are followed.

The generated mock envelope also passed the actual Managed Leads
`RunnerSubmission` schema, and its normalized payload hash matched the server's
digest function. Updated local checks: 117 tests passed (shadow state, adapter,
contract, API, mocked live client and existing public-signal regressions).

One authorized real shadow submission of the Qwen result returned 201; its exact
retry returned 200/replayed, and GET receipt matched both immutable receipts.
The receipt reported `shadow_received` and one candidate. Request/receipt and
Qwen evidence are retained privately outside Git; no credential or live tenant
identifier is included here. The runner credential cannot inspect operational
account/delivery totals; the Managed Leads operator owns that separate audit.
No claim of a full Qwen-plus-Jev pipeline is made.

CLI: `python -m local_brain.scripts.submit_research_shadow --credential-file
<private-file> --expected-workspace <explicit-workspace> --envelope-file
<saved-original-envelope> --receipt-file <new-private-receipt-file>`. Parent
directories must be owned by the operator and mode 0700; input files must be
0600. On failure, a reserved empty receipt file may remain: preserve the original
envelope/key and use a fresh receipt filename for an exact retry. Never generate
a new key to hide an uncertain submission outcome.
