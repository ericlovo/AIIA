# Business-development conductor

Studio's **BD conductor** (`#/bd`, also Morning note → Details) is a research desk with durable jobs, segment lanes, streaming activity, sourced proposals and local human triage. It never sends outreach, modifies a CRM, approves a prospect, or distributes a handoff.

## Run an isolated preview

Build the dashboard (`cd dashboard && npm ci && npm run build`), then from the repository root:

```sh
python -m local_brain.scripts.bd_preview --database /tmp/aiia-bd-preview/conductor.sqlite3 --port 8231
```

Open `http://127.0.0.1:8231/#/bd`. **Run shadow batch** queues four synthetic research jobs. Repeated clicks in the same configured interval reuse those jobs. Recorded Qwen extraction and Jev judgments are explicitly replay-only; no source retrieval, inference API, tenant API, or outreach occurs. Replay delays make state transitions visible; they are not performance measurements. The isolated preview deliberately does not start the legacy Brain, agent loops, or integrations. Legacy shell health may therefore show unavailable.

The normal Command Center mounts the same router and starts one worker during application lifespan, independent of an open browser. The worker is stopped cleanly on shutdown. A crashed attempt retains its lease and is recovered after expiry on restart.

## Configuration and durable state

`local_brain/command_center/config/bd_roster.v1.json` is the versioned roster:

- Healthcare, PE technical diligence, alumni/associations, and a synthetic SRG client lane.
- `mode: replay`, `schedule_enabled: false` (the one activation flag), and a
  standing `cadence` in America/Chicago: weekdays 08:00–18:00 every 30 minutes,
  weekends 09:00–17:00 hourly. The worker rotates one eligible bot per slot.
  Client bots join rotation only when they have a BD contract
  (`research_segment` plus `managed_leads`). SRG stays on `synthetic-srg` until
  an approved tenant binding exists. Studio **Activate daily research** persists
  the flag; the default roster stays off.
- Exact configured tenant per bot. API requests cannot override tenant or inference mode.
- One concurrent job across worker processes to protect the M4.
- Global daily, per-bot daily, and per-attempt caps for tokens, calls, and wall time.

Use `AIIA_BD_CONFIG` to select another reviewed config. Use `AIIA_BD_DATABASE` for dedicated persistent SQLite storage (default `local_brain/command_center/bd_data/conductor.sqlite3`, gitignored). Do not put it in legacy `agent_data`, memory stores, or runtime JSON. Separate replay/live databases. Changing mode or tenant configuration fences previously queued work; do not repurpose historical tenant records.

Each job moves `queued → leased → running → completed`, or to bounded retry/backoff, `failed`, or `cancelled`. Atomic leases and fencing tokens prevent stale workers from publishing results. Deduplication is `(tenant, entity, task)`; scheduled discovery uses a deterministic bot/time-window entity. Full per-attempt caps are reserved transactionally before work, with no refunds after failures or crashes. This is deliberately conservative; dashboard values are **reserved capacity**, not an invoice or measured inference spend. Reservations use the UTC start day, including an attempt crossing midnight.

Adapters reserve calls and conservative UTF-8 payload/output token envelopes before inference, enforce local output bounds, and reject provider usage exceeding the reservation. A remote provider's internal token accounting cannot be hard-capped locally; explicit spend approval remains required before live activation. Wall time is independently bounded. Stopping the conductor cancels active work and fences its results, while queued jobs remain durable; releasing the kill switch resumes the worker without enabling the schedule.

## Evidence and review

Public-news discovery uses fixed segment-specific RSS queries through the existing `news.fetch` egress gate. Each cycle checks up to eight returned sources and selects the first unseen URL/content hash from the bot and tenant's durable completed-evidence history. Qwen runs only at the fixed local Ollama endpoint with JSON-schema output.
Live extraction warms `qwen3:8b` with a tiny `keep_alive` ping and retries once
on `qwen_unavailable`. Jev uses Choice, Score and Noul over a public-only projection through `typesafe.signals`. Tenant identities, private records, and contact restrictions never enter the Jev state.

Exact source spans establish any extracted email or phone. An exact quote is necessary but not sufficient to support a market signal: Jev separately evaluates the proposed quote in context. Provider failures are `data_unavailable`, never an invented empty lead inventory. Consumer insights are unverified hypotheses, and public contacts have unverified identity. Discovery is not buying intent or qualification.

Qwen selects the index of a verbatim source span instead of generating quote text. The ordered spans preserve the complete source without normalization; local code resolves the selected index to its original text. Runtime validation still rejects invalid selections and unsupported organization names, and Jev still checks semantic support. Selection avoids copying long quotes into the capped model response or repeating source text in a schema enum. It does not prove the chosen span supports the proposed signal or establish live model quality. The complete request still counts toward the existing token reservation; budgets and retries are unchanged.

Packets retain evidence, provenance, source hash, observed time, model labels, usage, and typed judgments in SQLite. Studio shows source quote/URL, contact hold, signals, hypotheses, and local review status. **Research next / Watch / Dismiss / Restore** only update local review metadata; they grant no approval to contact anyone. All contacts remain held until tenant opt-outs and requested dates can be checked.

The live feed is newest-first and bounded to the latest 200 events. Jev statistics describe that same recent event window. Successful provider observations become stale after two minutes; they are not continuous health probes. SSE pushes changes with fallback polling in the dashboard. Only selected evidence quotes are exposed; raw provider responses, credentials, and lease tokens do not enter the public snapshot.

## Activation boundaries and missing integrations

A shadow preview is not live BD activation. Before enabling the all-day schedule:

1. Review and merge the draft through normal branch protections, then approve the Mini deployment/restart.
2. Approve the exact existing egress exceptions and provider spend. At inspection on 2026-10-23 the Mini had `AIIA_AIRGAP` on, `news.fetch` and `typesafe.signals` disabled, and Qwen3:8B installed. This change does not edit any of those flags or widen `AIRGAP_ALLOWED_EGRESS`.
3. Configure scoped credential environment references and the exact workspace identity for the implemented **read-only** Managed Leads adapter. Its account-list/detail contract is verified against the published API; authenticated access remains untested. Reads require the existing `web.fetch` policy gate, which this change does not open. Client bots also require an explicit research segment. Requested-contact-date semantics remain unavailable, so every contact stays held. No admin cookies, database-owner access, ZIP import, or cross-tenant access is used.
4. Review segment scope, exact tenant bindings, sources, cadence and budgets; switch a separate config/database to live. Keep the kill switch on until ready. Then either set `schedule_enabled: true` in that live config or click **Activate daily research** in Studio. Do not flip the flag on the Mini from this change. The standing cadence replaces the expired weekend display window; an old `schedule_window` is ignored when `cadence` is present.

Autonomous primary-page crawling is not implemented: the existing fetcher does not supply the SSRF protections needed for arbitrary discovered URLs. Headlines may yield no contact data. The explicit excerpt import path supports operator-supplied public primary evidence with hash/time/allowlist validation, without fetching arbitrary URLs; it remains human-supplied rather than independently fetched evidence.

## Verification

Run the repository backend tests, Ruff and its backlog ratchet, dashboard lint/unit/build/browser suites. New focused suites are `test_bd_conductor.py`, `test_bd_service.py`, `test_bd_routes.py`, `test_bd_pipeline.py`, and `test_bd_cadence.py`. The dashboard has fixture tests plus `dashboard/tests/bd-live.mjs` for the real isolated backend (hard-bound to loopback port 8231 and synthetic replay).

Tests cover durable state, concurrent dedup/leases, retry limits, deadline/heartbeat fencing, restart recovery, global/per-bot caps, kill/resume, schedule validation, strict tenant binding, same-origin mutations, evidence validation, provider denial, triage persistence, and no fake packet for a successful empty source result. Browser checks cover SSE running/completion, source evidence, triage, filters, error states, and phone layouts. Real local Qwen extraction was separately exercised on synthetic text; live cloud retrieval/Jev and authenticated tenant API access remain unverified because their activation boundaries are closed.

## Candidate quality and research history

Healthcare, PE technical diligence, and alumni/associations use the same strict
source-grounding checks. A candidate requires a named organization in the article
and selected quote, a concrete event, a supported claim, and a confident
segment-specific entity-role judgment. PE requires relevant investment,
acquisition, diligence, or technology operations; alumni requires relevant
membership, engagement, fundraising, or platform operations. Publisher names,
blank organizations, generic themes, and unrelated personal news are withheld.
These judgments use the existing Jev request and retain the per-job budget checks.

Studio separates **Supported candidates** from **Research history**. Completed
jobs can be rejected, inconclusive, or contain no new evidence; completion alone
is not a candidate. Historical packets remain intact, but missing role/evidence
checks keep them in research history. Local triage cannot upgrade their evidence
classification. Supported candidates still require human verification and remain
on contact hold; neither classification nor triage authorizes outreach.
