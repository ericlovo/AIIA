# Live BD activation

The conductor is deployed at revision `bf9d85a`. Its initial synthetic shadow cycle completed all four bots, produced 11 SSE updates during verification, and showed no browser JavaScript errors. These are replay results, not live research or model-quality evidence. Live research remains disabled until an operator explicitly approves its external requests and associated usage.

## What runs where

- **Local Qwen:** installed `qwen3:8b` through loopback Ollama. Extracts bounded public evidence into a schema. A local synthetic-input call was verified; this does not establish extraction accuracy across real sources.
- **Cloud Jev:** existing TypeSafe API, `jev-latest`. One request asks a Choice for research/watch/reject/uncertain, a three-level Score for evidence strength, and a Noul for whether a proposed quote supports its claim. Jev is not running locally on the M4.
- **Discovery:** a fixed Google News RSS query for the configured segment. Each cycle retrieves up to eight recent headlines and processes the first unseen source URL/content hash for that bot and tenant.
- **Output:** durable local research packets awaiting human review. No outreach, CRM writes, account approval, or contact verification writes occur.

The verified pre-activation configuration has airgap enabled, an existing TypeSafe credential configured, and `news.fetch`, `typesafe.signals`, and `web.fetch` disabled. Recheck these conditions before activation; a saved configuration is not proof of current process state.

## First pilot: one healthcare attempt per UTC day

Use an operator-owned configuration copied from `local_brain/command_center/config/bd_roster.v1.json`. Keep the repository default roster in replay mode. Use a **separate live database**; do not change a replay database into a live database.

Set these variables in the environment consumed by the service launcher, then restart the service through its existing operating procedure. The paths below are placeholders for operator-selected absolute paths, not literal defaults:

```sh
AIIA_BD_CONFIG=/absolute/operator-config/bd-healthcare-live.v1.json
AIIA_BD_DATABASE=/absolute/operator-state/bd-healthcare-live.sqlite3
AIIA_NEWS_ENABLED=1
AIIA_SIGNALS_ENABLED=1
```

Keep `AIIA_AIRGAP=1`. The existing news and signals flags enable their existing narrow airgap exceptions. Do not alter `AIRGAP_ALLOWED_EGRESS`, disable airgap, install another model, or create another API key for this pilot. Use the already configured `TYPESAFE_API_KEY`; never put its value in the roster or browser.

Apply these changes to the copied roster:

| Field | First-pilot value |
| --- | --- |
| `mode` | `"live"` |
| `schedule_enabled` | `true` |
| `max_attempts` | `1` |
| `max_concurrency` | `1` |
| `job_budget` | `{"tokens":8000,"calls":6,"seconds":120}` |
| `global_daily_budget` | `{"tokens":8000,"calls":6,"seconds":120}` |
| `bot_daily_budget` | `{"tokens":8000,"calls":6,"seconds":120}` |
| Healthcare `enabled` | `true` |
| Healthcare `schedule.enabled` | `true` |
| Healthcare `schedule.interval_seconds` | `86400` |
| Healthcare `tenant_id` | An explicit internal research scope, such as `research-healthcare`; this is not a Managed Leads client tenant |
| Every other bot `enabled` | `false` |
| Every other bot `schedule.enabled` | `false` |

Retain the existing lease duration and retry-backoff fields. Both the global schedule switch and the bot's schedule switch must be enabled. A 24-hour interval is an epoch-aligned UTC window, not “24 hours after the last run.” This configuration permits at most one attempt per UTC day; it is not a permanently self-disabling one-shot switch. After reviewing the first completed attempt, stop the conductor with the kill switch if further daily runs are not yet approved. There is no general “run one live discovery cycle” HTTP endpoint in this version; `/api/bd/shadow` is replay-only.

The worker begins scheduling when the service starts; an open dashboard is not required. Configuration is read when the service instance starts. Editing the JSON without restarting does not apply new settings.

## Verification and stop control

Read-only checks on the service host:

```sh
curl -fsS http://127.0.0.1:8100/health
curl -fsS http://127.0.0.1:11434/api/tags
curl -fsS http://127.0.0.1:8200/api/signal-jobs
curl -fsS http://127.0.0.1:8200/api/bd/snapshot
```

Confirm airgap remains enabled, the news/signals exceptions are enabled, Qwen is installed, the BD snapshot says `live`, only healthcare is enabled, and the kill switch is off. The signal-jobs status verifies shared retrieval/screening configuration; it does not prove a BD provider call succeeded. Enabling the shared flags can also make previously enabled legacy signal jobs eligible to run; inspect their existing enablement before activation rather than assuming Studio is the only consumer.

Observe the activity stream at `/api/bd/events` and the dashboard. Confirm one real source, local Qwen completion, a real Jev call/verdict, source provenance, measured usage, and a human-review packet. A successful HTTP response or green build alone is insufficient. If all current sources were previously seen, the correct result is `no_new_evidence` with no packet and no model calls. An unreachable provider must remain `data_unavailable`, not be interpreted as an empty market.

The operator stop action is:

```sh
curl -fsS -X POST http://127.0.0.1:8200/api/bd/kill-switch \
  -H 'Content-Type: application/json' -d '{"enabled":true}'
```

Use this only when intentionally stopping the conductor. It cancels active work and prevents further leases; it cannot retract an external request already sent. Resuming uses the same route with `enabled:false` after the cause of the stop is resolved.

## Expanding to a six-hour cadence

After the first live packet is reviewed, an operator can deliberately expand healthcare to a six-hour interval (`21600` seconds): approximately four scheduled windows per UTC day. Restore the roster's daily ceilings: 200,000 tokens / 200 calls / 7,200 seconds globally and 50,000 tokens / 50 calls / 1,800 seconds per bot. Retain the 8,000-token / 6-call / 120-second per-attempt budget and single-worker concurrency.

The ledger reserves the **whole attempt allowance before work starts**, including failed and retried attempts. The 50,000-token bot ceiling therefore admits at most **six 8,000-token attempts per UTC day**, not 50,000 measured model tokens plus unlimited attempts. Four six-hour windows leave room for at most two additional attempts across the day. Setting `max_attempts:3` permits up to two retries for an individual job but does not guarantee retry capacity for every job. Failures early in the day can consume capacity needed by later windows. An hourly schedule under these defaults would exhaust the reservation allowance long before 24 cycles.

These limits control reservations, request count, and elapsed time; they are **not a dollar spending cap**. Visible payload bytes and output limits provide conservative token preflight estimates. Provider-internal usage cannot be guaranteed before the response; reported usage exceeding its reservation stops downstream processing. Cold inference can also exceed the 120-second attempt wall limit. Do not raise limits merely to hide repeated failures.

## Evidence and enrichment limitations

RSS discovery supplies headlines and publisher links, not complete primary-source documents. The pipeline does not automatically resolve or fetch publisher pages, crawl websites, or search for contacts across the web. Consequently, automated discovery usually yields signals rather than email/telephone enrichment.

Emails and telephone numbers are retained only when present verbatim in the provided source text, with offsets and URL provenance. They are not verified identities or permission to contact. Consumer insights are labeled hypotheses, not demonstrated consumer findings. Source-key dedup does not merge different syndications of the same underlying event.

An operator can supply a longer public primary-source excerpt through `POST /api/bd/sources`, after configuring an exact `source_allowlist` for the segment bot. See [Public-source enrichment](BD-PUBLIC-SOURCES.md) for the strict hash/timestamp/schema contract. Imports are labeled human supplied and not independently fetched; they still invoke cloud Jev in live mode and consume the same budget. They are not an offline route around egress controls. Primary-source auto-fetch and autonomous contact enrichment remain future work.

## Client BD activation is a separate gate

Managed Leads already exposes tenant-bound read APIs; the issue is not a missing generic account API. The implemented adapter supports GET account list/detail, validates the response workspace, uses server-side read credential references, follows no redirects, and never uses browser/admin cookies. Its verified contract and source links are documented in [Public-source enrichment](BD-PUBLIC-SOURCES.md).

A client bot additionally needs:

1. The exact authorized Managed Leads workspace as its `tenant_id`.
2. An explicit `research_segment` from `healthcare`, `pe-technical-diligence`, or `alumni-associations`. Studio does not infer the client's industry from its name.
3. A server-only `managed_leads` block containing the fixed `base_url` `https://managed-leads.vercel.app`, matching `workspace_id`, and existing environment-variable references `client_id_env` and `token_env` for a least-privilege read integration.
4. Verified credential provisioning and a verified workspace-bound read. Credential presence alone does not establish access.
5. A reviewed **narrow egress design**. The adapter currently asks the existing `web.fetch` gate; that gate has no opt-in exception under the current airgap policy. News/signals flags do not enable it. Client reads therefore cannot activate with the current policy unchanged. Do not broadly allow `web.fetch` or disable airgap to work around this; a scoped read policy/code change requires explicit approval.

The Managed Leads detail response exposes contact `hold`, `optedOut`, and `verified`. It does **not** expose a requested-contact-date restriction. The human workspace has a separate `dueDate` workflow field, but Studio must not reinterpret it as that missing contract. Client research currently verifies account-list availability without guessing identity joins between discovered organizations and CRM accounts. Raw CRM data never enters Qwen or Jev public-evidence payloads. No configured read, high fit score, or completed packet lifts the contact hold: outreach remains unavailable pending verified identity, restrictions, requested dates, and human action.

## Showing an operator-managed schedule in Studio

The snapshot can describe an operator-managed start/stop window using optional
`schedule_window` configuration. For example, the approved October healthcare
weekend uses:

```json
{
  "schedule_window": {
    "start_at": "2026-10-24T06:00:00Z",
    "end_at": "2026-10-26T05:00:00Z",
    "timezone": "America/Chicago"
  }
}
```

These timestamps describe Saturday 01:00 Central through Monday 00:00 Central.
They are **display metadata**, not an execution timer. The separately installed
operator guard remains responsible for releasing the initial hold, stopping the
conductor at the deadline, disabling the shared news/signals exceptions, and
checking expiry before service startup. Do not configure a window without a
matching operational guard. Configuration changes take effect after restart.

Studio projects the next eligible epoch-aligned window from enabled bot intervals
and already-enqueued jobs. A paused conductor still needs its operator or guard;
the display is not a promise that a job will start. Budget limits, worker health,
and provider availability can also prevent execution. Old successful model
observations are labeled stale/unverified, not presented as a new provider outage.
