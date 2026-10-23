# Public-source enrichment boundary

BD discovery retrieves up to eight recent segment-specific news headlines and processes the first unseen headline per job through the existing `news.fetch` policy. It does not scrape arbitrary publisher links. Emails and telephone numbers are retained only when they occur verbatim in source text, with URL and character offsets. They are observations, not verified identities or permission to contact.

An operator can instead supply a public primary-source excerpt. This supports evidence from an organization's announcement or contact page beyond a short headline. `load_public_excerpt(data, allowed_hosts)` accepts exactly:

```json
{
  "url": "https://example.org/company/news",
  "text": "The exact public excerpt, including any public office contacts.",
  "source_hash": "SHA-256 of the exact UTF-8 text, lowercase hexadecimal",
  "captured_at": "2026-10-23T12:00:00Z",
  "public": true
}
```

The trusted bot configuration supplies `source_allowlist`, an exact hostname list, empty by default. An import request must never supply its own allowlist. The route selects the bot's configured tenant; the excerpt cannot override tenancy. Only HTTPS URLs on that list are accepted. Credentials, fragments, nonstandard ports, IP addresses, localhost and local/internal hostnames are rejected. The URL is provenance, **not fetched**. The operator attests that the excerpt is public; the system cannot independently verify that attestation. Private tenant records must not be pasted into this path because live Jev judgments send the public evidence projection to TypeSafe.

Text must be nonempty and at most 2,400 UTF-8 bytes. Hash mismatch, missing timezone, future capture timestamp, unknown fields or a false public attestation reject the import. Provenance is labeled `source_kind: imported_primary_excerpt` and `verification: human_supplied_not_independently_fetched`. The capture timestamp and processing timestamp are separate.

Pipeline integration uses `job.payload.public_source` for the strict excerpt and `job.source_allowlist` from trusted bot configuration. Imported sources skip news retrieval, then use full excerpt text for local Qwen extraction, exact-span contact extraction, and governed Jev judgments. No text is silently truncated. Input-byte plus output reservations can reject a large excerpt before inference if the remaining token budget is insufficient. These reservations bound visible inputs and outputs conservatively; provider-internal usage cannot be guaranteed before the response. Reported usage exceeding a reservation stops downstream processing.

Imports require live mode; recorded Jev responses cannot certify imported real-world evidence. Under the current airgap policy, live Jev remains unavailable until the existing screening gate is deliberately enabled. Client BD requires configured least-privilege credentials for the existing Managed Leads read API and an explicit research segment. Its account detail contract exposes holds and opt-outs, but not requested contact dates. All packets hold outreach pending those checks; no adapter writes to Managed Leads or sends messages.

The existing general research fetcher follows redirects and lacks the stricter public-IP controls needed for autonomous contact enrichment. It is intentionally not used here. A future automatic fetcher needs exact configured URL/host scope, bounded bodies, redirect rejection, DNS/IP validation without rebinding, and the existing `web.fetch` authorization check. This change does not widen egress.

Jev uses one Choice, one three-level Score (raw range 0–2, checked against its probability-weighted value), and one Noul. The Noul checks the proposed quote against its source context; an exact text match alone does not establish that a quote supports an event. Consumer insights remain hypotheses. See the [TypeSafe Score contract](https://docs.typesafe.ai/primitives/score) and [citation-validation pattern](https://docs.typesafe.ai/cookbooks/citation_check).

## Import API

`POST /api/bd/sources` accepts `{ "bot_id": "healthcare", "source": <excerpt object above> }`. It returns HTTP 202 with `job_id`, the configured `tenant_id`, `mode: live`, and the source kind. Repeating the same URL and content hash deduplicates within the configured tenant and task. Capture time changes alone do not create a second job.

For example, after an operator explicitly configures live mode and the selected segment bot's exact source hostname allowlist, create a request body locally:

```python
import hashlib
import json
from datetime import datetime, timezone

text = "Exact text copied from the public source."
body = {
    "bot_id": "healthcare",
    "source": {
        "url": "https://example.org/company/news",
        "text": text,
        "source_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "public": True,
    },
}
print(json.dumps(body))
```

Send the resulting JSON through the same-origin operator API. Replay mode, stopped conductors, disabled/unknown bots, client bots without the required contact restriction contract, unsupported domains and malformed evidence reject the request. Request bodies cannot override the tenant, runtime mode or hostname allowlist. A queued import does not bypass provider egress settings; its live Jev step can still report unavailable.


## Verified Managed Leads read integration

Verified against Managed Leads main commit `c3fd49b3a7b8e7450f4c81bb0a9fb667c8e348bf`, [integration source](https://github.com/ericlovold/managed-leads/blob/c3fd49b3a7b8e7450f4c81bb0a9fb667c8e348bf/backend/integrations.py), [contract documentation](https://github.com/ericlovold/managed-leads/blob/c3fd49b3a7b8e7450f4c81bb0a9fb667c8e348bf/docs/mindmoor-integration.md), and the public [deployed OpenAPI document](https://managed-leads.vercel.app/openapi.json). Inspection accessed source and route metadata only, not authenticated tenant records. Credential availability and live authenticated access have not been verified.

The service already exposes `GET /api/integrations/v1/accounts` with `limit`, `after`, and `q`, and `GET /api/integrations/v1/accounts/{account_id}`. Server configuration binds `X-Integration-Client` and a bearer read token to one workspace. The server applies tenant predicates, transaction-local RLS context, and PostgreSQL read-only transactions. Responses include `version: 1` and `workspaceId`; the client rejects a workspace mismatch. Account detail returns `people` with boolean `hold`, `optedOut`, and `verified`, plus signals. Requested contact dates are absent from this projection, even though the human workspace has a separate `dueDate` workflow field. No date is inferred.

The bot's optional server-only configuration is:

```json
{
  "research_segment": "healthcare",
  "managed_leads": {
    "base_url": "https://managed-leads.vercel.app",
    "workspace_id": "the-bots-exact-configured-tenant",
    "client_id_env": "EXISTING_READ_CLIENT_ID_ENV_NAME",
    "token_env": "EXISTING_READ_TOKEN_ENV_NAME"
  }
}
```

These are references to existing server environment variables, not raw credentials. No credentials are provisioned by Studio. The adapter permits only the verified fixed HTTPS origin and GET account list/detail routes, uses no cookies, follows no redirects, bounds response bodies, and authorizes every request through the existing `web.fetch` policy. **Current airgap policy blocks these reads**, independently of the news and Jev feature flags. Configured credentials therefore do not imply connectivity or permission to dial.

A configured client research job reads its bound tenant's account list, then researches only its explicitly configured public `research_segment`. It records read availability and account count locally; it does not guess identity matches between discovered companies and CRM accounts. CRM records are never included in Qwen or Jev public-evidence payloads. Packet contacts remain held for tenant restriction and requested-date checks. The standalone detail adapter preserves suppression flags for future explicit account matching; no write/verification endpoint is implemented.

News evidence keys are SHA-256 of the source URL, a NUL separator, and the SHA-256 of exact source text. `job.seen_evidence` contains prior keys for the same bot and tenant. A job selects the first unseen of up to eight current headlines; if all are seen, it returns `packets: []`, `reason: no_new_evidence` without model calls. The key is stored on each packet. Updated source text gets a new key. Source-key dedup does not assert that differently syndicated articles represent different real-world events.
