# Public-source enrichment boundary

BD discovery reads one recent segment-specific news headline per job through the existing `news.fetch` policy. It does not scrape arbitrary publisher links. Emails and telephone numbers are retained only when they occur verbatim in source text, with URL and character offsets. They are observations, not verified identities or permission to contact.

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

Imports require live mode; recorded Jev responses cannot certify imported real-world evidence. Under the current airgap policy, live Jev remains unavailable until the existing screening gate is deliberately enabled. Client BD still requires a tenant-scoped Managed Leads read API to check opt-outs and requested contact dates. All packets hold outreach pending those checks; no adapter writes to Managed Leads or sends messages.

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

Send the resulting JSON through the same-origin operator API. Replay mode, stopped conductors, disabled/unknown bots, client bots without the tenant API, unsupported domains and malformed evidence reject the request. Request bodies cannot override the tenant, runtime mode or hostname allowlist. A queued import does not bypass provider egress settings; its live Jev step can still report unavailable.
