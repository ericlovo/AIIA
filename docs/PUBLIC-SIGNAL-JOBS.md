# Public signal jobs

Two opt-in discovery jobs serve Performance Labs in Wisconsin, Minnesota and Iowa:

| Job | Retrieval | Output |
| --- | --- | --- |
| Market Signal Scout | Family capital, family-owned business and PE news | Evidence-linked research micro-stories |
| Lead Signal Scout | Acquisitions, expansion and leadership changes | Potential account triggers requiring primary-source verification |

## What runs where

The Mini retrieves fixed Google News RSS searches, filters publication dates to
14 days, and sends at most eight public headlines to Jev in one request. Jev
classifies each as lead, market, noise or uncertain. Application code assembles
research tasks from the evidence; Jev does not browse or generate prose.

Headlines are discovery evidence, not verified facts. Publication time is not
event time. A lead classification or confidence is not proof of buying intent,
budget, or account fit. No contact enrichment, email scraping, outreach, Slack
posting or publishing occurs. Uncertain items do not become work automatically.
Company websites and primary-source verification are the next research step,
not secretly fetched by this pilot. Google RSS search is not a guaranteed search
API: outages are shown as failed runs, without fallback to arbitrary websites.

Private memory, repository files, client notes and credentials never enter the
screening state. Only headline, publisher and publication date are projected.

## Enable deliberately

Keep `AIIA_AIRGAP=1`. Set these in the Mini's server environment, never the browser:

```dotenv
AIIA_NEWS_ENABLED=1
AIIA_SIGNALS_ENABLED=1
TYPESAFE_API_KEY=<existing server-side key>
```

The first flag permits only `news.fetch`; the second only `typesafe.signals`.
Neither unlocks generic `web.fetch` nor depends on the routing-advisor flag.
Both transports consult egress governance before each request and refuse
redirects. These flags are **off by default**, even outside airgap mode.

Restart the Command Center after deployment. Open `#/signals`, run a job once,
review its output, then opt in to its 12-hour schedule. Manual runs share the
same 12-hour cooldown as scheduled runs. Two jobs means at most four attempts
per rolling 24 hours; failures also consume their reservation. No live schedule
is activated by installing this code.

## State and limits

- `AIIA_SIGNAL_JOBS_PATH` optionally selects the SQLite job database; otherwise
  it is beside the memory inbox. Include both databases in runtime backups.
- Runs are reserved transactionally before networking. Multiple processes share
  the reservation. A 90-second execution deadline precedes the 180-second stale
  reservation recovery; interrupted attempts remain visible.
- At most three new inbox items per run, and six open public-signal items total.
  A full review queue stops retrieval and screening.
- Deduplication uses the Google article path across jobs, ignoring query tracking.
  It is not semantic event deduplication: syndicated versions can remain distinct.
- Each job's last run reports provider-returned input/output tokens. Failed
  requests without validated usage show unknown, not zero. The durable run rows
  retain usage history; this is not yet merged into the platform token ledger.
- The inbox's existing assignment and triage flow remains the human approval
  boundary. Accepting a proposal records `needs_work`; it does not send outreach.

## Endpoints

All endpoints use the Command Center's existing authenticated deployment boundary:

```text
GET  /api/signal-jobs
PUT  /api/signal-jobs/{market_news|lead_signals}    {"enabled": true|false}
POST /api/signal-jobs/{market_news|lead_signals}/run
```

Unknown jobs return 404; overlapping runs 409; cooldown 429; unavailable setup
or storage 503. Provider failures are persisted as a failed run, never presented
as no news. As with other Studio routes, do not expose an unauthenticated origin.

## Verification

Backend fixtures test feed validation, bounded output, failure reservations,
cross-process exclusion, cancellation, consent flags, payload projection,
backpressure, endpoint validation and cross-job deduplication. The browser test
uses fixture responses for desktop/mobile layouts and controls; it is not evidence
of live Jev quality. Validate the first real batch with Eric or Tony before
enabling recurrence. Measure pursue/watch/reject outcomes before adjusting policy.
