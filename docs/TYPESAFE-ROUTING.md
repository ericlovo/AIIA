# Jev routing advisor

Optional external advice for the New assignment form. Local execution remains
unchanged; Jev does not replace an agent, grant permissions, or approve reviews.

## Activation

Set these in the Command Center server's environment, not the browser:

```dotenv
AIIA_TYPESAFE_ENABLED=true
TYPESAFE_API_KEY=<your TypeSafe key>
TYPESAFE_MODEL=jev-latest
```

Deploy the code and restart Command Center through the normal release process.
`GET /api/integrations/typesafe/status` reports readiness without exposing the key.
No live credentials or runtime environment were changed during implementation.

## Egress governance

`typesafe.routing` is a registered egress point, so `/health` lists it and an
air-gapped install shows it as disabled. The advisor asks `authorize_egress()`
before it dials out, exactly like the Slack transports, which means a denial is
reported to Sanction and lands in the audit trail. Under `AIIA_AIRGAP` it is
denied unless `AIIA_TYPESAFE_ENABLED` is set; it is never a static exception in
`AIRGAP_ALLOWED_EGRESS`. Per-request consent from the caller is not
authorisation: both the governance decision and the consent are required.

## Data and behavior

The user writes a separate routing brief and checks the external-sharing consent.
Only that brief and candidate agent names/skills go to TypeSafe. Assignment context,
repository content, memory, personas, missions, and run history are not attached.
The brief itself is not automatically scrubbed: do not put sensitive information
in it. Agent *names* leave the Mini too, and some name a client (for example a
delivery watch named after one), so treat the agent roster as disclosed when
this is on. Agent IDs do not leave: candidates are aliased to `candidate_0..n`
and mapped back locally. This is cloud inference, not free local M4 inference.

Advice returns a candidate or no match, confidence, and measured input/output
tokens. Confidence is not a probability of workflow success. No automatic
confidence threshold or routing policy is enabled. The user explicitly applies
the suggestion and separately creates/runs the assignment. Editing the brief or
candidate descriptors hides stale advice.

Both server enablement and per-request consent are required. Requests are bounded
to 2,000 brief characters and 32 candidate agents, serialized with a ten-second
cooldown, a fifteen-second timeout, and no automatic retries. This is a
single-process guard, not a durable daily spend cap. Restarting resets the cooldown.
No scheduled loop calls Jev. Outages, malformed responses, missing configuration,
and no-match results leave manual assignment available. No provider error body is
shown or logged by the advisor.

Token usage is displayed per suggestion; it is not yet persisted or attributed to
an executing agent. Cloud cost is explicitly not estimated. Do not claim token or
cost savings until representative routing decisions have been evaluated.

## API

`POST /api/assignments/suggest-agent` accepts `brief`, `candidate_agent_ids`, and
`allow_external: true`. Unknown/duplicate candidates are rejected before any
external request. The response is advisory only and never mutates assignments.

Contract sources checked September 24, 2026:
- https://docs.typesafe.ai/api
- https://docs.typesafe.ai/primitives/choice
- https://docs.typesafe.ai/cookbooks/function_calling

The installed TypeSafe skill guides further decision/routing work. Keep scheduling,
permissions, repository access, and review-backlog limits deterministic and local.
