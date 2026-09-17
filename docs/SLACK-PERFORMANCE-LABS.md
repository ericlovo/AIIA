# Performance Labs Slack Capture

AIIA (`A0C1F9EG0M8`) was created and installed in Performance Labs on September
12, 2026, initially with `commands` and `app_mentions:read`; outbound receipts and
memory posts need `chat:write` as well (see Scope below). Workspace:
`performancelabs-hq.slack.com`, `T07LCJPNYJ1`. The separate Sanction product app
(`A0BSBESRBPG`) and its callbacks remain unchanged. The installed AIIA app has
the slash command configured; Events API activation awaits a verified callback.
`mindmoor-dev` is the proposed initial capture channel; its ID needs confirmation.
Capture deployment and end-to-end validation remain pending.

The Mini launcher loads private Slack settings from
`~/.config/aiia/slack.env` after the production `.env`. The file is mode 0600;
never copy its credentials into this repository, logs, or chat.

## Scope

This section is the current, authoritative statement of what crosses the Slack
boundary. Where an older paragraph below reads differently, this one wins.

| Direction | What | Egress point | Default |
|---|---|---|---|
| Inbound | `/aiia-capture` and `@AIIA` mentions from allowlisted channels, stored locally | none | on when capture is configured |
| Outbound | Fixed save and promotion receipts in the capture thread; never captured text | `slack.capture_ack` | off (`AIIA_SLACK_ACK_ENABLED=1`) |
| Outbound | Human-approved memory posts: the approved capture text, its priority and category, to one allowlisted channel | `slack.memory_post` | off (`AIIA_SLACK_MEMORY_POST_ENABLED=1`) |
| Outbound | Anything else, including general `slack.post` | `slack.post` | always denied |

Scopes: `commands` and `app_mentions:read` for capture; `chat:write` for either
outbound path, as in `config/slack-performance-labs-manifest.json`. No
`chat:write.public`, history, or impersonation scope is used.

**Policy change (2026-09-17).** Until this change, the rule was "captured text
is never transmitted". It is narrowly reversed by owner decision: a capture's
text may now be posted to Slack, but only when a person explicitly marks that
capture for Slack at the moment they log it to memory, only to the single
channel configured in `AIIA_SLACK_MEMORY_POST_CHANNEL_ID` (`#aiia-memory`), and
only back to the Performance Labs workspace it came from. Nothing is posted
automatically, receipts still never carry captured text, and the path is off
until the owner enables it.

## First Workflow

### Save acknowledgements

The receipt implementation adds `chat:write` to the app's existing scopes, without
`chat:write.public`, impersonation, history access, or an LLM/chat responder.
It sends a fixed thread reply only after the idea and receipt have committed in
the same SQLite transaction: "Saved to the local Mindmoor inbox for review.
Capture ID: ...". This confirms inbox storage, not promotion to confirmed Brain
memory. The captured text is never included in outbound receipt payloads.

Enable with `AIIA_SLACK_ACK_ENABLED=1` and enter `AIIA_SLACK_BOT_TOKEN` in the
private Mini environment after reinstalling AIIA with the added scope. Capture
continues without a token; receipts queue durably when acknowledgements are
enabled. Previously captured ideas without a receipt are not backfilled.

The background worker sends one receipt at a time, with a two-second poll,
ten-second HTTP timeout, fixed Slack destination, no redirects, and workspace/
channel checks at delivery time. The separate `slack.capture_ack` egress point
is opt-in even in air-gap mode; general `slack.post` remains denied.

Delivery retries back off, honor rate limits, and stop after eight failed
delivery attempts. Auth/scope/channel errors fail immediately. Restart recovery
uses a two-minute lease and a stable client message ID. This is not an exactly-once
network guarantee: a timeout after Slack accepts a message can produce a duplicate
receipt. Memory capture and receipt enqueue are independently deduplicated by the
original event identity.

The protected inbox API includes `acknowledgement_status`,
`acknowledgement_error`, and `acknowledgement_ts`. Integration status includes
queue counts and configured/enabled flags. After fixing an error, an authenticated
operator can POST `/api/memory-inbox/{idea_id}/acknowledgement/retry` to requeue
only a failed receipt, without recapturing the idea. No extra Cloudflare exception
is permitted for this operator endpoint.

Validation for this slice: 444 backend tests passed, 9 skipped, and the known
macOS audio MIME test failed. Live receipt delivery is pending token configuration
and an end-to-end test; the earlier successful live test covered capture only.

### Studio memory log and promotion receipts

Agent Studio has a Memory tab that lists the Mindmoor inbox: unreviewed,
logged, and dismissed captures with search, counts, and each capture's Slack
receipt state. The stored text is unchanged; the view strips the leading bot
mention for display and for the Brain fact.

Logging a capture ("Log to memory") calls the local Brain
`/v1/aiia/remember` with source `slack:mindmoor` and metadata carrying the
capture ID, project, workspace, channel, author, capture time, and any review
note, then in one SQLite transaction marks the capture `promoted` with the
memory ID and category and, when the capture arrived through a Slack thread,
queues one promotion receipt in `promotion_receipts`. The worker delivers it
through the same fixed-destination path as save receipts, with the text
"Logged to AIIA memory from the Mindmoor inbox. Capture ID: ... Memory ID: ...",
a distinct `client_msg_id`, and the same retry, rate-limit, and permanent-error
rules. Slash-command captures have no thread, so they get no receipt. At most
one promotion receipt is ever queued per capture. The egress point remains
`slack.capture_ack`, and receipts never transmit captured text. Posting the
text itself is the separate, opt-in memory post path below.

If the Brain rejects the fact (quality gate, 422) or does not answer (503), the
capture stays unreviewed and nothing is queued. If the Brain stores the fact but
the inbox update fails, the API returns 503 `memory_saved_inbox_update_failed`
rather than pretending nothing happened; do not log that capture again.

Dismiss keeps the record locally under Dismissed and sends nothing to Slack;
Restore returns a dismissed capture to Unreviewed. Logged captures cannot be
dismissed or restored from Studio because the Brain fact already exists.

### Memory posts to #aiia-memory

When logging a capture, the Memory log sets a priority (`urgent`, `high`,
`normal`, `low`; default `normal`) and, only when the status route reports
posting configured, offers "Post to #aiia-memory". Priority is stored on the
capture and the log can filter by it or sort priority-first.

If the post option is checked but posting is not enabled and configured, the
promote route returns 409 `memory_posting_disabled` before calling the Brain,
so nothing is logged and nothing is queued. Otherwise the Brain fact is stored,
then one SQLite transaction marks the capture promoted with its priority and
`post_requested`, and queues exactly one row in `memory_posts`, keyed by memory
ID. The row holds the body approved at that moment; later edits do not change
what is sent.

The body is plain text (`mrkdwn: false`):

```
[HIGH] Memory logged to decisions

<capture text, bot mention stripped>

Capture 1a2b3c4d · Memory decisions_4_1789
```

`&`, `<` and `>` are escaped, which neutralizes `<!channel>`, `<!here>`, user
and group mentions, and link syntax. Text over 3,000 characters is cut to 3,000
ending in `…`, and the footer gains `· Truncated`. The message is not threaded,
not broadcast, and unfurls nothing. `client_msg_id` is stable per memory.

A worker delivers one post at a time: urgent before high before normal before
low, then oldest due first. It reuses the receipt worker's lease, backoff,
rate-limit handling, permanent-error set and eight-attempt cap. At delivery it
re-checks that the capture's workspace is `AIIA_SLACK_TEAM_ID` (else it fails
with `source_not_allowed`) and that the queued channel still equals the
configured channel (else `destination_not_allowed`), and asks the
`slack.memory_post` egress point. As with receipts, a timeout after Slack
accepts a message can produce a duplicate.

Configuration, all in the private Mini environment:

- `AIIA_SLACK_MEMORY_POST_ENABLED=1` turns it on.
- `AIIA_SLACK_MEMORY_POST_CHANNEL_ID` is the one destination and must match
  `^C[A-Z0-9]{8,}$`. It is separate from the inbound `AIIA_SLACK_CHANNEL_IDS`
  and never extends it.
- `AIIA_SLACK_BOT_TOKEN` and `AIIA_SLACK_TEAM_ID` are reused.

Enabled and configured means all four are present and valid.
`/api/integrations/slack/status` reports `memory_posts_enabled`,
`memory_posts_configured`, `memory_post_channel_id` and `memory_posts` counts.
A failed post can be requeued with
`POST /api/memory-inbox/{id}/acknowledgement/retry?kind=memory_post`.

Activation, owner only, after deploy:

1. Confirm the AIIA bot is still a member of `#aiia-memory` and the app has
   `chat:write`.
2. Set `AIIA_SLACK_MEMORY_POST_ENABLED=1` and
   `AIIA_SLACK_MEMORY_POST_CHANNEL_ID` in the private service environment, then
   restart. Check the status route reports `memory_posts_configured: true`.
3. Log one clearly labelled synthetic capture with "Post to #aiia-memory"
   checked, and confirm exactly one message arrives, escaped and correctly
   headed, before logging anything real.

To turn it off, unset `AIIA_SLACK_MEMORY_POST_ENABLED` and restart; queued posts
stay pending locally and nothing is sent.

Operator routes, all behind Studio's existing access boundary:
`GET /api/memory-inbox?project=&query=&status=&offset=` (adds per-status
counts; optional `priority` filter and `sort=priority`),
`POST /api/memory-inbox/{id}/promote` (`category`, optional `note`, `priority`,
`post_to_slack`),
`POST /api/memory-inbox/{id}/dismiss` (optional `note`),
`POST /api/memory-inbox/{id}/restore`, and
`POST /api/memory-inbox/{id}/acknowledgement/retry?kind=capture|promotion|memory_post`.
`/api/integrations/slack/status` reports `promotion_acknowledgements` and
`memory_posts` counts. The ideas table gains `memory_id`, `memory_category`,
`review_note`, `reviewed_at`, `priority`, and `post_requested`, and a
`memory_posts` table is added; the migrations are additive and repeatable.

`/aiia-capture <idea>` explicitly saves the original text to a local SQLite inbox.
Records include workspace, channel, author, capture time, source and project
(`mindmoor`, the primary product repository is `tonybangert/mindmoor`). They start unreviewed. They are not automatically asserted
as Brain facts or included in agent prompts.

The app acknowledges a successful commit with an ephemeral capture ID. Retries
with the same Slack trigger ID return the existing capture. Signing secrets,
verification tokens and response URLs are not stored in the record. The capture
handler itself does not call a model, post channel messages, poll Slack history or
use response URLs; outbound messages come only from the opt-in workers above.

`@AIIA <idea>` is also supported through signed `app_mention` events. Original
mention text is retained; Slack event IDs deduplicate retries. Only allowlisted
workspace/channel events are stored. Bot messages and other event types are
ignored. Mentions are acknowledged to Slack after storage; they get a threaded
reply only when save receipts are enabled. Signed URL-verification challenges do
not create ideas.

Promotion to Brain memory through `/v1/aiia/remember` with provenance is done by
a person in the Studio Memory log (above). There is no curator agent; nothing is
promoted automatically.

## Installation

1. Confirm the Performance Labs workspace and the channel IDs allowed to capture.
2. Inspect the Performance Labs AIIA app and merge the required configuration from
   `config/slack-performance-labs-manifest.json`. Capture needs `commands` and
   `app_mentions:read`; receipts and memory posts also need `chat:write`. No
   channel-history, `chat:write.public` or impersonation scope is used. Reinstall
   when Slack requires updated scope consent, then invite AIIA to the chosen channel.
3. Put `AIIA_SLACK_SIGNING_SECRET`, `AIIA_SLACK_TEAM_ID`, and comma-separated
   `AIIA_SLACK_CHANNEL_IDS` in the Mini's private service environment. Enter secrets
   locally. No bot token is required for capture alone; receipts and memory posts
   need `AIIA_SLACK_BOT_TOKEN`.
4. Deploy the integration and restart the service. Verify
   `/api/integrations/slack/status` shows configured with the expected IDs.
5. Slack must reach the exact `/api/integrations/slack/commands` and
   `/api/integrations/slack/events` endpoints without an interactive Cloudflare
   Access login. Configure a narrowly scoped application/rule for that path;
   retain authentication on Studio, `/api/memory-inbox`, and every other API.
   The command endpoint authenticates Slack signatures and checks team/channel IDs.
   Do not bypass authentication for `/api/*` or the entire hostname.
6. Send one explicitly authorized synthetic capture from an allowed channel.
   Verify one local record and retry deduplication before broader use.

The HTTP response contains only capture acknowledgement or validation feedback.
General outbound `slack.post` remains denied in air-gap mode. The old Brain route
`POST /v1/aiia/slack`, which imported a module that was never committed, has been
removed. Any further outbound Slack use (notifications, conversations, other
channels) needs its own scoped transport and egress decision, as receipts and
memory posts each have.

## Local Access and Limits

- `GET /api/memory-inbox?project=mindmoor&query=idea&offset=0` lists up to
  50 captures, searches the original text, and reports the total.
- Storage defaults to `local_brain/command_center/memory_inbox.sqlite3`, already
  covered by the existing runtime SQLite ignore rule and backup glob. File mode
  is 0600. `AIIA_MEMORY_INBOX_PATH` may set an explicit alternate path.
- Captures allow up to 8,000 characters. Requests are capped at 32 KiB; signatures
  must match the original body and have a timestamp within five minutes.
- Invalid signatures, unconfigured identity or disallowed sources fail closed.
  Storage errors return 503 rather than claiming capture succeeded.
- The inbox API inherits Studio's existing access boundary; a project filter is
  not tenant authorization. Do not expose it through the Slack path exception.

## Sources

- [Request verification](https://docs.slack.dev/authentication/verifying-requests-from-slack/)
- [Slash command responses](https://docs.slack.dev/interactivity/implementing-slash-commands/)
- [App manifest fields](https://docs.slack.dev/reference/app-manifest/)

Slack expects an acknowledgement within three seconds; this handler performs only
bounded parsing and a local database write, with a one-second SQLite lock timeout.
End-to-end timing through Cloudflare still requires a live installation test.

Local validation: 23 capture regressions pass, covering authentication, source
restrictions, durable deduplication, credential exclusion and failed storage.
Full backend suite: 427 passed, 9 skipped, one previously confirmed macOS MIME
failure. No Slack messages were sent during implementation.
