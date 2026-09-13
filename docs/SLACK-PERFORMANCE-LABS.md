# Performance Labs Slack Capture

AIIA (`A0C1F9EG0M8`) was created and installed in Performance Labs on September
12, 2026, with `commands` and `app_mentions:read` only. Workspace:
`performancelabs-hq.slack.com`, `T07LCJPNYJ1`. The separate Sanction product app
(`A0BSBESRBPG`) and its callbacks remain unchanged. The installed AIIA app has
the slash command configured; Events API activation awaits a verified callback.
`mindmoor-dev` is the proposed initial capture channel; its ID needs confirmation.
Capture deployment and end-to-end validation remain pending.

The Mini launcher loads private Slack settings from
`~/.config/aiia/slack.env` after the production `.env`. The file is mode 0600;
never copy its credentials into this repository, logs, or chat.

## First Workflow

`/aiia-capture <idea>` explicitly saves the original text to a local SQLite inbox.
Records include workspace, channel, author, capture time, source and project
(`mindmoor`, the primary product repository is `tonybangert/mindmoor`). They start unreviewed. They are not automatically asserted
as Brain facts or included in agent prompts.

The app acknowledges a successful commit with an ephemeral capture ID. Retries
with the same Slack trigger ID return the existing capture. Signing secrets,
verification tokens and response URLs are not stored in the record. The handler
does not call a model, post channel messages, poll Slack history or use response URLs.

`@AIIA <idea>` is also supported through signed `app_mention` events. Original
mention text is retained; Slack event IDs deduplicate retries. Only allowlisted
workspace/channel events are stored. Bot messages and other event types are
ignored. Mentions are acknowledged to Slack after storage, but do not yet receive
an in-channel reply. Signed URL-verification challenges do not create ideas.

The local Brain already has `/v1/aiia/remember` for structured facts and semantic
indexing. A subsequent Studio inbox/review and curator-agent slice should promote
selected captures through that path with provenance. This integration provides
capture storage and a searchable API; it does not yet provide that UI or curator.

## Installation

1. Confirm the Performance Labs workspace and the channel IDs allowed to capture.
2. Inspect the Performance Labs AIIA app and merge the required configuration from
   `config/slack-performance-labs-manifest.json`. Required scopes are `commands`
   and `app_mentions:read`, without channel-history or write access. Reinstall
   when Slack requires updated scope consent, then invite AIIA to the chosen channel.
3. Put `AIIA_SLACK_SIGNING_SECRET`, `AIIA_SLACK_TEAM_ID`, and comma-separated
   `AIIA_SLACK_CHANNEL_IDS` in the Mini's private service environment. Enter secrets
   locally. No bot token is required for this synchronous capture workflow.
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
Existing outbound `slack.post` remains denied in air-gap mode. The old outbound
route references a missing `local_brain.slack_client`; it is not used or repaired
by this capture integration. Connecting outbound notifications or conversations
requires its own scoped transport implementation and egress decision.

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
