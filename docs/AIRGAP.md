# Air-Gap Mode

One flag turns the Brain into a local-only runtime: inference, embeddings,
retrieval, and memory all stay on the box, every cloud egress point is denied
except the explicit Voice Conductor allowlist (`xai.realtime`), and each
denied attempt is reported to Sanction as audit evidence. This is the
enforcement core of **Sanction Local** — the deny-list plus the audit export
*is* the "data never leaves the building" artifact.

## Enabling

```bash
# .env (or the launchd plist environment)
AIIA_AIRGAP=1
```

Effects, applied in `local_brain/config.py`:

- `execution_enabled` is forced **off** (the execution engine spawns the
  `claude` CLI — cloud egress).
- `autonomy.research_enabled` is forced **off** (the literature loop fetches
  arbitrary URLs).
- Every registered egress point (below) is denied by `local_brain/egress.py`,
  except `xai.realtime` (Voice Conductor ephemeral token mint).

Cloud API keys may remain set; they are inert except `XAI_API_KEY`, which
Voice Conductor may use to mint a short-lived xAI token. `aiia doctor`
reports other keys as "configured but inert under AIIA_AIRGAP".

## The egress kill list

| Tool name | Call site | Deny behavior |
|---|---|---|
| `anthropic.messages` / `openai.messages` / `groq.messages` | journal distiller | distillation skipped, raw transcript preserved |
| `groq.whisper` | journal transcription | `TranscriptionError` (local faster-whisper voice path unaffected) |
| `slack.post` | `POST /v1/aiia/slack` | 403 `EGRESS_DENIED_AIRGAP` |
| `google.tts` | speak endpoints | client never initialized; macOS `say` fallback |
| `anthropic.claude_code` | execution engine / story runner | engine refuses to start; runner exits at arg-parse |
| `web.fetch` | research literature loop | force-disabled + fetch guard |
| `xai.realtime` | Voice Conductor ephemeral token | **airgap exception** — status can be `connected` if a key is present; `POST /api/voice/session` may mint |

**Permitted egress:** the Sanction control plane (`SANCTION_API_URL`) —
governance metadata (tool names, token counts, decisions), never content —
plus the Voice Conductor exception (`xai.realtime`) so Studio/PWA can hold
the mic while the Mini mints an xAI ephemeral token. Do not add other tools
to `AIRGAP_ALLOWED_EGRESS`. For a fully offline install that must also
block voice, remove `xai.realtime` from that allowlist (or unset the xAI
key). Point `SANCTION_API_URL` at a local Sanction instance; the client is
config-driven, so this is an env swap, not a code change.

## Decision semantics (fail closed)

`local_brain/egress.py::authorize_egress()`:

- **Air-gap on** → deny, decided locally, unless the tool is in
  `AIRGAP_ALLOWED_EGRESS` (`xai.realtime` only). Denied attempts are still
  POSTed to Sanction `/authorize/tool` so the denial persists in the audit
  trail. A failed audit post never converts a deny into an allow. The
  Voice Conductor exception is a local allow; it does not unset
  `AIIA_AIRGAP` and does not open any other egress point.
- **Air-gap off, Sanction configured** → ask Sanction synchronously; timeout,
  transport error, or any non-`authorized: true` response ⇒ deny.
- **Air-gap off, Sanction unconfigured** → allow (vanilla OSS behavior;
  governance must never break core AIIA).

The Sanction-side policy that mirrors this posture: a non-empty
`allowed_tools` list (e.g. `["local.ollama", "local.chroma",
"sanction.control_plane"]`) — deny-by-default for every other tool name.

## Verifying nothing leaves the box

```bash
# 1. Status surfaces
curl -s localhost:8100/health | jq .airgap
aiia status          # shows AIRGAP=on + the disabled egress list
aiia doctor          # cloud keys reported as inert

# 2. Probe script — denies + audit rows
bash scripts/airgap_probe.sh

# 3. Unit suite
pytest local_brain/tests/test_airgap.py

# 4. Live network watch during a dogfood session
PID=$(pgrep -f local_brain.local_api)
while sleep 10; do
  lsof -i -P -a -p "$PID" -sTCP:ESTABLISHED | grep -v -e 127.0.0.1 -e localhost
done
# Expect ONLY the Sanction control-plane host — plus api.x.ai if Voice
# Conductor minted an ephemeral token (xai.realtime airgap exception).

sudo tcpdump -i any -n 'host api.anthropic.com or host api.groq.com or host api.openai.com or host generativelanguage.googleapis.com'
# Expect silence.
```

## Pulling the evidence artifact

Denied tool authorizations persist as `AuthorizationRequest` rows in Sanction
(kind `tool`, status `denied`, with the policy revision and the exact decision
context — replayable). Export:

```bash
curl -s "$SANCTION_API_URL/audit-events?type=authorization&format=csv" \
  -H "x-api-key: $SANCTION_API_KEY" > airgap-evidence.csv
```

That CSV — every egress attempt, every denial, the policy it ran under — is
the assessor-facing proof.
