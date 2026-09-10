# Voice Conductor

> First slice: talk to the Mini / Command Center with Grok Voice and
> orchestrate Agent Studio work as typed Assignments — not a free agent
> shell.
>
> Status: implemented (v1). Audience: Eric placing a key on the Mini, and
> an engineer extending the allowlist.

Voice is an interface over the executable-org model in
[`docs/EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md). The
human speaks. Grok reasons. Command Center creates or runs Assignments.
The Mini executes privately. Handoffs stay typed edges you inspect; this
slice does not fire them.

**Not this:** Journal voice memos (`docs/JOURNAL.md`), the legacy
`/voice` HTML page, or local faster-whisper `/v1/voice/transcribe`.
Those stay. Voice Conductor is Agent Studio push-to-talk on `:8200`.

---

## 1. Product bounds

| In | Out |
|---|---|
| Hold-to-talk / spacebar PTT on Agent Studio | Always-on room mic / Sonos |
| Live transcript + speaking state + tool chips | Phone / Twilio agent |
| Read agents, assignments, handoffs, repos, GitHub status, Mini busy | Custom voice cloning |
| Create an Assignment for an **existing** agent | World canvas (#49), GitHub App (#47) |
| Run / retry a `queued` or `failed` Assignment | `git push` / `open_pr` via voice |
| Fail closed without `XAI_API_KEY` | Arbitrary shell, file write, Sanction spend |

Depth is honest: **D2 orchestrator**. The conductor creates work for
specialists. It is not a D4 executive (no budgets, no Specialty
installs, no Handoff writes).

---

## 2. Architecture

```
Browser (Agent Studio :8200)
    │  GET  /api/voice/status          → connected | not_configured
    │  POST /api/voice/session         → ephemeral token + session.update
    │  POST /api/voice/tools           → allowlisted Command Center calls
    │
    ├─ WebSocket wss://api.x.ai/v1/realtime?model=grok-voice-latest
    │     auth: sec-websocket-protocol  xai-client-secret.<ephemeral>
    │     PCM 24 kHz  ·  PTT (turn_detection: null)  ·  voice eve
    │
Mini / Command Center
    │  XAI_API_KEY from env or ~/.aiia/keys.json   (never sent to the browser)
    │  Tool handlers → agent_registry / assignment_registry / github_status
    │  run_assignment → existing Mini runner (409 mini_busy)
    ▼
Mini local model (qwen / Ollama) executes the Assignment
```

Grok is the **voice + reasoning plane only**. Local-first still holds:
the Mini runs the work. Cloud sees speech, the system prompt, and the
tool schemas / results — not your long-lived key, not a shell.

Existing Whisper / Google TTS paths are untouched.

---

## 3. Key handling

Never put `XAI_API_KEY` in the dashboard bundle, query string, or
WebSocket from the browser.

Load order (first non-empty wins):

1. Environment `XAI_API_KEY`
2. `~/.aiia/keys.json` fields, in order: `XAI_API_KEY`, `xai_api_key`,
   `xai`, or `providers.xai`

That file is the same Mini secret pattern documented for the console
keystore (`docs/SECURITY-ARCHITECTURE.md`): `0700` on `~/.aiia`, `0600`
on `keys.json`.

```json
{
  "xai": "xai-..."
}
```

`GET /api/voice/status` reports `connected` or `not_configured`. It
never echoes the key. `POST /api/voice/session` mints a **300s**
ephemeral token from `POST https://api.x.ai/v1/realtime/client_secrets`
and returns only that token plus the session config (voice, tools,
instructions).

Under `AIIA_AIRGAP=1`, Voice Conductor is an **intentional exception**:
`xai.realtime` is on `AIRGAP_ALLOWED_EGRESS`. Status can be `connected`
when a key is present, and `POST /api/voice/session` may mint an
ephemeral token. Every other registered cloud egress point stays
denied. This is not a global air-gap off — see `docs/AIRGAP.md`.

---

## 4. Tool allowlist

Advertised to Grok as `function` tools only. xAI server tools
(`web_search`, `x_search`, `file_search`, `mcp`) are **not** included
in `session.update`. Execution is always server-side via
`POST /api/voice/tools`.

### Read

| Tool | Backing API |
|---|---|
| `list_agents` | `agent_registry.list()` |
| `list_assignments` | `assignment_registry.list_assignments()` |
| `list_handoffs` | `assignment_registry.list_handoffs()` |
| `list_resources` | `available_repos()` + `github_status()` |
| `mini_status` | `agent_run_lock.locked()` |

### Act (bounded)

| Tool | Backing API | Fail closed |
|---|---|---|
| `create_assignment` | `assignment_registry.create_assignment` | unknown `agent_id` → `agent_not_found` |
| `run_assignment` | `POST /api/assignments/{id}/run` | not `queued`/`failed` → 409; lock held → `mini_busy` |

### Forbidden (always 403)

`run_shell`, `shell`, `execute_command`, `git_push`, `push`, `open_pr`,
`git_write`, `write_file`, `approve_git_write`, `approve_git_workspace`,
`create_handoff`, `delete_agent`, `delete_assignment`, `sanction_spend`,
`authorize_spend`, `gh_token`, `web_search`, `x_search`, `file_search`,
`mcp`, plus any unknown name (`tool_not_allowlisted`).

A tampered browser can send a different `session.update` (xAI ephemeral
tokens do not bind session tools at mint time). That cannot widen
**our** allowlist. It could enable xAI-hosted search on their side —
mitigated by a 300s token, never shipping the long-lived key, and the
narrow air-gap exception (mint only; Command Center tools stay
fail-closed). A future slice can proxy the WebSocket if that gap
must close.

---

## 5. Threat model

| Threat | Mitigation |
|---|---|
| Long-lived key in DevTools | Key stays on Mini; browser gets ephemeral token only |
| Voice asks to `git push` / open a PR | Name not allowlisted; handler 403 |
| Voice invents a new agent id | `create_assignment` requires a live registry row |
| Voice piles work on a busy Mini | `run_assignment` returns `mini_busy` (409); Assignment stays queued |
| Air-gapped Mini with a key still set | `xai.realtime` is allowlisted: status `connected` if key present; mint allowed. Other egress stays denied |
| Prompt injection via repo text | Tools do not read file contents; repo mounts are status-only |
| Personal `gh` token leakage | `list_resources` returns connection status, never credentials |
| Sanction spend via voice | No spend / authorize tool |

Audio is processed by xAI in realtime (their retention: not stored / not
used for training per their Voice docs). Voice Conductor is the one
registered air-gap exception (`xai.realtime`); speech still leaves the
box to xAI. Keep `AIIA_AIRGAP=1` for every other cloud call.

---

## 6. Specialty (seed, not auto-installed)

```
slug: voice-conductor@0.1.0
depth: 2
maxDelegationDepth: 0
requiredApprovalLevel: supervised
requiredTools: list_agents, list_assignments, list_handoffs,
               list_resources, mini_status, create_assignment, run_assignment
```

This is a docs + API seed (`VOICE_CONDUCTOR_SPECIALTY` in
`local_brain/command_center/voice_conductor.py`). Command Center does
**not** auto-create an Agent Studio agent on boot. If you want a
visible card, create an agent named "Voice Conductor" with that
mission; voice still talks to the Mini through the panel, not as that
agent's loop.

---

## 7. How Eric tries this

On the Mini:

```bash
# either
echo 'XAI_API_KEY=xai-...' >> ~/.aiia/.env   # however you already load env
# or
printf '%s\n' '{"xai":"xai-..."}' > ~/.aiia/keys.json
chmod 600 ~/.aiia/keys.json
```

Restart Command Center (`python -m local_brain.command_center.server`
or the launchd / compose unit that serves `:8200`).

Open `http://<mini>:8200` (or localhost if you are on the box). Agent
Studio shows a **Voice Conductor** bar at the bottom.

- Without a key: pill says **not configured**. No crash.
- With a key: hold the mic (or hold spacebar outside a text field).
  Speak something like: “List the agents, then create a normal
  assignment for Research titled Scout the repo.” Release. Grok should
  call tools; chips appear; the Assignments tab updates.

Microphone permission is a browser prompt. Desktop and phone browsers
both work; the bar stays below Agents / Assignments / Handoffs and does
not occupy a World canvas.

---

## 8. Acceptance

- [x] `GET /api/voice/status` is `not_configured` without a key
- [x] `POST /api/voice/session` is 503 without a key
- [x] Allowlisted tools execute; forbidden / unknown tools 403
- [x] `run_assignment` respects `mini_busy`
- [x] Tests use fixture strings only (`xai-test-fixture`); no live secrets
- [x] Dashboard PTT + transcript + tool chips
- [x] `XAI_API_KEY` noted in `.env.example`
- [x] CHANGELOG `[Unreleased]`

### Later (explicitly out of scope)

Always-on room / Sonos, Twilio, voice cloning, World canvas merge,
enabling git push / open_pr, binding ephemeral tokens to a fixed
session tool set (needs an xAI API change or a server-proxied WS).
