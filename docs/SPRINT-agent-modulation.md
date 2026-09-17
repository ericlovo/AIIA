# Sprint: Agent Modulation and Prioritized Memory Posts

Cut from `main` at `486f0ed` on 2026-09-17. Integration branch `sprint/agent-modulation`.

Two epics, four tracks. Tracks build in parallel against the contracts below, so the
contracts are the source of truth. If a track must deviate, it records the deviation in
its summary rather than silently diverging, because another track is coding against it.

## Why

An audit on 2026-09-17 found the Agent Studio Map could change none of an agent's
fourteen editable settings. It saves node positions, and "Edit agent" only switches
tabs. Several settings that look like controls are inert: there is no per-agent model
at all, `suite` adds one prompt line, `memory_namespace` is never read, and the "Local
memory" tool claims retrieval but fetches nothing. Agent update is not a real partial
edit, and enabling a loop through `PUT` skips the check that a loop task exists.

A second audit found no path to post memory to a Slack channel. The only Slack egress
allowed in air-gap mode is a fixed receipt into the capture thread, general
`slack.post` is always denied, and its route imports a module that was never committed.
No priority is stored on memories or captures.

## Owner decisions

- The team is twelve Claude engineers: four tracks, each a builder, a reviewer and a
  verifier. Nothing deploys without the owner.
- Memory posts send **approved text to one channel**. A human sets a priority when a
  capture is logged and explicitly marks it for Slack. Only marked memories are posted,
  with their text, to one allowlisted Performance Labs channel, through a new opt-in
  egress point that is off by default. This narrowly reverses "captured text is never
  transmitted", behind that approval. The content returns to the workspace it came from.
- The destination is `#aiia-memory`, already allowlisted, with the AIIA bot a member.

## Standing rules for every track

- Work only in your assigned worktree, on your track branch. Never push, merge, rebase
  onto, or check out another branch.
- Never touch the production checkout, never restart a service, never call the running
  Command Center or Brain, and never run an agent or a model on the Mini.
- Never send a real Slack request. Slack is exercised only through `httpx.MockTransport`.
- Every Brain call you add must send `AIIA_HEADERS`; `test_brain_proxy_auth.py` enforces it.
- Every commit passes the full CI-equivalent check, plus a browser test for UI changes.
- Keep changes inside your track's files. The files you share with another track are
  listed; touch only your region of them so integration merges cleanly.

---

## Contracts

### C1. Agent partial update

`PATCH /api/agents/{agent_id}`

- Body: a JSON object with any subset of the editable fields: `name`, `mission`,
  `persona`, `skills`, `tools`, `repo_id`, `temperature`, `max_tokens`, `model`,
  `loop_enabled`, `loop_interval_minutes`, `loop_task`, `loop_max_runs_per_day`,
  `suite`, `memory_namespace`.
- Every present field uses the same limits as `AgentCreateRequest`. `model` is new, see C2.
- Unknown field: `422`. Empty body: `422` with detail `empty_patch`.
- Cross-field rule, applied to the merged result: an agent with `loop_enabled: true`
  must have a non-empty `loop_task`, else `422` with detail `loop_task_required`. The
  same rule is now enforced on `PUT /api/agents/{agent_id}` and `POST /api/agents`,
  which fixes the existing bypass.
- `repo_id` is validated exactly as today.
- Unknown agent: `404` with detail `agent_not_found`.
- Success: `200` with `{"agent": <full agent record>}`, and the same studio broadcast as `PUT`.
- Patching a running agent is allowed and affects the next run only; the in-flight run
  already read its settings.

### C2. Per-agent model

- The agent record gains `model: str`, default `""`. Empty means "use the task-role
  default". Existing agents are backfilled with `""` on load.
- `_execute_agent` sends `"model": agent["model"]` to the Brain when it is non-empty,
  and otherwise keeps `"model_role": "task"`. The run ledger records the model actually used.

`GET /api/agents/models`

- Lists installed local chat models from Ollama at `http://127.0.0.1:11434/api/tags`.
  This is localhost, not a cloud egress point.
- Excludes embedding models: a name containing `embed`, or a `details.family` of
  `nomic-bert` or `bert`.
- Response:
  `{"default": "<task model>", "models": [{"id": "qwen3:8b", "label": "qwen3:8b", "family": "qwen3", "parameter_size": "8.2B", "size_gb": 5.2, "default": true}]}`
- `default` is `LOCAL_TASK_MODEL` from the environment, falling back to the Brain default.
- Ollama unreachable: `503` with detail `models_unavailable`.
- Validation in C1 and C3: `model` must be `""` or an `id` present in this list; otherwise
  `422` with detail `unknown_model`. If Ollama is unreachable during validation, reject
  with `503 models_unavailable` rather than accepting an unverifiable model.
- Ollama's `/api/tags` item shape: `{"name", "size" (bytes), "details": {"family", "parameter_size"}}`.

### C3. Suite bulk modulation

`PATCH /api/agent-suites/{suite}/agents`

- Body: the C1 field subset **excluding** `name`, `mission` and `suite`. Identity and
  membership are never bulk-set.
- Applies to every agent whose `suite` equals `{suite}`.
- All-or-nothing: validate the merged result for every member first. If any member
  fails, change none and return `422` with `{"detail": "suite_patch_rejected", "failures": [{"agent_id", "detail"}]}`.
- No agent has that suite: `404` with detail `suite_not_found`.
- Success: `200` with `{"suite": "<suite>", "count": n, "agents": [<records>]}`, one
  studio broadcast per changed agent.
- `GET /api/agent-suites` already exists and stays; it must list every suite actually in use.

### C4. Map handoff creation

No new backend. The Map calls the existing routes directly.

- `POST /api/handoffs` with `{"source_assignment_id", "to_agent_id", "artifact_type": "brief", "instructions"}`.
- `DELETE /api/handoffs/{handoff_id}`.
- `GET /api/handoffs` for the edge list.

### C5. Memory posts

**Capture fields.** The memory inbox `ideas` table gains
`priority TEXT NOT NULL DEFAULT 'normal'`, one of `low`, `normal`, `high`, `urgent`,
and `post_requested INTEGER NOT NULL DEFAULT 0`. The migration is additive and repeatable.

**Promote request.** `POST /api/memory-inbox/{idea_id}/promote` gains
`priority` (default `normal`) and `post_to_slack` (bool, default `false`).
An invalid priority returns `422` with detail `invalid_priority`.

- If `post_to_slack` is true but memory posting is not enabled and configured, return
  `409` with detail `memory_posting_disabled` **before** calling the Brain. Nothing is
  written anywhere, so there is no half-promoted state.
- Otherwise promote as today, record `priority` and `post_requested`, and in the same
  SQLite transaction enqueue exactly one memory post.

**Outbox.** New table `memory_posts`, keyed `memory_id TEXT PRIMARY KEY`, with columns
`idea_id`, `channel_id`, `priority`, `body` (the formatted text snapshot approved at
promote time), `status`, `attempts`, `next_attempt`, `lease`, `error`, `slack_ts`.
It reuses the receipt worker's lease, backoff, 429 handling, permanent-error set and
eight-attempt cap. The claim order is priority first (`urgent`, `high`, `normal`, `low`),
then `next_attempt`. That ordering is what "prioritized" means for delivery.

**Egress.** A new egress point `slack.memory_post` described as
"human-approved memory post to one allowlisted channel (opt-in)". It is a conditional
air-gap exception, allowed only when `AIIA_SLACK_MEMORY_POST_ENABLED=1`, exactly
mirroring `slack.capture_ack`. It is **not** added to the `AIRGAP_ALLOWED_EGRESS`
frozenset. `slack.post` must remain denied, with a test proving it.

**Config.**
- `AIIA_SLACK_MEMORY_POST_ENABLED=1` enables it.
- `AIIA_SLACK_MEMORY_POST_CHANNEL_ID` is the single destination, matching
  `^C[A-Z0-9]{8,}$`. It is a separate outbound setting and does not reuse or extend
  the inbound `AIIA_SLACK_CHANNEL_IDS`.
- The existing `AIIA_SLACK_BOT_TOKEN` and `AIIA_SLACK_TEAM_ID` are reused. The team is
  re-checked at delivery; a mismatch fails permanently with `source_not_allowed`.
- Enabled and configured means: enabled, bot token set, team set, and a valid channel id.

**Formatting.** Plain text with `mrkdwn: false`.
- Strip the leading bot mention, reusing `capture_text`.
- Escape `&` to `&amp;`, `<` to `&lt;`, `>` to `&gt;`. This neutralizes `<!channel>`,
  `<!here>`, `<@U…>` and link syntax.
- Body layout: a header line `[<PRIORITY>] Memory logged to <category>`, a blank line,
  the text, a blank line, then `Capture <first 8 of idea id> · Memory <memory_id>`.
- Cap the text at 3,000 characters, ending with `…` and marking it truncated.
- `unfurl_links: false`, `unfurl_media: false`, `reply_broadcast: false`, no `thread_ts`.
- `client_msg_id` is `uuid5(NAMESPACE_URL, memory_id + ":memory_post")`.

**Receipts are unchanged.** Capture and promotion receipts still never carry captured
text, and their existing tests stay as they are. Memory posts are a separate path.

**Status.** `GET /api/integrations/slack/status` adds
`memory_posts_enabled`, `memory_posts_configured`, `memory_post_channel_id`, and
`memory_posts` status counts. The Memory log shows the "Post to #aiia-memory" control
only when posting is configured.

**Retry.** `POST /api/memory-inbox/{idea_id}/acknowledgement/retry?kind=memory_post`
requeues a failed memory post.

---

## Track 1 — Agent model backend

Worktree `t1`, branch `sprint/t1-agent-backend`.

Owns `agent_registry.py`, `agent_suites.py`, and in `server.py` the agent request
models, the agent and suite routes, `_agent_system_prompt` and `_execute_agent`.
Shares `server.py` with nobody else in this sprint.

**A1. Partial update and the loop fix** (contract C1)
- [ ] `PATCH /api/agents/{id}` changes only the supplied fields and returns the full record.
- [ ] Unknown field and empty body are refused with the specified codes.
- [ ] `loop_enabled: true` with an empty resulting `loop_task` is refused on PATCH, PUT and create.
- [ ] A registry test proves an omitted field is left untouched after reload from disk.

**A2. Per-agent model** (contract C2)
- [ ] `model` is stored, backfilled to `""` on load, and returned in every agent record.
- [ ] `GET /api/agents/models` lists chat models, excludes embeddings, and marks the default.
- [ ] `_execute_agent` sends the agent's model when set and the task role otherwise; a test captures the outgoing Brain payload in both cases.
- [ ] An unknown model is refused with `unknown_model`; unreachable Ollama gives `models_unavailable`.

**A7b. Suite bulk modulation** (contract C3)
- [ ] One request changes every agent in a suite.
- [ ] One failing member leaves every member unchanged, proven against the file on disk.
- [ ] Identity and membership fields are refused in a bulk patch.

**A8. Honest "Local memory"** — last in the track, droppable
- [ ] When an agent has the "Local memory" tool, the run injects real retrieved memories with their ids, at most 6 entries and 1,500 characters, filtered by `memory_namespace` when set, via the authenticated Brain memory endpoint.
- [ ] Any failure injects "Local memory was unavailable for this run" instead of claiming retrieval.
- [ ] If this cannot be done honestly, remove the claim and document why. Never claim retrieval that did not happen.

## Track 2 — Map inspector and controls

Worktree `t2`, branch `sprint/t2-map-inspector`, browser port `5191`.

Owns a new `dashboard/src/console/AgentInspector.tsx` for the **agent** node inspector.
In `AgentGraphOverlay.tsx`, the only change is rendering `<AgentInspector …/>` in place
of the agent inspector block. In `api.ts`, add `patchAgent` and `agentModels` only.
Track 3 owns edges, the handoff wire, and the assignment-node actions in the same overlay.

**A3. The inspector shows the real configuration**
- [ ] Shows model (or "Task default: <model>"), temperature, max tokens, tools, skills, repo, suite, loop state with interval and runs today of the daily maximum, persona, and the last result and last error, truncated.

**A4. Edit agents on the Map** (consumes C1 and C2)
- [ ] Inline controls for model, temperature, max tokens, suite, loop on or off, loop interval, and loop daily maximum.
- [ ] Each change sends a PATCH with only that field.
- [ ] Updates are optimistic and roll back with a visible message on error, including readable text for `loop_task_required`, `unknown_model` and `models_unavailable`.
- [ ] While the agent is running, controls stay usable and say the change applies to the next run.
- [ ] Controls are keyboard operable and labelled.

**A5. Run an agent from the Map**
- [ ] A "Run now" control takes a task and calls `POST /api/agents/{id}/run`.
- [ ] `409 mini_busy` shows a clear busy message; a pending run disables the control.

**Tests.** Unit tests for any pure logic, and `dashboard/tests/agent-inspector.browser.mjs`
using fully synthetic API responses at 1440 and 390 pixels. Cover the config display, a
successful PATCH, a rolled-back failed PATCH, the model picker, and a busy run.

## Track 3 — Map relationships and suites on the Map

Worktree `t3`, branch `sprint/t3-map-relationships`, browser port `5192`.

In `AgentGraphOverlay.tsx`, owns edge rendering, the handoff wire, and the
**assignment**-node actions. Owns `graphLayout.ts` suite grouping and `AgentStudio.tsx`
`routeHandoff`. In `api.ts`, add `createHandoff`, `deleteHandoff`, `agentSuites` and
`patchSuiteAgents` only, reusing any that already exist. Track 2 owns the agent inspector.

**A6. The handoff wire creates the handoff** (contract C4)
- [ ] Dropping a wire from a completed assignment onto an agent opens an inline confirm with editable instructions, then POSTs the handoff directly instead of switching tabs.
- [ ] The new edge appears without a reload, and errors are shown inline.

**A6b. Edges you can act on**
- [ ] Handoff edges are clickable and keyboard reachable, and show source, target, status and created time.
- [ ] A selected handoff edge can be removed with `DELETE`, with a confirm step.
- [ ] Selecting an agent-to-assignment edge opens that assignment.

**A7. Suites on the Map**
- [ ] Nodes show their suite with a consistent colour and label.
- [ ] A suite legend filters the Map to one suite, and clearing restores all agents.
- [ ] Layout keeps working with no suites and with many.

**A7c. Modulate a whole suite from the Map** (consumes C3)
- [ ] From the legend, a suite panel sets model, temperature, max tokens, and loop settings for every member in one request.
- [ ] A `suite_patch_rejected` response lists the failing agents and changes nothing in the UI.

**Tests.** Unit tests for layout and grouping, and `dashboard/tests/map-relationships.browser.mjs`
with synthetic responses at 1440 and 390 pixels. Cover wire to created handoff, edge
select and remove, suite filter, and a suite patch that succeeds and one that is rejected.

## Track 4 — Prioritized memory posts

Worktree `t4`, branch `sprint/t4-memory-posts`, browser port `5193`.

Owns `memory_inbox.py`, `slack_receipts.py` or a new `slack_memory_posts.py`,
`slack_capture.py`, `egress.py`, `local_api.py` for the dead route, `MemoryLog.tsx`,
`memoryText.ts`, and `docs/AIRGAP.md` plus `docs/SLACK-PERFORMANCE-LABS.md`. In
`api.ts`, extend only the memory inbox and Slack status types and calls.

**B1. Priority set at log time** (contract C5)
- [ ] `priority` and `post_requested` are stored with an additive, repeatable migration.
- [ ] Promote accepts `priority`, and the Memory log sets it when logging and can filter or sort by it.

**B2. Memory post outbox and egress** (contract C5)
- [ ] Promote with `post_to_slack` enqueues exactly one post in the same transaction, and never a second on retry.
- [ ] Posting disabled returns `memory_posting_disabled` before any Brain call, proven by a test whose Brain transport fails if called.
- [ ] Delivery goes only to the configured channel, re-checks the team, and claims urgent before low.
- [ ] `slack.memory_post` is allowed only with its flag, and `slack.post` stays denied in air-gap mode, with a test for both.
- [ ] Retries, 429s, permanent errors, the lease and the failed-post retry route all behave like receipts, with tests.

**B3. Slack-safe formatting**
- [ ] Escaping neutralizes `<!channel>`, `<!here>`, user mentions and links, with tests.
- [ ] Text over 3,000 characters is truncated and marked.
- [ ] The bot mention is stripped, and the header carries priority and category.

**B4. Retire the dead route and fix the docs**
- [ ] `POST /v1/aiia/slack` and its missing import are removed, and `scripts/airgap_probe.sh` and `tests/test_airgap.py` are updated so they no longer depend on it, while the `slack.post` egress point stays registered and denied.
- [ ] `docs/AIRGAP.md` documents both conditional exceptions and keeps "do not add to the frozenset".
- [ ] `docs/SLACK-PERFORMANCE-LABS.md` resolves its contradictory scope statements, documents memory posts and the activation steps, and states the policy change plainly.

**Tests.** Backend tests for every C5 rule, and `dashboard/tests/memory-posts.browser.mjs`
with synthetic responses at 1440 and 390 pixels. Cover setting priority, the post control
hidden when posting is unconfigured, a successful post-marked log, and the disabled error.

---

## Integration

Done by the owner's session after all tracks report, never by a track. Merge order is
Track 1, Track 4, Track 2, Track 3, resolving conflicts in `AgentGraphOverlay.tsx` and
`api.ts`. Then run the full check and every browser test on the integrated tree, and an
end-to-end pass wiring the Map controls to the real Track 1 routes. Deploying and turning
on memory posts are separate owner decisions.

## Activation, after deploy, owner only

1. Confirm the AIIA bot is still a member of `#aiia-memory`.
2. Set `AIIA_SLACK_MEMORY_POST_ENABLED=1` and `AIIA_SLACK_MEMORY_POST_CHANNEL_ID` in the private service environment, then restart.
3. Log one clearly labelled synthetic capture with "Post to #aiia-memory" checked, and confirm exactly one message arrives before logging anything real.
