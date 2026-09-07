# Handoff — Astra / Agent Studio tune-up (2026-09-07)

**From:** Astra (bigger-picture follow-up)  
**For:** Eric via AIIA Bot · Mini operators · next engineer on suite + Run ledger  
**Repo:** `ericlovo/AIIA` · base `main`  
**Air-gap:** stays **ON** (`AIIA_AIRGAP=1`). Do not open xAI Voice egress.

This is a session / product handoff. It is **not** the org-graph Handoff
entity. Typed edges live in
[`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md).
The suite contract that this pass produced is
[`MINDMOOR-AGENT-SUITE.md`](./MINDMOOR-AGENT-SUITE.md).

---

## TL;DR

A serial **23-agent** Agent Studio pass ran on the live Mini. One email
agent was skipped; one Mindmoor reviewer hard-failed empty at
`max_tokens=600`; eight agents were weak (low caps / format drift);
thirteen were OK.

Live Mini already has a runtime-only bump for **Mindmoor Cron Review
Gate** (`max_tokens` 600→1200 + output-contract nudge). That is **not**
in git and must not be committed as `agent_data.json`.

Follow-up in this repo: suite spec + optional Studio `suite` /
`memory_namespace` scaffold. Next after that: Run ledger. Still out of
scope: unset air-gap, Sonos, email, GitHub App (#47), Voice live demo.

---

## 1. Pass scoreboard

| Outcome | Count | Notes |
|---|---|---|
| Ran | 22 | Serial, one `agent_run_lock` |
| Skipped | 1 | Tony's Assistant (email) — out of scope |
| Failed | 1 | Mindmoor Cron Review Gate |
| Weak | 8 | Low caps and/or format drift |
| OK | 13 | Usable structured output |
| **Fleet** | **23** | 22 + 1 skipped |

Failed + weak + OK = 22 ran. Do not invent agent IDs; names below are
Studio display names from the pass.

### 1.1 Hard fail

| Agent | Error | Cause | Live Mini (runtime only) |
|---|---|---|---|
| Mindmoor Cron Review Gate | `empty_agent_result` | `max_tokens=600` at the 2000 ceiling's low end; model returned blank | Bumped to **1200** + output-contract nudge (State / Signals / Risks / Next). Not committed. |

`empty_agent_result` is already a first-class failed Run in code
(`agent_registry.finish_run`, `_execute_agent` → HTTP 502). Activity /
Needs Attention will show it. The miss was **budget + contract**, not
the failure detector.

Recommend 1600 for this reviewer on the next Mini edit (still ≤ 2000).

### 1.2 Weak (low caps / format drift)

- Repo Diff Watcher
- Memory Note
- Ambiguous Remote Canary
- Delivery Watch
- Cron Test Engineer
- Specialty Probe
- Mindmoor Scout
- Dependency Diplomat

Common pattern: token cap too tight for the mounted repo/GitHub
context, **or** no closed output contract, so the model wandered.
Several of these are Mindmoor suite members (see §2). Treat the weaks
as "raise floor to 1200 + pin the four-heading contract", not as
rename/replace.

### 1.3 OK exemplars (copy their shape)

- Mindmoor Discovery Bot
- Weekend Repo Brief
- CI Signal Officer
- Release Conductor
- Sanction Policy Reviewer
- Test Cartographer

These held a structured brief at 1200-class caps. New or retuned
agents should clone **their** heading contract and token floor, not
the failed Gate's old 600.

---

## 2. Product read: Mindmoor is a suite, not six islands

Six names in the pass belong together as the **Mindmoor Agent Suite**
inside Studio. Client-isolated, shared local memory, one mount.

| Member | Pass result | Depth (spec) |
|---|---|---|
| Delivery Watch | Weak | D0 Observe |
| Cron Review Gate | Fail (then live 1200) | D1 Review |
| Cron Test Engineer | Weak | D1 Interpret |
| Specialty Probe | Weak | D1 Interpret |
| Mindmoor Scout | Weak | D1→**D0** Observe in spec |
| Discovery Bot (Mindmoor Discovery Bot) | OK | D1 Interpret |

Full contract: [`MINDMOOR-AGENT-SUITE.md`](./MINDMOOR-AGENT-SUITE.md).

Implications Eric asked for:

1. **Air-gapped / client-isolated.** Suite members do not get Voice,
   email, or web tools. They do not read other clients' memory.
2. **Shared local AIIA memory namespace** `mindmoor` (`source=suite:mindmoor`).
   Reuse existing categories (`project`, `decisions`, `agents`,
   `lessons`). No tenth category.
3. **Mount / GitHub-read caching.** Serial suite ticks re-hit the same
   `mindmoor` snapshot. Spec calls for a 90s process-local cache;
   implementation may follow. Cache negatives (disconnected) too.
4. **Typed Handoffs** when we graduate islands → org graph. Empty
   Sanction grant still means no fire. Isolated loops keep working.
5. **Depth D0–D2 only.** No Lead/Executive on this client yet.
   `maxDelegationDepth = 0`.
6. **Mini serial budgets.** One lock. Stagger intervals. Token floors
   1200 (Gate 1600). Daily suite sketch: 24 runs / 48k tokens cap.

Not suite members (even if they ran in the same 23): Weekend Repo
Brief, CI Signal Officer, Release Conductor, Sanction Policy Reviewer,
Test Cartographer, Repo Diff Watcher, Memory Note, Ambiguous Remote
Canary, Dependency Diplomat, Tony's Assistant.

---

## 3. What landed in git (this follow-up)

| Item | Path |
|---|---|
| Suite spec | `docs/MINDMOOR-AGENT-SUITE.md` |
| This handoff | `docs/HANDOFF-ASTRA-STUDIO-TUNEUP-2026-09-07.md` |
| Catalog + match helpers | `local_brain/command_center/agent_suites.py` |
| Optional agent fields | `suite`, `memory_namespace` on registry + create API |
| Prompt line | `_agent_system_prompt` states suite + namespace when tagged |
| List endpoint | `GET /api/agent-suites` |
| Tests | `local_brain/tests/test_agent_suites.py` |

No runtime JSON. No air-gap flag change. No Voice demo work.

---

## 4. What remains runtime-only on Mini

Do **not** expect these in the PR checkout. Apply on Mini after pull
if still desired:

| Item | Where | Git? |
|---|---|---|
| Cron Review Gate `max_tokens=1200` (was 600) | Mini `agent_data.json` | no |
| Cron Review Gate output-contract nudge | Mini `loop_task` / instructions | no |
| Any other live cap bumps from the pass | Mini only | no |
| Live agent UUIDs / hex ids | Mini registry | never invent in docs |
| Process snapshot cache | not implemented yet | n/a |
| `AIIA_AIRGAP=1` | Mini env / launchd | stays on; not changed here |
| Voice key inert under air-gap | Mini | leave closed |

After deploy: restart Command Center / `com.aiia.brain` so
`GET /api/agent-suites` is live. Tagging existing cards `suite=mindmoor`
is an operator PUT; the catalog will also match the six names by alias
if the field is still empty.

---

## 5. Out of scope (repeat)

- Unset air-gap
- Sonos
- Email agents (Tony's Assistant)
- Merging #47 GitHub App path
- Full Run ledger (see suite spec §10 — next after this)
- Parallel Mini runs
- Live Voice demo / opening `xai.realtime`

---

## 6. Suggested next Assignments (human / D3 later)

1. **Operator on Mini:** set suite tags + memory namespace on the six
   members; raise remaining weaks to 1200 + four-heading contract;
   consider Gate at 1600.
2. **Engineer:** process-local mount/GitHub snapshot cache (90s TTL).
3. **Engineer:** Run ledger (`~/.aiia/org/runs/…`) so a 23-agent pass
   is queryable without scraping 12-deep `runs[]` arrays.
4. **Later:** install typed Handoffs from suite spec §7 with **local**
   Sanction grants. Do not fire edges into email or Voice.

---

## 7. Pointers

- Org model: [`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md)
- Air-gap: [`AIRGAP.md`](./AIRGAP.md)
- Voice (do not enable under this handoff): [`VOICE-CONDUCTOR.md`](./VOICE-CONDUCTOR.md)
- Session notebook index: [`HANDOFF.md`](./HANDOFF.md)
