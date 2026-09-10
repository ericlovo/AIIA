# Mindmoor Agent Suite

> Status: spec + thin Studio tagging scaffold. Live Mini agents stay on
> Mini (`agent_data.json` is not committed).
> Audience: an engineer wiring suite membership, shared memory, and
> typed Handoffs without re-reading the Studio tune-up pass.

**Aligns with:** [`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md)
(Assignments, typed Handoffs, depth `D0`–`D4`, Sanction-gated edges).
This file is the first **client-isolated suite** inside Agent Studio —
not a second organization model.

**Not this file:** [`HANDOFF.md`](./HANDOFF.md) (Mini ⇄ MacBook session
notebook) and [`HANDOFF-ASTRA-STUDIO-TUNEUP-2026-09-07.md`](./HANDOFF-ASTRA-STUDIO-TUNEUP-2026-09-07.md)
(the 2026-09-07 Studio pass findings that motivated this spec).

| Plane | Owns here | Does not own |
|---|---|---|
| **AIIA / Studio** | Suite catalog, member tagging, shared local memory namespace, mount/GitHub read cache | Client data leaving the Mini |
| **Sanction** | Whether a Handoff may fire | Running the model |
| **Mini** | Serialized local compute (`agent_run_lock`) | Policy. Air-gap stays on. |
| **Mindmoor (client)** | Product facts that land in the shared namespace | AIIA org graph, Voice, email |

---

## 1. Why a suite

Agent Studio today treats every loop agent as an island
([`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §3.1). That
is the correct safety default. Mindmoor needs a **named cluster** of
those islands that:

1. Share one **local** AIIA memory namespace (`mindmoor`).
2. Mount the same read-only `mindmoor` repo (and optional GitHub-read
   snapshot of that origin).
3. May later pass typed artifacts along **granted** Handoffs.
4. Stay **client-isolated** — no other tenant's memory, no Voice egress,
   no email, no web fetch.

Without a suite tag, six Mindmoor roles are six unrelated Studio cards.
With a suite tag, they are still isolated at runtime until a Handoff
fires, but operators and the scheduler can reason about them as one
budget and one memory scope.

---

## 2. Non-goals (this increment)

- Unsetting `AIIA_AIRGAP=1`. Air-gap stays **on**. See [`AIRGAP.md`](./AIRGAP.md).
- Opening xAI Voice egress (`xai.realtime`). Voice Conductor remains
  configured-but-inert under air-gap.
- Tony's Assistant / any email agent.
- Merging GitHub App path (#47).
- Full Run ledger implementation (outline only, §10).
- Raising Mini concurrency. The scheduler stays serialized.
- D3/D4 Mindmoor roles, org budget allocation, or auto-firing Handoffs
  without a local Sanction grant.
- Committing live Mini `agent_data.json` / assignment JSON.

---

## 3. Isolation and air-gap

Mindmoor is a **client**, not a SaaS tenant in the Brain. Isolation is
enforced by configuration and grants, not by a multi-org runtime.

| Bound | Rule |
|---|---|
| Air-gap | `AIIA_AIRGAP=1`. Isolated loop Runs keep working. Handoffs that need cloud-side effects stay denied. Observe-only local artifact movement still needs a grant from a **local** Sanction instance ([`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §8.1). |
| Memory | Suite members read/write only the `mindmoor` namespace (see §5). They do not query other clients. |
| Mounts | Default `repositoryId = mindmoor` (`REPO_MOUNTS` in `repository_tools.py`). Other mounts are out of role unless an operator explicitly retags. |
| GitHub | Read-only GET adapter, still disconnected unless `gh` is authenticated **and** air-gap policy later allows that control-plane call. Runs must not claim live remote state when the snapshot says disconnected. |
| Voice / email / web | Out of suite. Do not add those tools to member agents. |
| Execution | Propose-only system prompt until a Handoff + grant says otherwise. Existing AUTO / SUPERVISED / GATED engine is unchanged. |

---

## 4. Suite membership

Membership is by **display name / slug**, not by live Mini agent IDs.
Those IDs exist only on the Mini and must not be invented in this repo.

Canonical suite slug: `mindmoor`.

| Slug | Studio name (and aliases) | Depth | Role layer | Typical artifacts |
|---|---|---|---|---|
| `delivery-watch` | Delivery Watch | D0 | Observe | `signal_report` |
| `mindmoor-scout` | Mindmoor Scout | D0 | Observe | `signal_report` |
| `discovery-bot` | Discovery Bot, Mindmoor Discovery Bot | D1 | Interpret | `signal_report`, `documentation` |
| `specialty-probe` | Specialty Probe | D1 | Interpret | `signal_report`, `failure_analysis` |
| `cron-test-engineer` | Cron Test Engineer | D1 | Interpret | `signal_report`, `failure_analysis` |
| `cron-review-gate` | Cron Review Gate, Mindmoor Cron Review Gate | D1 | Decide (review) | `risk_review`, `audit_record` |

Suite ceiling is **D2**. No member is D3/D4. `maxDelegationDepth = 0`
for every migrated loop agent (same default as
[`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §10.1).

D2 is reserved for a future reversible operator (for example a release
or cron-apply role) and is **not** instantiated in this increment.
Cron Review Gate is a reviewer, not an actor: it emits `risk_review`,
it does not apply cron changes.

Code catalog: `local_brain/command_center/agent_suites.py`.
`GET /api/agent-suites` lists the catalog and any matching Studio
agents (explicit `suite` tag **or** name/alias match).

### 4.1 Not suite members

Named in the 2026-09-07 Studio pass but **outside** this suite. Do not
auto-tag them `mindmoor`.

| Name | Why listed in the pass | Suite? |
|---|---|---|
| Weekend Repo Brief | OK exemplar (format / cap) | no |
| CI Signal Officer | OK exemplar | no |
| Release Conductor | OK exemplar | no |
| Sanction Policy Reviewer | OK exemplar | no |
| Test Cartographer | OK exemplar | no |
| Repo Diff Watcher | Weak (low cap / format) | no |
| Memory Note | Weak | no |
| Ambiguous Remote Canary | Weak | no |
| Dependency Diplomat | Weak | no |
| Tony's Assistant | Skipped (email) | no — out of scope |

---

## 5. Shared local memory namespace

One namespace for the suite. Local AIIA memory only. No cloud memory
product, no per-agent private vault that the others cannot read.

```
MemoryScope {
  collections: ["project", "decisions", "agents", "lessons"]
  write: true
  namespace: "mindmoor"
  source: "suite:mindmoor"
}
```

This is the Organization.`memoryScope` shape from
[`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §6.1 plus a
`namespace` / `source` pair so existing `eq_brain.memory.Memory`
categories stay unchanged.

| Rule | Detail |
|---|---|
| Write | `Memory.remember(..., source="suite:mindmoor", metadata={"namespace": "mindmoor", "suite": "mindmoor"})` |
| Read | Filter recall by `source == "suite:mindmoor"` or `metadata.namespace == "mindmoor"`. Do not dump the whole vault into a Mindmoor Run. |
| Categories | Reuse `project`, `decisions`, `agents`, `lessons`. Do **not** add a tenth Memory category for the client. |
| Isolation | Other Studio agents omit this source. They must not write `suite:mindmoor`. |
| Air-gap | Memory I/O is local disk / Chroma. No egress. |

Studio field: optional `memory_namespace` on the agent record. When
`suite=mindmoor` and the field is empty, the registry fills
`mindmoor`. The system prompt then states the namespace so the model
does not invent a different one.

---

## 6. Mount and GitHub-read caching

Serial Mini passes re-read the same `mindmoor` mount (and the same
GitHub GET snapshot) for every member. Uncached, six due loops in a
row pay six `git` + optional `gh api` tax on one lock.

| Cache | Key | TTL | Invalidate |
|---|---|---|---|
| Repo snapshot | `repo_id` | 90s | `HEAD` changed, or worktree status mtime newer than cache |
| GitHub-read snapshot | `repo_id` + origin slug | 90s | same HEAD change, or `github_status` flipped |
| Disconnected / missing mount | same keys | 90s | cache the negative too — do not retry `gh` on every agent in the same tick window |

Implementation target: `local_brain/command_center/repository_tools.py`
(`repo_snapshot`, `github_snapshot`). Process-local, not a committed
file. Safe under air-gap: a cached "disconnected" string is still
honest.

This increment specifies the contract. A cache implementation may land
in a follow-up; suite tagging does not depend on it.

---

## 7. Typed Handoffs (legal graph)

Same invariant as the org spec: the only legal A→B path is a Handoff
with a matching `artifactType` and an **active** Sanction grant.
Empty grant ⇒ the edge does not fire. Isolated loops (no Handoff)
keep working.

```
Mindmoor Scout      --signal_report-->  Discovery Bot
Delivery Watch      --signal_report-->  Cron Review Gate
Specialty Probe     --signal_report-->  Discovery Bot
Cron Test Engineer  --failure_analysis-->  Cron Review Gate
Discovery Bot       --documentation-->  (human / memory curator)
Cron Review Gate    --risk_review-->    (human; no D2 actor yet)
```

Illegal (must refuse):

- Scout or Delivery Watch producing `risk_review` (`artifact_type_mismatch`).
- Any member `call_agent` / "please also ping X" (`handoff_required`).
- Cron Review Gate applying cron or writing the repo (propose-only;
  no action grant; Git writes stay approval-gated and out of this suite).
- A Handoff to Tony's Assistant or any email / Voice surface.

Today's Studio Handoff artifact enum is still
`brief | analysis | plan | decision | review`
(`assignment_registry.py`). Map as follows until org v1 artifacts
land: `signal_report`→`brief`, `failure_analysis`→`analysis`,
`risk_review`→`review`, `documentation`→`brief`. Do not invent new
runtime enum values in this increment.

---

## 8. Depth D0–D2 (authority, not IQ)

Depth is an authorization class
([`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §5).
Changing `qwen3:8b` for a stronger model does not raise depth.

| Depth | Suite use |
|---|---|
| **D0** | Scout, Delivery Watch. Read mount + optional GitHub-read. Emit `signal_report` only. |
| **D1** | Discovery Bot, Specialty Probe, Cron Test Engineer, Cron Review Gate. Produce bounded artifacts. No child Assignments. |
| **D2** | Reserved. Not assigned. Would be the first reversible operator under an active grant. |
| **D3+** | Out of suite. |

`maxDelegationDepth = 0` for all six. Cost brakes remain: Mini
serialization + Assignment `budgetCap` + this zero.

---

## 9. Mini serial capacity and budgets

Ground truth from Agent Studio
([`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §3.1):

| Bound | Value |
|---|---|
| Concurrent Runs | 1 (`agent_run_lock`) |
| `max_tokens` | 128–2000 |
| Temperature | 0–1 |
| Loop interval | 15–1440 min |
| Runs / agent / day | 1–48 |
| Agent cap | 40 Studio agents |
| Run history | last 12 rows / agent |

### 9.1 Recommended compute (not live Mini IDs)

From the 2026-09-07 serial pass: one hard fail at `max_tokens=600`
(`empty_agent_result`), eight weaks from low caps / format drift.
OK exemplars held a structured contract at 1200-class caps.

| Role | `maxTokensPerRun` floor | `maxRunsPerDay` | Interval hint |
|---|---|---|---|
| D0 observers | 1200 | 4 | 60–180 min, stagger starts |
| D1 interpret | 1200 | 4 | 60–180 min |
| Cron Review Gate | 1600 (live Mini already 1200 + output-contract nudge) | 4 | after test-engineer, not same tick |

Do not schedule all six as `due` with identical intervals. The
scheduler already picks **one** due agent per 15s tick, but a stacked
queue still occupies the Mini for the sum of wall times.

Budget sketch for the suite (tokens kind, period `day`):

```
Budget { kind: "tokens", cap: 48000, period: "day" }
  # 6 agents × 4 runs × ~2000 max_tokens ceiling
Budget { kind: "runs_per_day", cap: 24, period: "day" }
```

Operators may lower, not raise, without a GATED Assignment (org spec
§9). Live Mini bumps already applied for Cron Review Gate
(`600→1200` + output-contract) stay **runtime-only** until an operator
copies them; this repo does not commit those JSON rows.

### 9.2 Output contract (nudge)

Weak and failed Runs in the pass drifted off structure or returned
empty. Every suite member's `loop_task` / Assignment objective should
end with a closed contract, for example:

```
Return GitHub-flavored markdown with exactly these headings:
## State
## Signals
## Risks
## Next
No empty body. If evidence is missing, say so under Signals.
No emoji. No claim of file writes, email, or web fetch.
```

Cron Review Gate's live nudge is this class of contract. Commit the
*pattern* here; do not commit the Mini prompt text.

---

## 10. Run ledger (next, not this PR)

After this spec, the next increment is a first-class Run ledger —
today's `agent.runs[]` (last 12) is not enough to budget a suite or
debug a 23-agent serial pass.

Sketch only (do not implement here):

```
~/.aiia/org/runs/<assignmentId>/<runId>.json
```

Fields as in [`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md)
§6.6: `trigger`, `cost`, `evidence`, `outputs`, `error`. Suite
acceptance will later require: filter Runs by `suite=mindmoor`, sum
tokens for the day, and show `empty_agent_result` in Activity without
opening each agent card.

---

## 11. Mapping onto today's Studio records

| Suite concept | Today | v1 org |
|---|---|---|
| Suite slug | optional `suite` on agent | Organization.slug or Team |
| Member | Agent Studio name / alias | Agent.migratedFrom + Specialty pin |
| Memory namespace | `memory_namespace` + `source=suite:mindmoor` | Organization.memoryScope |
| Mount | `repo_id=mindmoor` | Agent.repositoryId |
| Loop | `loop_*` fields | recurrent Assignment |
| Handoff | `assignment_registry` artifact enum | org Handoff + Sanction grant |
| Depth | documented here; not yet a persisted Studio field | Agent.depth |
| Run | `agent.runs[]` | Run ledger (§10) |

Migration must not create Handoffs automatically
([`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) §10.3).
Tagging a suite is not consent to fire edges.

---

## 12. Acceptance criteria

1. **Catalog is name-based.** Six members above; no invented Mini
   agent IDs in repo or docs.
2. **Air-gap stays on.** No Voice, email, or web tools on members.
   Isolated no-Handoff Runs still run.
3. **Shared namespace.** Suite-tagged agents persist
   `suite=mindmoor` and `memory_namespace=mindmoor`. Prompt states the
   namespace. Writes use `source=suite:mindmoor`.
4. **Client isolation.** Non-members are not auto-tagged. Email agent
   remains skipped / out of suite.
5. **Depth ceiling D2.** Members are D0 or D1; `maxDelegationDepth=0`.
6. **Handoffs typed.** Legal graph in §7; illegal edges refuse with
   the org codes (`handoff_required`, `artifact_type_mismatch`,
   empty grant ⇒ no fire).
7. **Mini serialization preserved.** One Run at a time; `409` /
   `mini_busy` unchanged.
8. **Token floors documented.** Cron Review Gate ≥ 1200 (recommend
   1600); other members ≥ 1200; output contract required.
9. **No runtime JSON in git.** Live caps / nudges remain Mini-only
   until an operator applies them.
10. **`GET /api/agent-suites`** returns the catalog and any matching
    in-memory registry agents (tag or alias).
11. **This file + org spec agree** on Assignment / Handoff / depth
    vocabulary. Session notebook (`HANDOFF.md`) is not overwritten.

---

## 13. Doc-review checklist

- [ ] Engineer can tag a Studio agent `suite=mindmoor` and know which
      memory source, mount, depth, and artifacts apply.
- [ ] Six members and the explicit non-members list match the
      2026-09-07 pass names (no new IDs).
- [ ] Air-gap / Voice / email / #47 / Run ledger are out of scope.
- [ ] Handoff graph is typed and fail-closed on empty grant.
- [ ] CHANGELOG `[Unreleased]` points here.
- [ ] Tests cover catalog match + registry persistence.

---

## 14. Code that landed with this spec

| Path | Role |
|---|---|
| `local_brain/command_center/agent_suites.py` | Catalog, alias match, memory scope helper |
| `AgentRegistry` `suite` / `memory_namespace` | Optional persisted fields |
| `AgentCreateRequest` + `_agent_system_prompt` | Accept tag; state namespace when set |
| `GET /api/agent-suites` | Catalog + matched live agents |
| `local_brain/tests/test_agent_suites.py` | Unit coverage |

No change to `egress.py`, Voice routes, or air-gap config.
