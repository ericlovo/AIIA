# Executable Organization — Assignments + typed Handoffs

> **v1 increment.** Typed nodes, typed edges, and assignment contracts.
> The organizational grid UI is a later interface over this model — not the
> starting point.
>
> Status: spec (docs-only). No runtime change in the PR that lands this file.
> Audience: an engineer implementing persistence, API, and the Sanction gate
> without re-reading the sprint brief.

**Not this file:** [`docs/HANDOFF.md`](./HANDOFF.md) is the Mini ⇄ MacBook
Claude Code session notebook. It is a different concept. Do not put
Assignment/Handoff records there, and do not overwrite it.

| Plane | Owns | Does not own |
|---|---|---|
| **AIIA** | Organization graph, Assignments, Runs, memory, Mini execution | Whether an edge may fire |
| **Sanction** | Grants, budgets, approval level, audit trail | Running the model or writing the artifact |
| **Mini** | Serialized local compute for authorized Runs | Policy. It executes work it is handed. |

AIIA builds the organization. Sanction governs what the organization may do.
The Mini executes the work privately. The graph determines what work is
*allowed to happen* — it is not a visualization of workflows that already ran.

---

## 1. Motivation

Agent Studio today can define local agents, mount read-only repo context, and
loop a prompt on an interval. Each agent is an island: it cannot legally
delegate, cannot pass a typed artifact to another role, and cannot spend a
capability it has not been granted. That is the right safety posture for a
first increment. It is not an organization.

The next increment makes isolation *contractual* rather than accidental:

- Work is an **Assignment** (a durable contract), not a prompt in a loop field.
- A loop tick is a **Run** (one traceable attempt), not the work itself.
- Agent-to-agent movement is a **Handoff** (a typed, Sanction-gated edge).
- Roles are **Specialties** (versioned packages), not a bag of skill chips.

Without this model, a Policy Analyst will eventually "just call" a Grant
Operator. That path is illegal here. The graph exists to make that refusal a
data-model fact, not a prompt instruction.

---

## 2. Non-goals (this increment)

These are explicit. Do not pull them into v1 because they would be nice on a
canvas:

- Free-form drag-and-drop workflow canvas; arbitrary node shapes; "draw any
  edge."
- Organizational grid UI polish (columns/rows, animated Runs, blocked-edge
  chrome). Spec for it lives in §16; implementation is out of scope.
- Multi-tenant SaaS Organization hosting. v1 is one Organization per Brain.
- Agents invoking other agents as unconstrained tools (`call_agent`, A2A
  `message/send` used as a side channel, MCP tool that skips Handoff).
- Auto-upgrading Specialties under a running Agent.
- Replacing Command Center stories, the research loop, or the 3-tier
  execution engine. Those keep their own stores; this spec maps them (§10,
  §15) rather than absorbing them.
- GitHub write, web browse, or any new egress. GitHub remains disconnected /
  read-only intent, as today.
- Parallel Mini runs. The scheduler stays serialized with manual Runs
  (`agent_run_lock` in `local_brain/command_center/server.py`).

---

## 3. Today's product state (ground truth)

Read this section as the migration source. Paths are current as of the
Agent Studio loops landing (`feat(agent-studio): add tools and bounded loops`).

### 3.1 Agent Studio (isolated loop agents)

| Concern | Where it lives | Bound |
|---|---|---|
| Agent record | `local_brain/command_center/agent_registry.py` → `agent_data.json` | max 40 agents |
| Create/update API | `POST/PUT /api/agents` on Command Center `:8200` | `AgentCreateRequest` |
| Dashboard | `dashboard/src/console/AgentStudio.tsx` | — |
| TS type | `dashboard/src/lib/api.ts` `Agent` | — |
| Tools | labels: `Local memory`, `Repository read`, `GitHub read` | max 8 |
| Repo mounts | `REPO_MOUNTS`: `aiia`, `mindmoor`, `sanction`, `proxy-ai` | read-only git snapshot |
| GitHub | `{ status: "disconnected", mode: "read_only" }` | do not claim GitHub data |
| Model params | `temperature` 0–1, `max_tokens` 128–2000 | local `/v1/chat` |
| Loop | `loop_enabled`, interval 15–1440 min, 1–48 runs/day | requires `loop_task` |
| Scheduler | `agent_loop_runner()` every 15s; one agent at a time | `agent_run_lock` |
| Run history | last 12 `{ task, result, error, at }` on the agent | not a first-class Run |
| Status | `idle` \| `running` \| `error` | — |
| System prompt | `_agent_system_prompt()` — supervised, propose-only | no file/network claims |

Repo context already injected when `Repository read` is mounted: branch,
recent commits, working-tree status, README excerpt.

### 3.2 Ops loops (Brain-owned definitions)

`GET /v1/loops` on `:8100` merges `AGENT_DEFS` in `local_brain/local_api.py`
with `~/.aiia/loops-registry.json`. These are named ops agents (`standup`,
`code-review`, …), all `execution: "propose"` today. They are a second
isolated-loop surface. v1 maps them the same way as Agent Studio loops (§10)
but does not require migrating them in the first slice — Agent Studio is
the primary source.

### 3.3 Sanction as it exists here

| Call | Module | Behavior |
|---|---|---|
| `POST {SANCTION_API_URL}/authorize/tool` | `local_brain/egress.py` | fail-closed when Sanction is configured; allow if unconfigured (vanilla OSS) |
| `POST {SANCTION_API_URL}/tokens` | `local_brain/sanction.py` | fire-and-forget token/cost log |
| Air-gap | `AIIA_AIRGAP=1` | local deny + audit post; control-plane metadata only |

Handoffs **do not** reuse the vanilla-OSS "allow if unconfigured" shortcut.
See §8. Isolated loops with no Handoff still run as they do today.

### 3.4 Execution tiers (already real)

`local_brain/execution/` is AUTO / SUPERVISED / GATED. Assignment
`requiredApprovalLevel` maps onto that vocabulary (§6.4). Do not invent
a fourth runtime executor; gate the *Assignment*, then let the existing
engine run any side-effecting action the Run proposes.

### 3.5 A2A (do not conflate)

`local_brain/a2a/` exposes Agent Cards and JSON-RPC `message/send`. That is
an external protocol adapter. An inbound A2A message may *create* an
Assignment; it must not fire a Handoff or call another Agent as a tool.
A2A `Task` ≠ this spec's Assignment. A2A `Artifact` is a wire part list;
this spec's artifact is a typed org object with a schema.

---

## 4. Core model

Six durable entities. Everything else (artifact instance, grant, eval score)
hangs off them.

```
Organization
  ├── Specialty[]     versioned capability packages (installable)
  ├── Agent[]         identity = specialty + persona + memory + authority
  ├── Assignment[]    durable work contracts (optionally recurrent)
  │     └── Run[]     one attempt: evidence, cost, outputs, approvals
  └── Handoff[]       typed, Sanction-gated edges between Agents
```

**Invariant:** the only legal way for Agent A to cause Agent B to act is:

1. A produces an artifact that matches a Handoff's `artifactType`.
2. The Handoff's `conditions` hold.
3. Sanction has minted (or already attached) a grant for that edge.
4. A **child Assignment** is opened for B (or B's existing Assignment
   accepts the artifact as an input).
5. The Mini is scheduled to execute a **Run** of that Assignment.

Any other path (`call_agent`, shared scratchpad, "please also ping X" in a
prompt, A2A hop, MCP tool) is illegal and must fail closed with
`handoff_required`.

---

## 5. Agent depth (authority, not model IQ)

Depth is an authorization class. It is not a quality ranking of the model.
A D0 Observer on a strong model is still D0. A D3 Lead on `qwen3:8b` is
still D3, and is still bound by Mini serialization and grants.

| Depth | Name | May do | May not do |
|---|---|---|---|
| **D0** | Observer | Read mounted signals; emit `signal_report` | Produce binding artifacts; delegate; hold action grants |
| **D1** | Specialist | Produce bounded artifacts that validate against the Specialty schema | Reversible external actions; spawn children |
| **D2** | Operator | Reversible actions covered by an active grant (e.g. mint/revoke under scope) | Allocate budget; reassign teams; unbounded delegation |
| **D3** | Lead | Decompose an Assignment into children; pick assignees of depth ≤ own depth | Allocate org budget; change Organization policies |
| **D4** | Executive | Allocate budgets, priorities, teams; set Specialty installs | Bypass Sanction; mint grants itself |

Every Agent has `maxDelegationDepth` (int, `0..4`):

- `0` — cannot create child Assignments. **This is the default for migrated
  loop agents.** It is how cost stays predictable.
- `n` — a child Assignment consumes 1; the child's remaining allowance is
  `min(child.maxDelegationDepth, n - 1)`.
- Spawn that would exceed the remaining allowance is rejected
  (`delegation_depth_exceeded`) before Sanction is even asked.
- Depth of the *assignee* must be `>= assignment.authority.minDepth`.
- An Agent cannot assign work that requires a higher depth than it holds
  (`authority_insufficient`).

Infinite spawn is a budget bug, not a feature. `maxDelegationDepth` plus
Assignment `budgetCap` plus Mini serialization are the three cost brakes.

---

## 6. Entity schemas

Conventions: identifiers are UUID v4 strings. Timestamps are UTC ISO-8601.
Enums are lowercase snake_case. Unknown fields are rejected on write
(`extra: forbid`) so the contract stays typed.

### 6.1 Organization

A workspace. v1: exactly one per Brain. The first dogfood Organization is
**Sanction** (authorization is its domain) — see §11.

| Field | Type | Notes |
|---|---|---|
| `id` | `uuid` | Stable. |
| `slug` | `string` | `^[a-z0-9][a-z0-9-]{1,62}$`. v1 default `sanction`. |
| `name` | `string` | Display. |
| `mission` | `string` | What this org exists to do. |
| `repositoryIds` | `string[]` | Keys into the existing mount table (`aiia`, `mindmoor`, `sanction`, `proxy-ai`) plus future mounts. Read-only in v1. |
| `policyIds` | `string[]` | Opaque ids Sanction understands (policy revision). Empty = org cannot fire Handoffs. |
| `budgets` | `Budget[]` | See below. |
| `memoryScope` | `MemoryScope` | Vault / Chroma collections this org may read/write. |
| `teams` | `Team[]` | Named groups of Agents. Grid columns later. |
| `createdAt` / `updatedAt` | `datetime` | |

```
Budget {
  id: string
  kind: "tokens" | "usd" | "runs_per_day"
  cap: number
  period: "day" | "week" | "month" | "assignment"
  walletId?: string          // SANCTION_WALLET_ID when kind is usd/tokens
}

MemoryScope {
  collections: string[]      // existing eq_brain categories, e.g. decisions
  write: boolean
}

Team {
  id: string
  name: string
  specialtySlugs: string[]   // columns in the future grid
  memberAgentIds: string[]
}
```

### 6.2 Specialty

A versioned **capability package**. Treat it closer to software than to a
prompt: it is installable, pin-able, and testable. Bumping `version` does
not mutate Agents that already pin an older version.

| Field | Type | Notes |
|---|---|---|
| `id` | `uuid` | Identity of this version row. |
| `slug` | `string` | Stable across versions (`repo-analyst`). |
| `version` | `semver` | `MAJOR.MINOR.PATCH`. |
| `name` | `string` | Display (`Repo Analyst`). |
| `instructions` | `string` | System prompt body. |
| `requiredTools` | `ToolId[]` | Must be grantable. Agent cannot run without them. |
| `tools` | `ToolId[]` | Superset advertised; runtime uses intersection with grants. |
| `contextSources` | `ContextSource[]` | Memory, repo mount, GitHub-read (still disconnected). |
| `acceptedInputTypes` | `ArtifactType[]` | What may land on this Specialty. |
| `producedArtifacts` | `ArtifactType[]` | What it is allowed to emit. |
| `outputSchema` | `object` | JSON Schema for the primary artifact body. |
| `evalRubric` | `EvalCriterion[]` | How a Run is scored. |
| `defaultModelParams` | `{ temperature: number, maxTokens: number }` | Same bounds as Agent Studio. |
| `recommendedDepth` | `0..4` | Default Agent depth when instantiating. |
| `requiredApprovalLevel` | `ApprovalLevel` | Floor for Assignments of this Specialty. |
| `tests` | `SpecialtyTest[]` | Optional but expected for "installable." |
| `deprecated` | `boolean` | If true, new Agents may not pin it. |

```
ToolId = "local.memory" | "repo.read" | "github.read"
       | "sanction.authorize" | "sanction.grants"
       | "org.assign" | "org.handoff"
       // extend deliberately; do not accept free strings in v1

ContextSource {
  kind: "memory" | "repo" | "github" | "assignment_inputs"
  repoId?: string
  collections?: string[]
}

EvalCriterion {
  id: string
  description: string
  weight: number            // 0..1, weights sum to 1
  required: boolean         // required=true must pass for Run.status=done
}

SpecialtyTest {
  id: string
  input: object             // fixture matching acceptedInputTypes
  expect: { artifactType: ArtifactType, schemaValid: true }
}
```

**Shipped example slugs** (packages, not running Agents):

`repo-analyst`, `security-reviewer`, `authorization-architect`,
`approval-triage`, `customer-researcher`, `release-operator`,
`documentation-publisher`, `organizational-memory-curator`.

Plus the Sanction-grid roles in §11, which are the first ones to actually
instantiate.

A Specialty without `tests` may be drafted; it may not be installed onto an
Agent that will receive production Assignments (`specialty_untested`). v1
may ship tests as fixtures in `local_brain/org/specialties/<slug>/`.

### 6.3 Agent

Identity bound to one Organization. Combines a pinned Specialty with a
persona, a memory scope, an authority class, and a compute allocation.

| Field | Type | Notes |
|---|---|---|
| `id` | `uuid` | |
| `organizationId` | `uuid` | |
| `name` | `string` | |
| `persona` | `string` | Tone; does not grant tools. |
| `specialtySlug` | `string` | |
| `specialtyVersion` | `semver` | Pin. |
| `memoryScope` | `MemoryScope` | Intersection with Organization.memoryScope. |
| `depth` | `0..4` | Authority class. Must be `>= specialty.recommendedDepth` unless an Executive overrides, which is itself a GATED Assignment. |
| `maxDelegationDepth` | `0..4` | Default `0`. |
| `authority` | `AgentAuthority` | Redundant-but-queryable snapshot. |
| `computeAllocation` | `ComputeAllocation` | Mini quota. |
| `repositoryId` | `string?` | Default repo mount. |
| `toolGrants` | `ToolId[]` | Must be ⊇ specialty.requiredTools after Sanction. |
| `teamId` | `uuid?` | |
| `status` | `AgentStatus` | `idle` \| `running` \| `blocked` \| `disabled` \| `error` |
| `migratedFrom` | `{ surface: "agent_studio" \| "ops_loop", id: string }?` | Set on loop migration. |

```
AgentAuthority {
  depth: 0..4
  maxDelegationDepth: 0..4
  canHoldActionGrants: boolean   // true iff depth >= 2
}

ComputeAllocation {
  maxTokensPerRun: number        // maps today's max_tokens
  maxRunsPerDay: number          // 1..48, same bound as loops
  temperature: number            // 0..1
}
```

Persona + Specialty instructions are concatenated into the Run system
prompt. Persona cannot add tools, raise depth, or widen `acceptedInputTypes`.

### 6.4 Assignment

The durable work contract. This is the unit of authorization. Loops become
Assignments with `recurrence`; a prompt in `loop_task` is not a contract.

| Field | Type | Notes |
|---|---|---|
| `id` | `uuid` | |
| `organizationId` | `uuid` | |
| `objective` | `string` | What done looks like in prose. |
| `inputs` | `AssignmentInput[]` | Artifact refs and inline payloads. |
| `successCriteria` | `string[]` | Human-readable; evaluated against `evalRubric`. |
| `deadline` | `datetime?` | Null = no deadline. |
| `recurrence` | `Recurrence?` | Null = run once (or on demand). |
| `expectedArtifact` | `ArtifactType` | Must be in assignee Specialty.`producedArtifacts`. |
| `assigneeId` | `uuid` | Agent. |
| `requesterId` | `uuid` | Agent or `human:<user>`. |
| `parentAssignmentId` | `uuid?` | Set when a Lead decomposes. |
| `handoffId` | `uuid?` | The edge that created this child, if any. |
| `authority` | `AssignmentAuthority` | |
| `status` | `AssignmentStatus` | See §7.1. |
| `blockedReason` | `BlockedReason?` | Required when `status=blocked`. |
| `createdAt` / `updatedAt` | `datetime` | |

```
AssignmentInput {
  artifactType: ArtifactType
  artifactId?: uuid            // previously produced
  inline?: object              // v1: small JSON, max 64KiB
}

Recurrence {
  kind: "interval"             // v1 only
  minutes: number              // 15..1440
  maxRunsPerDay: number        // 1..48
  timezone: "UTC"
}

AssignmentAuthority {
  minDepth: 0..4
  requiredApprovalLevel: ApprovalLevel
  budgetCap: Budget            // period: "assignment" typical
}

ApprovalLevel = "auto" | "supervised" | "gated"

BlockedReason =
  | "awaiting_grant"
  | "awaiting_approval"
  | "sanction_unconfigured"
  | "handoff_required"
  | "delegation_depth_exceeded"
  | "authority_insufficient"
  | "budget_exhausted"
  | "specialty_untested"
  | "mini_busy"
  | "eval_failed"
  | "input_schema_invalid"
```

`requiredApprovalLevel` mapping onto existing execution:

| ApprovalLevel | Existing tier | Assignment may enter `running` when |
|---|---|---|
| `auto` | `SafetyTier.AUTO` | Grant is active; Mini free. |
| `supervised` | `SUPERVISED` | Grant active; Run emits an intervention event (same as today's supervised actions). |
| `gated` | `GATED` | Explicit human (or D4) approval record on the Assignment. |

A Specialty's `requiredApprovalLevel` is a **floor**. The Assignment may
raise it, never lower it.

### 6.5 Handoff

A typed connection between two Agents. It is an edge in the organization
graph, not a chat. Selecting the edge (in a future UI) reveals this
contract.

| Field | Type | Notes |
|---|---|---|
| `id` | `uuid` | |
| `organizationId` | `uuid` | |
| `fromAgentId` | `uuid` | Producer. |
| `toAgentId` | `uuid` | Consumer. |
| `artifactType` | `ArtifactType` | The only type that may pass. |
| `conditions` | `Condition` | Must all hold before fire. |
| `decisionOwner` | `DecisionOwner` | Who decides the next step after fire. |
| `sanctionGrant` | `SanctionGrant?` | **Null means the edge cannot fire.** |
| `enabled` | `boolean` | Disabled edges are visible but inert. |
| `createdAt` / `updatedAt` | `datetime` | |

```
Condition {
  all?: Predicate[]
  any?: Predicate[]            // v1: prefer `all`; `any` is optional
}

Predicate {
  field: string                // dotted path, allow-listed (§8.3)
  op: "eq" | "neq" | "gte" | "lte" | "in" | "exists"
  value?: unknown
}

DecisionOwner {
  kind: "from_agent" | "to_agent" | "human" | "sanction"
  agentId?: uuid
}

SanctionGrant {
  grantId: string              // Sanction-issued
  capability: string           // e.g. "sanction.grants.mint"
  scope: object                // resource constraints Sanction understands
  expiresAt: datetime
  budget?: Budget
  status: "pending" | "active" | "denied" | "expired" | "revoked"
}
```

**Empty grant rule (non-negotiable):** `sanctionGrant == null` OR
`sanctionGrant.status != "active"` OR `expiresAt <= now` ⇒ the edge
**does not fire**. The runtime must request a mint from Sanction
(`POST /authorize/handoff`, §8.1). It must not invent a local grant, copy a
sibling edge's grant, or treat "the user said go" as a grant.

Free agent-to-agent tool calls that skip this contract are illegal.

### 6.6 Run

One traceable attempt at an Assignment. Today's `agent.runs[]` rows collapse
into this.

| Field | Type | Notes |
|---|---|---|
| `id` | `uuid` | |
| `assignmentId` | `uuid` | |
| `agentId` | `uuid` | Snapshot of assignee at start. |
| `attempt` | `int` | 1-based. Recurrence and retries both increment. |
| `trigger` | `manual` \| `interval` \| `handoff` \| `parent` | |
| `status` | `RunStatus` | `queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` |
| `startedAt` / `endedAt` | `datetime?` | |
| `evidence` | `Evidence[]` | What was read. |
| `cost` | `Cost` | Tokens, usd estimate, wall time. |
| `decisions` | `Decision[]` | Including "did not fire Handoff X because …". |
| `outputs` | `ArtifactRef[]` | |
| `approvals` | `Approval[]` | |
| `evalScores` | `{ criterionId: string, pass: boolean, notes?: string }[]` | |
| `error` | `string?` | |

```
Evidence { kind: "memory" | "repo" | "input" | "grant"; ref: string; excerpt?: string }
Cost     { tokensIn: number; tokensOut: number; usd: number; latencyMs: number }
Decision { at: datetime; kind: string; summary: string; handoffId?: uuid; granted?: boolean }
ArtifactRef { artifactId: uuid; artifactType: ArtifactType; schemaValid: boolean }
Approval { by: string; level: ApprovalLevel; at: datetime; note?: string }
```

Mini serialization: at most one Run in `running` per Brain. Additional due
work stays `queued` (`mini_busy`). Manual Runs and interval Runs share the
same lock, as today.

### 6.7 Artifact types (v1 catalog)

Closed enum for v1. New types are a spec bump, not a runtime string.

| `ArtifactType` | Typical producer depth | Typical consumer |
|---|---|---|
| `signal_report` | D0 | D1 Interpret |
| `usage_snapshot` | D0 | D1 |
| `policy_analysis` | D1 | D3 Decide |
| `grant_request` | D1 | D3 / Sanction |
| `risk_review` | D1 | D3 |
| `approval_packet` | D3 | D2 Act / human |
| `grant_decision` | D2 (Grant Operator) | Learn / Audit |
| `integration_patch` | D2 | D3 / human |
| `documentation` | D1–D2 | Publisher / memory |
| `audit_record` | D0–D1 | Memory Curator |
| `failure_analysis` | D1 | D3 / Optimizer |
| `memory_entry` | D1 | org memory |

An Agent may only emit types in its Specialty.`producedArtifacts`, and may
only accept types in `acceptedInputTypes`. A Handoff whose `artifactType` is
outside either side's lists cannot be created (`artifact_type_mismatch`).

---

## 7. Status machines

### 7.1 Assignment

```
                    ┌──────────────┐
                    │    draft     │
                    └──────┬───────┘
                           │ requester complete;
                           │ schema valid; assignee exists
                           ▼
                    ┌──────────────┐
         ┌─────────│  authorized  │◄────────── grant minted /
         │         └──────┬───────┘            approval recorded
         │                │ Mini dequeues
         │                ▼
         │         ┌──────────────┐
         │         │   running    │
         │         └──────┬───────┘
         │                │
         │     ┌──────────┼──────────┐
         │     ▼          ▼          ▼
         │ ┌────────┐ ┌──────┐ ┌─────────┐
         │ │blocked │ │ done │ │ failed  │
         │ └───┬────┘ └──────┘ └─────────┘
         │     │
         │     │ grant/approval/budget restored
         └─────┘
```

| From | To | Guard |
|---|---|---|
| `draft` | `authorized` | Inputs validate; assignee Specialty accepts `expectedArtifact`; `minDepth` ≤ assignee.depth; remaining delegation allowance ≥ 0; Sanction grant active if this Assignment was created by a Handoff; `requiredApprovalLevel` satisfied. |
| `authorized` | `running` | Mini lock acquired; budget remaining. |
| `running` | `done` | Primary artifact schema-valid; every `evalRubric` criterion with `required=true` passed. |
| `running` | `failed` | Model/Mini error, or required eval failed with no retry budget. |
| `running` | `blocked` | Missing grant, missing approval, `mini_busy` after lock steal, budget hit, eval needs human. |
| `blocked` | `authorized` | `blockedReason` cleared. Never skip to `running` without Mini dequeue. |
| `draft` / `authorized` / `blocked` | `failed` | Requester or D4 cancels with reason. Terminal. |
| `done` / `failed` | — | Terminal. Recurrence opens a **new** Assignment or a new Run on the same Assignment (§7.1); it does not revive a failed one in place. |

v1 recurrence: keep one Assignment, enqueue a new Run when interval + daily
cap allow. Status while waiting between ticks is `authorized` (not `done`).
The last successful Run remains queryable. If a tick fails, Assignment
goes `blocked` or `failed` per retry policy (v1: one fail → `blocked` with
`eval_failed` or the Mini error; human or Lead unblocks).

### 7.2 Run

```
queued → running → succeeded
                 → failed
                 → cancelled
```

`cancelled` only if the Mini lock is released without a model result
(shutdown, kill switch). The Assignment then becomes `blocked` /
`mini_busy` or `failed`.

### 7.3 Handoff (edge state is the grant)

The edge itself is `enabled` or not. Fire-ability is a function:

```
canFire(h) =
    h.enabled
    && h.sanctionGrant.status == "active"
    && h.sanctionGrant.expiresAt > now
    && conditionsHold(h.conditions)
    && artifact.schemaValid
    && artifact.type == h.artifactType
```

No separate "firing" status on the Handoff. Each successful fire creates
or updates an Assignment and a Run, and appends a `Decision` on the
producer's Run.

### 7.4 Agent

`idle` → `running` (a Run started) → `idle` (Run ended success) or `error`
(Run failed) or `blocked` (Assignment blocked on this Agent). `disabled`
is operator-set and skips the scheduler.

---

## 8. Handoff authorization flow (Sanction-first)

This is the increment's load-bearing sequence. Implement it before any UI.

```
Producer Run (Agent A)
    │  emits artifact T, schema-valid
    ▼
Lookup Handoff (A → B, artifactType=T, enabled)
    │  none → Decision: no_handoff (ok if Assignment did not require one)
    │  found, conditions fail → Decision: conditions_unmet; do not fire
    ▼
Grant present and active?
    │  no → Assignment-for-B stays uncreated;
    │       POST Sanction /authorize/handoff
    │       on pending: block producer Assignment? no. Record Decision
    │                  awaiting_grant; edge is inert until callback/poll
    │       on denied: Decision grant_denied; illegal to retry via tool call
    │       on unconfigured: blockedReason=sanction_unconfigured (fail closed)
    ▼
Minted / already active
    ▼
Create child Assignment for B
    inputs = [{ artifactType: T, artifactId }]
    requesterId = A
    parentAssignmentId = A's assignment
    handoffId = h.id
    authority.minDepth from B.specialty.recommendedDepth
    remainingDelegation = min(B.maxDelegationDepth, A.remaining - 1)
    │  remaining < 0 → reject delegation_depth_exceeded (no Assignment)
    ▼
Authorize Assignment (approval floor + budget)
    ▼
Queue Run on Mini (serialized)
```

### 8.1 Sanction API this Brain will call

AIIA does not mint grants. It asks. Until Sanction exposes a dedicated
handoff route, the Brain adapter may wrap the existing
`POST /authorize/tool` **only if** the payload is unambiguous and the
response includes a `grantId`. Prefer a dedicated route; do not silently
reuse `tool=web.fetch` style checks for org edges.

Request (canonical, implement the client against this shape):

```http
POST /authorize/handoff
x-api-key: $SANCTION_API_KEY
```

```json
{
  "organization_id": "…",
  "handoff_id": "…",
  "from_agent_id": "…",
  "to_agent_id": "…",
  "assignment_id": "…",
  "artifact_type": "grant_request",
  "capability": "sanction.grants.mint",
  "scope": {
    "repo_id": "sanction",
    "max_ttl_seconds": 3600
  },
  "budget": {
    "kind": "usd",
    "cap": 5.0,
    "period": "assignment"
  },
  "expires_at": "2026-09-04T12:00:00Z",
  "evidence_artifact_ids": ["…"]
}
```

Success (`200`):

```json
{
  "authorized": true,
  "grant_id": "grn_…",
  "capability": "sanction.grants.mint",
  "scope": {},
  "expires_at": "2026-09-04T12:00:00Z",
  "status": "active"
}
```

Anything else (non-200, `authorized !== true`, timeout, transport error,
missing `grant_id`) is **deny**. Attach `status: "denied"` or leave the
grant null. Same fail-closed posture as `egress.py`, except there is **no**
"Sanction unconfigured ⇒ allow" branch on this path.

Audit: every attempt, including denials, is posted. A failed audit post
never converts a deny into an allow (same rule as air-gap).

Air-gap (`AIIA_AIRGAP=1`): Handoffs that would require cloud-side effects
stay denied. Observe-only artifact movement (`signal_report` between D0
agents whose tools are local) still needs a grant; that grant must come
from a local Sanction instance or the edge does not fire. Isolated loop
Runs that never touch a Handoff keep working.

### 8.2 What a Policy Analyst actually submits

Not a tool call. An Assignment whose `expectedArtifact` is `grant_request`
and whose inputs include:

- evidence artifact ids (what was observed / interpreted)
- requested `capability`
- `budget` cap
- `expires_at`
- target Operator `assigneeId` (Grant Operator)

Sanction authorizes the **Handoff** from Analyst → Architect/Triage (or
Analyst → Operator if such an edge exists and is granted). The Operator
does not run until that grant is active.

### 8.3 Condition field allow-list

v1 predicates may only reference:

- `artifact.type`
- `artifact.schemaValid`
- `assignment.status`
- `assignment.authority.minDepth`
- `grant.status`
- `grant.expiresAt`
- `from.depth` / `to.depth`
- `budget.remaining`

Unknown `field` values fail the condition (`false`), they do not throw in
the producer Run. Do not add a general expression language in v1.

---

## 9. Depth + `maxDelegationDepth` rules (normative)

Let `remaining(A, assignment)` be stored on the Assignment at creation:

```
if parent is null:
    remaining = A.maxDelegationDepth
else:
    remaining = min(A.maxDelegationDepth, parent.remaining - 1)
```

Then:

1. If `remaining < 0`, reject. (Should be unreachable if parents check first.)
2. Creating a child requires `remaining >= 1` on the parent Assignment
   **and** `parent.agent.depth >= 3` (only Leads and Executives decompose).
3. D0–D2 with `maxDelegationDepth > 0` is a misconfiguration. Reject on
   write (`delegation_reserved_for_leads`). v1: if depth < 3, force
   `maxDelegationDepth = 0`.
4. D3 default `maxDelegationDepth = 1`. D4 default `2`. Operators may
   lower, not raise, without a GATED Assignment.
5. Child `authority.minDepth` cannot exceed the assignee's `depth`.
6. Child `requiredApprovalLevel` ≥ max(parent's, specialty floor).
7. Child `budgetCap.cap` ≤ parent remaining budget.
8. Cycles: an Agent may not be assignee of a descendant Assignment that
   already lists it as requester in the parent chain
   (`delegation_cycle`).

These checks are local and cheap. Run them **before** calling Sanction so
we do not mint grants for work that cannot legally exist.

---

## 10. Mapping today's Loop agents → Assignment / Run

This is the compatibility story. Agent Studio remains usable. The org
graph is how those agents join an organization.

### 10.1 Field map (Agent Studio → org)

| Today (`Agent` in `api.ts` / `agent_registry`) | v1 |
|---|---|
| `id` | `Agent.migratedFrom.id`; new `Agent.id` (uuid) |
| `name`, `persona` | `Agent.name`, `Agent.persona` |
| `mission` | Organization-level or Assignment `objective` seed; do not overload persona |
| `skills[]` | Informational tags only. **Not** a Specialty. |
| `tools[]` | Map labels → `ToolId`: `Local memory`→`local.memory`, `Repository read`→`repo.read`, `GitHub read`→`github.read` |
| `repo_id` | `Agent.repositoryId` |
| `temperature`, `max_tokens` | `computeAllocation` |
| `loop_enabled` | If true, create a recurrent Assignment; else Agent has no standing Assignment |
| `loop_interval_minutes` | `recurrence.minutes` |
| `loop_max_runs_per_day` | `recurrence.maxRunsPerDay` and `computeAllocation.maxRunsPerDay` |
| `loop_task` | `Assignment.objective` (and the user-message of each Run) |
| `loop_runs_today` / `loop_day` | Derived from Runs that day, not stored on Agent |
| `status` | `Agent.status` (`error` stays `error`) |
| `runs[]` | Each row → `Run` (`trigger: interval` or `manual`, `attempt` by order) |
| — | `depth = 1` (Specialist), `maxDelegationDepth = 0` |
| — | Specialty: synthesize a private `studio-{slug}` package from instructions+tools, **or** pin a shipped Specialty if the operator picks one. v1 migration uses a generic `loop-specialist` package (`producedArtifacts: [signal_report]`, `requiredApprovalLevel: auto`). |

GitHub disconnected stays disconnected: a migrated Agent may list
`github.read` as a tool label, but Runs must keep the current prompt line
that forbids claiming GitHub data.

### 10.2 Semantics that must not change on migration

- Interval bounds 15 minutes–24 hours, 1–48 Runs/day.
- Mini: one Run at a time; manual Run returns `409 mini_busy` / Assignment
  `blockedReason=mini_busy` if the lock is held.
- Propose-only system prompt until a Handoff + grant says otherwise.
- No new egress.

### 10.3 What migration does *not* do

- Does not create Handoffs between existing studio Agents. They remain
  isolated until an operator (or D4 Assignment) installs edges.
- Does not raise `maxDelegationDepth` above 0.
- Does not interpret `skills` as authority.
- Does not auto-install the Sanction grid (§11).

### 10.4 Ops loops (`GET /v1/loops`)

Same map, `migratedFrom.surface = "ops_loop"`. Optional in v1; do Agent
Studio first. `execution: "propose"` ⇒ `requiredApprovalLevel: auto` and
no action grants.

### 10.5 After migration, a loop tick

```
due = Assignment.recurrence due and runs_today < maxRunsPerDay
  && Assignment.status in {authorized, blocked? no}
  && Agent.status != disabled
→ enqueue Run(trigger=interval)
→ Mini executes with Specialty instructions + persona + inputs
→ persist Run; update Agent.status
→ if outputs include a type that matches an enabled Handoff, enter §8
```

The loop field in Agent Studio can remain as a convenience editor for the
standing Assignment. It must write through to Assignment + Run, not to a
parallel `loop_task` that the org graph cannot see. Dual-write during
transition is acceptable; dual-source-of-truth is not.

---

## 11. Sanction Agent Grid (example org)

First Organization: **Sanction**. Layers are operational (grid rows), not
a second depth system. Depth still applies per Agent.

Columns ≈ teams / specialties. Rows ≈ Observe → Interpret → Decide → Act
→ Learn (or D0–D4). This table is the executable graph, not a mood board.

| Layer | Example Agents | Depth | Specialty slug | Produces |
|---|---|---|---|---|
| Observe | Repository Watcher | D0 | `repository-watcher` | `signal_report` |
| Observe | Usage Monitor | D0 | `usage-monitor` | `usage_snapshot` |
| Observe | Security Signal Scout | D0 | `security-signal-scout` | `signal_report` |
| Interpret | Policy Analyst | D1 | `policy-analyst` | `policy_analysis`, `grant_request` |
| Interpret | Grant Investigator | D1 | `grant-investigator` | `policy_analysis` |
| Interpret | Customer Context Analyst | D1 | `customer-researcher` | `signal_report` |
| Decide | Authorization Architect | D3 | `authorization-architect` | `approval_packet` |
| Decide | Risk Reviewer | D1 | `security-reviewer` | `risk_review` |
| Decide | Approval Triage | D3 | `approval-triage` | `approval_packet` |
| Act | Grant Operator | D2 | `grant-operator` | `grant_decision` |
| Act | Integration Builder | D2 | `integration-builder` | `integration_patch` |
| Act | Documentation Publisher | D1 | `documentation-publisher` | `documentation` |
| Learn | Audit Historian | D0 | `audit-historian` | `audit_record` |
| Learn | Policy Optimizer | D3 | `policy-optimizer` | `policy_analysis` |
| Learn | Failure Analyst | D1 | `failure-analyst` | `failure_analysis` |

### 11.1 Legal edge

```
Policy Analyst  --grant_request-->  Authorization Architect
                                    --approval_packet-->  Grant Operator
```

1. Watcher emits `signal_report` (Handoff Observe→Analyst, grant
   `org.observe` active).
2. Analyst Assignment completes with `grant_request` (capability, scope,
   budget, expiration, evidence ids).
3. Handoff Analyst→Architect has `artifactType: grant_request`. Grant is
   either pre-attached or minted via §8.1.
4. Architect (D3) may decompose: child Assignment to Risk Reviewer
   (`risk_review`) if `maxDelegationDepth >= 1` and remaining ≥ 1.
5. Architect emits `approval_packet`.
6. Handoff Architect→Operator fires only with an **active**
   `sanctionGrant.capability = "sanction.grants.mint"` (or whatever
   Sanction names the act). Empty grant ⇒ Operator never runs.
7. Operator Run performs the reversible act under that grant; emits
   `grant_decision`.
8. Handoff Operator→Audit Historian carries `grant_decision` /
   `audit_record`.

### 11.2 Illegal edge (must refuse)

```
Policy Analyst  --tool call-->  Grant Operator.mint(...)
```

There is no Handoff Analyst→Operator in the example graph. Even if someone
adds one, it cannot fire with a null grant. The runtime returns
`handoff_required` or `awaiting_grant`. The Analyst's prompt is not a
vote. Tests in §17 must include this case.

A second illegal edge: Watcher (D0) → Grant Operator (D2) with
`artifactType: grant_request`. D0 cannot produce `grant_request`
(`producedArtifacts` is `signal_report` only). Creating that Handoff fails
`artifact_type_mismatch`.

A third: Architect sets `maxDelegationDepth=0` then tries to spawn Risk
Reviewer. `delegation_depth_exceeded`.

### 11.3 ASCII grid (how the future UI should read)

```
                 Observe              Interpret             Decide                 Act                  Learn
D4
D3                                                          Arch  Triage                                Optimizer
D2                                                                               GrantOp  Integrator
D1                                    Analyst  Invest.      RiskRev                                    Failure
D0           Watcher  Usage  Scout                                                                  Historian
             |                         ^
             | signal_report + grant   |
             +-------------------------+
                                       | grant_request + grant
                                       +-----------------------> Arch
                                                                | approval_packet + grant
                                                                +-----------------------> GrantOp
```

Selecting an edge shows: artifact type, conditions, decision owner, grant
capability / scope / expiration / status. Runs animate along granted
edges. Approvals render on edges that are `awaiting_approval` or
`awaiting_grant`. **Do not implement this UI in v1.**

---

## 12. Type sketches

Implementations may use Pydantic v2 (Brain, py310) or these TypeScript
shapes (Console / dashboard). Names must match. This is the contract.

```ts
type Uuid = string;
type IsoDateTime = string;
type SemVer = string;
type Depth = 0 | 1 | 2 | 3 | 4;
type ApprovalLevel = "auto" | "supervised" | "gated";
type AssignmentStatus =
  | "draft"
  | "authorized"
  | "running"
  | "blocked"
  | "done"
  | "failed";
type AgentStatus = "idle" | "running" | "blocked" | "disabled" | "error";
type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
type ArtifactType =
  | "signal_report"
  | "usage_snapshot"
  | "policy_analysis"
  | "grant_request"
  | "risk_review"
  | "approval_packet"
  | "grant_decision"
  | "integration_patch"
  | "documentation"
  | "audit_record"
  | "failure_analysis"
  | "memory_entry";
type ToolId =
  | "local.memory"
  | "repo.read"
  | "github.read"
  | "sanction.authorize"
  | "sanction.grants"
  | "org.assign"
  | "org.handoff";
type BlockedReason =
  | "awaiting_grant"
  | "awaiting_approval"
  | "sanction_unconfigured"
  | "handoff_required"
  | "delegation_depth_exceeded"
  | "authority_insufficient"
  | "budget_exhausted"
  | "specialty_untested"
  | "mini_busy"
  | "eval_failed"
  | "input_schema_invalid";

interface Budget {
  id: string;
  kind: "tokens" | "usd" | "runs_per_day";
  cap: number;
  period: "day" | "week" | "month" | "assignment";
  walletId?: string;
}

interface MemoryScope {
  collections: string[];
  write: boolean;
}

interface Team {
  id: Uuid;
  name: string;
  specialtySlugs: string[];
  memberAgentIds: Uuid[];
}

interface Organization {
  id: Uuid;
  slug: string;
  name: string;
  mission: string;
  repositoryIds: string[];
  policyIds: string[];
  budgets: Budget[];
  memoryScope: MemoryScope;
  teams: Team[];
  createdAt: IsoDateTime;
  updatedAt: IsoDateTime;
}

interface EvalCriterion {
  id: string;
  description: string;
  weight: number;
  required: boolean;
}

interface SpecialtyTest {
  id: string;
  input: object;
  expect: { artifactType: ArtifactType; schemaValid: true };
}

interface Specialty {
  id: Uuid;
  slug: string;
  version: SemVer;
  name: string;
  instructions: string;
  requiredTools: ToolId[];
  tools: ToolId[];
  contextSources: {
    kind: "memory" | "repo" | "github" | "assignment_inputs";
    repoId?: string;
    collections?: string[];
  }[];
  acceptedInputTypes: ArtifactType[];
  producedArtifacts: ArtifactType[];
  outputSchema: object;
  evalRubric: EvalCriterion[];
  defaultModelParams: { temperature: number; maxTokens: number };
  recommendedDepth: Depth;
  requiredApprovalLevel: ApprovalLevel;
  tests: SpecialtyTest[];
  deprecated: boolean;
}

interface Agent {
  id: Uuid;
  organizationId: Uuid;
  name: string;
  persona: string;
  specialtySlug: string;
  specialtyVersion: SemVer;
  memoryScope: MemoryScope;
  depth: Depth;
  maxDelegationDepth: Depth;
  computeAllocation: {
    maxTokensPerRun: number;
    maxRunsPerDay: number;
    temperature: number;
  };
  repositoryId?: string;
  toolGrants: ToolId[];
  teamId?: Uuid;
  status: AgentStatus;
  migratedFrom?: { surface: "agent_studio" | "ops_loop"; id: string };
}

interface Assignment {
  id: Uuid;
  organizationId: Uuid;
  objective: string;
  inputs: {
    artifactType: ArtifactType;
    artifactId?: Uuid;
    inline?: object;
  }[];
  successCriteria: string[];
  deadline?: IsoDateTime;
  recurrence?: {
    kind: "interval";
    minutes: number;
    maxRunsPerDay: number;
    timezone: "UTC";
  };
  expectedArtifact: ArtifactType;
  assigneeId: Uuid;
  requesterId: string;
  parentAssignmentId?: Uuid;
  handoffId?: Uuid;
  authority: {
    minDepth: Depth;
    requiredApprovalLevel: ApprovalLevel;
    budgetCap: Budget;
  };
  status: AssignmentStatus;
  blockedReason?: BlockedReason;
  remainingDelegation: number;
  createdAt: IsoDateTime;
  updatedAt: IsoDateTime;
}

interface SanctionGrant {
  grantId: string;
  capability: string;
  scope: object;
  expiresAt: IsoDateTime;
  budget?: Budget;
  status: "pending" | "active" | "denied" | "expired" | "revoked";
}

interface Handoff {
  id: Uuid;
  organizationId: Uuid;
  fromAgentId: Uuid;
  toAgentId: Uuid;
  artifactType: ArtifactType;
  conditions: {
    all?: { field: string; op: "eq" | "neq" | "gte" | "lte" | "in" | "exists"; value?: unknown }[];
    any?: { field: string; op: "eq" | "neq" | "gte" | "lte" | "in" | "exists"; value?: unknown }[];
  };
  decisionOwner: {
    kind: "from_agent" | "to_agent" | "human" | "sanction";
    agentId?: Uuid;
  };
  sanctionGrant?: SanctionGrant | null;
  enabled: boolean;
  createdAt: IsoDateTime;
  updatedAt: IsoDateTime;
}

interface Run {
  id: Uuid;
  assignmentId: Uuid;
  agentId: Uuid;
  attempt: number;
  trigger: "manual" | "interval" | "handoff" | "parent";
  status: RunStatus;
  startedAt?: IsoDateTime;
  endedAt?: IsoDateTime;
  evidence: { kind: "memory" | "repo" | "input" | "grant"; ref: string; excerpt?: string }[];
  cost: { tokensIn: number; tokensOut: number; usd: number; latencyMs: number };
  decisions: {
    at: IsoDateTime;
    kind: string;
    summary: string;
    handoffId?: Uuid;
    granted?: boolean;
  }[];
  outputs: { artifactId: Uuid; artifactType: ArtifactType; schemaValid: boolean }[];
  approvals: { by: string; level: ApprovalLevel; at: IsoDateTime; note?: string }[];
  evalScores: { criterionId: string; pass: boolean; notes?: string }[];
  error?: string;
}
```

---

## 13. JSON Schema (draft 2020-12)

Enough to validate fixtures and API bodies. `additionalProperties: false`
everywhere. IDs and dates use `format` hints; Python may treat them as
strings.

```json
{
  "$id": "https://aiia.local/schemas/org-v1.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$defs": {
    "uuid": { "type": "string", "format": "uuid" },
    "datetime": { "type": "string", "format": "date-time" },
    "semver": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$" },
    "depth": { "type": "integer", "minimum": 0, "maximum": 4 },
    "approvalLevel": { "enum": ["auto", "supervised", "gated"] },
    "artifactType": {
      "enum": [
        "signal_report", "usage_snapshot", "policy_analysis", "grant_request",
        "risk_review", "approval_packet", "grant_decision", "integration_patch",
        "documentation", "audit_record", "failure_analysis", "memory_entry"
      ]
    },
    "toolId": {
      "enum": [
        "local.memory", "repo.read", "github.read",
        "sanction.authorize", "sanction.grants", "org.assign", "org.handoff"
      ]
    },
    "budget": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "kind", "cap", "period"],
      "properties": {
        "id": { "type": "string" },
        "kind": { "enum": ["tokens", "usd", "runs_per_day"] },
        "cap": { "type": "number", "minimum": 0 },
        "period": { "enum": ["day", "week", "month", "assignment"] },
        "walletId": { "type": "string" }
      }
    },
    "memoryScope": {
      "type": "object",
      "additionalProperties": false,
      "required": ["collections", "write"],
      "properties": {
        "collections": { "type": "array", "items": { "type": "string" } },
        "write": { "type": "boolean" }
      }
    },
    "sanctionGrant": {
      "type": "object",
      "additionalProperties": false,
      "required": ["grantId", "capability", "scope", "expiresAt", "status"],
      "properties": {
        "grantId": { "type": "string", "minLength": 1 },
        "capability": { "type": "string", "minLength": 1 },
        "scope": { "type": "object" },
        "expiresAt": { "$ref": "#/$defs/datetime" },
        "budget": { "$ref": "#/$defs/budget" },
        "status": { "enum": ["pending", "active", "denied", "expired", "revoked"] }
      }
    },
    "Organization": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "slug", "name", "mission", "repositoryIds", "policyIds", "budgets", "memoryScope", "teams", "createdAt", "updatedAt"],
      "properties": {
        "id": { "$ref": "#/$defs/uuid" },
        "slug": { "type": "string", "pattern": "^[a-z0-9][a-z0-9-]{1,62}$" },
        "name": { "type": "string", "minLength": 1, "maxLength": 80 },
        "mission": { "type": "string", "maxLength": 4000 },
        "repositoryIds": { "type": "array", "items": { "type": "string" } },
        "policyIds": { "type": "array", "items": { "type": "string" } },
        "budgets": { "type": "array", "items": { "$ref": "#/$defs/budget" } },
        "memoryScope": { "$ref": "#/$defs/memoryScope" },
        "teams": { "type": "array", "items": { "type": "object" } },
        "createdAt": { "$ref": "#/$defs/datetime" },
        "updatedAt": { "$ref": "#/$defs/datetime" }
      }
    },
    "Specialty": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "id", "slug", "version", "name", "instructions", "requiredTools", "tools",
        "contextSources", "acceptedInputTypes", "producedArtifacts", "outputSchema",
        "evalRubric", "defaultModelParams", "recommendedDepth", "requiredApprovalLevel",
        "tests", "deprecated"
      ],
      "properties": {
        "id": { "$ref": "#/$defs/uuid" },
        "slug": { "type": "string" },
        "version": { "$ref": "#/$defs/semver" },
        "name": { "type": "string" },
        "instructions": { "type": "string" },
        "requiredTools": { "type": "array", "items": { "$ref": "#/$defs/toolId" } },
        "tools": { "type": "array", "items": { "$ref": "#/$defs/toolId" } },
        "contextSources": { "type": "array", "items": { "type": "object" } },
        "acceptedInputTypes": { "type": "array", "items": { "$ref": "#/$defs/artifactType" } },
        "producedArtifacts": { "type": "array", "items": { "$ref": "#/$defs/artifactType" } },
        "outputSchema": { "type": "object" },
        "evalRubric": { "type": "array" },
        "defaultModelParams": {
          "type": "object",
          "additionalProperties": false,
          "required": ["temperature", "maxTokens"],
          "properties": {
            "temperature": { "type": "number", "minimum": 0, "maximum": 1 },
            "maxTokens": { "type": "integer", "minimum": 128, "maximum": 2000 }
          }
        },
        "recommendedDepth": { "$ref": "#/$defs/depth" },
        "requiredApprovalLevel": { "$ref": "#/$defs/approvalLevel" },
        "tests": { "type": "array" },
        "deprecated": { "type": "boolean" }
      }
    },
    "Agent": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "id", "organizationId", "name", "persona", "specialtySlug", "specialtyVersion",
        "memoryScope", "depth", "maxDelegationDepth", "computeAllocation", "toolGrants", "status"
      ],
      "properties": {
        "id": { "$ref": "#/$defs/uuid" },
        "organizationId": { "$ref": "#/$defs/uuid" },
        "name": { "type": "string", "minLength": 1, "maxLength": 80 },
        "persona": { "type": "string", "maxLength": 2000 },
        "specialtySlug": { "type": "string" },
        "specialtyVersion": { "$ref": "#/$defs/semver" },
        "memoryScope": { "$ref": "#/$defs/memoryScope" },
        "depth": { "$ref": "#/$defs/depth" },
        "maxDelegationDepth": { "$ref": "#/$defs/depth" },
        "computeAllocation": { "type": "object" },
        "repositoryId": { "type": "string" },
        "toolGrants": { "type": "array", "items": { "$ref": "#/$defs/toolId" } },
        "teamId": { "$ref": "#/$defs/uuid" },
        "status": { "enum": ["idle", "running", "blocked", "disabled", "error"] },
        "migratedFrom": { "type": "object" }
      }
    },
    "Assignment": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "id", "organizationId", "objective", "inputs", "successCriteria",
        "expectedArtifact", "assigneeId", "requesterId", "authority",
        "status", "remainingDelegation", "createdAt", "updatedAt"
      ],
      "properties": {
        "id": { "$ref": "#/$defs/uuid" },
        "organizationId": { "$ref": "#/$defs/uuid" },
        "objective": { "type": "string", "minLength": 1, "maxLength": 8000 },
        "inputs": { "type": "array" },
        "successCriteria": { "type": "array", "items": { "type": "string" } },
        "deadline": { "$ref": "#/$defs/datetime" },
        "recurrence": {
          "type": "object",
          "additionalProperties": false,
          "required": ["kind", "minutes", "maxRunsPerDay", "timezone"],
          "properties": {
            "kind": { "const": "interval" },
            "minutes": { "type": "integer", "minimum": 15, "maximum": 1440 },
            "maxRunsPerDay": { "type": "integer", "minimum": 1, "maximum": 48 },
            "timezone": { "const": "UTC" }
          }
        },
        "expectedArtifact": { "$ref": "#/$defs/artifactType" },
        "assigneeId": { "$ref": "#/$defs/uuid" },
        "requesterId": { "type": "string" },
        "parentAssignmentId": { "$ref": "#/$defs/uuid" },
        "handoffId": { "$ref": "#/$defs/uuid" },
        "authority": { "type": "object" },
        "status": { "enum": ["draft", "authorized", "running", "blocked", "done", "failed"] },
        "blockedReason": { "type": "string" },
        "remainingDelegation": { "type": "integer", "minimum": 0, "maximum": 4 },
        "createdAt": { "$ref": "#/$defs/datetime" },
        "updatedAt": { "$ref": "#/$defs/datetime" }
      }
    },
    "Handoff": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "id", "organizationId", "fromAgentId", "toAgentId", "artifactType",
        "conditions", "decisionOwner", "enabled", "createdAt", "updatedAt"
      ],
      "properties": {
        "id": { "$ref": "#/$defs/uuid" },
        "organizationId": { "$ref": "#/$defs/uuid" },
        "fromAgentId": { "$ref": "#/$defs/uuid" },
        "toAgentId": { "$ref": "#/$defs/uuid" },
        "artifactType": { "$ref": "#/$defs/artifactType" },
        "conditions": { "type": "object" },
        "decisionOwner": { "type": "object" },
        "sanctionGrant": { "anyOf": [{ "$ref": "#/$defs/sanctionGrant" }, { "type": "null" }] },
        "enabled": { "type": "boolean" },
        "createdAt": { "$ref": "#/$defs/datetime" },
        "updatedAt": { "$ref": "#/$defs/datetime" }
      }
    },
    "Run": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "id", "assignmentId", "agentId", "attempt", "trigger", "status",
        "evidence", "cost", "decisions", "outputs", "approvals", "evalScores"
      ],
      "properties": {
        "id": { "$ref": "#/$defs/uuid" },
        "assignmentId": { "$ref": "#/$defs/uuid" },
        "agentId": { "$ref": "#/$defs/uuid" },
        "attempt": { "type": "integer", "minimum": 1 },
        "trigger": { "enum": ["manual", "interval", "handoff", "parent"] },
        "status": { "enum": ["queued", "running", "succeeded", "failed", "cancelled"] },
        "startedAt": { "$ref": "#/$defs/datetime" },
        "endedAt": { "$ref": "#/$defs/datetime" },
        "evidence": { "type": "array" },
        "cost": { "type": "object" },
        "decisions": { "type": "array" },
        "outputs": { "type": "array" },
        "approvals": { "type": "array" },
        "evalScores": { "type": "array" },
        "error": { "type": "string" }
      }
    }
  }
}
```

---

## 14. Persistence, API, and suggested build order

Platform-level: **Brain owns the graph** (`:8100`). Command Center and
aiia-console are clients. Agent Studio's `agent_data.json` is a migration
source, not the long-term store. Mini scheduler calls Brain to dequeue
Runs (or Command Center continues to hold the lock but records Runs via
Brain). Dual-write is a transition tactic only.

Suggested on-disk layout (JSON files are enough for v1; SQLite later):

```
~/.aiia/org/
  organization.json
  specialties/<slug>@<version>.json
  agents.json
  assignments.json
  handoffs.json
  runs/<assignmentId>/<runId>.json
  artifacts/<artifactId>.json
```

Suggested Brain routes (all prefix `/v1/org`):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Organization singleton |
| `PUT` | `/` | Update mission/budgets/teams |
| `GET/POST` | `/specialties` | List / install package |
| `GET/POST` | `/agents` | |
| `POST` | `/assignments` | Create `draft` |
| `POST` | `/assignments/{id}/authorize` | draft → authorized (runs §8–§9) |
| `POST` | `/assignments/{id}/run` | Manual Run; `409` if Mini busy |
| `GET` | `/assignments/{id}/runs` | |
| `GET/POST` | `/handoffs` | Create edge (validates artifact types; grant may be null) |
| `POST` | `/handoffs/{id}/authorize` | Ask Sanction to mint; never local-mint |
| `POST` | `/handoffs/{id}/fire` | Internal; also invoked from Run completion |
| `POST` | `/migrate/agent-studio` | Idempotent map of §10 |

Fail closed on `POST /handoffs/{id}/fire` when grant is empty.

**Package location in-repo (when code lands):** `local_brain/org/` — new
package, not a folder inside `execution/` or `a2a/`. Execution stays the
side-effect engine; org stays the graph. Tests: `local_brain/tests/test_org_*.py`
with Sanction mocked (same style as `test_airgap.py`).

### Build order (v1 slices)

1. Pydantic models + JSON store + schema validation (no scheduler yet).
2. Assignment lifecycle `draft → authorized → running → done|failed` for a
   single Agent with `maxDelegationDepth=0` and **no** Handoffs. Migrate one
   Agent Studio loop onto this.
3. Handoff records + `canFire` + Sanction client. Illegal-edge tests.
4. Delegation rules + parentAssignment. Sanction grid fixtures.
5. Eval rubric on Run completion. Budget decrement + token log.
6. (Later, other increment) Organizational grid UI.

---

## 15. Relationship to existing systems

| System | Relationship |
|---|---|
| Agent Studio | Migration source; convenience editor; must not remain a second scheduler after write-through. |
| `GET /v1/loops` | Same map, lower priority. |
| `local_brain/execution/` | Runs that propose side effects still go through SafetyGate + AUTO/SUPERVISED/GATED. Assignment approval is the *work* gate; execution tier is the *action* gate. Both can block. |
| `local_brain/egress.py` | Unchanged for cloud tools. Handoff auth is a sibling client, stricter about unconfigured Sanction. |
| Research loop | Stays a topic/session engine (`docs/RESEARCH-LOOP.md`). A future Specialty may wrap it as Assignments; not v1. |
| Stories / roadmap | Human backlog. A D3 Lead may create Assignments *from* a story; stories are not Assignments. |
| A2A | Ingress only. |
| Air-gap | Isolated Runs ok; Handoffs need a grant from local Sanction or they do not fire. |
| `docs/HANDOFF.md` | Unrelated session notebook. |

---

## 16. Visual direction (spec only — do not implement)

Primary view becomes an **organizational grid**:

- **Columns** = teams / Specialty slugs.
- **Rows** = depth (D0–D4) or operational layer (Observe…Learn). Pick one
  scale and keep it; Sanction dogfood uses layers as rows and depth as a
  badge on the node.
- **Agent node** shows: status, open Assignment count, repository, tool
  grants, compute usage (runs today / cap, tokens).
- **Edges** show: delegation, artifact type, authorization boundary
  (grant active / pending / missing).
- **Select edge** → contract inspector (this spec's Handoff fields).
- **Runs** animate along granted edges only.
- **Approvals** appear at blocked edges (`awaiting_grant`,
  `awaiting_approval`).

This UI is an interface *over* the model. If the grid would need a Handoff
the store cannot represent, the store wins: do not draw illegal edges.

Out of scope for v1 (repeat): grid polish, free-form canvas, animation,
drag-to-connect that bypasses typed Handoff creation.

---

## 17. Acceptance criteria (v1 increment)

An implementation is done when all of the following are true. Doc review
for *this* PR uses the checklist in §19.

1. **Assignments are the unit of work.** A recurrent Agent Studio loop can
   be represented as one Assignment + many Runs without using `loop_task`
   as a source of truth.
2. **Handoffs are the only A→B path.** A helper that calls another Agent
   as a tool is rejected with `handoff_required`. Covered by a unit test.
3. **Empty grant cannot fire.** `Handoff.sanctionGrant` null / non-active /
   expired ⇒ `canFire == false`. Covered by a unit test. No local mint.
4. **Sanction unconfigured ⇒ Handoff does not fire**
   (`sanction_unconfigured`). Isolated no-Handoff Runs still run (vanilla
   OSS loops keep working).
5. **Illegal Sanction-grid edge refused:** Policy Analyst cannot invoke
   Grant Operator; must submit an Assignment carrying evidence,
   capability, budget, expiration. Fixture in §11.2 has a test.
6. **Legal chain works with mocked Sanction:** Analyst → Architect →
   Operator, each edge requiring a mocked `authorized: true` + `grant_id`.
7. **`maxDelegationDepth`:** migrated loop Agents have `0`; a D3 with `1`
   can spawn one child; the child cannot spawn; cycles rejected.
8. **Depth ≠ IQ:** changing the model does not change `depth`.
9. **Mini serialization preserved:** concurrent manual + interval Runs
   still collide on one lock.
10. **Specialty is a package:** installing `policy-analyst@1.0.0` onto two
    Agents shares instructions/schema/rubric; editing persona on one Agent
    does not edit the package.
11. **Approval floor cannot be lowered** below Specialty
    `requiredApprovalLevel`.
12. **No overwrite of `docs/HANDOFF.md`.** Session notebook remains.
13. **No canvas, no grid app** in the v1 implementation PR. API + store +
    tests only.
14. **GitHub stays disconnected** in prompts until a later increment.

---

## 18. Out of scope (explicit)

- Organizational grid UI, animation, drag-and-drop, free-form canvas.
- Multi-org / multi-tenant Brain.
- Raising Mini concurrency.
- GitHub write, web fetch as Agent tools, new egress points.
- Replacing research loop, stories, or A2A.
- Cron recurrence, event-bridge, webhooks into fire().
- Human-in-the-loop chat as a substitute for Handoff.
- Auto-approving GATED Assignments from a D4 Agent without a grant.
- Prompt-only "policies" that are not Sanction grants.

---

## 19. Doc-review checklist (this spec PR)

Use as the test plan until code exists:

- [ ] Engineer can implement models + `canFire` + Sanction client from this
      file alone.
- [ ] v1 is Assignments + typed Handoffs; UI is deferred (§2, §16, §18).
- [ ] Empty grant / unconfigured Sanction / illegal Analyst→Operator are
      unambiguous fails.
- [ ] Loop → Assignment/Run map matches live Agent Studio bounds
      (15–1440 min, 1–48/day, serialized Mini, disconnected GitHub).
- [ ] Depth table and `maxDelegationDepth` defaults are normative, not
      advisory.
- [ ] Six entities have fields, types, and status machines.
- [ ] TypeScript sketches and JSON Schema agree on names and enums.
- [ ] `docs/HANDOFF.md` still exists and is only the session notebook.
- [ ] CHANGELOG `[Unreleased]` points here.
- [ ] No application code required to land the spec.

---

## 20. Open questions that must not block v1

Resolved defaults (change only with a spec bump):

| Question | v1 default |
|---|---|
| One org or many? | One per Brain. |
| Where does the store live? | Brain, `~/.aiia/org/`. |
| Sanction route name? | `POST /authorize/handoff`; adapter may wrap `/authorize/tool` *only* if the response includes `grant_id` and `authorized`. |
| Recurrence model? | Interval only, same bounds as loops. |
| Who is `human:` requester? | `human:local` until Console auth exists. |
| Specialty tests required? | Required before production Assignments; drafts allowed without. |
| Eval executor? | Deterministic JSON-schema check + optional local-model rubric; schema failure always fails the Run. |

Unblock later, not now: Console-side grant inspector, grid, multi-org,
ops-loop migration, wrapping research topics as Assignments.
