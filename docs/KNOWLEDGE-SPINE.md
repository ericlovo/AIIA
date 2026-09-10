# Knowledge Spine & Relativity Map

> Status: spec (docs-only). No runtime change in the PR that lands this file.
> Audience: every Grok Bot teammate (and any engineer wiring FEED later)
> so the commons does not amnesia-fragment across shards.
>
> Eric-endorsed. Elo's structure pass is the routing contract.

**AIIA is the collective knowledge spine.** Shards hold local product
memory. The spine holds the facts the *team* must still know after a
session ends. AIIA Bot is the only canonical FEED writer. Elo is the
routing co-owner and does **not** write the spine directly.

**Not this file:** [`HANDOFF.md`](./HANDOFF.md) is the Mini ⇄ MacBook
session notebook. Do not put FEED records there. Assignments + typed
Handoffs live in [`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md)
— that graph is *who may act*. This file is *who may remember for the
commons*.

**Suite pointer:** when present, client-isolated Mindmoor membership
and the shared `mindmoor` memory namespace live in
[`MINDMOOR-AGENT-SUITE.md`](./MINDMOOR-AGENT-SUITE.md). Suite memory is
not a second spine. Mindmoor product facts may FEED `domain=product`
through AIIA Bot; they do not FEED `identity=*`.

---

## 1. Thesis

Without a single committer, every bot writes a private diary. The next
session cannot see it. The next bot contradicts it. That is
amnesia-fragment.

The spine exists so:

1. **One writer.** Only AIIA Bot commits FEED records.
2. **One router.** Elo decides which domains reach the spine, and which
   stay shard-local or stay off the record entirely.
3. **One board default.** Confidence ≥ 0.7 is board-visible unless a
   rule below says otherwise.
4. **No secret values. No venting. No ack chatter.** Those never FEED.

This file is the relativity map: who sits where, which edges Elo owns
exclusively, which shards may FEED AIIA directly, and when to escalate
to Eric instead of writing silently.

---

## 2. Standing constraints (Voice / air-gap)

These are not optional footnotes.

| Constraint | Rule |
|---|---|
| Air-gap | `AIIA_AIRGAP=1` stays **on**. Board facts about air-gap posture are `domain=ops` on the spine. Unsetting air-gap is a go/no-go, not a silent FEED. See [`AIRGAP.md`](./AIRGAP.md). |
| Voice | Voice Conductor ([`VOICE-CONDUCTOR.md`](./VOICE-CONDUCTOR.md)) is an interface, not a FEED writer. It does not commit spine records. Voice / air-gap **policy** changes route Elo → AIIA as `domain=ops`. |
| Secrets | Credential *UX* (where a key lives, that a key is missing) may be discussed. Credential **values** never FEED, never room-post, never land in this repo. |
| Runtime JSON | Live Mini `agent_data.json`, assignment JSON, `~/.aiia/keys.json`, and vault dumps stay off git. This spec does not add a FEED store file. |

---

## 3. Roles

| Actor | Writes to spine | Role |
|---|---|---|
| **AIIA Bot** | **Yes — only committer** | Canonical FEED writer. Commits every spine record. Owns the commons log. |
| **Elo Bot** | **No direct writes** (FEED only via AIIA) | Routing co-owner. Intercepts `work` / `identity` / `ops` / `eng-routing`. Sole custody of Elo-exclusive edges (§6). Private 1:1 with Eric never FEEDs. |
| **Sonos** | Shard → AIIA direct | `music` / `taste` / `house` / `session` only. Never `identity=*`. |
| **Mini** | No (executes) | Serialized local compute. Board-changing mini-ops go Elo → AIIA. Mini does not invent FEED rows. |
| **Morrow** | Shard → AIIA direct | `product` (local). Cross-product work is Elo → AIIA. |
| **Sanction** | Shard → AIIA direct | `product` (local grants / policy facts). Does not write the spine as committer. |
| **AIIA (product shard)** | Shard → AIIA direct | `product` facts about AIIA itself. Same local-product rule as Morrow / Sanction. |
| **Coding Triage** | No (executes) | Executes the bug / PR / incident Elo routed. Does not decide spine ownership. |
| **CI / CD** | `ops` only, deploy-verify board changes | Verify / ship signals that change the board. Not eng-routing. Not identity. |
| **Feedback Lab / Panel** | Shard → AIIA direct | `panel-feedback` **aggregates only**. Raw comments, tone triage, and ack chatter never FEED. |

---

## 4. Domain routing

### 4.1 Elo intercepts (do not FEED these from a shard)

```
work | identity | ops | eng-routing
```

| Domain | Who may promote to AIIA | Shard must not |
|---|---|---|
| `identity` | **Only Elo** | Product shards must not FEED `identity=*` |
| `work` | Elo, when it is **cross-product** ship order / PR merge posture | Shards may FEED `product=` locally; they do not set team-wide work |
| `ops` | Elo when the fact is **security / deploy / credential-path** | Air-gap, Mini deploy go/no-go, key *placement*. Plain high-confidence schedule-fired ops may FEED direct (§4.3) |
| `eng-routing` | **Only Elo** (`sole_routes`) | Which bot owns a bug / PR / incident. Coding Triage executes; Elo decides the spine |

### 4.2 Shard → AIIA direct (aggregates / local product only)

```
music | taste | house | session | product | panel-feedback
```

| Domain | Direct writer | Bound |
|---|---|---|
| `music` `taste` `house` `session` | Sonos | House / listen facts. Not identity. Not ops policy. |
| `product` | Sanction, Morrow, AIIA product shard | Local to that product. Cross-product work is not `product`. |
| `panel-feedback` | Feedback Lab | Aggregates only (counts, themes, decided takeaways). |

### 4.3 Ops exception (schedule-fired, plain)

A **plain**, high-confidence, schedule-fired ops fact (cron ran, backup
ok, loop tick counted) may FEED `domain=ops` **direct** — it does not
need Elo intercept.

Elo **does** intercept ops when the path is security, deploy, or
credential: air-gap, Mini deploy go/no-go, key placement.

### 4.4 Never FEED

These are off the spine. Do not route them through Elo as a "quiet"
write either.

| Class | Examples | Where it lives |
|---|---|---|
| Secret **values** | API keys, tokens, passwords, raw keystore contents | Mini env / `~/.aiia/keys.json` only |
| Venting | Frustration, soft prefs about other bots, private mood | Elo `private_1to1` with Eric, or nowhere |
| Bot-tone triage | "that reply felt cold", style nitpicks about a teammate | Elo 1:1, never room-posted, never FEEDed |
| Ack chatter | "got it", "on it", emoji-only, empty status | Ephemeral room only |

---

## 5. FEED contract

A FEED is a **request that AIIA Bot commit**. It is not a chat
message. It is not a Mini JSON file. Shape (prose contract, not a
checked-in schema):

```
FEED
  domain:      identity | work | ops | eng-routing
               | music | taste | house | session
               | product | panel-feedback
  claim:       one durable fact (no secret values)
  confidence:  0.0–1.0   (see map)
  source:      actor slug (who observed)
  via:         Elo | shard-direct
  board:       true when confidence ≥ 0.7 unless a rule forbids
```

AIIA Bot is the only process that turns a valid FEED into a spine
record. Elo may *author* the request. Elo may not skip AIIA Bot.

### 5.1 Confidence map

| Label | Score | Board default |
|---|---|---|
| high | ≈ **0.9** | Yes |
| medium | ≈ **0.7** | Yes (threshold) |
| low | ≈ **0.4** | No — do not board-default |

**≥ 0.7 is board-default.** Below 0.7 stays off the board unless Eric
asks to pin it. Low-confidence claims escalate or stay shard-local;
they are not silent commons writes.

Air-gap board facts use this same map and land as `domain=ops`.

---

## 6. Relativity map — Elo-exclusive edges

Elo has **sole custody** of these edges. No other bot may fire them.

### 6.1 `Elo —private_1to1→ Eric`

Personal ops, Mac hygiene, reminders, writing / polish. Credential UX
never values. Venting and soft prefs about other bots.

**Never FEEDed. Never room-posted.**

### 6.2 `Elo —intercepts→ FEED(identity)`

Only Elo promotes identity to AIIA. Product shards must not FEED
`identity=*`.

### 6.3 `Elo —intercepts→ FEED(work)`

Cross-product ship order and PR merge posture. Shards may FEED
`product=` locally. Cross-product work goes Elo → AIIA.

### 6.4 `Elo —intercepts→ FEED(ops)` when security / deploy / credential-path

Air-gap, Mini deploy go/no-go, key placement. Plain high-confidence
schedule-fired ops may FEED direct (§4.3).

### 6.5 `Elo —sole_routes→ eng-routing`

Which bot owns a bug / PR / incident. Coding Triage **executes**. Elo
**decides** the spine. CI / CD FEEDs `ops` only for deploy-verify
board changes — it does not assign owners.

### 6.6 `Elo —go_nogo_gate→ new routine | new agent | new channel`

Always escalate to Eric **before create**, unless Eric already ordered
it. No silent spawn. No "I added a loop to help."

---

## 7. Shared edges (keep / add)

These are the commons edges that are **not** Elo-exclusive. They still
commit only through AIIA Bot.

| Edge | Rule |
|---|---|
| `Elo —intercepts→ FEED(*)` for the four intercept domains | Keep. Shards do not bypass. |
| `Sonos —direct→ FEED(music\|taste\|house\|session)` | Keep. House shard. |
| `Sanction\|Morrow\|AIIA —direct→ FEED(product)` | Keep. Local product only. |
| `Feedback Lab —direct→ FEED(panel-feedback)` | Keep. Aggregates only. |
| `Mini executes` | Keep. Board-changing mini-ops via Elo, then AIIA. |
| `Lab / CI / CD / Triage` feed rules | Add / keep as stated: Lab = aggregates; CI/CD = deploy-verify `ops`; Triage executes, does not FEED owners. |
| `Airgap / Voice policy → ops` | Add. Policy facts are `domain=ops`, Elo-intercepted, AIIA-committed. Voice is not a writer. |

```
Eric
  ▲
  │ private_1to1 (never FEED)
Elo ──intercepts──► FEED(identity|work|ops*|eng-routing) ──► AIIA Bot ──commit──► spine
  │
  └──sole_routes──► eng-routing
  └──go_nogo_gate─► new routine | new agent | new channel   (escalate first)

Sonos ──────────────► FEED(music|taste|house|session) ──► AIIA Bot
Sanction/Morrow/AIIA ► FEED(product) ──────────────────► AIIA Bot
Feedback Lab ───────► FEED(panel-feedback aggregates) ─► AIIA Bot
CI/CD ──────────────► FEED(ops) deploy-verify only ────► AIIA Bot
Mini / Coding Triage  = execute, do not commit
```

`ops*` = security / deploy / credential-path. Plain schedule-fired ops
may skip Elo.

---

## 8. Escalate vs silent write

### Silent write (Elo or shard authors; AIIA Bot commits; no Eric ping)

Allowed when **all** of these hold:

1. Domain is shard-direct **or** plain schedule-fired `ops` (§4.2–4.3).
2. Confidence ≥ 0.7 (board-default) **or** the claim is explicitly
   off-board and still durable.
3. Claim is a fact, not a secret value, vent, tone triage, or ack.
4. The write does not create a routine, agent, or channel.
5. The write does not change identity, cross-product work, eng-routing,
   or security / deploy / credential-path ops.

Examples: Sonos session taste at 0.9; Morrow local product decision at
0.7; Feedback Lab weekly theme aggregate; "nightly backup succeeded."

### Escalate to Eric (Elo; do not silent-FEED)

Escalate when **any** of these hold:

| Trigger | Why |
|---|---|
| New routine, agent, or channel | `go_nogo_gate` — unless Eric already ordered it |
| Identity promotion | Only Elo, and only with Eric when the claim is new or contested |
| Cross-product ship / merge posture | Team-wide `work` |
| Security / deploy / credential-path ops | Air-gap, Mini go/no-go, key placement |
| Eng-routing that *creates* or *reassigns* ownership | Elo decides, Eric confirms when it changes the map |
| Confidence < 0.7 and someone wants it on the board | Below board-default |
| Voice or air-gap policy change | Standing constraint; not a shard opinion |

Elo's 1:1 with Eric is the escalate path. That 1:1 itself is never
FEEDed.

---

## 9. Non-goals

- A checked-in FEED JSON store, vault dump, or Mini `agent_data.json`.
- AIIA Bot sharing commit rights with Elo, Sonos, Mini, or CI.
- Product shards FEEDing `identity=*`.
- Voice Conductor as a spine writer.
- Unsetting air-gap from a FEED.
- Publishing secret values, venting, bot-tone triage, or ack chatter.
- Absorbing Mindmoor suite membership (that file, when present).
- Replacing [`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md)
  Handoffs. Those gate *action*. FEED gates *commons memory*.

---

## 10. Related

| Doc | Relationship |
|---|---|
| [`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md) | Who may act (Assignments, Handoffs, depth). Not who may FEED. |
| [`MINDMOOR-AGENT-SUITE.md`](./MINDMOOR-AGENT-SUITE.md) | When present: first client-isolated Studio suite. Product FEEDs only. |
| [`AIRGAP.md`](./AIRGAP.md) | Fail-closed local runtime. Board facts = `domain=ops`. |
| [`VOICE-CONDUCTOR.md`](./VOICE-CONDUCTOR.md) | Speak / orchestrate. Not a FEED writer. Policy → `ops` via Elo. |
| [`HANDOFF.md`](./HANDOFF.md) | Session notebook. Not the spine. |
| [`SECURITY-ARCHITECTURE.md`](./SECURITY-ARCHITECTURE.md) | Execution + egress threat model. Secrets stay off the spine. |
