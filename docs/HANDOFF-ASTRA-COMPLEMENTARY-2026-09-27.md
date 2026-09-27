# Handoff: Astra / complementary AIIA work (2026-09-27)

**From:** Claude Code session on the Studio UX overhaul
**For:** Astra (three-day window), Eric reviews and merges
**Repo:** `ericlovo/AIIA` · base `main` at `3cb10df`
**Air-gap:** stays **ON** (`AIIA_AIRGAP=1`). No new egress points.

This is a session handoff, not the org-graph Handoff entity
([`EXECUTABLE-ORGANIZATION.md`](./EXECUTABLE-ORGANIZATION.md)).

---

## TL;DR

The Agent Studio frontend is mid-overhaul in a separate lane (PRs 1 to 3
merged, PR 4 next). This handoff gives Astra five backend, tooling and docs
tracks that move the same roadmap forward **without touching the Studio
frontend files**. Track A unblocks Studio PR 5 directly. Every item below was
verified against `main` on 2026-09-27; nothing is from memory.

Recommended order for three days:

| Day | Tracks | Why this order |
|---|---|---|
| 1 | **A** action queue contract, **E** MCP launcher | Together they make human approvals reachable again |
| 2 | **B** Studio event contract, **C** ruff ratchet | B unblocks the shared socket hook; C stops a regression that is growing |
| 3 | **D** docs drift, then drain one ruff rule | Low risk, good buffer work |

---

## 0. Lane rules (read first)

**Do not edit** (owned by the Studio overhaul, PRs 4 to 8 restyle nearly all
of it and will conflict):

- `dashboard/src/**`, `dashboard/tests/**`
- `design/**`
- `docs/SPRINT-studio-ux-overhaul.md`

**Process:**

- One topic per PR, each branched from `main`. **Do not stack PRs.** On
  2026-09-27 two stacked PRs merged into their stacked bases instead of
  `main` and needed a recovery PR (#77).
- Conventional Commits (`feat`, `fix`, `docs`, `ci`, `test`, `refactor`).
- Before every push: `ruff check local_brain/`, `ruff format --check local_brain/`,
  `pytest local_brain/tests/` (CI enforces all three), and the sanitization
  grep in `.github/workflows/ci.yml`.
- No merges to `main`. Eric merges.
- Do not commit `.claude/` (a local mod lives there) or any runtime JSON
  (`action_data.json`, `agent_data.json`, `task_data.json`).
- Cursor Bugbot is hitting its usage limit on every PR, so do not rely on it
  as a reviewer. Self-review each diff adversarially before pushing.

---

## 1. Where the roadmap stands

Studio UX overhaul, per `docs/SPRINT-studio-ux-overhaul.md`:

| PR | Status | What it did |
|---|---|---|
| 1 | merged (#74) | Per-view error boundaries, one "needs attention" definition, Space key fix, deleted 3 orphan components |
| 2 | merged (#77) | Browser suites run in CI (`npm run test:browser`) |
| 3 | merged (#77) | One nav shell, hash routes for every view (`#/assignments/<id>` etc.) |
| 4 | next | Design tokens, 12px floor, contrast, blocking axe check |
| 5 | planned | Today absorbs Overview, **action approvals fold into Today**, shared socket hook |
| 6 to 8 | planned | Agent detail, Work, Inbox |

Tracks A and B below are the backend prerequisites for PR 5.

---

## Track A: action queue contract (unblocks Studio PR 5)

**Problem, verified:**

- `local_brain/command_center/action_queue.py` is the human-approval queue
  for automated findings (lifecycle `pending -> approved -> executing ->
  completed|failed`, `rejected`, `expired` after 72h). Producers include
  `execution/executor.py`, `execution/chains.py`, `execution/story_executor.py`,
  `autonomy/proactive_executor.py`, `autonomy/self_healing.py`,
  `autonomy/gated_downgrade.py`.
- HTTP routes exist in `command_center/server.py` (lines ~2433 to 2538):
  `GET/POST /api/actions`, `GET /api/actions/summary`,
  `POST /api/actions/{id}/approve|reject|complete`.
- **No test in `local_brain/tests/` exercises those HTTP routes.** Only
  `test_autonomy.py` touches the queue, at the unit level.
- The only UI for approvals was `dashboard/src/console/RightNow.tsx`, which
  has no importers. The Studio shows no pending approvals today.

**Deliverables:**

1. `local_brain/tests/test_action_routes.py`, using FastAPI's `TestClient`
   and a temp `ACTION_DATA_FILE`:
   - list, create, and summary shapes;
   - each legal transition;
   - illegal transitions (approve a rejected action, complete a pending one)
     return an error status, not 200;
   - an unknown id returns 404;
   - expiry: an action older than 72h reads as `expired`.

   Write the tests against current behavior first. If a test exposes a real
   bug (for example an illegal transition returning 200), fix it in the same
   PR and say so in the PR body.
2. `docs/ACTION-QUEUE.md`: the lifecycle diagram, every field with its type
   and meaning (especially `severity`, `source`, and whatever the UI needs to
   show "why this needs you"), who produces actions, and the 72h expiry rule.
3. A table comparing `Action` in `dashboard/src/lib/api.ts` with the real
   payload. **Report mismatches in the doc; do not edit `api.ts`** (lane
   rule). Studio PR 5 will consume the table.

**Acceptance:** CI green; the route tests fail if a transition guard is
removed (run that negative check once and note it in the PR).

---

## Track B: Studio event contract (unblocks the shared socket hook)

**Problem, verified:**

- `command_center/studio_events.py` projects records into
  `{entity, event, item}` messages with a per-entity field allowlist
  ("privacy-bounded"). `server.py` emits events such as `created`, `updated`,
  `deleted`, `closed`, `registered`, `agent_changed`, `session_attached`.
- The dashboard has two separate WebSocket consumers (Overview and Map) with
  their own ad hoc parsing. Studio PR 5 replaces them with one typed hook, and
  it needs the contract written down.

**Deliverables:**

1. `docs/STUDIO-EVENTS.md`: every `entity` × `event` pair that can reach
   `/ws`, the projected `item` fields per entity, and the snapshot message
   (`studio_snapshot`).
2. A unit test asserting the projection allowlist per entity. Its purpose is
   to catch a new sensitive field leaking to the browser: adding a field
   should require updating the test on purpose.

**Acceptance:** a fixture record carrying an extra secret-looking field
(`api_key`, `token`) never appears in the projected output.

---

## Track C: ruff backlog is growing, add a ratchet

**Problem, verified 2026-09-27:** `pyproject.toml` ignores 13 rules and
documents "73 sites". The actual count is **101**:

| Rule | Documented | Actual |
|---|---|---|
| E402 | 12 | **27** |
| B905 | 1 | **9** |
| SIM105 | 9 | 11 |
| SIM102 | 6 | 7 |
| SIM110 | 2 | 4 |
| others | = | = |

Ignored rules are accumulating new violations silently.

**Deliverables:**

1. **PR C1 (ratchet):** a small script, e.g. `scripts/ruff_ratchet.py`, that
   runs `ruff check --select <ignored rules> --statistics` and fails if any
   rule's count exceeds a checked-in baseline. Wire it into the existing
   `lint-and-test` CI job. Update the comment in `pyproject.toml` to the real
   counts.
2. **PR C2+:** drain one rule per PR (CLAUDE.md convention), then remove it
   from `ignore` and lower the baseline. Start with **B905**
   (`zip(..., strict=True)`, 9 sites) and **F841** (6 sites). Each fix needs a
   test run, because `strict=True` changes behavior when lengths differ:
   decide per call site and note it.

---

## Track D: docs and metadata drift (one PR)

All verified on `main`:

- `CLAUDE.md` says the dashboard has "no Tailwind yet". It runs Tailwind 4.3.
- `CLAUDE.md` says CI runs `pytest --collect-only` and that tests cannot run
  without Ollama. CI runs `pytest local_brain/tests/` for real
  (`ci.yml`, "pytest (enforced)").
- `local_brain/__version__.py`'s docstring says `pyproject.toml` reads its
  version from that module. `pyproject.toml` has its own literal
  `version = "0.7.0"`. **Recommended fix:** make the version dynamic
  (`[project] dynamic = ["version"]` plus
  `[tool.setuptools.dynamic] version = {attr = "local_brain.__version__.__version__"}`,
  or the equivalent for the build backend in use), then verify
  `pip install -e .` and the `aiia` CLI version output. If the backend makes
  that awkward, fix the docstring instead and say why.
- `docs/SPRINT-agent-studio-world-canvas.md` references `VoidStarWorld.tsx`
  and `voidstarProjection.ts`, which no longer exist. Mark the doc superseded
  at the top rather than rewriting it.

Leave `design/TODO.md` alone: Studio PR 4 rewrites it.

---

## Track E: MCP server config that works without hand edits

**Problem, verified:** `.mcp.json` in the repo root ships placeholder paths
(`/path/to/your/venv/bin/python3`, `/path/to/AIIA`). Every Claude Code session
that loads the repo, including cloud sessions, fails to start the `aiia` MCP
server (`ENOENT`). That also cuts off the MCP tools that surface pending
actions at session start, which is the queue from Track A.

**Deliverables:**

1. A launcher, e.g. `scripts/aiia-mcp`, that finds a usable Python (the
   repo's `.venv` if present, otherwise `python3` on `PATH`), checks that
   `local_brain` is importable, prints one actionable line if it is not, and
   execs `python -m local_brain.mcp_server`. `EQ_BRAIN_DATA_DIR` defaults to
   `~/.aiia/eq_data` unless set.
2. `.mcp.json` points at the launcher with repo-relative paths and no
   personal paths.
3. A short README or CLAUDE.md note on how to verify it (`claude mcp list`
   shows `aiia` connected).

**Security:** the launcher must not print environment values or keys. Keep
the server on stdio, with no network listener.

---

## Out of scope for this window

- Anything under the lane rules in §0.
- Unsetting air-gap, new egress, Voice, email, GitHub App work.
- Runtime changes on the Mini (budgets, agent edits). Note them in a PR body
  or the journal, never in committed JSON.

---

## Handback

At the end of the window, append to this file (or open a short
`docs/JOURNAL.md` entry) with:

- PRs opened, and their state;
- anything found but not fixed, with file references;
- any contract decision Studio PR 5 must respect (field names, event names).
