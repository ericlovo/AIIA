# Sprint: feat/agent-studio-world-canvas

Handed by AIIA Bot → Codex · 2026-09-06
Repo: AIIA · local checkout: this repository · base: main @ dc70bac (#48 merged)

## Goal
Land the World/Activity WIP as a reviewable PR, make durable layout + live updates work on Mini prod (:8200), keep Git writes approval-gated. No free-form agent shell.

## Context (already true)
- Merged #48: agents, assignments, handoffs, allowlisted repo/GitHub read (GET-only), approval-gated worktrees + writes (write_file, run_tests, commit). push/open_pr deferred.
- Uncommitted WIP already on disk (do not rewrite from scratch):
  - dashboard/src/console/VoidStarWorld.tsx, AgentGraphOverlay.tsx, voidstarProjection.ts, ActivityOverview.tsx
  - dashboard/public/voidstar/**
  - local_brain/command_center/studio_layout.py, studio_events.py
  - server.py routes GET/PUT/DELETE /api/agent-world/layout + WS broadcast
  - tests: test_studio_layout.py, test_studio_events.py, test_agent_registry.py
  - StudioTabs views: activity | agents | assignments | handoffs | world
- Live gap: GET http://127.0.0.1:8200/api/agent-world/layout → 404 until rebuild + com.aiia.brain restart loads new server.py. Mock may exist on :8217 — do not rely on it for prod.
- Also include pending ops-brief format system-prompt nudge in server.py if present (State/Signals/Risks/Next; no emoji).

## In scope
1. Branch feat/agent-studio-world-canvas from main.
2. Stage ONLY studio world/activity source + tests + necessary api.ts/AgentStudio/WorkBoard/StudioTabs/vite/public changes. EXCLUDE: monitor_data.json, action_data.json, chat_history.json, task_data.json, token_data.json, .env.backup*, unrelated dirt.
3. Verify APIs + UI: layout CRUD, drag/save/reload positions, Activity timeline, World tab, handoff wire → existing handoff create (human still runs downstream).
4. Rebuild dashboard/dist; REACT_DIST prefers dashboard/dist; restart gui/$UID/com.aiia.brain once; confirm layout endpoint 200.
5. UI: surface mini_busy when Run returns 409.
6. Tests green; open PR; CI green.

## Out of scope
Typed Specialty/depth D0–D4 on nodes; Sanction-gated handoffs in canvas; push/open_pr; fixing all Pulse reds; splitting Proxy AI remote.

## Acceptance
- [ ] PR open, CI green
- [ ] Mini: Activity + World usable after build/restart
- [ ] Layout survives refresh
- [ ] Drag → save → reload keeps positions
- [ ] Graph handoff creates normal handoff record
- [ ] 409 Run shows Mini busy in UI
- [ ] No runtime JSON data files in PR

## Key paths
dashboard/src/console/{AgentStudio,StudioTabs,WorkBoard,ActivityOverview,VoidStarWorld,AgentGraphOverlay,voidstarProjection}.*
dashboard/public/voidstar/**
dashboard/src/lib/api.ts
local_brain/command_center/{server,studio_layout,studio_events,agent_registry}.py
local_brain/tests/test_studio_*.py

## Report back
PR URL, SHAs, proof of /api/agent-world/layout 200, test summary.

## Report to AIIA Bot / Eric
What you did for the handoff (file path, clipboard, whether Codex thread was started).
