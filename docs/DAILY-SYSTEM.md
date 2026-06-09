# Eric's Daily System — design

The ritual layer that binds AIIA's machinery into a daily practice. Decided
2026-06-09 in conversation; this doc is the durable spec.

## The shape of the practice

| Decision | Choice |
|---|---|
| Front door | **AIIA Console app** — open it, you're logged in to the day |
| Briefing contents | What to work on · yesterday recap · system + repo status · one thinking prompt |
| Day shape | **Many threads** — context-parking and resumption are the core service |
| Organizing principle | **Eric's ventures**, of which AIIA is chapter one |

Everything built for this ritual ships in the product. Eric is user zero;
the DMG's first-run experience inherits his morning surface.

## Architecture

```
┌─ aiia-console ──────────────────────────────┐
│  TODAY view (opens on launch)               │
│  ├─ Briefing card: synthesis + risk flags   │
│  ├─ Thinking prompt                         │
│  └─ Workstreams: resumable threads          │
│     (next_action, branch, ci_status,        │
│      session history → tap to resume)       │
└──────────────┬──────────────────────────────┘
               │ GET /v1/aiia/briefing
┌──────────────▼──────────────────────────────┐
│  Brain :8100                                │
│  ├─ MorningBriefing.generate()              │
│  │   overnight reports → risk flags →       │
│  │   LLM synthesis (heuristic fallback)     │
│  └─ WorkstreamRegistry.list_active()        │
└─────────────────────────────────────────────┘
```

The same endpoint powers the CLI (`aiia briefing`) and mobile (iOS thin
client) — one briefing, every device.

## The daily loop

1. **Open the console** → Today view loads the briefing.
2. **Pick a thread** — a workstream with its `next_action`, branch, and last
   session context, or take the briefing's recommendation.
3. **Work** — sessions attach to the workstream (`session-start` /
   `session-end` endpoints already exist); decisions and lessons flow into
   memory as a side effect.
4. **Park** — ending a session updates the workstream's `next_action`, so
   tomorrow's resume is one tap.
5. **Capture anywhere** — voice journal from the phone (docs/JOURNAL.md),
   `aiia memory add`, or the console.

## Build status

- [x] `GET /v1/aiia/briefing` — Brain endpoint (briefing + active workstreams)
- [ ] Console **Today view** — the front door surface (opens on launch)
- [ ] Thinking prompt — one good question drawn from memory + recent work
      (generate inside MorningBriefing or console-side; decide at build time)
- [ ] Repo status in briefing — CI state + open PRs for both repos
- [ ] Workstream resume — tap a thread, console restores its context
- [ ] iOS Today view — same endpoint, pocket-sized

## Principles

- **Dogfood first.** Eric's ritual is the product's first-run experience.
- **Ventures, not projects.** The schema holds future chapters, not just AIIA.
- **Threads are sacred.** Parking a thread must be cheap; resuming must be
  instant. The system, not Eric's head, holds the thread state.
- **Capture is ambient.** If remembering requires stopping, it won't happen.
