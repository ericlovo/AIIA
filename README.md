# AIIA — AI Information Architecture

> Your AI coding assistant forgets everything between sessions. AIIA doesn't.

**Hardware:** Mac Mini M4, 24GB unified memory
**License:** Apache 2.0
**Status:** Production (since February 2026)

---

## What Is This?

AIIA is the missing runtime layer between you and your AI coding tools.

Today, AI assistants like Claude Code, Cursor, and Copilot are powerful — but stateless. Every session starts from zero. They don't remember what you decided last week, what broke in production yesterday, or which feature your client is waiting on. You re-explain context, re-discover patterns, and lose continuity across every conversation.

AIIA fixes that. It runs on a Mac Mini next to your cloud infrastructure and gives your AI tools **persistent memory, autonomous background work, and a prioritized backlog** — turning a chat assistant into a teammate that knows your codebase, remembers your decisions, and wakes up every morning with a plan.

### How It Works

```
You ←→ Claude Code (MCP) ←→ AIIA (Mac Mini) ←→ Your Cloud Services
                               │
                               ├── Remembers every decision, pattern, and lesson
                               ├── Indexes your entire codebase for instant RAG
                               ├── Runs security scans, health checks, and reports overnight
                               ├── Captures stories from your work sessions automatically
                               ├── Scores and prioritizes your backlog using business impact
                               └── Executes safe fixes autonomously (lint, formatting, deps)
```

### What Makes This Different

**vs. OpenClaw / Computer Use agents:** Those operate your computer. AIIA operates *alongside* you — it's infrastructure, not a screen driver. It doesn't click buttons; it maintains memory, schedules background work, and feeds context into your existing AI tools via MCP.

**vs. RAG-only solutions:** RAG gives you search. AIIA gives you search + structured memory + autonomous task scheduling + story capture + safety-gated execution. Memory isn't just "retrieve similar chunks" — it's 9 typed categories with sync tiers, quality scoring, and automatic decay.

**vs. Custom GPTs / System Prompts:** Those are prompt engineering. AIIA is a running service with a real API, background jobs, a dashboard, health monitoring, and a CLI. It persists across sessions, across tools, across days.

### Who Is This For?

- **Solo developers or small teams** running production SaaS who want their AI tools to have institutional memory
- **Platform engineers** building multi-tenant products who need a local intelligence layer that knows every service, every tenant, every deployment
- **Anyone tired of re-explaining context** to Claude/Cursor/Copilot every single session

### The 60-Second Version

1. **You work.** Write code, make decisions, ship features — using Claude Code, Cursor, whatever.
2. **AIIA remembers.** Decisions, patterns, lessons, and work-in-progress are captured to structured memory via MCP tools. Stories are auto-extracted from your session summaries.
3. **AIIA works overnight.** Security scans, memory consolidation, codebase re-indexing, morning briefings — all on a schedule, all on local hardware, all at $0 LLM cost.
4. **You come back.** Next session, AIIA loads your context: what you were doing, what decisions you made, what the security scan found, what to build next. No re-explaining.

That's the loop. Work → Remember → Background → Context → Work.

---

## Architecture Overview

```mermaid
graph TB
  subgraph "Mac Mini M4 (24GB)"
    subgraph "Local LLM Runtime"
      Ollama["Ollama :11434<br/>llama3.1:8b Q8_0 (10.2GB VRAM)<br/>nomic-embed-text<br/>deepseek-r1:14b"]
    end

    subgraph "Local Brain API :8100"
      API["FastAPI<br/>local_api.py"]
      AIIA["AIIA Brain<br/>brain.py"]
      Conductor["Smart Conductor<br/>Intent Classification"]
      Memory["Structured Memory<br/>9 categories, JSON"]
      Knowledge["ChromaDB<br/>5,512 docs indexed"]
      SessionIdx["Session Indexer<br/>Claude Code transcripts"]
      Prioritizer["Story Prioritizer<br/>5-filter framework"]
      RLM["Recursive Engine<br/>RLM Phase 4"]
    end

    subgraph "Command Center :8200"
      Dashboard["Web Dashboard<br/>4 views + WebSocket"]
      Tasks["Task Runner<br/>11 scheduled tasks"]
      Actions["Action Queue<br/>Approval workflow"]
      Monitor["Production Monitor<br/>30s check cycle"]
      Roadmap["Roadmap Store<br/>Stories + Pipeline"]
      Executor["Execution Engine<br/>Safety-gated"]
    end

    subgraph "Nightly Automation (launchd)"
      SecurityScan["Security Scan<br/>12:00am · 6 scanners"]
      MemorySync["Memory Sync<br/>Every 4h · Tier 1+2"]
      DailyReport["Daily Report<br/>2:30am · git analysis"]
      Consolidation["Consolidation<br/>3:00am · DeepSeek R1"]
      Briefing["Morning Briefing<br/>4:30am · DeepSeek R1"]
      SessionIndex["Session Index<br/>5:30am · JSONL → ChromaDB"]
      IntervalReport["Interval Reports<br/>Every 3h"]
    end
  end

  subgraph "Cloud Services"
    Supermemory["Supermemory<br/>21 containers<br/>3M tokens/month"]
    Render["Render (5 services)<br/>Product · Platform<br/>Marketing · Sales · Client"]
    Vercel["Vercel<br/>Per-tenant frontends"]
  end

  subgraph "Developer Tools"
    Claude["Claude Code<br/>MCP Server integration"]
    BrainCLI["brain CLI<br/>20+ commands"]
  end

  Claude -->|MCP tools| API
  BrainCLI -->|shell| API
  BrainCLI -->|shell| Dashboard

  API --> AIIA
  AIIA --> Ollama
  AIIA --> Memory
  AIIA --> Knowledge
  AIIA --> Conductor

  Dashboard --> Tasks
  Dashboard --> Actions
  Dashboard --> Monitor
  Dashboard --> Roadmap

  Tasks --> Actions
  Actions --> Executor
  Executor --> Ollama

  Memory -->|Metered sync| Supermemory
  Monitor -->|Health checks| Render
  API -->|Tailscale tunnel| Render

  Prioritizer --> Ollama
  Prioritizer --> Roadmap
  SessionIdx --> Knowledge
  SessionIdx --> Memory

  SecurityScan --> Actions
  Briefing --> Actions
  Consolidation --> Memory
  MemorySync --> Supermemory
```

---

## System Components

### Local LLM Stack

| Model | Role | Quantization | VRAM | Context | Temp | Max Tokens |
|-------|------|-------------|------|---------|------|------------|
| `llama3.1:8b-instruct-q8_0` | Routing, tasks, PII | Q8_0 (near-lossless) | ~10.2GB | 16K | 0.1-0.7 | 256-4096 |
| `nomic-embed-text` | Embeddings (RAG) | Native | ~0.5GB | — | — | — |
| `deepseek-r1:14b` | Deep reasoning (nightly) | Full | ~9GB | 8K | 0.6 | 8192 |

**Ollama Configuration:**
- `keep_alive: 30m` — Model stays warm between requests
- `num_batch: 512` — Parallel processing batch size
- `num_gpu: 99` — Full GPU offload
- VRAM headroom: ~14GB free after primary model loaded

### Memory System

Two-layer architecture: local JSON (instant, free) + Supermemory cloud (persistent, metered).

```mermaid
graph LR
  subgraph "Local (Mac Mini, $0)"
    D[decisions.json<br/>58KB · never expires]
    P[patterns.json<br/>78KB · never expires]
    L[lessons.json<br/>68KB · never expires]
    S[sessions.json<br/>20KB · 90d TTL]
    T[team.json<br/>1.4KB · never expires]
    A[agents.json<br/>1KB · never expires]
    M[meta.json<br/>62KB · 180d TTL]
    PR[project.json<br/>239KB · 60d TTL]
    W[wip.json<br/>516B · 24h TTL]
  end

  subgraph "Cloud (Supermemory)"
    SC["21 containers<br/>10 AIIA + 11 SME"]
  end

  D & P & L & S -->|"Tier 1: daily<br/>quality gate 4+/5"| SC
  T & A & M -->|"Tier 2: weekly<br/>no scoring"| SC
  PR & W -.-x|"Tier 3: never synced"| SC
```

**Memory Entry Structure:**
```json
{
  "id": "decisions_42_2026-03-12T15:00:00Z",
  "fact": "The fact to remember",
  "source": "claude-code|session|bootstrap",
  "created_at": "2026-03-12T15:00:00Z",
  "metadata": {}
}
```

**Current Stats (March 2026):**
| Category | Entries | Size | Tier | Sync | Decay |
|----------|---------|------|------|------|-------|
| decisions | ~60 | 58KB | 1 | Daily | Never |
| patterns | ~70 | 78KB | 1 | Daily | Never |
| lessons | ~60 | 68KB | 1 | Daily | Never |
| sessions | ~70 | 20KB | 1 | Daily | 90 days |
| team | ~2 | 1.4KB | 2 | Weekly | Never |
| agents | ~1 | 1KB | 2 | Weekly | Never |
| meta | ~40 | 62KB | 2 | Weekly | 180 days |
| project | ~200 | 239KB | 3 | Never | 60 days |
| wip | ~1 | 516B | 3 | Never | 24 hours |
| **Total** | **~500+** | **~529KB** | — | — | — |

**Metered Sync Pipeline:**
- Budget: 3M tokens/month, 200K/day
- Quality gate: local LLM scores each memory 1-5, only 4+ synced
- Dedup: SHA256 content hash prevents re-syncing
- Circuit breaker: halts after 5 consecutive API errors

### Knowledge Store (ChromaDB)

- **Collection:** `aplora_knowledge`
- **Documents:** 5,512 indexed
- **Storage:** 92MB on disk (`~/aplora-local-brain/eq_data/chroma/`)
- **Chunking:** 1,500 chars max, 200 char overlap, paragraph-aware breaks
- **Chunk IDs:** Deterministic SHA256 hash

**Indexed Content:**
- CLAUDE.md, ADRs, tenants.yaml, render.yaml
- 17 AI agents (Python source)
- Backend routes, services, models
- Knowledge base YAMLs (SME domain expertise)
- Product documentation, READMEs
- Local brain module code

### Smart Conductor (Intent Classification)

Replaces keyword matching with LLM-powered routing:

```
Query → SmartConductor → {domain, eq_level, complexity_score, recommended_path}
```

| Complexity | Path | Handler | Cost |
|-----------|------|---------|------|
| 0.0-0.3 | `local` | Ollama (Mac Mini) | $0 |
| 0.3-0.6 | `eos` | Single Claude call | ~$0.01 |
| 0.6-1.0 | `rlm` | Agentic loop (multi-step) | ~$0.10+ |

**Domains:** Finance, Legal, Compliance, Document, Memory, Crisis, Estate, Marketing, Social, General
**EQ Levels:** Fibonacci-scaled (1, 2, 3, 5, 8, 13, 21) — ANALYST through EMERGENCY

---

## Story Capture & Prioritization

### The Loop

```mermaid
flowchart LR
  A[Work Session] -->|"aiia_session_end<br/>next_steps + blockers"| B[Auto-Extract]
  B -->|"LLM extracts<br/>candidate stories"| C[Dedup Check]
  C -->|"SequenceMatcher<br/>85%+ = existing"| D[Roadmap Backlog]
  E[Manual] -->|"aiia_log_story<br/>tags + impact"| C
  D -->|"aiia_prioritize_backlog"| F[LLM Scoring]
  F -->|"5-filter framework<br/>weighted 0-150"| G[Ranked List]
  G -->|"aiia_execute_story"| H[Action Queue]
  H -->|"Decompose → Actions<br/>Safety-gated execution"| I[Shipped]
```

### 5-Filter Priority Framework

Every backlog story is scored against these filters (0-10 each, weighted):

| Filter | Weight | Question |
|--------|--------|----------|
| Closes Deal | 5x | Does this help close an active sales opportunity? |
| Retains Client | 4x | Does this fix a bug, improve UX, or add a feature for the paying client? |
| Reduces Cost | 3x | Does this reduce token spend, infra cost, or manual overhead? |
| Enables Tenants | 2x | Does this improve the platform for all products? |
| New Revenue | 1x | Does this create a new revenue stream (Content Engine, new product)? |

**Max Score:** 150 (all filters at 10)
**Priority Mapping:** P0 >= 90 | P1 >= 50 | P2 >= 25 | P3 < 25

### Story Model

```json
{
  "id": "a811afa1",
  "title": "AIIA as tenant-facing intelligence layer",
  "product": "platform",
  "priority": "P1",
  "status": "backlog",
  "description": "...",
  "source_session": "session-id",
  "source_type": "manual|auto-extracted",
  "tags": ["feature", "integration"],
  "client_impact": "All tenants benefit...",
  "related_stories": [],
  "priority_score": 64,
  "priority_reasoning": "...",
  "created_at": "2026-03-12T22:55:44Z",
  "updated_at": "2026-03-12T22:56:08Z"
}
```

**Valid Statuses:** backlog, active, in_progress, shipped, blocked, cancelled
**Valid Tags:** feature, bug, tech-debt, integration, ux, security, performance, devops

---

## Autonomous Task System

### Scheduled Tasks

| Task | Schedule | LLM | Purpose |
|------|----------|-----|---------|
| `health_journal` | Every 1h | No | Service health snapshots → AIIA memory |
| `ci_monitor` | Every 30m | No | CI/CD pipeline checks |
| `code_health` | Every 3h | No | Lint, test, dependency analysis |
| `security_scan` | Every 6h | No | 6-scanner security suite |
| `repo_sync` | Every 6h | No | Re-index repo into ChromaDB |
| `learning_loop` | Every 4h | Yes | Extract insights from recent actions |
| `test_runner` | Every 4h | No | Run platform test suite |
| `cross_tenant_analytics` | Daily 3am | Yes | Cross-tenant pattern analysis |
| `memory_digest` | Daily 6am | Yes | Memory consolidation digest |
| `daily_brief` | Daily 8am | Yes | Morning briefing generation |
| `weekly_client_status` | Every 7 days | Yes | Primary client health report |

### Nightly Automation (launchd)

| Time | Agent | What |
|------|-------|------|
| 12:00am | `com.aplora.securityscan` | 6 scanners: bandit, semgrep, trivy, trufflehog, shellcheck, hadolint |
| Every 4h | `com.aplora.memorysync` | Quality-scored sync: local memory → Supermemory cloud |
| 2:30am | `com.aplora.dailyreport` | Git log analysis grouped by product |
| 3:00am | `com.aplora.consolidate` | DeepSeek R1 memory consolidation (themes, contradictions, stale) |
| 4:30am | `com.aplora.briefing` | DeepSeek R1 alert synthesis from overnight reports |
| 5:30am | `com.aplora.sessionindex` | Claude Code JSONL transcripts → ChromaDB + memory |
| Every 3h | `com.aplora.intervalreport` | 3-hour code shipping windows |
| Always | `com.aplora.localbrain` | KeepAlive: auto-restart Brain API + Command Center |

### Action Queue

**Lifecycle:**
```
pending → approved → executing → completed
       ↘ rejected (terminal)
       ↘ expired (72h auto)
                           ↘ failed (terminal)
```

**Action Types:** lint_fix, test_fix, security_fix, ci_fix, review, tech_debt, post_commit_review, verify_lint, verify_test, verify_security, commit

**Safety Tiers:**

| Tier | Auto-Execute | Actions |
|------|-------------|---------|
| AUTO | Yes | lint_fix, verify_test, verify_lint, verify_security |
| SUPERVISED | 30s delay | test_fix, tech_debt, commit |
| GATED | Manual only | security_fix (critical/error), review |

**Forbidden Files:** `.env*`, `*.pem`, `*.key`, `*/migration/*`, `render.yaml`, `products/*/backend/main.py`

---

## Execution Engine

### Story Execution Flow

```mermaid
sequenceDiagram
  participant User
  participant MCP as MCP Server
  participant CC as Command Center
  participant SE as Story Executor
  participant LLM as Ollama (8B)
  participant AQ as Action Queue
  participant EE as Execution Engine

  User->>MCP: aiia_execute_story(story_id)
  MCP->>CC: POST /api/execution/story/{id}
  CC->>SE: execute_story()
  SE->>LLM: Decompose into 2-8 steps
  LLM-->>SE: JSON array of actions
  loop Each step
    SE->>AQ: create_action(type, severity, title, files)
  end
  SE->>CC: Story → in_progress
  Note over AQ: Actions sit in pending
  User->>AQ: Approve (or auto-approve)
  AQ->>EE: Poll picks up approved action
  EE->>EE: Safety gate check (tier)
  alt AUTO tier
    EE->>EE: Execute immediately
  else SUPERVISED
    EE->>EE: 30s notification delay
  else GATED
    Note over EE: Skip — needs explicit trigger
  end
  EE-->>AQ: Complete/fail action
  Note over SE: All complete → story shipped<br/>Any failed → story blocked
```

### Execution Strategies

| Strategy | Used For | Timeout | Method |
|----------|----------|---------|--------|
| DirectFixStrategy | lint_fix, dep bumps | 120s | `ruff check --fix` + `ruff format` |
| ClaudeCodeStrategy | Complex code changes | 600s | Claude CLI on `aiia/*` branch |
| CommitStrategy | Git commits | 60s | `git add` + `git commit` |

---

## MCP Integration (Claude Code)

### Available Tools

| Tool | Purpose |
|------|---------|
| `aiia_ask` | Search knowledge + memory + LLM reasoning |
| `aiia_remember` | Store fact in persistent memory |
| `aiia_search` | Fast vector search (no LLM) |
| `aiia_status` | Health + stats check |
| `aiia_session_start` | Load context at session start |
| `aiia_session_end` | Record summary + auto-extract stories |
| `aiia_save_wip` | Preserve work-in-progress state |
| `aiia_log_story` | Capture story to backlog (with dedup) |
| `aiia_prioritize_backlog` | Score backlog via 5-filter framework |
| `aiia_execute_story` | Decompose story into actions |
| `aiia_story_progress` | Check execution progress |
| `aiia_briefing` | Get/generate morning briefing |
| `aiia_ops_status` | Production health check |
| `aiia_tokens_today` | Token usage and costs |
| `aiia_what_was_i_doing` | Quick context catch-up |

### Session Protocol

```python
# START — Load context before work
aiia_session_start(task_description="What you're working on", branch="feat/xyz")

# DURING — Capture decisions and learnings
aiia_remember(fact="Chose X over Y because...", category="decisions")
aiia_log_story(title="Refactor auth middleware", tags=["tech-debt"])

# END — Preserve state for next session
aiia_save_wip(description="Halfway through auth refactor", next_steps=["Finish token validation"])
aiia_session_end(
    session_id="unique-id",
    summary="Refactored auth middleware",
    key_decisions=["Switched to jose library"],
    next_steps=["Add refresh token rotation"],
    blockers=["Need staging access"]
)
# ^ Auto-extracts stories from next_steps/blockers
```

---

## Command Center Dashboard

**URL:** `http://localhost:8200`

### Views

| View | URL | Purpose |
|------|-----|---------|
| Console | `/` | Platform constellation, routing stats, AIIA status, token tracking |
| Work | `/work` | Kanban board, check-in, activity, actions, prioritization |
| Voice | `/voice` | Voice interface with macOS TTS |
| Ops | `/old` | Legacy operations view |

### Work Dashboard Tabs

- **Check-in:** WIP, active/blocked stories, pending actions, commits, pipeline, "What to Build Next" (priority scoring)
- **Board:** Kanban (5 columns), drag-drop status changes, product/priority filters, "Prioritize" button
- **Activity:** Commits, heatmap, projects, uncommitted changes, daily report
- **Actions:** Pending action queue with approve/reject, severity filters

### API Endpoints (111 total — 74 server.py + 37 local_api.py)

<details>
<summary>Full endpoint list</summary>

**Pages:** GET `/`, `/old`, `/work`, `/voice`, WebSocket `/ws`

**Platform:** GET `/api/platform`, `/api/summary`, `/api/aiia`, `/api/health`

**Monitor:** GET `/api/monitor`, `/api/monitor/{service_id}`

**Tasks:** GET `/api/tasks`, `/api/tasks/history` | POST `/api/tasks/{id}/run`

**Actions:** GET `/api/actions`, `/api/actions/summary` | POST `/api/actions`, `/api/actions/{id}/approve`, `/api/actions/{id}/reject`, `/api/actions/{id}/complete`

**Reports:** GET `/api/briefing/latest`, `/api/reports/today`, `/api/reports/today-md`, `/api/reports`, `/api/reports/{date}`, `/api/reports/interval/latest` | POST `/api/briefing/generate`, `/api/reports/generate`, `/api/reports/interval`

**Metrics:** GET `/api/routing/stats`, `/api/routing/recent`, `/api/insights`, `/api/tokens/today`, `/api/tokens/recent` | POST `/ops/record-token-usage`, `/ops/record-latency`, `/ops/record-routing`

**Memory & Chat:** GET `/api/memories`, `/api/chat/history` | DELETE `/api/memories/{id}`, `/api/chat/history`, `/api/chat/history/{index}` | PUT `/api/chat/history/{index}` | POST `/api/chat`, `/api/chat/stream`, `/api/chat/stop`, `/api/tts`, `/api/voice`, `/api/speak`, `/api/speak/stop`

**Roadmap:** GET `/api/roadmap`, `/api/roadmap/similar/{title}`, `/api/roadmap/summary` | POST `/api/roadmap`, `/api/roadmap/extract`, `/api/roadmap/prioritize` | PUT `/api/roadmap/{id}` | DELETE `/api/roadmap/{id}`

**Pipeline:** GET `/api/pipeline` | POST `/api/pipeline` | PUT `/api/pipeline/{id}` | DELETE `/api/pipeline/{id}`

**Execution:** GET `/api/execution/status`, `/api/execution/log`, `/api/execution/story/{id}/progress` | POST `/api/execution/kill`, `/api/execution/execute/{id}`, `/api/execution/story/{id}`

**Work Context:** GET `/api/work/context`, `/api/checkin`, `/api/syntax`

</details>

---

## Brain CLI

```bash
brain start          # Start Ollama + Brain API + Command Center
brain stop           # Stop all services
brain restart        # Clean stop then start
brain status         # Service status + AIIA health

brain report         # Today's shipped code report
brain report 2026-03-10  # Report for specific date
brain report --interval  # 3-hour interval report

brain scan           # Full 6-scanner security suite
brain scan -q        # Quick scan (secrets + deps only)

brain sync           # Metered memory sync (local → cloud)
brain sync -w        # Weekly mode (includes Tier 2)

brain consolidate    # Deep memory consolidation (DeepSeek R1)
brain briefing       # Morning briefing (alert synthesis)
brain morning        # One-shot catch-up (nightly jobs + WIP + stories)
brain morning -v     # With voice output

brain chat           # Interactive AIIA chat
brain chat -v        # With voice

brain session-index  # Index Claude Code transcripts
brain commits        # Extract intelligence from git commits

brain idea "Title" product  # Quick-capture to backlog
brain actions list          # View pending actions
brain actions approve ID    # Approve action
brain actions reject ID "reason"

brain install-agents  # Install launchd agents (nightly automation)
brain test platform  # Run platform tests
brain logs           # Recent logs
brain logs -f        # Follow logs
brain logs err       # Error logs
brain pull           # Git pull latest code
brain help           # All commands
```

---

## Directory Structure

```
~/aplora-local-brain/
├── XCAi-AIIA/                              # Main monorepo
│   └── xcai_intelligence/
│       └── local_brain/
│           ├── local_api.py                # FastAPI :8100 (65KB)
│           ├── config.py                   # LocalBrainConfig dataclass
│           ├── ollama_client.py            # Ollama HTTP client
│           ├── smart_conductor.py          # LLM intent classification
│           ├── mcp_server.py               # MCP tools for Claude Code
│           │
│           ├── eq_brain/                   # AIIA Core Intelligence
│           │   ├── brain.py                # AIIA class (35KB)
│           │   ├── memory.py               # Structured JSON memory
│           │   ├── knowledge_store.py      # ChromaDB wrapper
│           │   ├── supermemory_bridge.py   # Cloud sync (SDK v3.27.0)
│           │   ├── memory_sync.py          # Metered sync + TokenLedger
│           │   ├── memory_consolidator.py  # DeepSeek R1 consolidation
│           │   ├── story_prioritizer.py    # 5-filter scoring engine
│           │   ├── session_indexer.py      # Claude Code transcript → ChromaDB
│           │   ├── morning_briefing.py     # Alert synthesis
│           │   ├── recursive_engine.py     # RLM Phase 4
│           │   ├── repl_env.py             # Variable-based exploration
│           │   └── bootstrap.py            # Knowledge indexing
│           │
│           ├── command_center/             # Web Dashboard :8200
│           │   ├── server.py               # FastAPI + WebSocket (84KB)
│           │   ├── aiia_tasks.py           # Task scheduling (91KB)
│           │   ├── action_queue.py         # Action lifecycle
│           │   ├── static/                 # Dashboard HTML/JS
│           │   │   ├── dashboard.html      # Console view
│           │   │   ├── work.html           # Kanban + prioritization
│           │   │   └── voice.html          # Voice interface
│           │   ├── task_data.json          # Persisted task state
│           │   ├── action_data.json        # Action queue
│           │   └── monitor_data.json       # Production health
│           │
│           ├── execution/                  # Safety-gated execution
│           │   ├── executor.py             # ExecutionEngine
│           │   ├── safety.py               # SafetyGate + tier mapping
│           │   ├── strategies.py           # Direct, Claude, Commit
│           │   ├── story_executor.py       # Story → action decomposition
│           │   ├── verification.py         # Post-execution checks
│           │   ├── subprocess_pool.py      # Subprocess management
│           │   ├── execution_log.py        # Execution history
│           │   ├── git_ops.py              # Git operations
│           │   └── chains.py              # Action chaining
│           │
│           ├── scripts/                    # Utilities & CLI runners
│           │   ├── roadmap_store.py        # Story CRUD + dedup
│           │   ├── pipeline_store.py       # Sales pipeline
│           │   ├── daily_report.py         # Git report generator
│           │   ├── memory_sync_runner.py   # CLI for brain sync
│           │   ├── consolidation_runner.py # CLI for brain consolidate
│           │   ├── morning_briefing_runner.py # CLI for brain briefing
│           │   ├── interval_report_runner.py  # 3-hour interval reports
│           │   ├── session_indexer_runner.py   # Claude Code transcript indexer
│           │   ├── briefing_cli.py         # Briefing generation CLI
│           │   ├── commit_intelligence.py  # Git commit analysis
│           │   ├── backfill_runner.py      # Data backfill utilities
│           │   └── syntax_checker.py       # Code syntax validation
│           │
│           └── pilot/                      # Mac Mini setup
│               └── start_brain.sh          # Startup script
│
├── eq_data/                                # AIIA Data
│   ├── memory/                             # 9 JSON memory files (~529KB)
│   ├── chroma/                             # ChromaDB (92MB, 5,512 docs)
│   ├── roadmap/stories.json                # Kanban stories
│   ├── sync/                               # Sync state + token ledger
│   ├── reports/                            # Daily/weekly reports
│   ├── execution/                          # Execution logs
│   ├── session_index/                      # Session memory index
│   └── trajectories/                       # Agent execution traces
│
├── logs/                                   # All automation logs
│   ├── brain.log                           # Main service log
│   ├── security/                           # Security scan reports
│   ├── sync/                               # Memory sync reports
│   ├── briefings/                          # Morning briefings
│   ├── consolidation/                      # Memory consolidation
│   └── session-index/                      # Session indexing
│
├── brain                                   # CLI (1,145 lines, bash)
├── start_brain.sh                          # Service startup
├── .env                                    # Environment variables
└── venv/                                   # Python virtualenv
```

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_LLM_URL` | `http://localhost:11434` | Ollama endpoint |
| `LOCAL_BRAIN_HOST` | `0.0.0.0` | Brain API listen address |
| `LOCAL_BRAIN_PORT` | `8100` | Brain API port |
| `LOCAL_ROUTING_MODEL` | `llama3.1:8b-instruct-q8_0` | Conductor model |
| `LOCAL_TASK_MODEL` | `llama3.1:8b-instruct-q8_0` | Task/extraction model |
| `LOCAL_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `LOCAL_DEEP_MODEL` | `deepseek-r1:14b` | Nightly deep reasoning |
| `EQ_BRAIN_DATA_DIR` | `~/aplora-local-brain/eq_data` | Data directory |
| `EXECUTION_ENABLED` | `false` | Enable execution engine |
| `SUPERMEMORY_API_KEY` | — | Supermemory cloud access |
| `SUPERMEMORY_ENABLED` | `true` | Cloud sync kill switch |
| `SUPERMEMORY_TIMEOUT` | `8.0` | Per-call timeout (seconds) |
| `ANTHROPIC_API_KEY` | — | Claude API (for Claude strategy) |
| `GOOGLE_API_KEY` | — | Google TTS |

### Key Limits

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Context window | 16,384 tokens | Ollama `num_ctx` |
| Output tokens | 3,072 | Max generation per request |
| Recursive iterations | 15 | RLM max loop count |
| Recursive token budget | 50,000 | RLM session cap |
| Sync daily budget | 200,000 tokens | Supermemory daily cap |
| Sync monthly budget | 3,000,000 tokens | Supermemory monthly cap |
| Execution timeout | 600s | Max per-action execution |
| Execution max retries | 2 | Retry count before failing |
| Execution concurrency | 1 | Max simultaneous actions |
| Monitor check interval | 30s | Production health polling |
| Scheduler interval | 10s | Task due-check frequency |
| Knowledge chunk size | 1,500 chars | ChromaDB chunk max |
| Knowledge chunk overlap | 200 chars | Cross-chunk context |

---

## Production Monitor

Checks 4 services every 30 seconds with 24-hour history retention:

| Service | Check URL | Timeout | Category |
|---------|-----------|---------|----------|
| AIIA Local Brain | `localhost:8100/v1/aiia/status` | 5s | intelligence |
| Product Backend | `{product}.onrender.com/health` | 10s | backend |
| Platform API | `{platform}.onrender.com/health` | 10s | backend |
| Ollama | `localhost:11434/api/tags` | 3s | local |

**Status Values:** online, degraded (slow/4xx), offline (timeout/5xx/error)

---

## Security

### 6-Scanner Suite (Nightly)

| Scanner | What | Fail Condition |
|---------|------|----------------|
| trufflehog | Secret detection | Any secrets found |
| trivy | CVE scanning (pip/npm) | Critical severity |
| bandit | Python SAST | High severity |
| semgrep | Pattern-based security | Error-level findings |
| shellcheck | Shell script analysis | Errors |
| hadolint | Dockerfile best practices | Errors |

### Execution Safety

- **Forbidden files** cannot be touched by automated execution
- **Safety tiers** gate what runs automatically vs. needs approval
- **Git isolation:** Execution creates `aiia/*` branches
- **Max 20 files** per action, 1 concurrent action

---

## Network Architecture

```
┌─── Mac Mini M4 (24GB) ────────────────┐
│  Ollama            :11434              │
│  Local Brain API   :8100               │
│  Command Center    :8200               │
│  AIIA EQ Brain + Memory               │
│  Supermemory Bridge (cloud sync)       │
└──────────┬─────────────────────────────┘
           │ Tailscale tunnel (WireGuard)
┌──────────┼────────────────────────────────────────┐
│          │            │              │             │
│  Platform       Product A     Product B     Product C
│  (platform)     (product-a)   (product-b)   (product-c)
│          │            │
│     Render.com    Render.com
└───────────────────────────────────────────────────┘
```

**Three-provider LLM stack:** LOCAL ($0) → ANTHROPIC (Claude, primary) → GOOGLE (Gemini, fallback)

---

## Getting Started

### Prerequisites

- Mac Mini M4 (or Apple Silicon with 24GB+ RAM)
- Ollama installed (`brew install ollama`)
- Python 3.12+ with virtualenv
- Tailscale for production tunnel (optional)

### Setup

```bash
# Clone and setup
cd ~/aplora-local-brain
git clone https://github.com/ericlovo/XCAi-AIIA.git
python3 -m venv venv
source venv/bin/activate
pip install -r XCAi-AIIA/requirements.txt

# Pull models
ollama pull llama3.1:8b-instruct-q8_0
ollama pull nomic-embed-text
ollama pull deepseek-r1:14b

# Configure
cp .env.example .env  # Edit with your API keys

# Start
brain start

# Install nightly automation
brain install-agents

# Bootstrap knowledge
cd XCAi-AIIA
python -m xcai_intelligence.local_brain.eq_brain.bootstrap
```

### Verify

```bash
brain status           # All services green
curl localhost:8100/health  # Brain API healthy
curl localhost:8200/api/aiia  # AIIA status + doc count
open http://localhost:8200    # Dashboard
```

---

## Key Design Decisions

1. **JSON over PostgreSQL** for memory/state — simplicity, zero-config, portable, git-diffable
2. **Quality-gated sync** — don't push everything to cloud; local LLM scores quality for free
3. **Safety tiers for execution** — automated lint is fine, security fixes need human eyes
4. **Fibonacci EQ scale** — non-linear emotional sensitivity maps well to real crisis escalation
5. **Story dedup** — SequenceMatcher at 85% catches "Lint execution module" vs "Lint check execution module"
6. **Weighted priority framework** — business impact (deals, revenue) outweighs technical elegance
7. **Variable-based RLM** — store docs as handles, not full context; LLM peeks only what it needs
8. **DeepSeek R1 for nightly** — chain-of-thought reasoning at $0 for consolidation and briefings
9. **Single concurrent action** — execution engine processes one action at a time for safety
10. **Deterministic chunk IDs** — SHA256 prevents re-indexing unchanged content
