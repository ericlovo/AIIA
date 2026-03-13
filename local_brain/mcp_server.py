"""
AIIA MCP Server — Gives Claude Code direct access to AIIA's brain.

This MCP server exposes AIIA's capabilities as tools that Claude Code
can call during any session. It communicates with the AIIA Local Brain
API running on the Mac Mini over Tailscale.

Tools:
    aiia_ask            — Ask AIIA a question (knowledge + memory + LLM reasoning)
    aiia_remember       — Teach AIIA a fact (persisted on Mac Mini)
    aiia_search         — Fast vector search (no LLM, just document matches)
    aiia_status         — Check AIIA health + knowledge/memory stats
    aiia_session_end    — Record session summary + decisions + lessons
    aiia_session_start  — Load context from past sessions, WIP, decisions at session start
    aiia_what_was_i_doing — Quick "catch me up" on recent work, decisions, open tasks
    aiia_ops_status     — Production health check via Command Center
    aiia_save_wip       — Save work-in-progress state for next session
    aiia_tokens_today   — Query today's token usage and costs
    aiia_log_story      — Log a story/idea to the roadmap backlog (with dedup)
    aiia_prioritize_backlog — Score and rank backlog stories using 5-filter framework

Usage:
    Configured in .mcp.json — Claude Code launches this automatically.

    Or manually:
        AIIA_URL=http://localhost:8100 python3 mcp_server.py

Requires:
    pip install mcp httpx
"""

import os
import logging
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("aiia.mcp")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

AIIA_URL = os.environ.get("AIIA_URL", "http://localhost:8100")
AIIA_API_KEY = os.environ.get("AIIA_API_KEY", "")
COMMAND_CENTER_URL = os.environ.get("COMMAND_CENTER_URL", "http://localhost:8200")
TIMEOUT = float(
    os.environ.get("AIIA_TIMEOUT", "120")
)  # seconds — ask can be slow on 7B

# ─────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────

mcp = FastMCP(
    "aiia",
    instructions=(
        "AIIA (AI Information Architecture) is the persistent AI teammate "
        "running on a Mac Mini M4. She has a ChromaDB knowledge base with "
        "5,500+ indexed documents from the project repo, and a "
        "structured memory system that persists across sessions. Use aiia_ask "
        "for questions that benefit from AIIA's knowledge and reasoning. Use "
        "aiia_remember to teach her facts, decisions, or lessons. Use "
        "aiia_search for fast document lookups without LLM reasoning. Use "
        "aiia_session_start at the beginning of work sessions to load context. "
        "Use aiia_save_wip before ending sessions to preserve work state."
    ),
)


async def _call_aiia(method: str, path: str, body: Optional[dict] = None) -> dict:
    """Make an HTTP request to the AIIA Local Brain API."""
    url = f"{AIIA_URL}{path}"
    headers = {}
    if AIIA_API_KEY:
        headers["X-API-Key"] = AIIA_API_KEY

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                resp = await client.post(url, json=body or {}, headers=headers)

            if resp.status_code != 200:
                return {"error": f"AIIA returned {resp.status_code}: {resp.text}"}
            return resp.json()
    except httpx.ConnectError:
        return {
            "error": (
                f"Cannot reach AIIA at {AIIA_URL}. "
                "Is the Mac Mini on? Is Tailscale connected?"
            )
        }
    except httpx.ReadTimeout:
        return {
            "error": f"AIIA timed out after {TIMEOUT}s. The model may be processing a complex query."
        }
    except Exception as e:
        return {"error": f"AIIA request failed: {str(e)}"}


async def _call_command_center(
    method: str, path: str, body: Optional[dict] = None
) -> dict:
    """Make an HTTP request to the Command Center dashboard."""
    url = f"{COMMAND_CENTER_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if method == "GET":
                resp = await client.get(url)
            else:
                resp = await client.post(url, json=body or {})

            if resp.status_code != 200:
                return {
                    "error": f"Command Center returned {resp.status_code}: {resp.text}"
                }
            return resp.json()
    except httpx.ConnectError:
        return {"error": f"Cannot reach Command Center at {COMMAND_CENTER_URL}"}
    except Exception as e:
        return {"error": f"Command Center request failed: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# Core Tools (existing)
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_ask(
    question: str, context: str = "", n_results: int = 5, depth: str = "fast"
) -> str:
    """Ask AIIA a question. She searches her knowledge base (5,500+ indexed
    docs from the project repo) and her persistent memory, then reasons
    with her local LLM (llama3.1:8b) to give a grounded answer.

    Use this for:
    - "What does AIIA know about the conductor?"
    - "How does the RLM engine work?"
    - "What decisions have we made about auth?"
    - Any question about the codebase, architecture, or project history

    Args:
        question: The question to ask AIIA
        context: Optional additional context to help AIIA answer
        n_results: Number of knowledge docs to search (default 5)
        depth: Search depth — "fast" (local only), "hybrid" (local + Supermemory cloud), "deep" (+ DeepSeek R1)
    """
    body = {"question": question, "n_results": n_results, "depth": depth}
    if context:
        body["context"] = context

    result = await _call_aiia("POST", "/v1/aiia/ask", body)

    if "error" in result:
        return f"AIIA Error: {result['error']}"

    answer = result.get("answer", "No answer")
    model = result.get("model", "unknown")
    sources = result.get("sources_used", 0)
    cloud_hits = result.get("cloud_hits", 0)
    suffix = f"[Model: {model} | Sources: {sources}"
    if cloud_hits:
        suffix += f" | Cloud: {cloud_hits}"
    suffix += "]"
    return f"{answer}\n\n{suffix}"


@mcp.tool()
async def aiia_remember(
    fact: str, category: str = "lessons", source: str = "claude-code"
) -> str:
    """Teach AIIA a fact so she remembers it permanently. This is stored on
    the Mac Mini and persists across all sessions forever.

    Use this for:
    - Decisions made: "We decided to use qwen2.5 for routing because..."
    - Lessons learned: "ChromaDB 0.5.23 must be pinned for compatibility"
    - Team knowledge: "Eric prefers snake_case for Python"
    - Patterns discovered: "The conductor routes finance queries to..."

    Categories: decisions, patterns, lessons, team_knowledge, agents, sessions, wip

    Args:
        fact: The fact to remember
        category: Memory category (default: "lessons")
        source: Where this fact came from (default: "claude-code")
    """
    body = {"fact": fact, "category": category, "source": source}
    result = await _call_aiia("POST", "/v1/aiia/remember", body)

    if "error" in result:
        return f"AIIA Error: {result['error']}"

    return f'AIIA remembered: "{fact}" [category: {category}]'


@mcp.tool()
async def aiia_search(query: str, n_results: int = 5) -> str:
    """Fast vector search across AIIA's knowledge base — no LLM reasoning,
    just returns the most relevant document chunks. Much faster than aiia_ask.

    Use this when you need raw document matches:
    - "Find code related to tenant configuration"
    - "Search for Dockerfile patterns"
    - "Find references to CORS setup"

    Args:
        query: Search query
        n_results: Number of results to return (default 5)
    """
    body = {"question": query, "n_results": n_results}
    result = await _call_aiia("POST", "/v1/aiia/search", body)

    if "error" in result:
        return f"AIIA Error: {result['error']}"

    results = result.get("results", [])
    if not results:
        return "No matching documents found."

    output = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        text = r.get("text", "")[:500]  # Truncate long chunks
        score = r.get("relevance_score", r.get("distance", "?"))
        output.append(f"[{i}] {source} (score: {score})\n{text}")

    return "\n\n".join(output)


@mcp.tool()
async def aiia_status() -> str:
    """Check AIIA's health — knowledge doc count, memory stats, model info,
    and whether she's online. Quick way to verify the Mac Mini connection.
    """
    result = await _call_aiia("GET", "/v1/aiia/status")

    if "error" in result:
        return f"AIIA is unreachable: {result['error']}"

    identity = result.get("identity", "AIIA")
    name = result.get("name", "AI Information Architecture")
    model = result.get("model", "unknown")
    knowledge = result.get("knowledge", {})
    memory = result.get("memory", {})
    categories = memory.get("by_category", memory.get("categories", {}))

    total_docs = knowledge.get("knowledge_docs", knowledge.get("total_documents", 0))
    session_docs = knowledge.get("session_docs", 0)
    total_mem = memory.get("total_memories", sum(categories.values()))

    lines = [
        f"{identity} — {name}",
        "Status: online",
        f"Model: {model}",
        f"Knowledge: {total_docs} documents ({session_docs} session docs)",
        f"Memories: {total_mem} total",
    ]
    for cat, count in categories.items():
        lines.append(f"  {cat}: {count}")

    return "\n".join(lines)


@mcp.tool()
async def aiia_session_end(
    session_id: str,
    summary: str,
    key_decisions: list[str] | None = None,
    lessons_learned: list[str] | None = None,
    next_steps: list[str] | None = None,
    blockers: list[str] | None = None,
) -> str:
    """Record the end of a Claude Code session. AIIA stores the summary
    and extracts decisions + lessons into her long-term memory.

    Also auto-extracts candidate stories from next_steps and blockers,
    deduplicates against existing backlog, and logs them with priority scoring.

    Call this at the end of significant work sessions to build AIIA's
    institutional knowledge over time.

    Args:
        session_id: Unique session identifier
        summary: What was accomplished in this session
        key_decisions: List of decisions made (optional)
        lessons_learned: List of lessons learned (optional)
        next_steps: List of next steps / future work items (optional)
        blockers: List of blockers or open questions (optional)
    """
    body = {"session_id": session_id, "summary": summary}
    if key_decisions:
        body["key_decisions"] = key_decisions
    if lessons_learned:
        body["lessons_learned"] = lessons_learned

    result = await _call_aiia("POST", "/v1/aiia/session-end", body)

    if "error" in result:
        return f"AIIA Error: {result['error']}"

    lines = [
        f"Session recorded. AIIA stored summary + {len(key_decisions or [])} decisions + {len(lessons_learned or [])} lessons."
    ]

    # Auto-extract stories from next_steps and blockers
    if next_steps or blockers:
        extracted = await _auto_extract_stories(
            summary=summary,
            next_steps=next_steps,
            blockers=blockers,
            key_decisions=key_decisions,
            session_id=session_id,
        )
        if extracted:
            lines.append(f"\n### Auto-captured {len(extracted)} stories from session:")
            for s in extracted:
                deduped = s.get("_deduped", False)
                tag = " (deduped -> existing)" if deduped else ""
                lines.append(f"  - [{s.get('priority', 'P2')}] {s['title']} ({s.get('product', '?')}){tag}")

    return "\n".join(lines)


async def _auto_extract_stories(
    summary: str,
    next_steps: list[str] | None = None,
    blockers: list[str] | None = None,
    key_decisions: list[str] | None = None,
    session_id: str = "",
) -> list[dict]:
    """Extract and log candidate stories from session data via Command Center."""
    # Call the new extraction endpoint on Command Center
    body = {
        "summary": summary,
        "next_steps": next_steps or [],
        "blockers": blockers or [],
        "key_decisions": key_decisions or [],
        "session_id": session_id,
    }
    result = await _call_command_center("POST", "/api/roadmap/extract", body)

    if "error" in result:
        logger.warning(f"Story extraction failed: {result['error']}")
        return []

    return result.get("stories", [])


# ─────────────────────────────────────────────────────────────
# Session & Workflow Tools (new)
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_session_start(
    task_description: str,
    branch: str = "",
    files: list[str] | None = None,
) -> str:
    """Load relevant context from past sessions, decisions, lessons, and
    work-in-progress at the start of a Claude Code session. Call this when
    beginning significant work to get AIIA's institutional memory.

    Returns: recent decisions, WIP items, relevant past sessions, and
    knowledge matches for the task at hand.

    Args:
        task_description: What you're about to work on
        branch: Current git branch (optional)
        files: Key files you'll be touching (optional)
    """
    body = {
        "task_description": task_description,
        "branch": branch or "",
        "files": files or [],
    }

    result = await _call_aiia("POST", "/v1/aiia/session-start", body)

    if "error" in result:
        # Fallback: make individual calls if session-start endpoint doesn't exist yet
        if "404" in str(result.get("error", "")):
            return await _session_start_fallback(task_description, files)
        return f"AIIA Error: {result['error']}"

    # Format the context package as a structured brief
    lines = ["=== AIIA Session Brief ===\n"]

    # Pending actions from action queue
    actions_result = await _call_command_center(
        "GET", "/api/actions?status=pending&limit=10"
    )
    if "error" not in actions_result:
        pending_actions = actions_result.get("actions", [])
        if pending_actions:
            lines.append(f"### Pending Actions ({len(pending_actions)} items)")
            for a in pending_actions:
                sev = a.get("severity", "info").upper()
                atype = a.get("type", "?")
                title = a.get("title", "")
                aid = a.get("id", "?")
                lines.append(f"- [{sev}] {atype}: {title} (id: {aid})")
            lines.append("")

    # Token & cost summary
    token_summary = result.get("token_summary", {})
    if token_summary.get("total_tokens"):
        total = token_summary.get("total_tokens", 0)
        cost = token_summary.get("total_cost", 0)
        reqs = token_summary.get("total_requests", 0)
        lines.append(f"Today: {reqs} LLM calls, ${cost:.4f}, {total:,} tokens")

    # Routing stats
    routing = result.get("routing_stats", {})
    if routing.get("total_requests"):
        eos_pct = routing.get("eos_pct", 0)
        rlm_pct = routing.get("rlm_pct", 0)
        lines.append(
            f"Routing: {routing['total_requests']} requests (EOS {eos_pct}% / RLM {rlm_pct}%)"
        )

    # WIP items
    wip = result.get("wip_items", [])
    if wip:
        lines.append(f"\n### Work in Progress ({len(wip)} items)")
        for item in wip:
            lines.append(f"- {item.get('fact', '')}")

    # Recent decisions
    decisions = result.get("recent_decisions", [])
    if decisions:
        lines.append(f"\n### Recent Decisions ({len(decisions)})")
        for d in decisions[:5]:
            lines.append(f"- {d.get('fact', '')}")

    # Recent insights
    insights = result.get("recent_insights", [])
    if insights:
        lines.append(f"\n### Recent Insights ({len(insights)})")
        for i in insights[:5]:
            sev_icon = {
                "success": "OK",
                "warn": "WARN",
                "error": "ERR",
                "info": "INFO",
            }.get(i.get("severity", ""), "")
            lines.append(f"- [{sev_icon}] {i.get('title', '')}")

    # Relevant knowledge
    knowledge = result.get("relevant_knowledge", [])
    if knowledge:
        lines.append(f"\n### Relevant Knowledge ({len(knowledge)} matches)")
        for k in knowledge[:3]:
            source = k.get("source", "unknown")
            text = k.get("text", "")[:200]
            lines.append(f"- [{source}] {text}")

    # Recent sessions
    sessions = result.get("recent_sessions", [])
    if sessions:
        lines.append(f"\n### Recent Sessions ({len(sessions)})")
        for s in sessions[:3]:
            lines.append(f"- {s.get('fact', '')}")

    # Related past sessions from session index
    history_result = await _call_aiia(
        "POST",
        "/v1/aiia/search-sessions",
        {"query": task_description, "limit": 3},
    )
    if "error" not in history_result:
        history_sessions = history_result.get("sessions", [])
        if history_sessions:
            lines.append(f"\n### Related Past Sessions ({len(history_sessions)} found)")
            for hs in history_sessions:
                sid = hs.get("session_id", "?")
                summary = hs.get("summary", "")
                ts = hs.get("start_timestamp", "?")
                branch = hs.get("branch", "?")
                lines.append(
                    f"- [{sid[:8]}...] ({ts[:10] if len(ts) >= 10 else ts}, {branch}) "
                    f"{summary[:200]}"
                )

    # Morning briefing (if available)
    briefing_result = await _call_command_center("GET", "/api/briefing/latest")
    if "error" not in briefing_result:
        briefing_text = briefing_result.get("briefing", "")
        briefing_run = briefing_result.get("last_run", "")
        if briefing_text:
            # Include a condensed version — first 800 chars
            lines.append(f"\n### Morning Briefing (from {briefing_run})")
            condensed = briefing_text[:800]
            if len(briefing_text) > 800:
                condensed += "\n... (use aiia_briefing for full text)"
            lines.append(condensed)

    session_id = result.get("session_id", "")
    if session_id:
        lines.append(f"\nSession ID: {session_id}")

    # Enrich with check-in data from Command Center
    try:
        checkin_data = await _call_command_center("GET", "/api/checkin")
        if "error" not in checkin_data:
            # Add action summary
            actions = checkin_data.get("actions", {})
            action_summary = actions.get("summary", {})
            pending_count = (
                action_summary.get("pending", 0)
                if isinstance(action_summary, dict)
                else 0
            )

            if pending_count > 0:
                lines.append(
                    f"\n### Pending Actions: {pending_count} items need review"
                )
                top_critical = actions.get("top_critical", [])
                for a in top_critical[:5]:
                    lines.append(
                        f"  - [{a.get('type', '?')}] {a.get('title', 'untitled')}"
                    )
                lines.append(
                    "  Use aiia_approve_action or aiia_reject_action to triage."
                )

            # Add blocked stories
            stories = checkin_data.get("stories", {})
            blocked = stories.get("blocked", [])
            if blocked:
                lines.append(f"\n### Blocked Stories: {len(blocked)}")
                for s in blocked:
                    lines.append(
                        f"  - {s.get('title', 'untitled')} ({s.get('product', '?')})"
                    )

            # Add nightly job status
            jobs = checkin_data.get("nightly_jobs", {})
            stale_jobs = [
                name
                for name, info in jobs.items()
                if isinstance(info, dict)
                and info.get("status") in ("stale", "missing")
            ]
            if stale_jobs:
                lines.append(
                    f"\n### Stale Nightly Jobs: {', '.join(stale_jobs)}"
                )
    except Exception:
        pass  # Check-in enrichment is optional

    return (
        "\n".join(lines)
        if len(lines) > 1
        else "No prior context found. Starting fresh."
    )


async def _session_start_fallback(
    task_description: str, files: list[str] | None
) -> str:
    """Fallback if /v1/aiia/session-start doesn't exist yet — make individual calls."""
    lines = ["## Session Context from AIIA\n"]

    # Get WIP items
    wip_result = await _call_aiia("GET", "/v1/aiia/memory?category=wip&limit=10")
    if "error" not in wip_result:
        wip = wip_result.get("memories", [])
        if wip:
            lines.append(f"### Work in Progress ({len(wip)} items)")
            for item in wip:
                lines.append(f"- {item.get('fact', '')}")
            lines.append("")

    # Get recent decisions
    dec_result = await _call_aiia("GET", "/v1/aiia/memory?category=decisions&limit=5")
    if "error" not in dec_result:
        decisions = dec_result.get("memories", [])
        if decisions:
            lines.append(f"### Recent Decisions ({len(decisions)})")
            for d in decisions[:5]:
                lines.append(f"- {d.get('fact', '')}")
            lines.append("")

    # Search knowledge for the task
    if task_description:
        search_result = await _call_aiia(
            "POST",
            "/v1/aiia/search",
            {
                "question": task_description,
                "n_results": 3,
            },
        )
        if "error" not in search_result:
            results = search_result.get("results", [])
            if results:
                lines.append(f"### Relevant Knowledge ({len(results)} matches)")
                for k in results[:3]:
                    source = k.get("source", "unknown")
                    text = k.get("text", "")[:200]
                    lines.append(f"- [{source}] {text}")
                lines.append("")

    # Morning briefing
    briefing_result = await _call_command_center("GET", "/api/briefing/latest")
    if "error" not in briefing_result:
        briefing_text = briefing_result.get("briefing", "")
        briefing_run = briefing_result.get("last_run", "")
        if briefing_text:
            lines.append(f"### Morning Briefing (from {briefing_run})")
            condensed = briefing_text[:800]
            if len(briefing_text) > 800:
                condensed += "\n... (use aiia_briefing for full text)"
            lines.append(condensed)
            lines.append("")

    return (
        "\n".join(lines)
        if len(lines) > 1
        else "No prior context found. Starting fresh."
    )


@mcp.tool()
async def aiia_what_was_i_doing() -> str:
    """Quick "catch me up" — asks AIIA to summarize recent work, decisions,
    WIP items, and open tasks. Use this when starting a new session and you
    need to remember where you left off.
    """
    lines = ["## What You Were Working On\n"]

    # Get WIP items
    wip_result = await _call_aiia("GET", "/v1/aiia/memory?category=wip&limit=10")
    if "error" not in wip_result:
        wip = wip_result.get("memories", [])
        if wip:
            lines.append(f"### Work in Progress ({len(wip)} items)")
            for item in wip:
                fact = item.get("fact", "")
                created = item.get("created_at", "")
                lines.append(f"- {fact}" + (f" ({created})" if created else ""))
            lines.append("")

    # Get recent sessions
    sess_result = await _call_aiia("GET", "/v1/aiia/memory?category=sessions&limit=5")
    if "error" not in sess_result:
        sessions = sess_result.get("memories", [])
        if sessions:
            lines.append(f"### Recent Sessions ({len(sessions)})")
            for s in sessions[:5]:
                lines.append(f"- {s.get('fact', '')}")
            lines.append("")

    # Get recent decisions
    dec_result = await _call_aiia("GET", "/v1/aiia/memory?category=decisions&limit=5")
    if "error" not in dec_result:
        decisions = dec_result.get("memories", [])
        if decisions:
            lines.append(f"### Recent Decisions ({len(decisions)})")
            for d in decisions[:5]:
                lines.append(f"- {d.get('fact', '')}")
            lines.append("")

    # Get recent lessons
    les_result = await _call_aiia("GET", "/v1/aiia/memory?category=lessons&limit=3")
    if "error" not in les_result:
        lessons = les_result.get("memories", [])
        if lessons:
            lines.append(f"### Recent Lessons ({len(lessons)})")
            for l in lessons[:3]:
                lines.append(f"- {l.get('fact', '')}")
            lines.append("")

    return (
        "\n".join(lines)
        if len(lines) > 1
        else "No recent work context found. This may be a fresh start."
    )


@mcp.tool()
async def aiia_ops_status() -> str:
    """Production health check via Command Center — shows status of all
    services (AIIA, Backend, Platform API, Ollama), latency,
    uptime, and recent status transitions.

    Use this to check if production is healthy before deploying or to
    diagnose issues.
    """
    result = await _call_command_center("GET", "/api/monitor")

    if "error" in result:
        return f"Command Center unreachable: {result['error']}"

    services = result.get("services", {})
    transitions = result.get("transitions", [])
    cycle_count = result.get("cycle_count", 0)

    lines = ["## Production Status\n"]

    for sid, svc in services.items():
        status = svc.get("status", "unknown")
        name = svc.get("name", sid)
        latency = svc.get("response_time_ms")
        avg_latency = svc.get("avg_response_time_ms")
        uptime = svc.get("uptime_pct")
        errors = svc.get("error_count", 0)

        icon = (
            "OK" if status == "online" else "WARN" if status == "degraded" else "DOWN"
        )
        latency_str = f"{latency:.0f}ms" if latency is not None else "--"
        avg_str = f"avg {avg_latency:.0f}ms" if avg_latency else ""
        uptime_str = f"{uptime:.1f}% uptime" if uptime is not None else ""

        parts = [f for f in [latency_str, avg_str, uptime_str] if f]
        lines.append(
            f"[{icon}] {name}: {status} ({', '.join(parts)})"
            + (f" [{errors} errors]" if errors else "")
        )

    if transitions:
        lines.append("\n### Recent Status Changes")
        for t in transitions[:5]:
            lines.append(
                f"- {t.get('service', '?')}: {t.get('from', '?')} -> {t.get('to', '?')} at {t.get('at', '?')}"
            )

    lines.append(f"\nMonitor cycles: {cycle_count}")

    return "\n".join(lines)


@mcp.tool()
async def aiia_save_wip(
    description: str,
    files: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> str:
    """Save work-in-progress state so the next session picks up where you
    left off. Call this before ending a session or switching tasks.

    The WIP is stored in AIIA's persistent memory under the "wip" category
    and will appear when you call aiia_session_start or aiia_what_was_i_doing.

    Args:
        description: What you're in the middle of doing
        files: Key files being modified (optional)
        next_steps: What needs to happen next (optional)
    """
    # Build a structured WIP fact
    parts = [description]
    if files:
        parts.append(f"Files: {', '.join(files[:10])}")
    if next_steps:
        parts.append("Next: " + "; ".join(next_steps[:5]))

    fact = " | ".join(parts)

    body = {
        "fact": fact,
        "category": "wip",
        "source": "claude-code",
    }
    result = await _call_aiia("POST", "/v1/aiia/remember", body)

    if "error" in result:
        return f"AIIA Error: {result['error']}"

    return f'WIP saved. AIIA will recall this in your next session.\n"{description}"'


# ─────────────────────────────────────────────────────────────
# Morning Briefing Tool
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_briefing(generate: bool = False) -> str:
    """Get AIIA's latest morning briefing — per-product commit summary,
    priority recommendations, platform health, and key numbers.

    The briefing runs automatically at 8am daily. Use generate=True to
    trigger a fresh one on demand.

    Use this at session start to see what shipped and what to focus on.

    Args:
        generate: If True, trigger a fresh briefing before fetching (default: False)
    """
    if generate:
        gen_result = await _call_command_center("POST", "/api/briefing/generate")
        if "error" in gen_result:
            return f"Failed to trigger briefing: {gen_result['error']}"
        # Wait briefly for the task to produce output
        import asyncio

        await asyncio.sleep(3)

    result = await _call_command_center("GET", "/api/briefing/latest")

    if "error" in result:
        return f"Briefing unavailable: {result['error']}"

    briefing = result.get("briefing", "")
    last_run = result.get("last_run", "never")
    status = result.get("status", "unknown")

    if not briefing:
        if generate:
            return (
                "Briefing triggered but not yet ready. "
                "The daily_brief task is running — try again in ~30 seconds."
            )
        return (
            "No briefing available yet. Use generate=True to create one, "
            "or wait for the next 8am auto-run."
        )

    header = f"## AIIA Morning Briefing\nGenerated: {last_run} | Status: {status}\n"
    return f"{header}\n{briefing}"


# ─────────────────────────────────────────────────────────────
# Voice Output Tool
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_speak(text: str, voice: str = "aiia") -> str:
    """AIIA speaks aloud on the Mac Mini via Google Cloud TTS. Non-blocking —
    returns immediately while speech plays. Falls back to macOS say if Google TTS unavailable.

    Use this to give voice feedback:
    - Read an insight aloud: aiia_speak("3 Dependabot alerts open")
    - Greet at session start: aiia_speak("Good morning. All services healthy.")
    - Read a summary: aiia_speak(brief_text)

    Args:
        text: The text for AIIA to speak
        voice: Google TTS preset (default: "aiia"). Others: "mia", or full Google voice name
    """
    result = await _call_aiia(
        "POST",
        "/v1/aiia/speak",
        {
            "text": text,
            "voice": voice,
        },
    )

    if "error" in result:
        return f"AIIA speak failed: {result['error']}"

    return f"AIIA is speaking ({len(text)} chars, voice: {voice})"


# ─────────────────────────────────────────────────────────────
# Supermemory Cloud Sync & Search Tools
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_sync_supermemory(
    categories: list[str] | None = None,
    limit_per_category: int = 50,
) -> str:
    """Push AIIA's local memories to Supermemory cloud backup. Safe to re-run —
    uses deterministic IDs so duplicates are skipped.

    Syncs decisions, lessons, patterns, sessions, and other memory categories
    to their matching Supermemory containers.

    Use this to:
    - Back up AIIA's local memories to the cloud
    - Ensure cloud and local are in sync after significant sessions
    - Backfill after AIIA learns many new things

    Args:
        categories: Which categories to sync (default: all). Options: decisions, lessons, patterns, sessions, team, project, agents, wip, meta
        limit_per_category: Max memories per category to sync (default: 50)
    """
    body: dict = {"limit_per_category": limit_per_category}
    if categories:
        body["categories"] = categories

    result = await _call_aiia("POST", "/v1/aiia/supermemory/sync", body)

    if "error" in result:
        return f"Sync failed: {result['error']}"

    total_synced = result.get("total_synced", 0)
    total_errors = result.get("total_errors", 0)
    by_cat = result.get("by_category", {})

    lines = [f"Synced {total_synced} memories to Supermemory cloud"]
    if total_errors:
        lines[0] += f" ({total_errors} errors)"

    for cat, stats in by_cat.items():
        synced = stats.get("synced", 0)
        total = stats.get("total", 0)
        if total > 0:
            lines.append(f"  {cat}: {synced}/{total} synced")

    return "\n".join(lines)


@mcp.tool()
async def aiia_search_supermemory(
    query: str,
    search_type: str = "sme",
    domains: list[str] | None = None,
    tenant_id: str = "default",
    limit: int = 5,
) -> str:
    """Search Supermemory cloud — either SME domain knowledge vaults or
    AIIA's own cloud-backed memories.

    search_type="sme": Search configured SME containers for domain expertise.

    search_type="aiia": Search AIIA's cloud-synced memories (decisions,
    lessons, patterns). Useful for finding past knowledge that may not
    be in local search.

    Use this for:
    - "What does the Rule of Thirds say?" (search_type="sme", domains=["finance"])
    - "Find knowledge about damages" (search_type="sme", domains=["legal"])
    - "What decisions did we make about auth?" (search_type="aiia")

    Args:
        query: What to search for
        search_type: "sme" for domain knowledge, "aiia" for AIIA's memories
        domains: SME domains to search (for sme type). Configure via AIIA_SME_CONFIG.
        tenant_id: Tenant for SME search (default: default)
        limit: Max results (default: 5)
    """
    body: dict = {
        "query": query,
        "search_type": search_type,
        "tenant_id": tenant_id,
        "limit": limit,
    }
    if domains:
        body["domains"] = domains

    result = await _call_aiia("POST", "/v1/aiia/supermemory/search", body)

    if "error" in result:
        return f"Search failed: {result['error']}"

    results = result.get("results", [])
    if not results:
        return f"No results found for '{query}' (search_type={search_type})"

    count = result.get("count", len(results))
    lines = [f"Found {count} results (search_type={search_type}):\n"]

    for i, r in enumerate(results, 1):
        content = r.get("content", "")[:400]
        score = r.get("score", 0)
        # Show domain for SME, category for AIIA
        label = r.get("domain", r.get("category", ""))
        lines.append(f"[{i}] {label} (score: {score:.3f})\n{content}\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Token & Cost Tracking Tool
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_tokens_today() -> str:
    """Query today's token usage and costs from the Command Center. Shows
    breakdown by provider (local/anthropic/google), total tokens, total
    cost, and request count.

    Use this to monitor spending or check how much local (free) processing
    is being used vs paid API calls.
    """
    result = await _call_command_center("GET", "/api/tokens/today")

    if "error" in result:
        return f"Token data unavailable: {result['error']}"

    date = result.get("date", "today")
    total_tokens = result.get("total_tokens", 0)
    total_cost = result.get("total_cost", 0)
    total_requests = result.get("total_requests", 0)
    providers = result.get("by_provider", {})

    lines = [f"## Token Usage — {date}\n"]
    lines.append(
        f"Total: {total_tokens:,} tokens | ${total_cost:.4f} | {total_requests} requests\n"
    )

    if providers:
        lines.append("### By Provider")
        for provider, stats in providers.items():
            tokens = stats.get("tokens", 0)
            cost = stats.get("cost", 0)
            reqs = stats.get("requests", 0)
            cost_str = "FREE" if provider == "local" else f"${cost:.4f}"
            lines.append(
                f"- {provider}: {tokens:,} tokens | {cost_str} | {reqs} requests"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Action Queue Tools — Dev Loop Orchestration
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_get_actions(status: str = "pending", limit: int = 10) -> str:
    """List action items from AIIA's automated tasks. Actions are created when
    AIIA detects issues (syntax errors, test failures, security CVEs, risky
    code patterns) during scheduled scans.

    Use this to see what needs attention. Pending actions await Eric's approval
    before anyone acts on them.

    Args:
        status: Filter by status — "pending" (default), "approved", "rejected", "completed", "expired"
        limit: Max results (default: 10)
    """
    result = await _call_command_center(
        "GET", f"/api/actions?status={status}&limit={limit}"
    )

    if "error" in result:
        return f"Action queue unavailable: {result['error']}"

    actions = result.get("actions", [])
    summary = result.get("summary", {})

    if not actions:
        pending = summary.get("by_status", {}).get("pending", 0)
        if pending:
            return f"No {status} actions found, but {pending} pending actions exist."
        return f"No {status} actions in the queue."

    lines = [f"### Action Items ({len(actions)} {status})\n"]

    for a in actions:
        sev = a.get("severity", "info").upper()
        atype = a.get("type", "unknown")
        title = a.get("title", "")
        aid = a.get("id", "?")
        desc = a.get("description", "")
        files = a.get("files_affected", [])

        lines.append(f"**[{sev}]** `{atype}` — {title}")
        lines.append(f"  ID: `{aid}`")
        if desc:
            lines.append(f"  {desc[:200]}")
        if files:
            lines.append(f"  Files: {', '.join(files[:5])}")
        lines.append("")

    # Summary footer
    by_sev = summary.get("pending_by_severity", {})
    if by_sev:
        sev_parts = [f"{v} {k}" for k, v in by_sev.items() if v > 0]
        lines.append(f"Pending: {', '.join(sev_parts)}")

    return "\n".join(lines)


@mcp.tool()
async def aiia_approve_action(action_id: str) -> str:
    """Approve a pending action item for execution. After approval, Claude Code
    can proceed to fix the issue.

    Only call this after Eric has reviewed the action and said to fix it.

    Args:
        action_id: The action ID (from aiia_get_actions output)
    """
    result = await _call_command_center("POST", f"/api/actions/{action_id}/approve")

    if "error" in result:
        return f"Approve failed: {result['error']}"

    title = result.get("title", "?")
    return f"Action `{action_id}` approved: {title}\nYou can now proceed to fix this issue."


@mcp.tool()
async def aiia_reject_action(action_id: str, reason: str = "") -> str:
    """Reject a pending action item (won't fix, not relevant, already handled).

    Args:
        action_id: The action ID (from aiia_get_actions output)
        reason: Why this action was rejected (optional)
    """
    result = await _call_command_center(
        "POST",
        f"/api/actions/{action_id}/reject",
        {"reason": reason},
    )

    if "error" in result:
        return f"Reject failed: {result['error']}"

    title = result.get("title", "?")
    return f"Action `{action_id}` rejected: {title}" + (
        f" (reason: {reason})" if reason else ""
    )


@mcp.tool()
async def aiia_complete_action(action_id: str, result_text: str = "") -> str:
    """Mark an approved action as completed after the fix has been applied.

    Call this after you've made the fix so AIIA tracks it as resolved.

    Args:
        action_id: The action ID
        result_text: Brief description of what was done (optional)
    """
    result = await _call_command_center(
        "POST",
        f"/api/actions/{action_id}/complete",
        {"result": result_text},
    )

    if "error" in result:
        return f"Complete failed: {result['error']}"

    title = result.get("title", "?")
    return f"Action `{action_id}` completed: {title}"


# ─────────────────────────────────────────────────────────────
# Session History Search
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_search_history(
    query: str,
    project: str = "",
    domain: str = "",
    limit: int = 5,
) -> str:
    """Search all indexed Claude Code session transcripts. Use this to find
    past work sessions by topic, file, or domain.

    Examples:
        - "When did we last work on the conductor?"
        - "Find sessions where we modified tenants.yaml"
        - "What sessions involved security scanning?"

    Args:
        query: Natural language search query
        project: Filter by project path substring (optional)
        domain: Filter by domain: platform, backend, marketing, local-brain, etc. (optional)
        limit: Max results (1-20, default 5)
    """
    result = await _call_aiia(
        "POST",
        "/v1/aiia/search-sessions",
        {"query": query, "project": project, "domain": domain, "limit": limit},
    )

    if "error" in result:
        return f"Search failed: {result['error']}"

    sessions = result.get("sessions", [])
    if not sessions:
        return "No matching sessions found."

    lines = [f"Found {len(sessions)} matching session(s):\n"]
    for s in sessions:
        sid = s.get("session_id", "?")
        summary = s.get("summary", "no summary")
        branch = s.get("branch", "?")
        domain_val = s.get("domain", "?")
        ts = s.get("start_timestamp", "?")
        dur = s.get("duration_seconds", 0)
        dur_str = f"{dur / 60:.0f}min" if dur else "?"
        files = s.get("files_count", 0)

        lines.append(f"### {sid[:8]}... ({ts[:10] if len(ts) >= 10 else ts})")
        lines.append(
            f"  Branch: {branch} | Domain: {domain_val} | Duration: {dur_str} | Files: {files}"
        )
        lines.append(f"  {summary[:300]}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def aiia_find_solutions(
    error_or_problem: str,
    limit: int = 5,
) -> str:
    """Search past sessions AND AIIA lessons memory for solutions to a
    specific error or problem. Checks both indexed session history and
    structured memory for past fixes.

    Examples:
        - "ECONNREFUSED error on port 8100"
        - "ChromaDB collection not found"
        - "Render deploy failing with memory error"

    Args:
        error_or_problem: Description of the error or problem
        limit: Max results (default 5)
    """
    lines = ["## Past Solutions Search\n"]

    # Search AIIA lessons memory
    memory_result = await _call_aiia(
        "POST",
        "/v1/aiia/search",
        {"question": error_or_problem, "n_results": limit},
    )

    if "error" not in memory_result:
        results = memory_result.get("results", [])
        lesson_matches = [
            r
            for r in results
            if "lesson" in r.get("source", "").lower()
            or "fix" in r.get("content", "").lower()
            or "solution" in r.get("content", "").lower()
            or "problem" in r.get("content", "").lower()
        ]
        if lesson_matches:
            lines.append(f"### Knowledge Base ({len(lesson_matches)} matches)")
            for r in lesson_matches[:3]:
                lines.append(f"- {r.get('content', '')[:300]}")
                lines.append(f"  (source: {r.get('source', '?')})")
            lines.append("")

    # Search AIIA structured memory (lessons category)
    mem_result = await _call_aiia("GET", "/v1/aiia/memory?category=lessons")
    if "error" not in mem_result:
        memories = mem_result.get("memories", [])
        query_lower = error_or_problem.lower()
        matched = [
            m
            for m in memories
            if any(
                word in m.get("fact", "").lower()
                for word in query_lower.split()
                if len(word) > 3
            )
        ]
        if matched:
            lines.append(f"### AIIA Lessons ({len(matched)} matches)")
            for m in matched[:5]:
                lines.append(f"- {m.get('fact', '')[:300]}")
                src = m.get("source", "")
                if src:
                    lines.append(f"  (source: {src})")
            lines.append("")

    # Search session history
    session_result = await _call_aiia(
        "POST",
        "/v1/aiia/search-sessions",
        {"query": error_or_problem, "limit": limit},
    )

    if "error" not in session_result:
        sessions = session_result.get("sessions", [])
        if sessions:
            lines.append(f"### Related Sessions ({len(sessions)} matches)")
            for s in sessions[:3]:
                sid = s.get("session_id", "?")
                summary = s.get("summary", "")
                ts = s.get("start_timestamp", "?")
                lines.append(
                    f"- [{sid[:8]}...] ({ts[:10] if len(ts) >= 10 else ts}) {summary[:200]}"
                )
            lines.append("")

    if len(lines) <= 2:
        return "No past solutions found for this problem."

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Execution Engine Tools
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_execution_status() -> str:
    """Check execution engine status — what's running, recent results,
    kill switch state."""
    result = await _call_command_center("GET", "/api/execution/status")

    if "error" in result:
        return f"Execution engine unavailable: {result['error']}"

    if not result.get("enabled", True):
        return "Execution engine is not enabled."

    lines = ["### Execution Engine Status\n"]
    lines.append(f"- Kill switch: {'ACTIVE' if result.get('killed') else 'off'}")
    lines.append(f"- Running: {result.get('running_count', 0)}")

    running = result.get("running", [])
    for r in running:
        lines.append(f"  - `{r.get('action_id', '?')}` {r.get('tier', '?')}")

    recent = result.get("recent", [])
    if recent:
        lines.append(f"\nRecent ({len(recent)}):")
        for rec in recent[:5]:
            status = rec.get("status", "?")
            aid = rec.get("action_id", "?")
            dur = rec.get("duration_s", "?")
            lines.append(f"  - `{aid}` {status} ({dur}s)")

    return "\n".join(lines)


@mcp.tool()
async def aiia_execute_action(action_id: str) -> str:
    """Explicitly trigger execution of an approved action. Use for GATED-tier
    actions that need manual trigger.

    Args:
        action_id: The action ID to execute (from aiia_get_actions output)
    """
    result = await _call_command_center(
        "POST", f"/api/execution/execute/{action_id}"
    )

    if "error" in result:
        return f"Execute failed: {result['error']}"

    return f"Execution triggered for `{action_id}`: {result}"


@mcp.tool()
async def aiia_kill_execution() -> str:
    """Emergency stop — kill all running executions immediately."""
    result = await _call_command_center("POST", "/api/execution/kill")

    if "error" in result:
        return f"Kill failed: {result['error']}"

    return "Execution engine killed. All running processes terminated."


@mcp.tool()
async def aiia_execution_log(limit: int = 10) -> str:
    """View recent execution history — successes, failures, durations.

    Args:
        limit: Max records to return (default: 10)
    """
    result = await _call_command_center(
        "GET", f"/api/execution/log?limit={limit}"
    )

    if "error" in result:
        return f"Execution log unavailable: {result['error']}"

    records = result.get("records", [])
    if not records:
        return "No execution records yet."

    lines = [f"### Execution Log ({len(records)} records)\n"]
    for rec in records:
        aid = rec.get("action_id", "?")
        status = rec.get("status", "?")
        tier = rec.get("tier", "?")
        dur = rec.get("duration_s", "?")
        ts = rec.get("completed_at", rec.get("started_at", "?"))
        if isinstance(ts, str) and len(ts) >= 19:
            ts = ts[:19]
        lines.append(f"- `{aid}` [{tier}] {status} ({dur}s) @ {ts}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Story Execution
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_execute_story(
    story_id: str, auto_approve: bool = False
) -> str:
    """Decompose a kanban story into executable actions and start execution.

    Takes an active/backlog story, uses local LLM to break it into action steps,
    creates actions in the queue, and optionally auto-approves them.

    Args:
        story_id: The roadmap story ID to execute
        auto_approve: If True, auto-approve all created actions for immediate execution
    """
    result = await _call_command_center(
        "POST",
        f"/api/execution/story/{story_id}",
        json={"auto_approve": auto_approve},
    )

    if "error" in result:
        return f"Story execution failed: {result['error']}"

    status = result.get("status", "unknown")
    if status == "error":
        return f"Story execution error: {result.get('reason', 'unknown')}"

    steps = result.get("steps", 0)
    action_ids = result.get("action_ids", [])
    approved = result.get("auto_approved", False)

    lines = [
        f"### Story Decomposed",
        f"- **Story:** {story_id}",
        f"- **Steps:** {steps}",
        f"- **Auto-approved:** {approved}",
        f"- **Actions:** {', '.join(action_ids)}",
        "",
        "Story moved to `in_progress`. "
        + (
            "Actions are executing."
            if approved
            else "Use `aiia_approve_action` to approve each step."
        ),
    ]
    return "\n".join(lines)


@mcp.tool()
async def aiia_story_progress(story_id: str) -> str:
    """Check execution progress of a story — how many actions completed, failed, pending.

    Args:
        story_id: The roadmap story ID to check
    """
    result = await _call_command_center(
        "GET", f"/api/execution/story/{story_id}/progress"
    )

    if "error" in result:
        return f"Progress unavailable: {result['error']}"

    story = result.get("story", {})
    progress = result.get("progress", 0)
    total = result.get("total", 0)
    completed = result.get("completed", 0)
    failed = result.get("failed", 0)
    pending = result.get("pending", 0)

    lines = [
        f"### Story Progress: {story.get('title', story_id)}",
        f"- **Status:** {story.get('status', '?')}",
        f"- **Progress:** {progress}% ({completed}/{total} complete)",
        f"- **Pending:** {pending}",
        f"- **Failed:** {failed}",
    ]

    actions = result.get("actions", [])
    if actions:
        lines.append("\n**Steps:**")
        for a in actions:
            step = a.get("step", "?")
            icon = {"completed": "+", "failed": "x", "executing": "~"}.get(
                a["status"], " "
            )
            lines.append(
                f"  {step}. [{icon}] {a.get('title', '?')} ({a['status']})"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Quick Capture
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def aiia_log_story(
    title: str,
    product: str = "platform",
    priority: str = "P2",
    description: str = "",
    tags: list[str] | None = None,
    client_impact: str = "",
) -> str:
    """Log a new story or idea to the roadmap backlog. Use this to capture
    feature ideas, bugs, or tasks that come up during work sessions.

    Automatically deduplicates against existing backlog stories — if a similar
    story already exists, it will be returned instead of creating a duplicate.

    Args:
        title: Short title for the story/idea
        product: Product this belongs to (default: "platform")
        priority: Priority level — P0 (critical) through P3 (low). Default: P2
        description: Optional longer description or context
        tags: Optional tags — feature, bug, tech-debt, integration, ux, security, performance, devops
        client_impact: Optional — who benefits and why
    """
    body = {
        "title": title,
        "product": product,
        "priority": priority,
        "status": "backlog",
        "description": description,
        "tags": tags or [],
        "client_impact": client_impact,
        "source_type": "manual",
    }

    result = await _call_command_center("POST", "/api/roadmap", body)

    if "error" in result:
        return f"Failed to log story: {result['error']}"

    story = result.get("story", result)
    story_id = story.get("id", "?")
    deduped = story.get("_deduped", False)

    if deduped:
        return (
            f"Story already exists: '{story.get('title', title)}' "
            f"[{story.get('status', '?')}] ({product}) [id: {story_id}]"
        )

    return f"Story logged: {title} -> backlog ({product}) [id: {story_id}]"


@mcp.tool()
async def aiia_prioritize_backlog(limit: int = 10) -> str:
    """Score and rank backlog stories using a 5-filter priority framework.

    Evaluates each backlog story against:
    1. Does it close a deal? (weight 5)
    2. Does it retain key clients? (weight 4)
    3. Does it reduce cost? (weight 3)
    4. Does it enable multiple tenants? (weight 2)
    5. Does it create a new revenue stream? (weight 1)

    Returns the top stories ranked by weighted score with priority suggestions.

    Args:
        limit: Max number of stories to score (default: 10)
    """
    result = await _call_command_center(
        "POST", "/api/roadmap/prioritize", {"limit": limit}
    )

    if "error" in result:
        return f"Prioritization failed: {result['error']}"

    stories = result.get("stories", [])
    if not stories:
        return "No backlog stories to prioritize."

    lines = ["### Backlog Priority Ranking", ""]
    for i, s in enumerate(stories, 1):
        score = s.get("priority_score", 0)
        suggested = s.get("suggested_priority", "?")
        current = s.get("priority", "?")
        reasoning = s.get("priority_reasoning", "")
        title = s.get("title", "untitled")
        product = s.get("product", "?")

        priority_change = ""
        if suggested != current:
            priority_change = f" (suggest: {suggested})"

        lines.append(
            f"{i}. **{title}** [{current}{priority_change}] — score {score}"
        )
        lines.append(f"   {product} | {reasoning}")

        # Show filter breakdown
        scores = s.get("filter_scores", {})
        if scores:
            parts = []
            for key in ("closes_deal", "retains_client", "reduces_cost", "enables_tenants", "new_revenue"):
                val = scores.get(key, 0)
                if val > 0:
                    short = key.replace("_", " ").title()
                    parts.append(f"{short}:{val}")
            if parts:
                lines.append(f"   Filters: {', '.join(parts)}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
