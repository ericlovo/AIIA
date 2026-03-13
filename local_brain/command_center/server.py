"""
AIIA Command Center — Platform Intelligence Dashboard

Real-time visualization of the AIIA platform.
Shows products, agent workflows, AIIA intelligence, and system connections.
Includes autonomous Production Monitor that checks all services every 30s.

Start:
    python -m local_brain.command_center.server
Or:
    uvicorn local_brain.command_center.server:app --port 8200
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("aiia.command_center")

# ─────────────────────────────────────────────────────────────
# Platform Registry — defines the full system topology
# Configure via AIIA_PRODUCTS_CONFIG env var pointing to a JSON file,
# or edit the PRODUCTS list below to match your deployment.
# ─────────────────────────────────────────────────────────────

def _load_products() -> list:
    """Load product registry from env-configured JSON file, or use defaults.

    Set AIIA_PRODUCTS_CONFIG to a JSON file path to override.
    Each product entry should have: id, name, subtitle, client, status, color,
    agents (list), components (list), knowledge_files (int), knowledge_words (str),
    and optionally frontend/backend URLs.
    """
    config_path = os.getenv("AIIA_PRODUCTS_CONFIG")
    if config_path and Path(config_path).exists():
        import json
        with open(config_path) as f:
            return json.load(f)
    return _DEFAULT_PRODUCTS


_DEFAULT_PRODUCTS = [
    {
        "id": "my_app",
        "name": "My App",
        "subtitle": "AI-powered application",
        "client": "Example Client",
        "status": "production",
        "color": "#3b82f6",
        "agents": ["conductor", "eos", "rlm", "finance", "advisor"],
        "components": ["Document Parsing", "Report Generation"],
        "knowledge_files": 5,
        "knowledge_words": "50K",
        "frontend": os.getenv("APP_FRONTEND_URL", "http://localhost:3000"),
        "backend": os.getenv("APP_BACKEND_URL", "http://localhost:9000"),
    },
    {
        "id": "demo",
        "name": "Demo",
        "subtitle": "Development & Testing",
        "client": "Internal",
        "status": "active",
        "color": "#475569",
        "agents": [],
        "components": [],
        "knowledge_files": 0,
        "knowledge_words": "0",
    },
    # Add your products here. Each entry represents a tenant/application
    # that AIIA monitors. See docs/PRODUCTS.md for the full schema.
]

PRODUCTS = _load_products()

# Future products (not market-facing yet)
FUTURE_PRODUCTS = [
    {
        "id": "guru",
        "name": "GURU",
        "subtitle": "Guided Self-Mastery — BE, DO, FAITH",
        "client": "AC Ping",
        "status": "future",
        "color": "#8b5cf6",
        "note": "Eric = user #1, co-designed",
    },
    {
        "id": "liz",
        "name": "Liz",
        "subtitle": "Health Coaching with Liz Josefsberg",
        "client": "ReaLIZ",
        "status": "uncertain",
        "color": "#10b981",
    },
    {
        "id": "estate",
        "name": "Estate Planner",
        "subtitle": "Estate Planning AI",
        "client": "Pipeline",
        "status": "pipeline",
        "color": "#f59e0b",
    },
]

AGENTS = [
    {
        "id": "conductor",
        "name": "Conductor",
        "type": "router",
        "color": "#3b82f6",
        "detail": "EQ + Thomson + Complexity",
    },
    {
        "id": "ping_orchestrator",
        "name": "PingOrchestrator",
        "type": "router",
        "color": "#a855f7",
        "detail": "Ethics routing + pattern detection",
    },
    {
        "id": "eos",
        "name": "Traction EOS",
        "type": "chat",
        "color": "#3b82f6",
        "detail": "Single-shot fast path",
    },
    {
        "id": "rlm",
        "name": "RLM Engine",
        "type": "reasoning",
        "color": "#a855f7",
        "detail": "Recursive agentic reasoning",
    },
    {
        "id": "mia",
        "name": "MIA",
        "type": "ethics",
        "color": "#c084fc",
        "detail": "Moral Intention Analyst — 19 frameworks",
    },
    {
        "id": "guru",
        "name": "GURU",
        "type": "development",
        "color": "#10b981",
        "detail": "Personal Development Guide",
    },
    {
        "id": "finance",
        "name": "Finance Analyst",
        "type": "specialist",
        "color": "#f59e0b",
        "detail": "Rule of Thirds, P&L",
    },
    {
        "id": "legal",
        "name": "Legal Analyst",
        "type": "specialist",
        "color": "#64d2ff",
        "detail": "Document analysis",
    },
    {
        "id": "estate_advisor",
        "name": "Estate Advisor",
        "type": "specialist",
        "color": "#f59e0b",
        "detail": "Trust & estate guidance",
    },
]

INFRASTRUCTURE = [
    {
        "id": "claude",
        "name": "Claude API",
        "type": "llm",
        "color": "#f59e0b",
        "detail": "claude-sonnet-4 (primary)",
    },
    {
        "id": "gemini",
        "name": "Gemini",
        "type": "llm",
        "color": "#10b981",
        "detail": "gemini-1.5-pro (fallback)",
    },
    {
        "id": "aiia",
        "name": "AIIA",
        "type": "intelligence",
        "color": "#a855f7",
        "detail": "llama3.1:8b on Mac Mini M4",
    },
    {
        "id": "chromadb",
        "name": "ChromaDB",
        "type": "database",
        "color": "#64d2ff",
        "detail": "Vector search",
    },
    {
        "id": "postgresql",
        "name": "PostgreSQL",
        "type": "database",
        "color": "#3b82f6",
        "detail": "Application data",
    },
]

EDGES = [
    # Products → Routers
    {"from": "analyst", "to": "conductor", "type": "chat"},
    {"from": "mia", "to": "ping_orchestrator", "type": "chat"},
    {"from": "codeword", "to": "conductor", "type": "chat"},
    {"from": "psg", "to": "conductor", "type": "chat"},
    {"from": "ck_marketing", "to": "conductor", "type": "chat"},
    {"from": "lns", "to": "conductor", "type": "chat"},
    # Routers → Agents
    {"from": "conductor", "to": "eos", "type": "routing", "label": "< 0.6"},
    {"from": "conductor", "to": "rlm", "type": "routing", "label": ">= 0.6"},
    {"from": "ping_orchestrator", "to": "mia", "type": "routing", "label": "ethics"},
    {
        "from": "ping_orchestrator",
        "to": "guru",
        "type": "routing",
        "label": "self-mastery",
    },
    # Agents → LLMs
    {"from": "eos", "to": "claude", "type": "inference"},
    {"from": "rlm", "to": "claude", "type": "inference"},
    {"from": "mia", "to": "claude", "type": "inference"},
    {"from": "guru", "to": "claude", "type": "inference"},
    # Data connections
    {"from": "rlm", "to": "chromadb", "type": "data"},
    {"from": "aiia", "to": "chromadb", "type": "data"},
    {"from": "conductor", "to": "aiia", "type": "intelligence"},
]


# ─────────────────────────────────────────────────────────────
# Production Monitor — Autonomous Service Health Tracking
# ─────────────────────────────────────────────────────────────

MONITOR_INTERVAL = 30  # seconds between checks
MONITOR_DATA_FILE = Path(__file__).parent / "monitor_data.json"
MAX_HISTORY = 2880  # 24h at 30s intervals

MONITORED_SERVICES = {
    "aiia": {
        "name": "AIIA Local Brain",
        "url": "http://localhost:8100/v1/aiia/status",
        "timeout": 5.0,
        "category": "intelligence",
    },
    "backend": {
        "name": "Primary Backend",
        "url": os.getenv("PRIMARY_BACKEND_URL", "http://localhost:9000") + "/health",
        "metrics_url": os.getenv("PRIMARY_BACKEND_URL", "http://localhost:9000")
        + "/metrics",
        "timeout": 10.0,
        "category": "backend",
    },
    "platform": {
        "name": "Platform API",
        "url": os.getenv("PLATFORM_API_URL", "http://localhost:8000") + "/health",
        "timeout": 10.0,
        "category": "backend",
    },
    "ollama": {
        "name": "Ollama",
        "url": "http://localhost:11434/api/tags",
        "timeout": 3.0,
        "category": "local",
    },
}


@dataclass
class ServiceHealth:
    service_id: str
    status: str  # "online", "degraded", "offline"
    response_time_ms: float
    checked_at: str
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class MonitorState:
    def __init__(self):
        self.response_times: Dict[str, deque] = {
            sid: deque(maxlen=MAX_HISTORY) for sid in MONITORED_SERVICES
        }
        self.statuses: Dict[str, str] = {sid: "unknown" for sid in MONITORED_SERVICES}
        self.error_counts: Dict[str, int] = {sid: 0 for sid in MONITORED_SERVICES}
        self.consecutive_up: Dict[str, int] = {sid: 0 for sid in MONITORED_SERVICES}
        self.total_checks: Dict[str, int] = {sid: 0 for sid in MONITORED_SERVICES}
        self.online_checks: Dict[str, int] = {sid: 0 for sid in MONITORED_SERVICES}
        self.transitions: deque = deque(maxlen=50)
        self.meta: Dict[str, Dict[str, Any]] = {sid: {} for sid in MONITORED_SERVICES}
        self.cycle_count: int = 0

    def record(self, health: ServiceHealth):
        sid = health.service_id
        self.total_checks[sid] += 1

        # Record response time
        self.response_times[sid].append(
            {
                "ms": health.response_time_ms,
                "ts": health.checked_at,
                "ok": health.status == "online",
            }
        )

        # Track uptime
        if health.status == "online":
            self.online_checks[sid] += 1
            self.consecutive_up[sid] += 1
        else:
            self.consecutive_up[sid] = 0
            self.error_counts[sid] += 1

        # Detect transitions
        old_status = self.statuses[sid]
        if old_status != health.status and old_status != "unknown":
            self.transitions.appendleft(
                {
                    "service": MONITORED_SERVICES[sid]["name"],
                    "from": old_status,
                    "to": health.status,
                    "at": health.checked_at,
                }
            )
        self.statuses[sid] = health.status

        # Store extra metadata
        if health.meta:
            self.meta[sid] = health.meta

    def get_service_snapshot(self, sid: str) -> Dict[str, Any]:
        cfg = MONITORED_SERVICES[sid]
        times = list(self.response_times[sid])
        recent = times[-40:] if times else []
        total = self.total_checks[sid]
        online = self.online_checks[sid]

        avg_ms = 0.0
        if times:
            ok_times = [t["ms"] for t in times if t["ok"]]
            if ok_times:
                avg_ms = round(sum(ok_times) / len(ok_times), 1)

        return {
            "id": sid,
            "name": cfg["name"],
            "category": cfg["category"],
            "status": self.statuses[sid],
            "response_time_ms": times[-1]["ms"] if times else None,
            "avg_response_time_ms": avg_ms,
            "uptime_pct": round((online / total) * 100, 1) if total > 0 else None,
            "total_checks": total,
            "error_count": self.error_counts[sid],
            "consecutive_up": self.consecutive_up[sid],
            "sparkline": [{"ms": t["ms"], "ok": t["ok"]} for t in recent],
            "meta": self.meta.get(sid, {}),
        }

    def get_full_snapshot(self) -> Dict[str, Any]:
        return {
            "services": {
                sid: self.get_service_snapshot(sid) for sid in MONITORED_SERVICES
            },
            "transitions": list(self.transitions),
            "cycle_count": self.cycle_count,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def to_persist(self) -> Dict[str, Any]:
        return {
            "response_times": {
                sid: list(dq) for sid, dq in self.response_times.items()
            },
            "statuses": self.statuses,
            "error_counts": self.error_counts,
            "consecutive_up": self.consecutive_up,
            "total_checks": self.total_checks,
            "online_checks": self.online_checks,
            "transitions": list(self.transitions),
            "meta": self.meta,
            "cycle_count": self.cycle_count,
        }

    def load_persisted(self, data: Dict[str, Any]):
        for sid in MONITORED_SERVICES:
            if sid in data.get("response_times", {}):
                for entry in data["response_times"][sid]:
                    self.response_times[sid].append(entry)
            self.statuses[sid] = data.get("statuses", {}).get(sid, "unknown")
            self.error_counts[sid] = data.get("error_counts", {}).get(sid, 0)
            self.consecutive_up[sid] = data.get("consecutive_up", {}).get(sid, 0)
            self.total_checks[sid] = data.get("total_checks", {}).get(sid, 0)
            self.online_checks[sid] = data.get("online_checks", {}).get(sid, 0)
            self.meta[sid] = data.get("meta", {}).get(sid, {})
        for t in reversed(data.get("transitions", [])):
            self.transitions.appendleft(t)
        self.cycle_count = data.get("cycle_count", 0)


# ─────────────────────────────────────────────────────────────
# Platform State
# ─────────────────────────────────────────────────────────────


class PlatformState:
    """Central state for the product console."""

    def __init__(self):
        self.start_time = time.time()
        self.health: Dict[str, Any] = {}
        self.aiia_status: Dict[str, Any] = {}

    def get_platform(self):
        return {
            "products": PRODUCTS,
            "future": FUTURE_PRODUCTS,
            "agents": AGENTS,
            "infrastructure": INFRASTRUCTURE,
            "edges": EDGES,
        }

    def get_summary(self):
        return {
            "uptime": int(time.time() - self.start_time),
            "health": self.health,
            "aiia": self.aiia_status,
        }


# ─────────────────────────────────────────────────────────────
# WebSocket Manager
# ─────────────────────────────────────────────────────────────


class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, event_type: str, data: Any):
        if not self.connections:
            return
        msg = json.dumps({"type": event_type, "data": data})
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.connections:
                self.connections.remove(ws)


# ─────────────────────────────────────────────────────────────
# Service Health Checker
# ─────────────────────────────────────────────────────────────


async def check_service(client: httpx.AsyncClient, service_id: str) -> ServiceHealth:
    """Check a single service. Never raises — always returns a ServiceHealth."""
    cfg = MONITORED_SERVICES[service_id]
    now = datetime.now(timezone.utc).isoformat()

    try:
        start = time.monotonic()
        resp = await client.get(cfg["url"], timeout=cfg["timeout"])
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        if resp.status_code == 200:
            meta = {}
            try:
                body = resp.json()
                # Extract useful metadata per service
                if service_id == "aiia":
                    knowledge = body.get("knowledge", {})
                    memory = body.get("memory", {})
                    meta = {
                        "docs": knowledge.get("knowledge_docs", 0),
                        "memories": memory.get("total_memories", 0),
                        "model": body.get("model", "unknown"),
                    }
                    # Also update the AIIA panel state
                    state.aiia_status = body
                elif service_id == "default":
                    meta = {
                        "status": body.get("status", "unknown"),
                        "database": body.get("database", "unknown"),
                    }
                elif service_id == "ollama":
                    models = body.get("models", [])
                    meta = {
                        "models": [m.get("name", "?") for m in models[:5]],
                        "model_count": len(models),
                    }
            except Exception:
                pass

            # Fetch memory metrics if service exposes /metrics
            metrics_url = cfg.get("metrics_url")
            if metrics_url:
                try:
                    mr = await client.get(metrics_url, timeout=5.0)
                    if mr.status_code == 200:
                        mdata = mr.json()
                        mem = mdata.get("memory", {})
                        if mem:
                            meta["rss_mb"] = mem.get("rss_mb", 0)
                            meta["rss_gb"] = mem.get("rss_gb", 0)
                        uptime = mdata.get("uptime", {})
                        if uptime:
                            meta["uptime"] = uptime.get("formatted", "")
                except Exception:
                    pass

            status = "degraded" if elapsed_ms > 5000 else "online"
            return ServiceHealth(
                service_id=service_id,
                status=status,
                response_time_ms=elapsed_ms,
                checked_at=now,
                meta=meta,
            )
        else:
            return ServiceHealth(
                service_id=service_id,
                status="offline",
                response_time_ms=round((time.monotonic() - start) * 1000, 1),
                checked_at=now,
                error=f"HTTP {resp.status_code}",
            )
    except httpx.TimeoutException:
        return ServiceHealth(
            service_id=service_id,
            status="offline",
            response_time_ms=cfg["timeout"] * 1000,
            checked_at=now,
            error="timeout",
        )
    except Exception as e:
        return ServiceHealth(
            service_id=service_id,
            status="offline",
            response_time_ms=0,
            checked_at=now,
            error=str(e)[:100],
        )


async def production_monitor_loop():
    """Autonomous monitor — checks all services every MONITOR_INTERVAL seconds."""
    await asyncio.sleep(3)  # let server finish startup
    logger.info(
        f"Production Monitor started — checking {len(MONITORED_SERVICES)} services every {MONITOR_INTERVAL}s"
    )

    while True:
        try:
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    *[check_service(client, sid) for sid in MONITORED_SERVICES],
                    return_exceptions=True,
                )

            for result in results:
                if isinstance(result, ServiceHealth):
                    monitor.record(result)
                elif isinstance(result, Exception):
                    logger.error(f"Monitor check exception: {result}")

            monitor.cycle_count += 1

            # Broadcast to all connected WebSocket clients
            snapshot = monitor.get_full_snapshot()
            await manager.broadcast("monitor_update", snapshot)

            # Also update legacy health dict
            state.health = {
                sid: {"status": monitor.statuses[sid]} for sid in MONITORED_SERVICES
            }

            # Persist every 5 cycles (~2.5 min)
            if monitor.cycle_count % 5 == 0:
                try:
                    MONITOR_DATA_FILE.write_text(
                        json.dumps(monitor.to_persist(), default=str)
                    )
                    token_tracker.save()
                except Exception as e:
                    logger.error(f"Failed to persist data: {e}")

        except Exception as e:
            logger.error(f"Monitor loop error: {e}")

        await asyncio.sleep(MONITOR_INTERVAL)


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="AIIA Command Center", version="2.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

state = PlatformState()
manager = ConnectionManager()
monitor = MonitorState()


# ─────────────────────────────────────────────────────────────
# Token & Cost Tracking
# ─────────────────────────────────────────────────────────────

# Per-model pricing (USD per 1M tokens): {model_prefix: (input_per_1m, output_per_1m)}
MODEL_PRICING = {
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3.5-haiku": (0.80, 4.0),
    "claude-opus-4": (15.0, 75.0),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
    # Local models are free
}


class TokenTrackingState:
    """Tracks token usage and costs, aggregated by day and provider."""

    def __init__(self):
        self.daily: Dict[str, Dict[str, Any]] = {}  # date -> provider -> stats
        self._data_file = Path(__file__).parent / "token_data.json"
        self._load()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_day(self, date: str):
        if date not in self.daily:
            self.daily[date] = {}

    def _ensure_provider(self, date: str, provider: str):
        self._ensure_day(date)
        if provider not in self.daily[date]:
            self.daily[date][provider] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0,
                "cost": 0.0,
            }

    def _calc_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request based on model pricing."""
        for prefix, (inp_price, out_price) in MODEL_PRICING.items():
            if model and model.startswith(prefix):
                return (input_tokens * inp_price / 1_000_000) + (
                    output_tokens * out_price / 1_000_000
                )
        return 0.0  # Unknown model or local — free

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Record a token usage event."""
        date = self._today()
        self._ensure_provider(date, provider)

        entry = self.daily[date][provider]
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["requests"] += 1
        entry["cost"] += self._calc_cost(model, input_tokens, output_tokens)

    def get_today(self) -> Dict[str, Any]:
        date = self._today()
        self._ensure_day(date)

        total_tokens = 0
        total_cost = 0.0
        total_requests = 0
        by_provider = {}

        for provider, stats in self.daily.get(date, {}).items():
            tokens = stats["input_tokens"] + stats["output_tokens"]
            total_tokens += tokens
            total_cost += stats["cost"]
            total_requests += stats["requests"]
            by_provider[provider] = {
                "tokens": tokens,
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "cost": round(stats["cost"], 6),
                "requests": stats["requests"],
            }

        return {
            "date": date,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "total_requests": total_requests,
            "by_provider": by_provider,
        }

    def get_recent(self, days: int = 7) -> List[Dict[str, Any]]:
        from datetime import timedelta

        result = []
        today = datetime.now(timezone.utc).date()
        for i in range(days):
            date = (today - timedelta(days=i)).isoformat()
            self._ensure_day(date)
            total_tokens = sum(
                s["input_tokens"] + s["output_tokens"]
                for s in self.daily.get(date, {}).values()
            )
            total_cost = sum(s["cost"] for s in self.daily.get(date, {}).values())
            total_requests = sum(
                s["requests"] for s in self.daily.get(date, {}).values()
            )
            result.append(
                {
                    "date": date,
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost, 6),
                    "total_requests": total_requests,
                }
            )
        return result

    def save(self):
        try:
            # Keep only last 30 days
            dates = sorted(self.daily.keys())
            if len(dates) > 30:
                for old in dates[:-30]:
                    del self.daily[old]
            self._data_file.write_text(json.dumps(self.daily, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save token data: {e}")

    def _load(self):
        if self._data_file.exists():
            try:
                self.daily = json.loads(self._data_file.read_text())
                logger.info(f"Loaded token data: {len(self.daily)} days")
            except Exception as e:
                logger.warning(f"Could not load token data: {e}")
                self.daily = {}


token_tracker = TokenTrackingState()


# ─────────────────────────────────────────────────────────────
# LLM Routing History — Persists routing decisions for visualization
# ─────────────────────────────────────────────────────────────


class RoutingHistoryState:
    """Tracks LLM routing decisions for the orchestration dashboard."""

    def __init__(self):
        self.history: deque = deque(maxlen=100)  # last 100 decisions
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "eos_count": 0,
            "rlm_count": 0,
            "by_provider": {},
            "by_domain": {},
            "by_eq_level": {},
            "complexity_scores": [],
        }

    def record(self, decision: Dict[str, Any]):
        """Record a routing decision."""
        decision["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.history.appendleft(decision)

        # Update rolling stats
        self.stats["total_requests"] += 1

        path = decision.get("recommended_path", "").lower()
        if "rlm" in path:
            self.stats["rlm_count"] += 1
        elif "eos" in path or path in ("local", "anthropic", "google"):
            self.stats["eos_count"] += 1

        # Provider distribution
        provider = decision.get("recommended_path", "unknown")
        self.stats["by_provider"][provider] = (
            self.stats["by_provider"].get(provider, 0) + 1
        )

        # Domain distribution
        domain = decision.get("domain", "general")
        if domain:
            self.stats["by_domain"][domain] = self.stats["by_domain"].get(domain, 0) + 1

        # Complexity scores (keep last 100)
        complexity = decision.get("complexity_score")
        if complexity is not None:
            scores = self.stats.setdefault("complexity_scores", [])
            scores.append(complexity)
            self.stats["complexity_scores"] = scores[-100:]

        # EQ level distribution
        eq = decision.get("eq_level")
        if eq is not None:
            bucket = (
                "1-2"
                if eq <= 2
                else "3-5"
                if eq <= 5
                else "8-12"
                if eq <= 12
                else "13+"
            )
            self.stats["by_eq_level"][bucket] = (
                self.stats["by_eq_level"].get(bucket, 0) + 1
            )

    def get_stats(self) -> Dict[str, Any]:
        """Return routing stats for dashboard."""
        return {
            **self.stats,
            "recent_decisions": list(self.history)[:20],
            "eos_pct": round(
                self.stats["eos_count"] / max(self.stats["total_requests"], 1) * 100, 1
            ),
            "rlm_pct": round(
                self.stats["rlm_count"] / max(self.stats["total_requests"], 1) * 100, 1
            ),
        }

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.history)[:limit]


routing_history = RoutingHistoryState()

# ─── Action Queue + Task Runner ───────────────────────────
from local_brain.command_center.action_queue import ActionQueue
from local_brain.command_center.aiia_tasks import TaskRunner

action_queue = ActionQueue()
REPO_PATH = str(Path(__file__).parent.parent.parent.parent)
task_runner = TaskRunner(
    broadcast_fn=manager.broadcast,
    repo_path=REPO_PATH,
    monitor_state=monitor,
    action_queue=action_queue,
)

# ─── Execution Engine ────────────────────────────────────
from local_brain.execution.executor import ExecutionEngine
from local_brain.config import LocalBrainConfig

_execution_engine: ExecutionEngine | None = None

# ─── Chat with AIIA ──────────────────────────────────────

AIIA_ASK_URL = "http://localhost:8100/v1/aiia/ask"
CHAT_HISTORY_FILE = Path(__file__).parent / "chat_history.json"
CHAT_HISTORY_MAX = 200

# Stream cancellation flag — set by /api/chat/stop, checked during streaming
_stream_cancel: asyncio.Event = asyncio.Event()


def _load_chat_history() -> List[Dict[str, str]]:
    """Load persisted chat history from disk."""
    if CHAT_HISTORY_FILE.exists():
        try:
            data = json.loads(CHAT_HISTORY_FILE.read_text())
            if isinstance(data, list):
                return data[-CHAT_HISTORY_MAX:]
        except Exception as e:
            logger.warning(f"Could not load chat history: {e}")
    return []


def _save_chat_history():
    """Persist chat history to disk (atomic write)."""
    try:
        tmp = CHAT_HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(chat_history[-CHAT_HISTORY_MAX:], indent=2, default=str)
        )
        tmp.rename(CHAT_HISTORY_FILE)
    except Exception as e:
        logger.warning(f"Could not save chat history: {e}")


chat_history: List[Dict[str, str]] = _load_chat_history()


class ChatMessage(BaseModel):
    message: str
    mode: str = Field(
        default="text",
        description="'voice' for short conversational replies, 'text' for full markdown",
    )


# ─── Routes ──────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_console():
    """Serve the Product Console (new) or fallback to old dashboard."""
    html_path = STATIC_DIR / "console.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    html_path = STATIC_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(
        content="<h1>AIIA Command Center</h1><p>No dashboard found.</p>"
    )


@app.get("/old", response_class=HTMLResponse)
async def serve_old_dashboard():
    """Serve the original ops dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<p>Old dashboard not found.</p>")


@app.get("/work", response_class=HTMLResponse)
async def serve_work_dashboard():
    """Serve the Work Dashboard — project tracking, commits, pipeline."""
    html_path = STATIC_DIR / "work.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<p>Work dashboard not found.</p>")


@app.get("/voice", response_class=HTMLResponse)
async def serve_voice():
    """Serve the AIIA Voice ambient interface."""
    html_path = STATIC_DIR / "voice.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<p>Voice interface not found.</p>")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(
            json.dumps(
                {
                    "type": "init",
                    "data": {
                        "platform": state.get_platform(),
                        "summary": state.get_summary(),
                        "monitor": monitor.get_full_snapshot(),
                        "tasks": task_runner.get_all_tasks(),
                        "insights": task_runner._extra.get("insights", []),
                        "task_extra": {
                            "code_health_trends": task_runner._extra.get(
                                "code_health_trends", []
                            ),
                            "test_trends": task_runner._extra.get("test_trends", []),
                            "security_snapshot": task_runner._extra.get(
                                "security_snapshot", {}
                            ),
                            "security_trends": task_runner._extra.get(
                                "security_trends", []
                            ),
                        },
                        "routing": routing_history.get_stats(),
                        "tokens": token_tracker.get_today(),
                        "actions": action_queue.list_actions(
                            status="pending", limit=20
                        ),
                        "action_summary": action_queue.summary(),
                    },
                }
            )
        )
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "get_platform":
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "platform_update",
                            "data": state.get_platform(),
                        }
                    )
                )
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


@app.get("/api/platform")
async def get_platform():
    return state.get_platform()


@app.get("/api/summary")
async def get_summary():
    return state.get_summary()


@app.get("/api/aiia")
async def get_aiia():
    """Proxy to AIIA status on Mac Mini."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:8100/v1/aiia/status")
            if resp.status_code == 200:
                data = resp.json()
                state.aiia_status = data
                return data
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
    return {"status": "unknown"}


@app.get("/api/health")
async def get_health():
    return state.health


@app.get("/api/monitor")
async def get_monitor():
    """Full production monitor snapshot — all services, history, transitions."""
    return monitor.get_full_snapshot()


@app.get("/api/monitor/{service_id}")
async def get_monitor_service(service_id: str):
    """Single service monitor data."""
    if service_id not in MONITORED_SERVICES:
        return {"error": f"Unknown service: {service_id}"}
    return monitor.get_service_snapshot(service_id)


# ─── Task API ─────────────────────────────────────────────


@app.get("/api/tasks")
async def get_tasks():
    """All AIIA tasks with current status."""
    return task_runner.get_all_tasks()


@app.post("/api/tasks/{task_id}/run")
async def run_task(task_id: str):
    """Manually trigger an AIIA task."""
    return await task_runner.trigger_task(task_id)


@app.get("/api/tasks/history")
async def get_task_history():
    """Recent task run history across all tasks."""
    return task_runner.get_history()


# ─── Action Queue API ─────────────────────────────────────


@app.get("/api/actions")
async def get_actions(
    status: Optional[str] = None, action_type: Optional[str] = None, limit: int = 50
):
    """List action items, optionally filtered."""
    return {
        "actions": action_queue.list_actions(
            status=status, action_type=action_type, limit=limit
        ),
        "summary": action_queue.summary(),
    }


@app.post("/api/actions")
async def create_action(body: dict = Body(...)):
    """Create a new action item via API."""
    action_type = body.get("action_type")
    severity = body.get("severity")
    title = body.get("title")

    if not action_type or not severity or not title:
        return JSONResponse(
            status_code=400,
            content={
                "error": "action_type, severity, and title are required"
            },
        )

    if action_type not in action_queue.VALID_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Invalid action_type. Valid:"
                    f" {sorted(action_queue.VALID_TYPES)}"
                )
            },
        )

    if severity not in action_queue.VALID_SEVERITIES:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Invalid severity. Valid:"
                    f" {sorted(action_queue.VALID_SEVERITIES)}"
                )
            },
        )

    action = action_queue.create_action(
        action_type=action_type,
        severity=severity,
        title=title,
        description=body.get("description", ""),
        proposed_fix=body.get("proposed_fix", ""),
        source_task=body.get("source_task", "api"),
        files_affected=body.get("files_affected"),
    )

    if body.get("auto_approve"):
        action_queue.approve(action["id"])
        action = action_queue.get_action(action["id"])

    return {"action": action}


@app.get("/api/actions/summary")
async def get_action_summary():
    """Count of actions by status and severity."""
    return action_queue.summary()


@app.post("/api/actions/{action_id}/approve")
async def approve_action(action_id: str):
    """Approve an action for execution."""
    action = action_queue.approve(action_id)
    if not action:
        return {"error": f"Action {action_id} not found"}
    await manager.broadcast("action_updated", action)
    return action


@app.post("/api/actions/{action_id}/reject")
async def reject_action(action_id: str, body: Dict[str, Any] = {}):
    """Reject an action with optional reason."""
    reason = body.get("reason", "")
    action = action_queue.reject(action_id, reason=reason)
    if not action:
        return {"error": f"Action {action_id} not found"}
    await manager.broadcast("action_updated", action)
    return action


@app.post("/api/actions/{action_id}/complete")
async def complete_action(action_id: str, body: Dict[str, Any] = {}):
    """Mark an approved action as completed."""
    result = body.get("result", "")
    action = action_queue.complete(action_id, result=result)
    if not action:
        return {"error": f"Action {action_id} not found"}
    await manager.broadcast("action_updated", action)
    return action


# ─── Briefing API ────────────────────────────────────────────


@app.post("/api/briefing/generate")
async def generate_briefing():
    """Trigger an on-demand morning briefing (runs daily_brief task)."""
    return await task_runner.trigger_task("daily_brief")


@app.get("/api/briefing/latest")
async def get_latest_briefing():
    """Return the most recent briefing output, last_run time, and status."""
    task = task_runner.tasks.get("daily_brief")
    if not task:
        return {"error": "daily_brief task not registered"}
    return {
        "briefing": task.get("last_output") or task.get("last_result", ""),
        "last_run": task.get("last_run"),
        "status": task.get("status", "unknown"),
        "duration_ms": task.get("last_duration_ms"),
        "run_count": task.get("run_count", 0),
    }


# ─── Interval Reports ───────────────────────────────────────


_latest_interval_report: Dict[str, Any] = {}


@app.post("/api/reports/interval")
async def receive_interval_report(payload: Dict[str, Any] = Body(...)):
    """Receive an interval code report and broadcast to dashboard."""
    global _latest_interval_report
    report = payload.get("report", {})
    _latest_interval_report = report
    await manager.broadcast("interval_report", report)
    return {"status": "received"}


@app.get("/api/reports/interval/latest")
async def get_latest_interval_report():
    """Return the most recent interval report."""
    return _latest_interval_report or {"summary": {"total_commits": 0}, "mode": "none"}


# ─── Ops Endpoints (receive metrics from local_api.py) ──────


class TokenUsageReport(BaseModel):
    provider: str = "local"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    endpoint: str = ""


class LatencyReport(BaseModel):
    provider: str = "local"
    latency_ms: float = 0.0
    model: str = ""


class RoutingReport(BaseModel):
    recommended_path: str = ""
    domain: str = ""
    model: str = ""
    usage: Dict[str, Any] = {}


@app.post("/ops/record-token-usage")
async def record_token_usage(report: TokenUsageReport):
    """Receive token usage reports from local_api or cloud backends."""
    token_tracker.record(
        provider=report.provider,
        model=report.model,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
    )
    # Broadcast live update
    await manager.broadcast("token_update", token_tracker.get_today())
    return {"status": "recorded"}


@app.post("/ops/record-latency")
async def record_latency(report: LatencyReport):
    """Receive latency samples from local_api."""
    # Latency is already tracked by the production monitor; this endpoint
    # accepts reports from local_api.py fire-and-forget calls
    return {"status": "recorded"}


@app.post("/ops/record-routing")
async def record_routing(report: RoutingReport):
    """Receive routing decision reports (includes token usage)."""
    usage = report.usage or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if input_tokens or output_tokens:
        token_tracker.record(
            provider=report.recommended_path or "local",
            model=report.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        await manager.broadcast("token_update", token_tracker.get_today())

    # Record routing decision for orchestration visualization
    routing_history.record(
        {
            "recommended_path": report.recommended_path,
            "domain": report.domain,
            "model": report.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "complexity_score": usage.get("complexity_score"),
            "eq_level": usage.get("eq_level"),
            "routing_mode": usage.get("routing_mode"),
            "latency_ms": usage.get("latency_ms"),
        }
    )
    await manager.broadcast("routing_update", routing_history.get_stats())

    return {"status": "recorded"}


# ─── Token API ─────────────────────────────────────────────


@app.get("/api/routing/stats")
async def get_routing_stats():
    """LLM routing statistics — provider/domain/EQ distribution, recent decisions."""
    return routing_history.get_stats()


@app.get("/api/routing/recent")
async def get_routing_recent(limit: int = 20):
    """Recent routing decisions."""
    return {"decisions": routing_history.get_recent(limit)}


@app.get("/api/insights")
async def get_insights():
    """Return stored insights and trend data from task runner."""
    insights = task_runner._extra.get("insights", [])
    return {
        "insights": insights,
        "count": len(insights),
        "task_extra": {
            "code_health_trends": task_runner._extra.get("code_health_trends", []),
            "test_trends": task_runner._extra.get("test_trends", []),
            "security_snapshot": task_runner._extra.get("security_snapshot", {}),
            "security_trends": task_runner._extra.get("security_trends", []),
        },
    }


@app.get("/api/tokens/today")
async def get_tokens_today():
    """Today's token usage and cost breakdown by provider."""
    return token_tracker.get_today()


@app.get("/api/tokens/recent")
async def get_tokens_recent(days: int = 7):
    """Recent daily token usage (last N days)."""
    return {"days": token_tracker.get_recent(days)}


# ─── Memory Browser Proxy ─────────────────────────────────

AIIA_BASE_URL = "http://localhost:8100"

# Shared httpx client for AIIA calls — connection pooling, avoids per-request overhead
async def get_aiia_client() -> httpx.AsyncClient:
    """Create a fresh httpx client per request — persistent clients go stale after long AIIA calls."""
    return httpx.AsyncClient(
        base_url=AIIA_BASE_URL,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


@app.get("/api/memories")
async def get_memories(category: Optional[str] = None, limit: int = 50):
    """Proxy to AIIA memory API for dashboard memory browser."""
    params = f"?limit={limit}"
    if category:
        params += f"&category={category}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{AIIA_BASE_URL}/v1/aiia/memory{params}")
            if resp.status_code == 200:
                return resp.json()
            return {
                "error": f"AIIA returned {resp.status_code}",
                "memories": [],
                "count": 0,
            }
    except Exception as e:
        return {"error": str(e), "memories": [], "count": 0}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Proxy delete to AIIA memory API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{AIIA_BASE_URL}/v1/aiia/memory/{memory_id}")
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ─── Chat API ─────────────────────────────────────────────


@app.post("/api/chat")
async def chat_with_aiia(msg: ChatMessage):
    """Proxy chat to AIIA's ask endpoint with conversation context."""
    logger.info(f"[CHAT] received: {msg.message[:30]}")

    now = datetime.now(timezone.utc).isoformat()
    recent = chat_history[-20:]
    context_lines = []
    for entry in recent:
        role = "User" if entry["role"] == "user" else "AIIA"
        context_lines.append(f"{role}: {entry['content']}")
    context = "\n".join(context_lines)
    if msg.mode == "voice":
        context = VOICE_MODE_INSTRUCTION + (f"\n## Recent Conversation\n{context}" if context else "")

    chat_history.append({"role": "user", "content": msg.message, "ts": now, "mode": msg.mode})

    reply = "No response"
    model = "unknown"
    sources = 0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            resp = await client.post(
                AIIA_ASK_URL,
                json={"question": msg.message, "context": context, "n_results": 5},
            )
            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("answer", "No response from AIIA.")
                model = data.get("model", "unknown")
                sources = data.get("sources_used", 0)
            else:
                reply = f"AIIA returned HTTP {resp.status_code}"
                model = "error"
    except Exception as e:
        logger.error(f"[CHAT] AIIA error: {e}")
        reply = f"Could not reach AIIA: {str(e)[:120]}"
        model = "error"

    chat_history.append({"role": "aiia", "content": reply, "ts": datetime.now(timezone.utc).isoformat()})
    _save_chat_history()
    return {"reply": reply, "model": model, "sources": sources}


VOICE_MODE_INSTRUCTION = """## Voice Mode — CRITICAL INSTRUCTIONS
You are in a VOICE CONVERSATION via AirPods. The user is speaking to you and will hear your response read aloud.

Rules for voice mode:
- Keep responses to 1-3 sentences MAX. Be concise like a real conversation.
- Be warm, natural, and conversational — not robotic or formal.
- NEVER use markdown, bullet points, numbered lists, or code formatting.
- NEVER dump long explanations. If the topic is complex, give a brief answer and offer to elaborate.
- Match the user's energy — if they're casual, be casual. If they ask something specific, answer directly.
- You are a teammate talking to Eric, not writing a report.
- If you don't know something, just say so in one sentence.
"""


@app.post("/api/chat/stream")
async def chat_with_aiia_stream(msg: ChatMessage):
    """Streaming proxy to AIIA's ask/stream endpoint. Mode-aware (voice/text)."""
    now = datetime.now(timezone.utc).isoformat()
    _stream_cancel.clear()

    # Build context from last 10 exchanges
    recent = chat_history[-20:]
    context_lines = []
    for entry in recent:
        role = "User" if entry["role"] == "user" else "AIIA"
        context_lines.append(f"{role}: {entry['content']}")
    conversation_context = "\n".join(context_lines)

    # Mode-aware context and token limits
    if msg.mode == "voice":
        context = VOICE_MODE_INSTRUCTION
        if conversation_context:
            context += f"\n## Recent Conversation\n{conversation_context}"
        max_tokens = 256
        n_results = 3
    else:
        context = conversation_context
        max_tokens = 4096
        n_results = 5

    chat_history.append(
        {"role": "user", "content": msg.message, "ts": now, "mode": msg.mode}
    )
    _save_chat_history()

    async def proxy_stream():
        full_answer = []
        cancelled = False
        got_done = False
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    AIIA_ASK_URL + "/stream",
                    json={
                        "question": msg.message,
                        "context": context,
                        "n_results": n_results,
                        "max_tokens": max_tokens,
                        "num_ctx": 8192,
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if _stream_cancel.is_set():
                            cancelled = True
                            break
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data.get("type") == "chunk":
                                full_answer.append(data["content"])
                            elif data.get("type") == "done":
                                got_done = True
                            yield line + "\n\n"
        except Exception as e:
            err = json.dumps({"type": "error", "message": str(e)[:200]})
            yield f"data: {err}\n\n"

        if cancelled:
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"

        # Safety net: emit synthetic done if upstream didn't send one
        if not got_done and not cancelled and full_answer:
            synthetic = json.dumps(
                {
                    "type": "done",
                    "answer": "".join(full_answer),
                }
            )
            yield f"data: {synthetic}\n\n"

        # Append the full reply to chat history after stream completes
        reply = "".join(full_answer) if full_answer else "No response from AIIA."
        if cancelled:
            reply += " [stopped]"
        chat_history.append(
            {
                "role": "aiia",
                "content": reply,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save_chat_history()

    return StreamingResponse(proxy_stream(), media_type="text/event-stream")


@app.post("/api/tts")
async def tts_proxy(body: Dict[str, Any] = {}):
    """Proxy TTS request to AIIA Local Brain's Google Cloud TTS endpoint. Returns audio bytes."""
    text = body.get("text", "")
    voice = body.get("voice", "aiia")
    speaking_rate = body.get("speaking_rate", 1.0)
    if not text:
        return {"status": "empty"}
    try:
        async with await get_aiia_client() as client:
            resp = await client.post(
                "/v1/aiia/tts",
                json={"text": text, "voice": voice, "speaking_rate": speaking_rate},
            )
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "audio" in content_type:
                    return Response(content=resp.content, media_type=content_type)
                else:
                    return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/voice")
async def voice_proxy(body: Dict[str, Any] = {}):
    """Proxy to AIIA combined ask+TTS endpoint. Returns MP3 audio of AIIA's spoken answer."""
    question = body.get("question", "")
    voice = body.get("voice", "aiia")
    if not question:
        return {"error": "question is required"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{AIIA_BASE_URL}/v1/aiia/voice",
                json={"question": question, "voice": voice},
            )
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "audio" in content_type:
                    return Response(content=resp.content, media_type=content_type)
                else:
                    return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/speak")
async def speak_proxy(body: Dict[str, Any] = {}):
    """Proxy speak request to AIIA Local Brain."""
    text = body.get("text", "")
    voice = body.get("voice", "aiia")
    if not text:
        return {"status": "empty"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{AIIA_BASE_URL}/v1/aiia/speak",
                json={"text": text, "voice": voice},
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/speak/stop")
async def stop_speak_proxy():
    """Proxy stop-speak request to AIIA Local Brain."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{AIIA_BASE_URL}/v1/aiia/speak/stop")
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ops/speech-done")
async def speech_done():
    """Called by local_api when TTS finishes. Broadcasts to all WS clients."""
    await manager.broadcast("speech_done", {})
    return {"status": "broadcast_sent"}


@app.get("/api/chat/history")
async def get_chat_history():
    """Return current conversation history."""
    return {"history": chat_history}


@app.delete("/api/chat/history")
async def clear_chat_history():
    """Clear conversation history."""
    chat_history.clear()
    _save_chat_history()
    return {"status": "cleared"}


@app.put("/api/chat/history/{index}")
async def edit_chat_message(index: int, body: Dict[str, Any]):
    """Edit a user message and truncate all subsequent messages."""
    if index < 0 or index >= len(chat_history):
        return {"error": "Index out of range"}
    if chat_history[index]["role"] != "user":
        return {"error": "Can only edit user messages"}
    new_text = body.get("message", "").strip()
    if not new_text:
        return {"error": "Message cannot be empty"}
    # Truncate everything after this message (including AIIA reply)
    del chat_history[index + 1 :]
    chat_history[index]["content"] = new_text
    chat_history[index]["ts"] = datetime.now(timezone.utc).isoformat()
    _save_chat_history()
    return {"status": "edited", "history": chat_history}


@app.delete("/api/chat/history/{index}")
async def delete_chat_message(index: int):
    """Delete a single message from chat history."""
    if index < 0 or index >= len(chat_history):
        return {"error": "Index out of range"}
    deleted = chat_history.pop(index)
    _save_chat_history()
    return {"status": "deleted", "deleted": deleted}


@app.post("/api/chat/stop")
async def stop_chat_stream():
    """Signal the active stream to stop."""
    _stream_cancel.set()
    return {"status": "cancel_requested"}


# ─── AIIA Proxy Endpoints ────────────────────────────────


@app.post("/api/aiia/session-start")
async def aiia_session_start(body: Dict[str, Any] = {}):
    """Proxy to AIIA session-start (load WIP, decisions, knowledge)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{AIIA_BASE_URL}/v1/aiia/session-start",
                json=body,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/aiia/remember")
async def aiia_remember(body: Dict[str, Any] = {}):
    """Proxy to AIIA remember (teach from console)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{AIIA_BASE_URL}/v1/aiia/remember",
                json=body,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ─── Daily Reports + Roadmap + Syntax ────────────────────────

from local_brain.scripts.daily_report import (
    generate_report,
    save_report,
    load_report,
    list_reports,
)
from local_brain.scripts.roadmap_store import RoadmapStore
from local_brain.scripts.pipeline_store import PipelineStore
from local_brain.eq_brain.story_prioritizer import StoryPrioritizer
from local_brain.scripts.syntax_checker import check_syntax

_roadmap = RoadmapStore()
_pipeline = PipelineStore()
_story_prioritizer: StoryPrioritizer | None = None


@app.get("/api/reports/today")
async def report_today():
    from datetime import date as date_type

    today = date_type.today().isoformat()
    existing = load_report(today)
    if existing:
        return existing
    report = generate_report(date=today, repo_dir=REPO_PATH)
    save_report(report)
    return report


@app.get("/api/reports/today-md")
async def report_today_md():
    """Return the rolling daily markdown report (today.md)."""
    md_path = Path.home() / ".aiia" / "eq_data" / "reports" / "today.md"
    if md_path.exists():
        return {"content": md_path.read_text()}
    return {"content": ""}


@app.get("/api/reports")
async def report_list():
    return {"dates": list_reports()}


@app.get("/api/reports/{date}")
async def report_by_date(date: str):
    existing = load_report(date)
    if existing:
        return existing
    report = generate_report(date=date, repo_dir=REPO_PATH)
    save_report(report)
    return report


@app.post("/api/reports/generate")
async def report_generate(body: Dict[str, Any] = {}):
    date = body.get("date")
    report = generate_report(date=date, repo_dir=REPO_PATH)
    save_report(report)
    return report


@app.get("/api/roadmap")
async def roadmap_list(product: Optional[str] = None, status: Optional[str] = None):
    stories = _roadmap.list(product=product, status=status)
    return {"stories": stories, "count": len(stories)}


@app.post("/api/roadmap")
async def roadmap_create(body: Dict[str, Any]):
    try:
        story = _roadmap.create(
            title=body["title"],
            product=body.get("product", ""),
            priority=body.get("priority", "P2"),
            status=body.get("status", "backlog"),
            description=body.get("description", ""),
            tags=body.get("tags", []),
            client_impact=body.get("client_impact", ""),
            source_session=body.get("source_session", ""),
            source_type=body.get("source_type", "manual"),
            dedup=body.get("dedup", True),
        )
        return {"story": story}
    except (KeyError, ValueError) as e:
        return {"error": str(e)}


@app.put("/api/roadmap/{story_id}")
async def roadmap_update(story_id: str, body: Dict[str, Any]):
    try:
        story = _roadmap.update(story_id, **body)
        if story is None:
            return {"error": "Story not found"}
        return {"story": story}
    except ValueError as e:
        return {"error": str(e)}


@app.delete("/api/roadmap/{story_id}")
async def roadmap_delete(story_id: str):
    return {"deleted": _roadmap.delete(story_id)}


@app.post("/api/roadmap/extract")
async def roadmap_extract(body: Dict[str, Any]):
    """Extract candidate stories from session data and log them."""
    if not _story_prioritizer:
        return {"error": "Story prioritizer not available", "stories": []}

    try:
        candidates = await _story_prioritizer.extract_stories_from_session(
            summary=body.get("summary", ""),
            next_steps=body.get("next_steps", []),
            blockers=body.get("blockers", []),
            key_decisions=body.get("key_decisions", []),
            session_id=body.get("session_id", ""),
        )
    except Exception as e:
        logger.error(f"Story extraction failed: {e}")
        return {"error": str(e), "stories": []}

    created = []
    for c in candidates:
        story = _roadmap.create(
            title=c["title"],
            product=c.get("product", "platform"),
            description=c.get("description", ""),
            tags=c.get("tags", []),
            client_impact=c.get("client_impact", ""),
            source_session=c.get("source_session", ""),
            source_type="auto-extracted",
            dedup=True,
        )
        created.append(story)

    return {"stories": created, "count": len(created)}


@app.post("/api/roadmap/prioritize")
async def roadmap_prioritize(body: Dict[str, Any]):
    """Score and rank backlog stories."""
    if not _story_prioritizer:
        return {"error": "Story prioritizer not available"}

    limit = body.get("limit", 10)
    stories = _roadmap.list()
    try:
        ranked = await _story_prioritizer.prioritize_backlog(stories, limit=limit)
        return {"stories": ranked, "count": len(ranked)}
    except Exception as e:
        logger.error(f"Prioritization failed: {e}")
        return {"error": str(e)}


@app.get("/api/roadmap/similar/{title}")
async def roadmap_similar(title: str):
    """Find similar existing stories (dedup check)."""
    matches = _roadmap.find_similar(title)
    return {"matches": matches, "count": len(matches)}


@app.get("/api/roadmap/summary")
async def roadmap_summary():
    """Backlog summary stats."""
    return _roadmap.backlog_summary()


# ─── Pipeline API ─────────────────────────────────────────


@app.get("/api/pipeline")
async def pipeline_list(stage: Optional[str] = None):
    deals = _pipeline.list(stage=stage)
    return {"deals": deals, "count": len(deals), "summary": _pipeline.summary()}


@app.post("/api/pipeline")
async def pipeline_create(body: Dict[str, Any]):
    try:
        deal = _pipeline.create(
            company=body.get("company", ""),
            contact=body.get("contact", ""),
            stage=body.get("stage", "lead"),
            value=body.get("value", 0),
            product=body.get("product", ""),
            notes=body.get("notes", ""),
        )
        return deal
    except (KeyError, ValueError) as e:
        return {"error": str(e)}


@app.put("/api/pipeline/{deal_id}")
async def pipeline_update(deal_id: str, body: Dict[str, Any]):
    try:
        deal = _pipeline.update(deal_id, **body)
        if deal is None:
            return {"error": "Deal not found"}
        return deal
    except ValueError as e:
        return {"error": str(e)}


@app.delete("/api/pipeline/{deal_id}")
async def pipeline_delete(deal_id: str):
    return {"deleted": _pipeline.delete(deal_id)}


@app.get("/api/syntax")
async def syntax_check():
    return check_syntax(REPO_PATH)


# ─── Work Context (aggregated view for work dashboard) ────


@app.get("/api/work/context")
async def work_context():
    """Aggregated work context: recent commits, pipeline, uncommitted, weekly activity."""
    import subprocess
    from datetime import date as date_type

    result: Dict[str, Any] = {}

    # 1. Today's report (always regenerate for freshness)
    today = date_type.today().isoformat()
    existing = generate_report(date=today, repo_dir=REPO_PATH)
    save_report(existing)
    result["today"] = {
        "date": existing.get("date"),
        "summary": existing.get("summary", {}),
        "products": existing.get("products", {}),
        "categories": existing.get("categories", {}),
    }

    # 2. Latest interval report
    result["interval"] = _latest_interval_report or {}

    # 3. Pipeline deals
    result["pipeline"] = _pipeline.list()

    # 4. Uncommitted changes
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_PATH,
        )
        staged_result = subprocess.run(
            ["git", "diff", "--stat", "--cached", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_PATH,
        )
        uncommitted_files = []
        for line in diff_result.stdout.strip().splitlines():
            if "|" in line:
                uncommitted_files.append(
                    {"file": line.split("|")[0].strip(), "status": "modified"}
                )
        for line in staged_result.stdout.strip().splitlines():
            if "|" in line:
                uncommitted_files.append(
                    {"file": line.split("|")[0].strip(), "status": "staged"}
                )
        result["uncommitted"] = uncommitted_files
    except Exception:
        result["uncommitted"] = []

    # 5. Weekly commit heatmap (last 7 days)
    try:
        week_log = subprocess.run(
            ["git", "log", "--since=7 days ago", "--format=%aI", "--all"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=REPO_PATH,
        )
        day_counts: Dict[str, int] = {}
        for line in week_log.stdout.strip().splitlines():
            if line:
                day = line[:10]
                day_counts[day] = day_counts.get(day, 0) + 1
        result["weekly_heatmap"] = day_counts
    except Exception:
        result["weekly_heatmap"] = {}

    # 6. Roadmap stories
    result["roadmap"] = _roadmap.list()

    # 7. Recent commits (last 12 hours for the feed)
    try:
        recent = generate_report(repo_dir=REPO_PATH, since_hours=12)
        commits_flat = []
        for product, data in recent.get("products", {}).items():
            for c in data.get("commits", []):
                commits_flat.append({**c, "product": product})
        result["recent_commits"] = commits_flat
    except Exception:
        result["recent_commits"] = []

    return result


# ─── Morning Check-in (aggregated dashboard) ─────────────────


@app.get("/api/checkin")
async def checkin():
    """Aggregated morning check-in: WIP, sessions, commits, nightly jobs, actions, stories, pipeline."""
    now = datetime.now(timezone.utc)
    result: Dict[str, Any] = {"timestamp": now.isoformat()}

    # 1. WIP state from AIIA memory
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{AIIA_BASE_URL}/v1/aiia/memory?category=wip")
            if resp.status_code == 200:
                result["wip"] = resp.json().get("memories", resp.json() if isinstance(resp.json(), list) else [])
            else:
                result["wip"] = {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        result["wip"] = {"error": str(e)}

    # 2. Recent sessions (last 3) from AIIA memory
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{AIIA_BASE_URL}/v1/aiia/memory?category=sessions")
            if resp.status_code == 200:
                data = resp.json()
                memories = data.get("memories", data if isinstance(data, list) else [])
                result["recent_sessions"] = memories[-3:] if len(memories) > 3 else memories
            else:
                result["recent_sessions"] = {"error": f"AIIA returned {resp.status_code}"}
    except Exception as e:
        result["recent_sessions"] = {"error": str(e)}

    # 3. Recent commits (last 24h, grouped by product) — reuses generate_report pattern
    try:
        report = generate_report(repo_dir=REPO_PATH, since_hours=24)
        commits_flat = []
        by_product: Dict[str, int] = {}
        for product, data in report.get("products", {}).items():
            prod_commits = data.get("commits", [])
            by_product[product] = len(prod_commits)
            for c in prod_commits:
                commits_flat.append({**c, "product": product})
        result["recent_commits"] = {
            "total": len(commits_flat),
            "by_product": by_product,
            "commits": commits_flat,
        }
    except Exception as e:
        result["recent_commits"] = {"total": 0, "by_product": {}, "commits": [], "error": str(e)}

    # 4. Nightly job freshness — check file modification timestamps
    nightly: Dict[str, Any] = {}
    stale_threshold_hours = 26

    def _check_job_file(path: Path) -> Dict[str, Any]:
        """Check if a nightly job output file exists and how old it is."""
        if not path.exists():
            return {"status": "missing", "age_hours": None, "path": str(path)}
        try:
            mtime = path.stat().st_mtime
            age_hours = round((time.time() - mtime) / 3600, 1)
            status = "ok" if age_hours <= stale_threshold_hours else "stale"
            return {"status": status, "age_hours": age_hours, "path": str(path)}
        except Exception as e:
            return {"status": "missing", "age_hours": None, "path": str(path), "error": str(e)}

    home = Path.home()
    nightly["security_scan"] = _check_job_file(
        home / ".aiia" / "logs" / "security" / "latest.txt"
    )
    nightly["memory_sync"] = _check_job_file(
        home / ".aiia" / "logs" / "sync" / "latest.txt"
    )

    # Daily report — find latest file in reports dir
    reports_dir = home / ".aiia" / "eq_data" / "reports"
    try:
        if reports_dir.exists():
            report_files = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if report_files:
                nightly["daily_report"] = _check_job_file(report_files[0])
            else:
                nightly["daily_report"] = {"status": "missing", "age_hours": None, "path": str(reports_dir)}
        else:
            nightly["daily_report"] = {"status": "missing", "age_hours": None, "path": str(reports_dir)}
    except Exception as e:
        nightly["daily_report"] = {"status": "missing", "age_hours": None, "error": str(e)}

    result["nightly_jobs"] = nightly

    # 5. Pending actions — summary + top 10 critical/error severity
    try:
        result_actions: Dict[str, Any] = {"summary": action_queue.summary()}
        pending = action_queue.list_actions(status="pending")
        top_critical = [
            a for a in pending if a.get("severity") in ("critical", "error")
        ][:10]
        result_actions["top_critical"] = top_critical
        result["actions"] = result_actions
    except Exception as e:
        result["actions"] = {"summary": {}, "top_critical": [], "error": str(e)}

    # 6. Active/blocked stories from roadmap
    try:
        all_stories = _roadmap.list()
        active = [s for s in all_stories if s.get("status") in ("active", "in_progress")]
        blocked = [s for s in all_stories if s.get("status") == "blocked"]
        backlog = [s for s in all_stories if s.get("status") not in ("active", "in_progress", "blocked", "done")]
        result["stories"] = {
            "active": active,
            "blocked": blocked,
            "backlog_count": len(backlog),
        }
    except Exception as e:
        result["stories"] = {"active": [], "blocked": [], "backlog_count": 0, "error": str(e)}

    # 7. Pipeline snapshot — deals grouped by stage
    try:
        pipeline_summary = _pipeline.summary()
        result["pipeline"] = {
            "deals_by_stage": pipeline_summary.get("by_stage", {}),
            "total_value": pipeline_summary.get("total_value", 0),
        }
    except Exception as e:
        result["pipeline"] = {"deals_by_stage": {}, "total_value": 0, "error": str(e)}

    return result


# ─── Execution Engine Endpoints ──────────────────────────────


@app.get("/api/execution/status")
async def execution_status():
    if not _execution_engine:
        return {"enabled": False}
    return await _execution_engine.get_status()


@app.post("/api/execution/kill")
async def execution_kill():
    if not _execution_engine:
        return {"error": "Execution engine not initialized"}
    await _execution_engine.emergency_stop()
    return {"status": "killed"}


@app.post("/api/execution/execute/{action_id}")
async def execution_execute(action_id: str):
    """Explicitly trigger execution of a GATED action."""
    if not _execution_engine:
        return {"error": "Execution engine not initialized"}
    result = await _execution_engine.execute_now(action_id)
    return result


@app.get("/api/execution/log")
async def execution_log(limit: int = 20):
    if not _execution_engine:
        return {"records": []}
    return {"records": _execution_engine.execution_log.list_recent(limit)}


@app.post("/api/execution/story/{story_id}")
async def execution_story(story_id: str, body: Dict[str, Any] = {}):
    """Decompose a story into actions and start execution."""
    if not _execution_engine:
        return {"error": "Execution engine not running"}
    auto_approve = body.get("auto_approve", False)
    result = await _execution_engine.execute_story(
        story_id, auto_approve=auto_approve
    )
    return result


@app.get("/api/execution/story/{story_id}/progress")
async def execution_story_progress(story_id: str):
    """Get execution progress for a story."""
    if not _execution_engine:
        return {"error": "Execution engine not running"}
    return _execution_engine.get_story_progress(story_id)


# ─── Background Tasks ────────────────────────────────────────


@app.on_event("startup")
async def startup():
    # Load persisted monitor data if available
    if MONITOR_DATA_FILE.exists():
        try:
            data = json.loads(MONITOR_DATA_FILE.read_text())
            monitor.load_persisted(data)
            logger.info(
                f"Loaded monitor history: {monitor.cycle_count} cycles, {sum(monitor.total_checks.values())} total checks"
            )
        except Exception as e:
            logger.warning(f"Could not load monitor data: {e}")

    asyncio.create_task(production_monitor_loop())

    # Start task runner
    task_runner.load_state()
    asyncio.create_task(task_runner.run_loop())

    # Start execution engine
    global _execution_engine
    try:
        config = LocalBrainConfig()
        _execution_engine = ExecutionEngine(
            action_queue=action_queue,
            config=config,
            broadcast_fn=manager.broadcast,
            roadmap_store=_roadmap,
        )
        await _execution_engine.start()
        logger.info("Execution engine started")
    except Exception as e:
        logger.warning(f"Execution engine failed to start: {e}")

    # Initialize story prioritizer (uses Ollama for scoring)
    global _story_prioritizer
    try:
        from local_brain.ollama_client import OllamaClient

        ollama = OllamaClient()
        model = config.models.get("task")
        model_name = model.model_name if model else "llama3.1:8b-instruct-q8_0"
        _story_prioritizer = StoryPrioritizer(ollama_client=ollama, model=model_name)
        logger.info("Story prioritizer initialized")
    except Exception as e:
        logger.warning(f"Story prioritizer failed to init: {e}")

    # Expire stale actions on startup
    expired = action_queue.expire_old(hours=72)
    if expired:
        logger.info(f"Expired {expired} stale action items")

    logger.info("AIIA Command Center started on :8200")


@app.on_event("shutdown")
async def shutdown():
    # Persist on shutdown
    try:
        MONITOR_DATA_FILE.write_text(json.dumps(monitor.to_persist(), default=str))
        logger.info("Monitor data persisted on shutdown")
    except Exception as e:
        logger.warning(f"Could not persist monitor data on shutdown: {e}")

    task_runner.save_state()
    token_tracker.save()
    logger.info("Task runner + token data persisted on shutdown")


# ─── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8200)  # nosec B104
