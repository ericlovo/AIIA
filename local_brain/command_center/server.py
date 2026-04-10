"""
AIIA Command Center — Platform Intelligence Dashboard

Real-time visualization of the entire AIIA multi-tenant AI platform.
Shows microproducts, agent workflows, AIIA intelligence, and system connections.
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

logger = logging.getLogger("aiia.console")

# ─────────────────────────────────────────────────────────────
# Platform Registry — defines the full system topology
# ─────────────────────────────────────────────────────────────

# Default products — customize this list for your deployment.
# Each product represents a tenant in the multi-tenant architecture.
# Override via AIIA_PRODUCTS_CONFIG env var (path to JSON file).
_DEFAULT_PRODUCTS = [
    {
        "id": "my_app",
        "name": "My App",
        "subtitle": "Your AI Application",
        "client": "Default",
        "status": "production",
        "color": "#3b82f6",
        "agents": [],
        "components": [],
        "knowledge_files": 0,
        "knowledge_words": "0",
        "frontend": os.getenv("APP_FRONTEND_URL", "http://localhost:3000"),
        "backend": os.getenv("APP_BACKEND_URL", "http://localhost:9000"),
    },
    {
        "id": "demo",
        "name": "Demo",
        "subtitle": "Development & Testing Tenant",
        "client": "Internal",
        "status": "active",
        "color": "#475569",
        "agents": [],
        "components": [],
        "knowledge_files": 0,
        "knowledge_words": "0",
    },
]

def _load_products():
    """Load product config from env var JSON file, or use defaults."""
    config_path = os.getenv("AIIA_PRODUCTS_CONFIG")
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path) as f:
            return json.load(f)
    return _DEFAULT_PRODUCTS

PRODUCTS = _load_products()

# Future products — populate via AIIA_FUTURE_PRODUCTS_CONFIG env var (JSON file).
FUTURE_PRODUCTS: List[Dict[str, Any]] = []

AGENTS = [
    {
        "id": "conductor",
        "name": "Conductor",
        "type": "router",
        "color": "#3b82f6",
        "detail": "EQ + complexity routing",
    },
    {
        "id": "fast",
        "name": "Fast Path",
        "type": "chat",
        "color": "#3b82f6",
        "detail": "Single-shot inference",
    },
    {
        "id": "rlm",
        "name": "RLM Engine",
        "type": "reasoning",
        "color": "#a855f7",
        "detail": "Recursive reasoning with tool calls",
    },
    {
        "id": "finance",
        "name": "Finance Analyst",
        "type": "specialist",
        "color": "#f59e0b",
        "detail": "Numerical analysis example",
    },
    {
        "id": "legal",
        "name": "Legal Analyst",
        "type": "specialist",
        "color": "#64d2ff",
        "detail": "Document analysis example",
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
        "detail": "llama3.1:8b-instruct-q8_0 on Mac Mini M4",
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
    {"from": "my_app", "to": "conductor", "type": "chat"},
    {"from": "demo", "to": "conductor", "type": "chat"},
    # Routers → Agents
    {"from": "conductor", "to": "fast", "type": "routing", "label": "< 0.6"},
    {"from": "conductor", "to": "rlm", "type": "routing", "label": ">= 0.6"},
    # Agents → LLMs
    {"from": "fast", "to": "claude", "type": "inference"},
    {"from": "rlm", "to": "claude", "type": "inference"},
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
    "default": {
        "name": "App Analyst",
        "url": os.getenv(
            "APP_BACKEND_URL", "https://localhost:9000"
        )
        + "/health",
        "metrics_url": os.getenv(
            "APP_BACKEND_URL", "https://localhost:9000"
        )
        + "/metrics",
        "timeout": 10.0,
        "category": "backend",
    },
    "platform": {
        "name": "AIIA Platform",
        "url": os.getenv(
            "AIIA_PLATFORM_URL", "http://localhost:9000"
        )
        + "/health",
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

from local_brain.__version__ import __version__

app = FastAPI(title="AIIA Command Center", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8200",
        "http://127.0.0.1:8200",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

STATIC_DIR = Path(__file__).parent / "static"

# React dashboard (built assets from products/command-center/frontend/dist)
REACT_DIST = (
    Path(__file__).parents[3] / "products" / "command-center" / "frontend" / "dist"
)
if REACT_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(REACT_DIST / "assets")),
        name="react-assets",
    )

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from local_brain.command_center.session_registry import (
    SessionRegistry,
)
from local_brain.command_center.workstream_registry import (
    WorkstreamRegistry,
)

state = PlatformState()
manager = ConnectionManager()
monitor = MonitorState()
session_registry = SessionRegistry(
    data_dir=os.path.expanduser("~/.aiia/eq_data"),
    broadcast_fn=lambda et, d: manager.broadcast(et, d),
)
workstream_registry = WorkstreamRegistry(
    data_dir=os.path.expanduser("~/.aiia/eq_data"),
)


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
    """Serve the React Command Center dashboard, or fallback to legacy."""
    react_index = REACT_DIST / "index.html" if REACT_DIST.exists() else None
    if react_index and react_index.exists():
        return HTMLResponse(content=react_index.read_text())
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
                        "sessions": session_registry.summary(),
                        "workstreams": workstream_registry.summary(),
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
            content={"error": "action_type, severity, and title are required"},
        )

    if action_type not in action_queue.VALID_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Invalid action_type. Valid: {sorted(action_queue.VALID_TYPES)}"
                )
            },
        )

    if severity not in action_queue.VALID_SEVERITIES:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Invalid severity. Valid: {sorted(action_queue.VALID_SEVERITIES)}"
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
    """Approve an action for execution.

    AUTO-tier actions (lint_fix, verify_*) execute immediately on approval.
    SUPERVISED/GATED actions stay in 'approved' status awaiting explicit trigger.
    """
    action = action_queue.approve(action_id)
    if not action:
        return {"error": f"Action {action_id} not found"}
    await manager.broadcast("action_updated", action)

    # Auto-execute for AUTO tier actions
    if _execution_engine and action.get("status") == "approved":
        from local_brain.execution.safety import (
            SafetyGate,
            SafetyTier,
        )

        gate = SafetyGate()
        tier = gate.get_tier(action)
        if tier == SafetyTier.AUTO:
            logger.info(
                f"Auto-executing AUTO-tier action: {action_id} ({action.get('type')})"
            )
            try:
                result = await _execution_engine.execute_now(action_id)
                action = action_queue.get_action(action_id)
                await manager.broadcast("action_updated", action)
                return {**action, "auto_executed": True, "execution_result": result}
            except Exception as e:
                logger.warning(f"Auto-execute failed for {action_id}: {e}")
                # Action stays approved, user can retry manually

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


# ─── Session Registry API ────────────────────────────────────


class SessionRegisterRequest(BaseModel):
    description: str
    working_directory: str = ""
    tags: list[str] = Field(default_factory=list)
    agent_name: str = ""
    machine_id: str = ""


class SessionUpdateRequest(BaseModel):
    description: str | None = None
    status: str | None = None
    milestone: str | None = None
    commits_delta: int = 0
    stories_delta: int = 0
    decisions_delta: int = 0
    files_changed: list[str] = Field(default_factory=list)


@app.post("/api/sessions/register")
async def register_session(body: SessionRegisterRequest):
    """Register a new Claude Code session. Returns session with ID."""
    session = session_registry.register(
        description=body.description,
        working_directory=body.working_directory,
        tags=body.tags,
        agent_name=body.agent_name,
        machine_id=body.machine_id,
    )
    await manager.broadcast(
        "session_update",
        {
            "event": "registered",
            "session": session.to_dict(),
        },
    )
    return session.to_dict()


@app.post("/api/sessions/{session_id}/update")
async def update_session(session_id: str, body: SessionUpdateRequest):
    """Update an active session with progress info."""
    session = session_registry.update(
        session_id=session_id,
        description=body.description,
        status=body.status,
        milestone=body.milestone,
        commits_delta=body.commits_delta,
        stories_delta=body.stories_delta,
        decisions_delta=body.decisions_delta,
        files_changed=body.files_changed or None,
    )
    if not session:
        return {"error": f"Session {session_id} not found"}
    await manager.broadcast(
        "session_update",
        {
            "event": "updated",
            "session": session.to_dict(),
        },
    )
    return session.to_dict()


@app.post("/api/sessions/{session_id}/close")
async def close_session(session_id: str, body: Dict[str, Any] = {}):
    """Close a session when Claude Code stops."""
    summary = body.get("summary", "")
    session = session_registry.close(session_id, summary=summary)
    if not session:
        return {"error": f"Session {session_id} not found"}
    await manager.broadcast(
        "session_update",
        {
            "event": "closed",
            "session": session.to_dict(),
        },
    )
    return session.to_dict()


@app.post("/api/sessions/{session_id}/heartbeat")
async def heartbeat_session(session_id: str):
    """Touch session timestamp to keep it active."""
    ok = session_registry.heartbeat(session_id)
    if not ok:
        return {"error": f"Session {session_id} not found"}
    return {"status": "ok"}


class AgentUpdateRequest(BaseModel):
    agent_name: str
    task_summary: str = ""
    chain_id: str = ""
    chain_position: int = 0


@app.post("/api/sessions/{session_id}/agent")
async def set_session_agent(session_id: str, body: AgentUpdateRequest):
    """Update which agent is active in a session. Records agent transitions."""
    session = session_registry.set_agent(
        session_id=session_id,
        agent_name=body.agent_name,
        task_summary=body.task_summary,
        chain_id=body.chain_id,
        chain_position=body.chain_position,
    )
    if not session:
        return {"error": f"Session {session_id} not found"}
    await manager.broadcast(
        "agent_update",
        {
            "event": "agent_changed",
            "session_id": session_id,
            "agent_name": session.agent_name,
            "agent_tier": session.agent_tier,
            "agent_color": session.agent_color,
            "machine_id": session.machine_id,
            "current_task": session.current_task,
            "chain_id": session.chain_id,
            "chain_position": session.chain_position,
        },
    )
    return session.to_dict()


@app.get("/api/sessions")
async def list_sessions(active_only: bool = False, limit: int = 50):
    """List sessions."""
    if active_only:
        return session_registry.list_active()
    return session_registry.list_all(limit=limit)


@app.get("/api/sessions/summary")
async def sessions_summary():
    """Return session summary stats for the dashboard."""
    return session_registry.summary()


# ─── Workstream API ──────────────────────────────────────────


class WorkstreamCreateRequest(BaseModel):
    name: str
    type: str = "product"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    story_ids: list[str] = Field(default_factory=list)
    color: str = ""


@app.get("/api/workstreams")
async def list_workstreams(active_only: bool = True):
    if active_only:
        return workstream_registry.list_active()
    return workstream_registry.list_all()


@app.get("/api/workstreams/summary")
async def workstreams_summary():
    return workstream_registry.summary()


@app.post("/api/workstreams")
async def create_workstream(body: WorkstreamCreateRequest):
    ws = workstream_registry.create(
        name=body.name,
        type=body.type,
        description=body.description,
        tags=body.tags,
        story_ids=body.story_ids,
        color=body.color,
    )
    await manager.broadcast(
        "workstream_update", {"event": "created", "workstream": ws.to_dict()}
    )
    return ws.to_dict()


@app.put("/api/workstreams/{workstream_id}")
async def update_workstream(workstream_id: str, body: Dict[str, Any]):
    ws = workstream_registry.update(workstream_id, **body)
    if not ws:
        return {"error": f"Workstream {workstream_id} not found"}
    await manager.broadcast(
        "workstream_update", {"event": "updated", "workstream": ws.to_dict()}
    )
    return ws.to_dict()


@app.delete("/api/workstreams/{workstream_id}")
async def delete_workstream(workstream_id: str):
    deleted = workstream_registry.delete(workstream_id)
    if deleted:
        await manager.broadcast(
            "workstream_update", {"event": "deleted", "id": workstream_id}
        )
    return {"deleted": deleted}


@app.post("/api/workstreams/{workstream_id}/attach")
async def attach_session_to_workstream(workstream_id: str, body: Dict[str, Any] = {}):
    session_id = body.get("session_id", "")
    description = body.get("description", "")
    ok = workstream_registry.attach_session(workstream_id, session_id, description)
    if not ok:
        return {"error": f"Workstream {workstream_id} not found"}
    ws = workstream_registry.get(workstream_id)
    await manager.broadcast(
        "workstream_update", {"event": "session_attached", "workstream": ws.to_dict()}
    )
    return ws.to_dict()


@app.post("/api/workstreams/{workstream_id}/stories")
async def add_stories_to_workstream(workstream_id: str, body: Dict[str, Any]):
    story_ids = body.get("story_ids", [])
    ok = workstream_registry.add_stories(workstream_id, story_ids)
    if not ok:
        return {"error": f"Workstream {workstream_id} not found"}
    ws = workstream_registry.get(workstream_id)
    return ws.to_dict()


@app.get("/api/workstreams/suggest")
async def suggest_workstream(
    directory: str = "", branch: str = "", description: str = ""
):
    ws = workstream_registry.suggest_workstream(directory, branch, description)
    if ws:
        return {"suggested": ws.to_dict()}
    return {"suggested": None}


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
        context = VOICE_MODE_INSTRUCTION + (
            f"\n## Recent Conversation\n{context}" if context else ""
        )

    chat_history.append(
        {"role": "user", "content": msg.message, "ts": now, "mode": msg.mode}
    )

    reply = "No response"
    model = "unknown"
    sources = 0
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=10.0)
        ) as client:
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

    chat_history.append(
        {"role": "aiia", "content": reply, "ts": datetime.now(timezone.utc).isoformat()}
    )
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
                        "num_ctx": 32768,
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


@app.get("/api/work/context-v1")
async def work_context_v1():
    """DEPRECATED: Use /api/work/context (v2) instead.

    Original work context endpoint. Kept for backward compat.
    """
    import subprocess
    from datetime import date as date_type

    today_str = date_type.today().isoformat()

    # Today's report (git analysis)
    existing = load_report(today_str)
    if not existing:
        existing = generate_report(date=today_str, repo_dir=REPO_PATH)
        save_report(existing)

    # Roadmap stories
    stories = _roadmap.list()

    # Pipeline deals
    deals = _pipeline.list()

    # Git uncommitted files
    uncommitted = []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--no-renames"],
            capture_output=True,
            text=True,
            cwd=REPO_PATH,
            timeout=10,
        )
        for line in result.stdout.splitlines()[:25]:
            if len(line) < 4:
                continue
            idx = line[0]
            wt = line[1]
            fname = line[3:].strip()
            if idx != " " and idx != "?":
                uncommitted.append({"file": fname, "status": "staged"})
            elif wt != " " or idx == "?":
                uncommitted.append({"file": fname, "status": "modified"})
    except Exception:
        pass

    # Recent commits (last 20)
    recent_commits = []
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "-20",
                "--format=%H|%s|%an",
                "--no-merges",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_PATH,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            sha, subject, author = parts
            # Parse conventional commit type and product
            ctype = "other"
            product = ""
            if "(" in subject and ")" in subject:
                prefix = subject.split(")")[0]
                if "(" in prefix:
                    ctype = prefix.split("(")[0].strip()
                    product = prefix.split("(")[1].strip()
            elif ":" in subject:
                ctype = subject.split(":")[0].strip()
            recent_commits.append(
                {
                    "hash": sha[:8],
                    "subject": subject,
                    "author": author,
                    "type": ctype,
                    "product": product,
                }
            )
    except Exception:
        pass

    # Weekly heatmap (commits per day, last 7 days)
    weekly_heatmap = {}
    try:
        result = subprocess.run(
            ["git", "log", "--since=7 days ago", "--format=%ai", "--no-merges"],
            capture_output=True,
            text=True,
            cwd=REPO_PATH,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            day = line.strip()[:10]
            weekly_heatmap[day] = weekly_heatmap.get(day, 0) + 1
    except Exception:
        pass

    return {
        "today": {
            "date": today_str,
            "summary": existing.get("summary", {}),
            "products": existing.get("products", {}),
            "categories": existing.get("categories", {}),
        },
        "interval": {},
        "pipeline": deals,
        "uncommitted": uncommitted,
        "weekly_heatmap": weekly_heatmap,
        "roadmap": stories,
        "recent_commits": recent_commits,
    }


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


async def _index_story_in_aiia(story: Dict[str, Any]) -> None:
    """Fire-and-forget: index a story in AIIA's ChromaDB."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                "http://localhost:8100/v1/aiia/index-story",
                json={"story": story},
            )
    except Exception as e:
        logger.debug(f"Story index fire-and-forget failed: {e}")


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
        # Index in ChromaDB for semantic search
        asyncio.create_task(_index_story_in_aiia(story))
        return {"story": story}
    except (KeyError, ValueError) as e:
        return {"error": str(e)}


@app.put("/api/roadmap/{story_id}")
async def roadmap_update(story_id: str, body: Dict[str, Any]):
    try:
        story = _roadmap.update(story_id, **body)
        if story is None:
            return {"error": "Story not found"}
        # Re-index in ChromaDB
        asyncio.create_task(_index_story_in_aiia(story))
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
                result["wip"] = resp.json().get(
                    "memories", resp.json() if isinstance(resp.json(), list) else []
                )
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
                result["recent_sessions"] = (
                    memories[-3:] if len(memories) > 3 else memories
                )
            else:
                result["recent_sessions"] = {
                    "error": f"AIIA returned {resp.status_code}"
                }
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
        result["recent_commits"] = {
            "total": 0,
            "by_product": {},
            "commits": [],
            "error": str(e),
        }

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
            return {
                "status": "missing",
                "age_hours": None,
                "path": str(path),
                "error": str(e),
            }

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
            report_files = sorted(
                reports_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if report_files:
                nightly["daily_report"] = _check_job_file(report_files[0])
            else:
                nightly["daily_report"] = {
                    "status": "missing",
                    "age_hours": None,
                    "path": str(reports_dir),
                }
        else:
            nightly["daily_report"] = {
                "status": "missing",
                "age_hours": None,
                "path": str(reports_dir),
            }
    except Exception as e:
        nightly["daily_report"] = {
            "status": "missing",
            "age_hours": None,
            "error": str(e),
        }

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
        active = [
            s for s in all_stories if s.get("status") in ("active", "in_progress")
        ]
        blocked = [s for s in all_stories if s.get("status") == "blocked"]
        backlog = [
            s
            for s in all_stories
            if s.get("status") not in ("active", "in_progress", "blocked", "done")
        ]
        result["stories"] = {
            "active": active,
            "blocked": blocked,
            "backlog_count": len(backlog),
        }
    except Exception as e:
        result["stories"] = {
            "active": [],
            "blocked": [],
            "backlog_count": 0,
            "error": str(e),
        }

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


# ─── Attention Aggregator (Dashboard Inbox) ──────────────────


@app.get("/api/attention")
async def get_attention_items():
    """Aggregate everything that needs Eric's attention into one response.

    This is the dashboard's inbox view. One call, all actionable items.
    """
    items = {
        "pending_actions": [],
        "failed_executions": [],
        "unscored_stories": [],
        "stale_sessions": [],
        "failed_tasks": [],
        "total_attention": 0,
    }

    # 1. Pending actions
    try:
        pending = action_queue.list_actions(status="pending", limit=20)
        items["pending_actions"] = [
            {
                "id": a["id"],
                "type": a["type"],
                "severity": a["severity"],
                "title": a["title"],
                "description": (a.get("description") or "")[:200],
                "files_affected": len(a.get("files_affected", [])),
                "created_at": a.get("created_at"),
            }
            for a in pending
        ]
    except Exception as e:
        logger.warning(f"Attention: actions check failed: {e}")

    # 2. Failed executions
    try:
        if _execution_engine:
            status = await _execution_engine.get_status()
            recent = status.get("recent", [])
            items["failed_executions"] = [
                {
                    "id": r.get("id"),
                    "action_id": r.get("action_id"),
                    "action_type": r.get("action_type"),
                    "error": r.get("error", "")[:200],
                    "completed_at": r.get("completed_at"),
                }
                for r in recent
                if r.get("status") == "failed"
            ][:5]
    except Exception as e:
        logger.warning(f"Attention: execution check failed: {e}")

    # 3. Unscored backlog stories (no priority or P3+)
    try:
        stories = _roadmap.list_stories() if _roadmap else []
        items["unscored_stories"] = [
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "product": s.get("product"),
                "status": s.get("status"),
            }
            for s in stories
            if s.get("status") == "backlog"
            and (not s.get("priority") or s.get("priority", 99) > 2)
        ][:10]
    except Exception as e:
        logger.warning(f"Attention: stories check failed: {e}")

    # 4. Stale sessions (active but idle > 2 hours)
    try:
        if session_registry:
            all_sessions = session_registry.list_sessions(active_only=True)
            now = datetime.now(timezone.utc)
            for sess in all_sessions:
                updated = sess.get("updated_at") or sess.get("started_at", "")
                if updated:
                    try:
                        ts = datetime.fromisoformat(updated)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        hours_idle = (now - ts).total_seconds() / 3600
                        if hours_idle > 2:
                            items["stale_sessions"].append(
                                {
                                    "id": sess.get("id"),
                                    "description": sess.get("description"),
                                    "hours_idle": round(hours_idle, 1),
                                }
                            )
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        logger.warning(f"Attention: sessions check failed: {e}")

    # 5. Failed background tasks
    try:
        for task in task_runner.tasks:
            if task.get("status") in ("error", "failed"):
                items["failed_tasks"].append(
                    {
                        "id": task.get("id"),
                        "name": task.get("name"),
                        "last_result": (task.get("last_result") or "")[:200],
                        "last_run": task.get("last_run"),
                    }
                )
    except Exception as e:
        logger.warning(f"Attention: tasks check failed: {e}")

    # Total attention count
    items["total_attention"] = (
        len(items["pending_actions"])
        + len(items["failed_executions"])
        + len(items["unscored_stories"])
        + len(items["stale_sessions"])
        + len(items["failed_tasks"])
    )

    return items


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
    result = await _execution_engine.execute_story(story_id, auto_approve=auto_approve)
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

    # Index all stories in AIIA's ChromaDB (fire-and-forget after delay)
    async def _index_all_stories():
        await asyncio.sleep(10)  # Wait for Brain API to be ready
        try:
            stories = _roadmap.list()
            if stories:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        "http://localhost:8100/v1/aiia/index-stories",
                        json={"stories": stories},
                    )
                    if resp.status_code == 200:
                        logger.info(f"Indexed {len(stories)} stories in AIIA ChromaDB")
        except Exception as e:
            logger.debug(f"Story indexing on startup failed: {e}")

    asyncio.create_task(_index_all_stories())

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
