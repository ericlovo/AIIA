"""Session Registry — Tracks active Claude Code sessions across terminals.

Each Claude Code terminal registers a session on start and updates it
during work. The dashboard shows all active sessions in real time.

Sessions are persisted to JSON so they survive brain restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("aiia.sessions")

DEFAULT_DATA_DIR = os.path.expanduser("~/.aiia/eq_data")


AGENT_TIERS = {
    "SCOUT": "engineering",
    "BUILDER": "engineering",
    "GUARDIAN": "engineering",
    "SMOKE": "engineering",
    "SENTINEL": "engineering",
    "SHIPPER": "operations",
    "WATCHDOG": "operations",
    "PITCH": "operations",
    "STYLIST": "design",
    "CRITIC": "design",
    "SCRIBE": "content",
}

AGENT_COLORS = {
    "SCOUT": "#64d2ff",
    "BUILDER": "#0a84ff",
    "GUARDIAN": "#ff453a",
    "SMOKE": "#30d158",
    "SENTINEL": "#ff9f0a",
    "SHIPPER": "#bf5af2",
    "WATCHDOG": "#ff9f0a",
    "PITCH": "#ffd60a",
    "STYLIST": "#bf5af2",
    "CRITIC": "#ff375f",
    "SCRIBE": "#64d2ff",
}


@dataclass
class Session:
    id: str
    description: str
    working_directory: str = ""
    status: str = "active"  # active, idle, completed
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    commits: int = 0
    stories_captured: int = 0
    decisions_made: int = 0
    files_changed: list[str] = field(default_factory=list)
    milestones: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Agent identity
    agent_name: str = ""
    agent_tier: str = ""
    agent_color: str = ""
    machine_id: str = ""
    current_task: str = ""
    chain_id: str = ""
    chain_position: int = 0
    agent_history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionRegistry:
    """Tracks N concurrent Claude Code sessions with persistence."""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        broadcast_fn: Callable[..., Coroutine] | None = None,
    ):
        self._data_dir = data_dir
        self._file = os.path.join(data_dir, "sessions_active.json")
        self._broadcast = broadcast_fn
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file) as f:
                data = json.load(f)
            for sid, sdata in data.items():
                self._sessions[sid] = Session(**sdata)
            logger.info(f"Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")

    def _save(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        try:
            with open(self._file, "w") as f:
                json.dump(
                    {sid: s.to_dict() for sid, s in self._sessions.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Failed to save sessions: {e}")

    async def _emit(self, event: str, data: dict) -> None:
        if self._broadcast:
            try:
                await self._broadcast(
                    "session_update",
                    {
                        "event": event,
                        **data,
                    },
                )
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    def register(
        self,
        description: str,
        working_directory: str = "",
        tags: list[str] | None = None,
        agent_name: str = "",
        machine_id: str = "",
    ) -> Session:
        """Register a new Claude Code session. Returns the session."""
        now = datetime.now(timezone.utc).isoformat()
        agent_upper = agent_name.upper() if agent_name else ""
        session = Session(
            id=uuid.uuid4().hex[:12],
            description=description,
            working_directory=working_directory,
            status="active",
            started_at=now,
            updated_at=now,
            tags=tags or [],
            agent_name=agent_upper,
            agent_tier=AGENT_TIERS.get(agent_upper, ""),
            agent_color=AGENT_COLORS.get(agent_upper, ""),
            machine_id=machine_id,
        )
        if agent_upper:
            session.agent_history.append(
                {"agent": agent_upper, "started_at": now, "ended_at": "", "verdict": ""}
            )
        self._sessions[session.id] = session
        self._save()
        logger.info(
            f"Session registered: {session.id} — {description}"
            f"{f' [{agent_upper}]' if agent_upper else ''}"
            f"{f' on {machine_id}' if machine_id else ''}"
        )
        return session

    def set_agent(
        self,
        session_id: str,
        agent_name: str,
        task_summary: str = "",
        chain_id: str = "",
        chain_position: int = 0,
    ) -> Session | None:
        """Set or change the active agent in a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        now = datetime.now(timezone.utc).isoformat()
        agent_upper = agent_name.upper()

        # Close previous agent entry
        if session.agent_history:
            last = session.agent_history[-1]
            if not last.get("ended_at"):
                last["ended_at"] = now

        prev_agent = session.agent_name
        session.agent_name = agent_upper
        session.agent_tier = AGENT_TIERS.get(agent_upper, "")
        session.agent_color = AGENT_COLORS.get(agent_upper, "")
        session.current_task = task_summary
        session.chain_id = chain_id
        session.chain_position = chain_position
        session.updated_at = now

        session.agent_history.append(
            {"agent": agent_upper, "started_at": now, "ended_at": "", "verdict": ""}
        )
        # Cap history
        if len(session.agent_history) > 30:
            session.agent_history = session.agent_history[-30:]

        self._save()
        logger.info(
            f"Session {session_id}: agent {prev_agent or 'none'} → {agent_upper}"
            f"{f' (chain {chain_id} step {chain_position})' if chain_id else ''}"
        )
        return session

    def update(
        self,
        session_id: str,
        description: str | None = None,
        status: str | None = None,
        milestone: str | None = None,
        commits_delta: int = 0,
        stories_delta: int = 0,
        decisions_delta: int = 0,
        files_changed: list[str] | None = None,
    ) -> Session | None:
        """Update an existing session. Returns updated session or None."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        now = datetime.now(timezone.utc).isoformat()
        session.updated_at = now

        if description:
            session.description = description
        if status:
            session.status = status
            if status == "completed":
                session.completed_at = now
        if milestone:
            session.milestones.append({"text": milestone, "at": now})
        if commits_delta:
            session.commits += commits_delta
        if stories_delta:
            session.stories_captured += stories_delta
        if decisions_delta:
            session.decisions_made += decisions_delta
        if files_changed:
            for f in files_changed:
                if f not in session.files_changed:
                    session.files_changed.append(f)

        self._save()
        return session

    def close(self, session_id: str, summary: str = "") -> Session | None:
        """Mark a session as completed."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        now = datetime.now(timezone.utc).isoformat()
        session.status = "completed"
        session.completed_at = now
        session.updated_at = now
        if summary:
            session.milestones.append({"text": f"Closed: {summary}", "at": now})

        self._save()
        logger.info(f"Session closed: {session_id}")
        return session

    def heartbeat(self, session_id: str) -> bool:
        """Touch session updated_at. Returns False if session not found."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_active(self) -> list[dict]:
        """Return all active sessions, sorted by most recently updated."""
        active = [s.to_dict() for s in self._sessions.values() if s.status == "active"]
        active.sort(key=lambda s: s["updated_at"], reverse=True)
        return active

    def list_all(self, limit: int = 50) -> list[dict]:
        """Return all sessions, sorted by most recently updated."""
        all_sessions = [s.to_dict() for s in self._sessions.values()]
        all_sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return all_sessions[:limit]

    def cleanup_stale(self, max_idle_hours: int = 4) -> int:
        """Mark sessions as completed if no update in N hours."""
        now = time.time()
        closed = 0
        for session in list(self._sessions.values()):
            if session.status != "active":
                continue
            try:
                updated = datetime.fromisoformat(session.updated_at).timestamp()
                if now - updated > max_idle_hours * 3600:
                    session.status = "completed"
                    session.completed_at = datetime.now(timezone.utc).isoformat()
                    closed += 1
            except (ValueError, TypeError):
                continue
        if closed:
            self._save()
            logger.info(f"Cleaned up {closed} stale sessions")
        return closed

    def summary(self) -> dict[str, Any]:
        """Return summary stats for the dashboard."""
        active = [s for s in self._sessions.values() if s.status == "active"]
        completed = [s for s in self._sessions.values() if s.status == "completed"]

        # Agent breakdown
        agents_active: dict[str, int] = {}
        machines: dict[str, int] = {}
        for s in active:
            if s.agent_name:
                agents_active[s.agent_name] = agents_active.get(s.agent_name, 0) + 1
            if s.machine_id:
                machines[s.machine_id] = machines.get(s.machine_id, 0) + 1

        return {
            "active_count": len(active),
            "completed_count": len(completed),
            "total_commits": sum(s.commits for s in active),
            "total_stories": sum(s.stories_captured for s in active),
            "total_decisions": sum(s.decisions_made for s in active),
            "sessions": [s.to_dict() for s in active],
            "agents_active": agents_active,
            "machines": machines,
        }
