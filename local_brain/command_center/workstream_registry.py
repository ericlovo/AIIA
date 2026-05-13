"""Workstream Registry — Persistent, named work contexts that Claude sessions attach to.

A workstream is NOT a terminal session. It's a named body of work that persists
across days, sessions, and machines. Examples: "DefaultApp Product Owner",
"Security & Infrastructure", "CK Marketing Product Owner".

Claude terminals attach to workstreams when they start. AIIA suggests the best
workstream based on the directory, branch, and recent git history. The developer
can reassign or create new workstreams at any time.

Workstreams own:
  - A description and type (client, ops, maintenance, product, meta)
  - Assigned roadmap story IDs
  - A history of Claude sessions that worked on them
  - A "next action" recommendation
  - Tags for filtering
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("aiia.workstreams")

DEFAULT_DATA_DIR = os.path.expanduser("~/.aiia/eq_data")

WORKSTREAM_TYPES = ("client", "ops", "maintenance", "product", "meta")


@dataclass
class Workstream:
    id: str
    name: str
    type: str = "product"  # client, ops, maintenance, product, meta
    description: str = ""
    status: str = "active"  # active, paused, completed
    story_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    next_action: str = ""
    session_history: list[dict[str, str]] = field(default_factory=list)
    active_session_id: str = ""
    branch: str = ""
    ci_status: str = ""  # passing, failing, unknown
    created_at: str = ""
    updated_at: str = ""
    color: str = ""  # for dashboard display

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default color palette for workstream types
TYPE_COLORS = {
    "client": "#3b82f6",  # blue
    "ops": "#f59e0b",  # amber
    "maintenance": "#6b7280",  # gray
    "product": "#8b5cf6",  # violet
    "meta": "#06b6d4",  # cyan
}


class WorkstreamRegistry:
    """Manages persistent workstreams with JSON backing."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self._data_dir = data_dir
        self._file = os.path.join(data_dir, "workstreams.json")
        self._workstreams: dict[str, Workstream] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file) as f:
                data = json.load(f)
            for wid, wdata in data.items():
                self._workstreams[wid] = Workstream(**wdata)
            logger.info(f"Loaded {len(self._workstreams)} workstreams")
        except Exception as e:
            logger.warning(f"Failed to load workstreams: {e}")

    def _save(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        try:
            with open(self._file, "w") as f:
                json.dump(
                    {wid: w.to_dict() for wid, w in self._workstreams.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Failed to save workstreams: {e}")

    def create(
        self,
        name: str,
        type: str = "product",
        description: str = "",
        tags: list[str] | None = None,
        story_ids: list[str] | None = None,
        color: str = "",
    ) -> Workstream:
        now = datetime.now(timezone.utc).isoformat()
        wid = uuid.uuid4().hex[:10]
        ws = Workstream(
            id=wid,
            name=name,
            type=type if type in WORKSTREAM_TYPES else "product",
            description=description,
            status="active",
            story_ids=story_ids or [],
            tags=tags or [],
            color=color or TYPE_COLORS.get(type, "#6b7280"),
            created_at=now,
            updated_at=now,
        )
        self._workstreams[wid] = ws
        self._save()
        logger.info(f"Workstream created: {wid} — {name}")
        return ws

    def update(self, workstream_id: str, **kwargs: Any) -> Workstream | None:
        ws = self._workstreams.get(workstream_id)
        if not ws:
            return None
        for key, val in kwargs.items():
            if hasattr(ws, key) and val is not None:
                setattr(ws, key, val)
        ws.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return ws

    def attach_session(self, workstream_id: str, session_id: str, description: str = "") -> bool:
        ws = self._workstreams.get(workstream_id)
        if not ws:
            return False
        ws.active_session_id = session_id
        ws.session_history.append(
            {
                "session_id": session_id,
                "attached_at": datetime.now(timezone.utc).isoformat(),
                "description": description,
            }
        )
        # Keep last 20 session entries
        if len(ws.session_history) > 20:
            ws.session_history = ws.session_history[-20:]
        ws.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def detach_session(self, workstream_id: str) -> bool:
        ws = self._workstreams.get(workstream_id)
        if not ws:
            return False
        ws.active_session_id = ""
        ws.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def add_stories(self, workstream_id: str, story_ids: list[str]) -> bool:
        ws = self._workstreams.get(workstream_id)
        if not ws:
            return False
        for sid in story_ids:
            if sid not in ws.story_ids:
                ws.story_ids.append(sid)
        ws.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def get(self, workstream_id: str) -> Workstream | None:
        return self._workstreams.get(workstream_id)

    def find_by_name(self, name: str) -> Workstream | None:
        name_lower = name.lower()
        for ws in self._workstreams.values():
            if ws.name.lower() == name_lower:
                return ws
        return None

    def list_active(self) -> list[dict]:
        active = [
            w.to_dict() for w in self._workstreams.values() if w.status in ("active", "paused")
        ]
        active.sort(key=lambda w: w["updated_at"], reverse=True)
        return active

    def list_all(self) -> list[dict]:
        all_ws = [w.to_dict() for w in self._workstreams.values()]
        all_ws.sort(key=lambda w: w["updated_at"], reverse=True)
        return all_ws

    def suggest_workstream(
        self, directory: str = "", branch: str = "", description: str = ""
    ) -> Workstream | None:
        """Suggest the best workstream for a new session based on context signals."""
        context = f"{directory} {branch} {description}".lower()
        best_score = 0
        best_ws = None

        for ws in self._workstreams.values():
            if ws.status != "active":
                continue
            score = 0
            # Match on tags
            for tag in ws.tags:
                if tag.lower() in context:
                    score += 3
            # Match on name
            for word in ws.name.lower().split():
                if len(word) > 2 and word in context:
                    score += 2
            # Match on type keywords
            if ws.type == "client" and any(t in context for t in ws.tags):
                score += 1
            if score > best_score:
                best_score = score
                best_ws = ws

        return best_ws if best_score >= 2 else None

    def summary(self) -> dict[str, Any]:
        active = [w for w in self._workstreams.values() if w.status == "active"]
        paused = [w for w in self._workstreams.values() if w.status == "paused"]
        with_sessions = [w for w in active if w.active_session_id]
        return {
            "active_count": len(active),
            "paused_count": len(paused),
            "agents_working": len(with_sessions),
            "total_stories": sum(len(w.story_ids) for w in active),
            "workstreams": [w.to_dict() for w in active],
        }

    def delete(self, workstream_id: str) -> bool:
        if workstream_id in self._workstreams:
            del self._workstreams[workstream_id]
            self._save()
            return True
        return False
