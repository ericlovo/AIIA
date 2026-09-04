"""Persistent, supervised definitions for local AIIA agents."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("aiia.agents")

AGENT_DATA_FILE = Path(__file__).parent / "agent_data.json"
MAX_AGENTS = 40
MAX_RUNS = 12
MAX_TOOLS = 8


class AgentRegistry:
    """Store agent definitions and their bounded local run history."""

    def __init__(self):
        self.agents: list[dict[str, Any]] = []
        self.load()

    def list(self) -> list[dict[str, Any]]:
        return sorted(self.agents, key=lambda agent: agent["updated_at"], reverse=True)

    def get(self, agent_id: str) -> dict[str, Any] | None:
        return next((agent for agent in self.agents if agent["id"] == agent_id), None)

    def create(
        self,
        name: str,
        mission: str,
        persona: str,
        skills: list[str],
        tools: list[str] | None = None,
        repo_id: str = "",
        temperature: float = 0.35,
        max_tokens: int = 1200,
        loop_enabled: bool = False,
        loop_interval_minutes: int = 60,
        loop_task: str = "",
        loop_max_runs_per_day: int = 4,
    ) -> dict[str, Any]:
        if len(self.agents) >= MAX_AGENTS:
            raise ValueError("agent_limit_reached")
        now = datetime.now(timezone.utc).isoformat()
        agent = {
            "id": uuid.uuid4().hex[:12],
            "name": name.strip(),
            "mission": mission.strip(),
            "persona": persona.strip(),
            "skills": self._skills(skills),
            "tools": self._tools(tools),
            "repo_id": str(repo_id).strip()[:80],
            "temperature": self._temperature(temperature),
            "max_tokens": self._max_tokens(max_tokens),
            "loop_enabled": bool(loop_enabled),
            "loop_interval_minutes": self._interval(loop_interval_minutes),
            "loop_task": str(loop_task).strip()[:8_000],
            "loop_max_runs_per_day": self._daily_limit(loop_max_runs_per_day),
            "loop_runs_today": 0,
            "loop_day": "",
            "status": "idle",
            "last_run_at": None,
            "last_result": "",
            "last_error": "",
            "runs": [],
            "created_at": now,
            "updated_at": now,
        }
        self.agents.append(agent)
        self.save()
        return agent

    def update(self, agent_id: str, **changes: Any) -> dict[str, Any] | None:
        agent = self.get(agent_id)
        if not agent:
            return None
        for field in ("name", "mission", "persona"):
            if field in changes:
                agent[field] = str(changes[field]).strip()
        if "skills" in changes:
            agent["skills"] = self._skills(changes["skills"])
        if "tools" in changes:
            agent["tools"] = self._tools(changes["tools"])
        if "repo_id" in changes:
            agent["repo_id"] = str(changes["repo_id"]).strip()[:80]
        if "temperature" in changes:
            agent["temperature"] = self._temperature(changes["temperature"])
        if "max_tokens" in changes:
            agent["max_tokens"] = self._max_tokens(changes["max_tokens"])
        if "loop_enabled" in changes:
            agent["loop_enabled"] = bool(changes["loop_enabled"])
        if "loop_interval_minutes" in changes:
            agent["loop_interval_minutes"] = self._interval(changes["loop_interval_minutes"])
        if "loop_task" in changes:
            agent["loop_task"] = str(changes["loop_task"]).strip()[:8_000]
        if "loop_max_runs_per_day" in changes:
            agent["loop_max_runs_per_day"] = self._daily_limit(changes["loop_max_runs_per_day"])
        agent["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return agent

    def due_loop(self) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        for agent in self.agents:
            if not agent.get("loop_enabled") or not agent.get("loop_task"):
                continue
            if agent.get("status") == "running":
                continue
            if agent.get("loop_day") != today:
                agent["loop_day"] = today
                agent["loop_runs_today"] = 0
            if agent.get("loop_runs_today", 0) >= agent.get("loop_max_runs_per_day", 4):
                continue
            last_run = agent.get("last_run_at")
            if not last_run:
                return agent
            try:
                elapsed = now - datetime.fromisoformat(last_run)
            except ValueError:
                return agent
            if elapsed >= timedelta(minutes=agent.get("loop_interval_minutes", 60)):
                return agent
        self.save()
        return None

    def record_loop_run(self, agent_id: str) -> None:
        agent = self.get(agent_id)
        if not agent:
            return
        agent["loop_day"] = datetime.now(timezone.utc).date().isoformat()
        agent["loop_runs_today"] = agent.get("loop_runs_today", 0) + 1
        self.save()

    def set_running(self, agent_id: str) -> dict[str, Any] | None:
        agent = self.get(agent_id)
        if not agent:
            return None
        agent["status"] = "running"
        agent["last_error"] = ""
        agent["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        return agent

    def finish_run(
        self, agent_id: str, task: str, result: str = "", error: str = ""
    ) -> dict[str, Any] | None:
        agent = self.get(agent_id)
        if not agent:
            return None
        now = datetime.now(timezone.utc).isoformat()
        agent["status"] = "error" if error else "idle"
        agent["last_run_at"] = now
        agent["last_result"] = result
        agent["last_error"] = error
        agent["updated_at"] = now
        agent["runs"] = (
            [{"task": task, "result": result, "error": error, "at": now}] + agent["runs"]
        )[:MAX_RUNS]
        self.save()
        return agent

    def delete(self, agent_id: str) -> bool:
        agent = self.get(agent_id)
        if not agent:
            return False
        self.agents.remove(agent)
        self.save()
        return True

    @staticmethod
    def _skills(skills: Any) -> list[str]:
        if not isinstance(skills, list):
            return []
        return [str(skill).strip()[:80] for skill in skills if str(skill).strip()][:12]

    @staticmethod
    def _tools(tools: Any) -> list[str]:
        if not isinstance(tools, list):
            return []
        return [str(tool).strip()[:80] for tool in tools if str(tool).strip()][:MAX_TOOLS]

    @staticmethod
    def _interval(value: Any) -> int:
        return max(15, min(int(value), 1_440))

    @staticmethod
    def _daily_limit(value: Any) -> int:
        return max(1, min(int(value), 48))

    @staticmethod
    def _temperature(value: Any) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _max_tokens(value: Any) -> int:
        return max(128, min(int(value), 2_000))

    def save(self) -> None:
        try:
            AGENT_DATA_FILE.write_text(json.dumps({"agents": self.agents}, indent=2))
        except OSError as exc:
            logger.error("Could not save agents: %s", exc)

    def load(self) -> None:
        if not AGENT_DATA_FILE.exists():
            return
        try:
            self.agents = json.loads(AGENT_DATA_FILE.read_text()).get("agents", [])[:MAX_AGENTS]
            for agent in self.agents:
                agent.setdefault("tools", [])
                agent.setdefault("repo_id", "")
                agent.setdefault("temperature", 0.35)
                agent.setdefault("max_tokens", 1_200)
                agent.setdefault("loop_enabled", False)
                agent.setdefault("loop_interval_minutes", 60)
                agent.setdefault("loop_task", "")
                agent.setdefault("loop_max_runs_per_day", 4)
                agent.setdefault("loop_runs_today", 0)
                agent.setdefault("loop_day", "")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load agents: %s", exc)
