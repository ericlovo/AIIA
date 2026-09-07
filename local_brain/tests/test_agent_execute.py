"""Empty model output is a failed run at the shared `_execute_agent` boundary."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.assignment_registry import AssignmentRegistry


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.posts: list[dict[str, Any]] = []

    def __call__(self, *args: object, **kwargs: object) -> _FakeAsyncClient:
        return self

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, headers: object = None, json: object = None) -> _FakeResponse:
        self.posts.append({"url": url, "json": json})
        return self._response


def _studio(tmp_path, monkeypatch, content: str | None = ""):
    from local_brain.command_center import server as cc

    agents = AgentRegistry(tmp_path / "agents.json")
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    events: list[tuple[str, str, dict[str, Any]]] = []
    fake = _FakeAsyncClient(
        _FakeResponse(200, {"content": content, "model": "qwen3:8b", "latency_ms": 12})
    )

    async def capture(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))

    monkeypatch.setattr(cc, "agent_registry", agents)
    monkeypatch.setattr(cc, "assignment_registry", assignments)
    monkeypatch.setattr(cc, "broadcast_studio_event", capture)
    monkeypatch.setattr(cc.httpx, "AsyncClient", fake)
    return cc, agents, assignments, events, fake


def _create_agent(agents: AgentRegistry, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Signal Officer",
        "mission": "Report current CI state.",
        "persona": "Evidence first.",
        "skills": ["Analysis"],
    }
    payload.update(overrides)
    return agents.create(**payload)


async def test_manual_empty_output_is_not_success(tmp_path, monkeypatch):
    cc, agents, _assignments, events, _fake = _studio(tmp_path, monkeypatch, content="   ")
    agent = _create_agent(agents)

    with pytest.raises(HTTPException) as exc:
        await cc._execute_agent(agent["id"], "Inspect current checks.")

    updated = agents.get(agent["id"])
    assert exc.value.status_code == 502
    assert exc.value.detail == "empty_agent_result"
    assert updated["status"] == "error"
    assert updated["last_error"] == "empty_agent_result"
    assert updated["runs"][0]["trigger"] == "manual"
    assert updated["runs"][0]["error"] == "empty_agent_result"
    assert any(
        entity == "agent" and event == "failed" and item["id"] == agent["id"]
        for entity, event, item in events
    )
    assert not any(event == "completed" for _entity, event, _item in events)


async def test_interval_empty_output_is_scheduler_safe(tmp_path, monkeypatch):
    cc, agents, _assignments, events, _fake = _studio(tmp_path, monkeypatch, content="")
    agent = _create_agent(agents, loop_enabled=True, loop_task="Poll CI.", loop_interval_minutes=15)

    try:
        await cc._execute_agent(agent["id"], agent["loop_task"], loop_run=True)
    except HTTPException as exc:
        assert exc.detail == "empty_agent_result"
    else:
        raise AssertionError("interval empty output must fail at the shared boundary")

    updated = agents.get(agent["id"])
    assert updated["status"] == "error"
    assert updated["last_error"] == "empty_agent_result"
    assert updated["runs"][0]["trigger"] == "interval"
    assert any(entity == "agent" and event == "failed" for entity, event, _item in events)
    assert updated["status"] != "running"


async def test_assignment_empty_output_fails_agent_and_assignment(tmp_path, monkeypatch):
    cc, agents, assignments, events, _fake = _studio(tmp_path, monkeypatch, content=None)
    agent = _create_agent(agents)
    assignment = assignments.create_assignment(
        title="Map the authorization surface",
        objective="Return the five highest-leverage integration points.",
        agent_id=agent["id"],
        priority="high",
    )

    with pytest.raises(HTTPException) as exc:
        await cc.run_assignment(assignment["id"])

    failed_assignment = assignments.get_assignment(assignment["id"])
    failed_agent = agents.get(agent["id"])
    assert exc.value.status_code == 502
    assert exc.value.detail == "empty_agent_result"
    assert failed_agent["status"] == "error"
    assert failed_agent["last_error"] == "empty_agent_result"
    assert failed_agent["runs"][0]["trigger"] == "assignment"
    assert failed_assignment["status"] == "failed"
    assert failed_assignment["error"] == "empty_agent_result"
    assert any(entity == "agent" and event == "failed" for entity, event, _item in events)
    assert any(entity == "assignment" and event == "failed" for entity, event, _item in events)


async def test_nonempty_output_still_completes(tmp_path, monkeypatch):
    cc, agents, _assignments, events, _fake = _studio(tmp_path, monkeypatch, content="GREEN")
    agent = _create_agent(agents)

    result = await cc._execute_agent(agent["id"], "Inspect current checks.")

    updated = agents.get(agent["id"])
    assert result["agent"]["last_result"] == "GREEN"
    assert updated["status"] == "idle"
    assert updated["last_error"] == ""
    assert any(event == "completed" for _entity, event, _item in events)


def test_finish_run_treats_blank_success_as_error(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    agent = _create_agent(registry)

    updated = registry.finish_run(agent["id"], "Inspect current checks.", result="  ")

    assert updated["status"] == "error"
    assert updated["last_error"] == "empty_agent_result"
    assert updated["runs"][0]["error"] == "empty_agent_result"
