"""PATCH /api/agents/{id} is a real partial edit, and no write path enables an empty loop."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    events: list[tuple[str, str, dict[str, Any]]] = []

    async def capture(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))

    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "broadcast_studio_event", capture)
    return server, registry, events


def _call(server, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(go())


def _agent(registry: AgentRegistry, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Signal Officer",
        "mission": "Report current CI state.",
        "persona": "Evidence first.",
        "skills": ["Analysis"],
        "temperature": 0.2,
        "max_tokens": 900,
        "loop_task": "Poll CI.",
        "suite": "mindmoor",
    }
    payload.update(overrides)
    return registry.create(**payload)


def test_patch_changes_only_supplied_fields_and_persists(studio):
    server, registry, events = studio
    agent = _agent(registry)
    before = dict(agent)

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"temperature": 0.9})

    assert response.status_code == 200
    record = response.json()["agent"]
    assert record["temperature"] == 0.9
    unchanged = {k: v for k, v in before.items() if k not in {"temperature", "updated_at"}}
    assert {k: record[k] for k in unchanged} == unchanged
    restored = AgentRegistry(registry.data_file).get(agent["id"])
    assert restored["temperature"] == 0.9
    assert {k: restored[k] for k in unchanged} == unchanged
    assert events == [("agent", "updated", record)]


def test_registry_update_leaves_omitted_fields_untouched_on_disk(tmp_path):
    data_file = tmp_path / "agents.json"
    registry = AgentRegistry(data_file)
    agent = _agent(registry, tools=["Local memory"], loop_enabled=True, loop_interval_minutes=30)

    registry.update(agent["id"], max_tokens=1_500)

    restored = AgentRegistry(data_file).get(agent["id"])
    assert restored["max_tokens"] == 1_500
    for field in (
        "name",
        "mission",
        "persona",
        "skills",
        "tools",
        "temperature",
        "loop_enabled",
        "loop_interval_minutes",
        "loop_task",
        "suite",
        "memory_namespace",
    ):
        assert restored[field] == agent[field], field
    assert restored["loop_enabled"] is True
    assert restored["tools"] == ["Local memory"]


@pytest.mark.parametrize(
    "body",
    [
        {"temperature": 0.5, "colour": "red"},
        {"id": "hijack"},
        {"status": "idle"},
        {"temperature": 1.5},
        {"max_tokens": 50},
        {"name": ""},
        {"name": None},
        {"loop_enabled": None},
    ],
)
def test_patch_rejects_unknown_null_and_out_of_range_fields(studio, body):
    server, registry, events = studio
    agent = _agent(registry)
    disk = registry.data_file.read_bytes()

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json=body)

    assert response.status_code == 422
    assert registry.data_file.read_bytes() == disk
    assert events == []


@pytest.mark.parametrize("kwargs", [{"json": {}}, {}])
def test_patch_empty_body_is_refused(studio, kwargs):
    server, registry, events = studio
    agent = _agent(registry)

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", **kwargs)

    assert response.status_code == 422
    assert response.json()["detail"] == "empty_patch"
    assert events == []


def test_patch_unknown_agent_is_404(studio):
    server, _registry, events = studio
    response = _call(server, "PATCH", "/api/agents/missing", json={"temperature": 0.5})
    assert response.status_code == 404
    assert response.json()["detail"] == "agent_not_found"
    assert events == []


def test_patch_agent_deleted_during_model_check_is_404(studio, monkeypatch):
    server, registry, events = studio
    agent = _agent(registry)

    async def delete_while_checking() -> list[dict[str, Any]]:
        registry.delete(agent["id"])
        return [{"id": "qwen3:8b"}]

    monkeypatch.setattr(server, "_installed_chat_models", delete_while_checking)

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "qwen3:8b"})

    assert response.status_code == 404
    assert response.json()["detail"] == "agent_not_found"
    assert events == []


def test_patch_refuses_loop_without_task_on_merged_result(studio):
    server, registry, events = studio
    blank = _agent(registry, name="Blank", loop_task="")
    looping = _agent(registry, name="Looping", loop_enabled=True)

    enable = _call(server, "PATCH", f"/api/agents/{blank['id']}", json={"loop_enabled": True})
    clear = _call(server, "PATCH", f"/api/agents/{looping['id']}", json={"loop_task": "  "})

    for response in (enable, clear):
        assert response.status_code == 422
        assert response.json()["detail"] == "loop_task_required"
    restored = AgentRegistry(registry.data_file)
    assert restored.get(blank["id"])["loop_enabled"] is False
    assert restored.get(looping["id"])["loop_task"] == "Poll CI."
    assert events == []

    both = _call(
        server,
        "PATCH",
        f"/api/agents/{blank['id']}",
        json={"loop_enabled": True, "loop_task": "Watch deploys."},
    )
    assert both.status_code == 200
    assert both.json()["agent"]["loop_enabled"] is True


def test_put_and_create_refuse_loop_without_task(studio):
    server, registry, events = studio
    agent = _agent(registry, loop_task="")
    body = {"name": "Signal Officer", "mission": "Report.", "loop_enabled": True}

    created = _call(server, "POST", "/api/agents", json=body)
    replaced = _call(server, "PUT", f"/api/agents/{agent['id']}", json=body)

    for response in (created, replaced):
        assert response.status_code == 422
        assert response.json()["detail"] == "loop_task_required"
    assert len(registry.agents) == 1
    assert AgentRegistry(registry.data_file).get(agent["id"])["loop_enabled"] is False
    assert events == []


def test_registry_create_refuses_loop_without_task(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    with pytest.raises(ValueError, match="loop_task_required"):
        registry.create("Loop", "Mission", "Persona", [], loop_enabled=True, loop_task=" ")
    assert registry.agents == []


def test_patch_validates_repository_against_merged_record(studio):
    server, registry, events = studio
    agent = _agent(registry)

    response = _call(
        server, "PATCH", f"/api/agents/{agent['id']}", json={"tools": ["Repository read"]}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "mounted_repository_required"
    assert AgentRegistry(registry.data_file).get(agent["id"])["tools"] == []
    assert events == []


def test_patch_invalid_suite_maps_registry_error(studio):
    server, registry, _events = studio
    agent = _agent(registry)
    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"suite": "Bad Suite"})
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_suite"


def test_patch_running_agent_is_allowed(studio):
    server, registry, _events = studio
    agent = _agent(registry)
    registry.set_running(agent["id"])

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"max_tokens": 400})

    assert response.status_code == 200
    assert response.json()["agent"]["status"] == "running"
    assert response.json()["agent"]["max_tokens"] == 400
