"""PATCH /api/agent-suites/{suite}/agents modulates a whole suite, all or nothing."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry, BulkUpdateRejected

REAL_ASYNC_CLIENT = httpx.AsyncClient
TAGS = {
    "models": [
        {"name": "qwen3:8b", "size": 5_225_388_164, "details": {"family": "qwen3"}},
        {"name": "gemma3:4b", "size": 3_338_801_804, "details": {"family": "gemma3"}},
    ]
}


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    events: list[tuple[str, str, dict[str, Any]]] = []
    ollama = {"tags": TAGS, "calls": 0}

    async def capture(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.port == 11434, f"unexpected upstream call {request.url}"
        ollama["calls"] += 1
        if ollama["tags"] is None:
            raise httpx.ConnectError("ollama down", request=request)
        return httpx.Response(200, json=ollama["tags"])

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "broadcast_studio_event", capture)
    monkeypatch.setattr(server.httpx, "AsyncClient", client_factory)
    ops = [
        registry.create("Ops One", "Watch.", "Terse.", [], suite="ops", loop_task="Poll."),
        registry.create("Ops Two", "Watch.", "Terse.", [], suite="ops", loop_task="Poll."),
        registry.create("Ops Three", "Watch.", "Terse.", [], suite="ops", loop_task="Poll."),
    ]
    outsider = registry.create("Outsider", "Other.", "Terse.", [], loop_task="Poll.")
    return server, registry, events, ollama, ops, outsider


def _call(server, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        async with REAL_ASYNC_CLIENT(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(go())


def test_one_request_changes_every_member(studio):
    server, registry, events, _ollama, ops, outsider = studio
    outsider_before = dict(outsider)
    body = {
        "model": "gemma3:4b",
        "temperature": 0.1,
        "max_tokens": 700,
        "loop_enabled": True,
        "loop_interval_minutes": 30,
        "loop_max_runs_per_day": 6,
    }

    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["suite"] == "ops"
    assert data["count"] == 3
    assert {agent["id"] for agent in data["agents"]} == {agent["id"] for agent in ops}
    restored = AgentRegistry(registry.data_file)
    for agent in ops:
        saved = restored.get(agent["id"])
        assert {key: saved[key] for key in body} == body
        assert saved["name"] == agent["name"]
        assert saved["suite"] == "ops"
    assert restored.get(outsider["id"]) == outsider_before
    assert sorted(item["id"] for _, event, item in events if event == "updated") == sorted(
        agent["id"] for agent in ops
    )
    assert len(events) == 3


def test_one_failing_member_leaves_every_member_unchanged_on_disk(studio):
    server, registry, events, _ollama, ops, _outsider = studio
    registry.update(ops[1]["id"], loop_task="")
    disk = registry.data_file.read_bytes()
    memory = [dict(agent) for agent in registry.agents]

    response = _call(
        server,
        "PATCH",
        "/api/agent-suites/ops/agents",
        json={"loop_enabled": True, "temperature": 0.9},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "suite_patch_rejected",
        "failures": [{"agent_id": ops[1]["id"], "detail": "loop_task_required"}],
    }
    assert registry.data_file.read_bytes() == disk
    assert registry.agents == memory
    assert events == []


def test_unknown_model_rejects_the_suite(studio):
    server, registry, events, _ollama, ops, _outsider = studio
    disk = registry.data_file.read_bytes()

    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json={"model": "nope:1b"})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "suite_patch_rejected"
    assert len(body["failures"]) == 3
    assert {row["agent_id"] for row in body["failures"]} == {agent["id"] for agent in ops}
    assert {row["detail"] for row in body["failures"]} == {"unknown_model"}
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_unreachable_ollama_refuses_a_suite_model_change(studio):
    server, registry, events, ollama, _ops, _outsider = studio
    ollama["tags"] = None
    disk = registry.data_file.read_bytes()
    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json={"model": "qwen3:8b"})
    assert response.status_code == 503
    assert response.json()["detail"] == "models_unavailable"
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_mounted_repository_is_checked_per_member(studio):
    server, registry, events, _ollama, _ops, _outsider = studio
    disk = registry.data_file.read_bytes()
    response = _call(
        server, "PATCH", "/api/agent-suites/ops/agents", json={"tools": ["GitHub read"]}
    )
    assert response.status_code == 422
    assert {row["detail"] for row in response.json()["failures"]} == {"mounted_repository_required"}
    assert registry.data_file.read_bytes() == disk
    assert events == []


@pytest.mark.parametrize(
    "body",
    [
        {"name": "Renamed"},
        {"mission": "New mission"},
        {"suite": "other"},
        {"suite": "ops", "temperature": 0.5},
        {"id": "x"},
    ],
)
def test_identity_and_membership_are_refused(studio, body):
    server, registry, events, _ollama, _ops, _outsider = studio
    disk = registry.data_file.read_bytes()
    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json=body)
    assert response.status_code == 422
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_unknown_suite_and_empty_patch(studio):
    server, _registry, events, ollama, _ops, _outsider = studio
    missing = _call(server, "PATCH", "/api/agent-suites/ghost/agents", json={"temperature": 0.2})
    empty = _call(server, "PATCH", "/api/agent-suites/ops/agents", json={})
    assert missing.status_code == 404
    assert missing.json()["detail"] == "suite_not_found"
    assert empty.status_code == 422
    assert empty.json()["detail"] == "empty_patch"
    assert events == []
    assert ollama["calls"] == 0


def test_suite_list_includes_every_suite_in_use(studio):
    server, _registry, _events, _ollama, ops, _outsider = studio
    suites = {
        suite["slug"]: suite for suite in _call(server, "GET", "/api/agent-suites").json()["suites"]
    }
    assert set(suites) == {"mindmoor", "ops"}
    assert suites["ops"]["catalogued"] is False
    assert suites["ops"]["memory_namespace"] == "ops"
    assert {row["id"] for row in suites["ops"]["agents"]} == {agent["id"] for agent in ops}
    assert suites["mindmoor"]["catalogued"] is True
    assert suites["mindmoor"]["agents"] == []


def test_registry_update_many_is_all_or_nothing(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    good = registry.create("Good", "M", "P", [], loop_task="Poll.")
    bad = registry.create("Bad", "M", "P", [])
    disk = registry.data_file.read_bytes()

    with pytest.raises(BulkUpdateRejected) as exc:
        registry.update_many([good["id"], bad["id"]], {"loop_enabled": True, "max_tokens": 500})

    assert exc.value.failures == [{"agent_id": bad["id"], "detail": "loop_task_required"}]
    assert registry.data_file.read_bytes() == disk
    assert registry.get(good["id"])["max_tokens"] == 1_200

    updated = registry.update_many([good["id"], bad["id"]], {"max_tokens": 500})
    assert [agent["max_tokens"] for agent in updated] == [500, 500]
    restored = AgentRegistry(registry.data_file)
    assert restored.get(good["id"])["max_tokens"] == restored.get(bad["id"])["max_tokens"] == 500
