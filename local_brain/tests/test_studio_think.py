"""Studio and MCP /v1/chat callers default think:false so qwen3 does not spend the budget."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry


def _unwrap(tool: Any):
    return getattr(tool, "fn", tool)


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


def _run(server, agent_id: str, task: str = "Inspect"):
    return asyncio.run(server._execute_agent(agent_id, task))


def _brain(monkeypatch, server, payload: dict[str, Any] | None = None) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = httpx.Response(
        200,
        json=payload
        or {
            "content": "Observed evidence",
            "model": "qwen3:8b",
            "latency_ms": 12,
            "done_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    )
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: client)
    return client


def test_execute_agent_sends_think_false_by_default(studio, monkeypatch):
    server, registry, _events = studio
    agent = registry.create("Review", "Inspect", "Precise", [])
    client = _brain(monkeypatch, server)

    _run(server, agent["id"])

    request = client.post.call_args.kwargs["json"]
    assert request["think"] is False
    run = registry.ledger.get(agent["runs"][0]["id"])
    assert run["done_reason"] == "stop"
    assert run["think"] is False


def test_execute_agent_opts_in_when_agent_think_is_true(studio, monkeypatch):
    server, registry, _events = studio
    agent = registry.create("Reasoner", "Inspect", "Precise", [], think=True)
    client = _brain(monkeypatch, server)

    _run(server, agent["id"])

    request = client.post.call_args.kwargs["json"]
    assert request["think"] is True
    assert registry.ledger.get(agent["runs"][0]["id"])["think"] is True


def test_think_is_stored_returned_and_backfilled_on_load(tmp_path):
    import json

    data_file = tmp_path / "agents.json"
    registry = AgentRegistry(data_file)
    opted = registry.create("Reasoner", "Inspect", "Precise", [], think=True)
    default = registry.create("Default", "Inspect", "Precise", [])
    assert opted["think"] is True
    assert default["think"] is False

    stored = json.loads(data_file.read_text())
    for agent in stored["agents"]:
        agent.pop("think", None)
    data_file.write_text(json.dumps(stored))
    legacy = AgentRegistry(data_file)
    assert legacy.get(opted["id"])["think"] is False
    assert legacy.get(default["id"])["think"] is False


def test_patch_think_reaches_the_next_run(studio, monkeypatch):
    server, registry, _events = studio
    agent = registry.create("Review", "Inspect", "Precise", [])

    async def patch() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.patch(f"/api/agents/{agent['id']}", json={"think": True})

    response = asyncio.run(patch())
    assert response.status_code == 200
    assert response.json()["agent"]["think"] is True
    assert AgentRegistry(registry.data_file).get(agent["id"])["think"] is True

    client = _brain(monkeypatch, server)
    _run(server, agent["id"])
    assert client.post.call_args.kwargs["json"]["think"] is True


@pytest.mark.parametrize("tool_name", ["aiia_offload", "aiia_digest"])
def test_mcp_sidecar_chat_sends_think_false(monkeypatch, tool_name):
    from local_brain import mcp_server

    captured: dict[str, Any] = {}

    async def fake_call(
        method: str, path: str, body: dict | None = None, timeout: float | None = None
    ):
        captured.update({"method": method, "path": path, "body": body})
        return {
            "content": "ok",
            "model": "qwen3:8b",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "latency_ms": 10,
        }

    monkeypatch.setattr(mcp_server, "_call_aiia", fake_call)
    tool = _unwrap(getattr(mcp_server, tool_name))
    if tool_name == "aiia_offload":
        asyncio.run(tool("summarize", "a long log"))
    else:
        asyncio.run(tool("a long log", focus="errors"))

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/chat"
    assert captured["body"]["think"] is False
