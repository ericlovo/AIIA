"""The "Local memory" tool injects memories that were really retrieved, or says it failed."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry

REAL_ASYNC_CLIENT = httpx.AsyncClient


def _memories(count: int, **extra: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": f"mem{index:03d}",
            "fact": f"Synthetic fact number {index}.",
            "source": "command_center",
            "created_at": f"2026-09-{17 - index:02d}T00:00:00+00:00",
            "metadata": {},
            "category": "decisions",
            **extra,
        }
        for index in range(count)
    ]


class Brain:
    def __init__(self, memory_response: Any = None):
        self.memory_response = memory_response
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/aiia/memory":
            if isinstance(self.memory_response, Exception):
                raise self.memory_response
            return self.memory_response
        if request.url.path == "/v1/chat":
            return httpx.Response(200, json={"content": "GREEN", "model": "qwen3:8b"})
        raise AssertionError(f"unexpected upstream call {request.url}")

    @property
    def memory_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/v1/aiia/memory"]

    @property
    def system_prompt(self) -> str:
        chats = [json.loads(r.content) for r in self.requests if r.url.path == "/v1/chat"]
        assert len(chats) == 1
        return chats[0]["system"]


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    brain = Brain()

    async def ignore(*_args: Any) -> None:
        return None

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(brain.handler))
        return REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "broadcast_studio_event", ignore)
    monkeypatch.setattr(server.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(server, "AIIA_HEADERS", {"x-api-key": "synthetic-key"})
    return server, registry, brain


def _run(server, registry: AgentRegistry, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Signal Officer",
        "mission": "Report.",
        "persona": "Terse.",
        "skills": [],
        "tools": ["Local memory"],
    }
    payload.update(overrides)
    agent = registry.create(**payload)
    return asyncio.run(server._execute_agent(agent["id"], "Inspect"))


def test_run_injects_retrieved_memories_with_ids(studio):
    server, registry, brain = studio
    brain.memory_response = httpx.Response(200, json={"memories": _memories(9), "count": 9})

    result = _run(server, registry)

    request = brain.memory_requests[0]
    assert request.headers["x-api-key"] == "synthetic-key"
    assert request.url.params["limit"] == "6"
    prompt = brain.system_prompt
    for index in range(6):
        assert f"[mem{index:03d}] (decisions) Synthetic fact number {index}." in prompt
    assert "mem006" not in prompt
    assert "Local memory retrieved for this run (6 entries" in prompt
    assert "available through the Mini's private context" not in prompt
    assert server.LOCAL_MEMORY_UNAVAILABLE not in prompt
    assert result["agent"]["last_result"] == "GREEN"


def test_injected_memory_is_capped_at_1500_characters(studio):
    server, registry, brain = studio
    memories = _memories(6)
    for memory in memories:
        memory["fact"] = "x" * 600
    brain.memory_response = httpx.Response(200, json={"memories": memories})

    _run(server, registry)

    prompt = brain.system_prompt
    block = prompt.split("cite a memory by its id.\n", 1)[1].split("\n\n", 1)[0]
    entries = block.splitlines()
    assert len(entries) == 3
    assert len(block) == 1_500
    assert all(entry.startswith("- [mem") for entry in entries)
    assert entries[-1].endswith("…")


def test_namespace_filters_retrieved_memories(studio):
    server, registry, brain = studio
    memories = _memories(5)
    memories[0]["source"] = "suite:mindmoor"
    memories[2]["metadata"] = {"namespace": "mindmoor"}
    memories[3]["source"] = "suite:othertenant"
    brain.memory_response = httpx.Response(200, json={"memories": memories})

    _run(server, registry, suite="mindmoor")

    assert brain.memory_requests[0].url.params["limit"] == str(server.LOCAL_MEMORY_NAMESPACE_SCAN)
    prompt = brain.system_prompt
    assert "[mem000]" in prompt and "[mem002]" in prompt
    assert "mem001" not in prompt and "mem003" not in prompt and "mem004" not in prompt
    assert "in namespace mindmoor (2 entries" in prompt


def test_namespace_with_no_matches_says_so(studio):
    server, registry, brain = studio
    brain.memory_response = httpx.Response(200, json={"memories": _memories(3)})

    _run(server, registry, memory_namespace="quiet")

    prompt = brain.system_prompt
    assert (
        "Local memory was retrieved for this run and had no entries in namespace quiet." in prompt
    )
    assert "mem000" not in prompt


@pytest.mark.parametrize(
    "response",
    [
        httpx.ConnectError("brain down"),
        httpx.ReadTimeout("slow brain"),
        httpx.Response(401, json={"detail": "Invalid API key"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"memories": "nope"}),
        httpx.Response(200, json={"count": 0}),
    ],
)
def test_any_retrieval_failure_is_stated_not_hidden(studio, response):
    server, registry, brain = studio
    brain.memory_response = response

    result = _run(server, registry)

    prompt = brain.system_prompt
    assert "Local memory was unavailable for this run." in prompt
    assert "retrieved" not in prompt
    assert result["agent"]["status"] == "idle"


def test_agent_without_the_tool_does_not_read_memory(studio):
    server, registry, brain = studio
    brain.memory_response = httpx.Response(200, json={"memories": _memories(2)})

    _run(server, registry, tools=[])

    assert brain.memory_requests == []
    assert "Local memory" not in brain.system_prompt
