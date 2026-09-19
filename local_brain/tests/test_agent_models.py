"""Per-agent model: stored on the record, verified against Ollama, sent to the Brain."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry

REAL_ASYNC_CLIENT = httpx.AsyncClient

OLLAMA_TAGS = {
    "models": [
        {
            "name": "qwen3:8b",
            "size": 5_225_388_164,
            "details": {"family": "qwen3", "parameter_size": "8.2B"},
        },
        {
            "name": "gemma3:4b",
            "size": 3_338_801_804,
            "details": {"family": "gemma3", "parameter_size": "4.3B"},
        },
        {
            "name": "nomic-embed-text:latest",
            "size": 274_302_450,
            "details": {"family": "nomic-bert", "parameter_size": "137M"},
        },
        {
            "name": "mxbai-embed-large:latest",
            "size": 669_615_493,
            "details": {"family": "llama", "parameter_size": "334M"},
        },
        {
            "name": "bge-small:latest",
            "size": 133_000_000,
            "details": {"family": "bert", "parameter_size": "33M"},
        },
    ]
}


class Upstream:
    """Stands in for Ollama and the Brain; nothing leaves the process."""

    def __init__(self, tags: Any = OLLAMA_TAGS, tags_status: int = 200, brain_model: Any = None):
        self.tags = tags
        self.tags_status = tags_status
        self.brain_model = brain_model
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.port == 11434:
            if self.tags is None:
                raise httpx.ConnectError("ollama down", request=request)
            return httpx.Response(self.tags_status, json=self.tags)
        if request.url.path == "/v1/chat":
            payload = json.loads(request.content)
            model = self.brain_model if self.brain_model is not None else payload.get("model")
            return httpx.Response(
                200, json={"content": "GREEN", "model": model or "", "latency_ms": 5}
            )
        raise AssertionError(f"unexpected upstream call {request.url}")

    @property
    def ollama_calls(self) -> int:
        return sum(1 for request in self.requests if request.url.port == 11434)

    @property
    def chat_payloads(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/v1/chat"]


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    events: list[tuple[str, str, dict[str, Any]]] = []
    upstream = Upstream()

    async def capture(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        if "transport" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(upstream.handler)
        return REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "broadcast_studio_event", capture)
    monkeypatch.setattr(server.httpx, "AsyncClient", client_factory)
    monkeypatch.setenv("LOCAL_TASK_MODEL", "qwen3:8b")
    return server, registry, events, upstream


def _call(server, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        async with REAL_ASYNC_CLIENT(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(go())


def _agent(registry: AgentRegistry, **overrides: Any) -> dict[str, Any]:
    payload = {"name": "Signal Officer", "mission": "Report.", "persona": "Terse.", "skills": []}
    payload.update(overrides)
    return registry.create(**payload)


def test_model_is_stored_returned_and_backfilled_on_load(tmp_path):
    data_file = tmp_path / "agents.json"
    registry = AgentRegistry(data_file)
    chosen = _agent(registry, model=" gemma3:4b ")
    default = _agent(registry, name="Default")
    assert chosen["model"] == "gemma3:4b"
    assert default["model"] == ""
    assert AgentRegistry(data_file).get(chosen["id"])["model"] == "gemma3:4b"

    stored = json.loads(data_file.read_text())
    for agent in stored["agents"]:
        del agent["model"]
    data_file.write_text(json.dumps(stored))
    legacy = AgentRegistry(data_file)
    assert all(agent["model"] == "" for agent in legacy.list())


def test_models_route_lists_chat_models_and_marks_default(studio):
    server, _registry, _events, _upstream = studio

    response = _call(server, "GET", "/api/agents/models")

    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "qwen3:8b"
    assert body["models"] == [
        {
            "id": "qwen3:8b",
            "label": "qwen3:8b",
            "family": "qwen3",
            "parameter_size": "8.2B",
            "size_gb": 5.2,
            "default": True,
        },
        {
            "id": "gemma3:4b",
            "label": "gemma3:4b",
            "family": "gemma3",
            "parameter_size": "4.3B",
            "size_gb": 3.3,
            "default": False,
        },
    ]


def test_models_default_falls_back_to_brain_task_default(studio, monkeypatch):
    server, _registry, _events, _upstream = studio
    monkeypatch.delenv("LOCAL_TASK_MODEL")
    body = _call(server, "GET", "/api/agents/models").json()
    assert body["default"] == server.BRAIN_TASK_MODEL_FALLBACK
    assert not any(model["default"] for model in body["models"])


@pytest.mark.parametrize(
    ("tags", "status"), [(None, 200), ({"models": []}, 500), ("not an object", 200)]
)
def test_models_route_reports_unreachable_ollama(studio, tags, status):
    server, _registry, _events, upstream = studio
    upstream.tags, upstream.tags_status = tags, status
    response = _call(server, "GET", "/api/agents/models")
    assert response.status_code == 503
    assert response.json()["detail"] == "models_unavailable"


def test_patch_and_create_accept_installed_model_and_refuse_unknown(studio):
    server, registry, _events, _upstream = studio
    agent = _agent(registry)
    disk = registry.data_file.read_bytes()

    for response in (
        _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "llama9:70b"}),
        _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "nomic-embed-text"}),
        _call(server, "POST", "/api/agents", json={"name": "N", "mission": "M", "model": "x:1b"}),
        _call(
            server,
            "PUT",
            f"/api/agents/{agent['id']}",
            json={"name": "N", "mission": "M", "model": "x:1b"},
        ),
    ):
        assert response.status_code == 422
        assert response.json()["detail"] == "unknown_model"
    assert registry.data_file.read_bytes() == disk
    assert len(registry.agents) == 1

    patched = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "gemma3:4b"})
    assert patched.status_code == 200
    assert patched.json()["agent"]["model"] == "gemma3:4b"
    assert AgentRegistry(registry.data_file).get(agent["id"])["model"] == "gemma3:4b"
    created = _call(
        server, "POST", "/api/agents", json={"name": "N", "mission": "M", "model": "qwen3:8b"}
    )
    assert created.status_code == 200
    assert created.json()["agent"]["model"] == "qwen3:8b"


def test_unreachable_ollama_rejects_model_change_but_not_clearing_it(studio):
    server, registry, events, upstream = studio
    agent = _agent(registry, model="gemma3:4b")
    upstream.tags = None
    disk = registry.data_file.read_bytes()

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "qwen3:8b"})

    assert response.status_code == 503
    assert response.json()["detail"] == "models_unavailable"
    assert registry.data_file.read_bytes() == disk
    assert events == []

    cleared = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": ""})
    unchanged = _call(
        server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "", "temperature": 0.1}
    )
    assert cleared.status_code == unchanged.status_code == 200
    assert cleared.json()["agent"]["model"] == ""
    assert upstream.ollama_calls == 1


def test_agent_without_model_uses_task_role(studio):
    server, registry, _events, upstream = studio
    agent = _agent(registry)

    result = asyncio.run(server._execute_agent(agent["id"], "Inspect"))

    payload = upstream.chat_payloads[0]
    assert payload["model_role"] == "task"
    assert payload["think"] is False
    assert "model" not in payload
    assert result["agent"]["runs"][0]["model"] == ""
    assert upstream.requests[0].headers.get("x-api-key") == server.AIIA_HEADERS.get("x-api-key")


def test_agent_with_model_sends_it_and_ledger_records_model_used(studio):
    server, registry, _events, upstream = studio
    agent = _agent(registry, model="gemma3:4b")

    result = asyncio.run(server._execute_agent(agent["id"], "Inspect"))

    payload = upstream.chat_payloads[0]
    assert payload["model"] == "gemma3:4b"
    assert "model_role" not in payload
    run = result["agent"]["runs"][0]
    assert run["model"] == "gemma3:4b"
    assert registry.ledger.get(run["id"])["model"] == "gemma3:4b"

    upstream.brain_model = "qwen3:8b"
    second = asyncio.run(server._execute_agent(agent["id"], "Inspect again"))
    assert second["agent"]["runs"][0]["model"] == "qwen3:8b"


def test_in_flight_run_keeps_the_settings_it_started_with(studio, monkeypatch):
    server, registry, events, upstream = studio
    agent = _agent(registry, model="gemma3:4b", max_tokens=900)

    async def patch_while_running(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))
        if event == "running":
            registry.update(agent["id"], model="qwen3:8b", max_tokens=300)

    monkeypatch.setattr(server, "broadcast_studio_event", patch_while_running)

    asyncio.run(server._execute_agent(agent["id"], "Inspect"))

    payload = upstream.chat_payloads[0]
    assert payload["model"] == "gemma3:4b"
    assert payload["max_tokens"] == 900
    assert registry.get(agent["id"])["model"] == "qwen3:8b"
