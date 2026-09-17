"""Edge cases for agent PATCH, per-agent models, suite modulation and Local memory."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry

REAL_ASYNC_CLIENT = httpx.AsyncClient
TAGS = {
    "models": [
        {"name": "qwen3:8b", "size": 5_225_388_164, "details": {"family": "qwen3"}},
        {"name": "Snowflake-Arctic-EMBED:latest", "size": 1, "details": {"family": "x"}},
        {"name": "gemma3:4b", "size": 3_338_801_804, "details": None},
    ]
}


class Upstream:
    def __init__(self) -> None:
        self.tags: Any = TAGS
        self.memory: Any = httpx.Response(200, json={"memories": []})
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.port == 11434:
            if self.tags is None:
                raise httpx.ConnectError("ollama down", request=request)
            return httpx.Response(200, json=self.tags)
        if request.url.path == "/v1/aiia/memory":
            return self.memory
        if request.url.path == "/v1/chat":
            return httpx.Response(200, json={"content": "OK", "model": "qwen3:8b"})
        raise AssertionError(f"unexpected upstream call {request.url}")

    @property
    def ollama_calls(self) -> int:
        return sum(1 for request in self.requests if request.url.port == 11434)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    events: list[tuple[str, str, dict[str, Any]]] = []
    upstream = Upstream()

    async def capture(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(upstream.handler))
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


def _legacy_empty_loop(registry: AgentRegistry, **overrides: Any) -> dict[str, Any]:
    # An agent saved through the old PUT bypass: looping with no task.
    agent = registry.create("Legacy", "Mission.", "Persona.", [], loop_task="Poll.", **overrides)
    agent["loop_enabled"] = True
    agent["loop_task"] = ""
    registry.save()
    return agent


def test_failed_patch_rolls_back_in_memory_as_well_as_on_disk(studio):
    server, registry, events, _upstream = studio
    agent = registry.create("Blank", "Mission.", "Persona.", [])
    before = dict(agent)

    response = _call(
        server,
        "PATCH",
        f"/api/agents/{agent['id']}",
        json={"temperature": 0.9, "model": "", "loop_enabled": True},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "loop_task_required"
    assert registry.get(agent["id"]) == before
    assert events == []


@pytest.mark.parametrize("body", [[], "text", 5, [{"temperature": 0.5}]])
def test_patch_non_object_body_is_refused(studio, body):
    server, registry, events, _upstream = studio
    agent = registry.create("A", "Mission.", "Persona.", [])
    disk = registry.data_file.read_bytes()

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json=body)

    assert response.status_code == 422
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_legacy_empty_loop_agent_can_be_repaired_by_patch_and_loop_toggle(studio):
    server, registry, _events, _upstream = studio
    agent = _legacy_empty_loop(registry)

    blocked = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"max_tokens": 500})
    assert blocked.status_code == 422
    assert blocked.json()["detail"] == "loop_task_required"

    disabled = _call(server, "POST", f"/api/agents/{agent['id']}/loop", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json()["agent"]["loop_enabled"] is False

    restored = _legacy_empty_loop(registry, suite="ops")
    fixed = _call(
        server, "PATCH", f"/api/agents/{restored['id']}", json={"loop_task": "Watch deploys."}
    )
    assert fixed.status_code == 200
    assert AgentRegistry(registry.data_file).get(restored["id"])["loop_task"] == "Watch deploys."


def test_due_loop_never_selects_a_whitespace_loop_task(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    agent = registry.create("Loop", "Mission.", "Persona.", [], loop_task="Poll.")
    agent["loop_enabled"] = True
    agent["loop_task"] = "   "
    assert registry.due_loop() is None


def test_models_route_matches_embed_case_insensitively_and_tolerates_null_details(studio):
    server, _registry, _events, _upstream = studio

    body = _call(server, "GET", "/api/agents/models").json()

    assert [model["id"] for model in body["models"]] == ["qwen3:8b", "gemma3:4b"]
    assert body["models"][1]["family"] == ""
    assert body["models"][1]["parameter_size"] == ""


@pytest.mark.parametrize(
    "tags",
    [
        {"models": [1]},
        {"models": [{"name": "x", "details": "nope"}]},
        {"models": [{"name": "x", "size": "big"}]},
        ["qwen3:8b"],
    ],
)
def test_models_route_reports_malformed_ollama_payload_as_unavailable(studio, tags):
    server, registry, _events, upstream = studio
    upstream.tags = tags
    agent = registry.create("A", "Mission.", "Persona.", [])

    listed = _call(server, "GET", "/api/agents/models")
    patched = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": "x"})

    for response in (listed, patched):
        assert response.status_code == 503
        assert response.json()["detail"] == "models_unavailable"
    assert registry.get(agent["id"])["model"] == ""


def test_patch_model_whitespace_is_trimmed_before_verification(studio):
    server, registry, _events, _upstream = studio
    agent = registry.create("A", "Mission.", "Persona.", [])

    response = _call(server, "PATCH", f"/api/agents/{agent['id']}", json={"model": " gemma3:4b "})

    assert response.status_code == 200
    assert response.json()["agent"]["model"] == "gemma3:4b"


def test_put_without_model_keeps_the_stored_model_and_skips_ollama(studio):
    server, registry, _events, upstream = studio
    agent = registry.create("A", "Mission.", "Persona.", [], model="gemma3:4b")
    upstream.tags = None

    response = _call(
        server, "PUT", f"/api/agents/{agent['id']}", json={"name": "B", "mission": "Mission."}
    )

    assert response.status_code == 200
    assert response.json()["agent"]["model"] == "gemma3:4b"
    assert upstream.ollama_calls == 0


def test_suite_patch_clearing_model_needs_no_ollama(studio):
    server, registry, events, upstream = studio
    members = [
        registry.create(f"Ops {n}", "Watch.", "Terse.", [], suite="ops", model="gemma3:4b")
        for n in range(2)
    ]
    upstream.tags = None

    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json={"model": ""})

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert all(
        AgentRegistry(registry.data_file).get(agent["id"])["model"] == "" for agent in members
    )
    assert upstream.ollama_calls == 0
    assert len(events) == 2


@pytest.mark.parametrize(
    "body", [{"model": None}, {"loop_enabled": None}, {}, {"max_tokens": 5_000}, []]
)
def test_suite_patch_rejects_null_empty_and_out_of_range_bodies(studio, body):
    server, registry, events, _upstream = studio
    registry.create("Ops", "Watch.", "Terse.", [], suite="ops")
    disk = registry.data_file.read_bytes()

    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json=body)

    assert response.status_code == 422
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_suite_patch_invalid_namespace_is_rejected_for_every_member(studio):
    server, registry, events, _upstream = studio
    members = [registry.create(f"Ops {n}", "Watch.", "Terse.", [], suite="ops") for n in range(2)]
    disk = registry.data_file.read_bytes()

    response = _call(
        server, "PATCH", "/api/agent-suites/ops/agents", json={"memory_namespace": "Bad Space!"}
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "suite_patch_rejected"
    assert {row["agent_id"] for row in body["failures"]} == {agent["id"] for agent in members}
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_suite_patch_only_touches_tagged_members_not_alias_matches(studio):
    server, registry, _events, _upstream = studio
    tagged = registry.create("Ops", "Watch.", "Terse.", [], suite="ops")
    other = registry.create("Ops", "Watch.", "Terse.", [], suite="other")
    untagged = registry.create("Ops", "Watch.", "Terse.", [])

    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json={"temperature": 0.9})

    assert response.status_code == 200
    assert [agent["id"] for agent in response.json()["agents"]] == [tagged["id"]]
    restored = AgentRegistry(registry.data_file)
    assert restored.get(other["id"])["temperature"] == 0.35
    assert restored.get(untagged["id"])["temperature"] == 0.35


def test_local_memory_request_is_authenticated_and_widened_for_a_namespace(studio):
    server, registry, _events, upstream = studio
    memories = [
        {"id": f"m{n}", "fact": f"Fact {n}.", "source": "suite:ops", "category": "decisions"}
        for n in range(10)
    ]
    upstream.memory = httpx.Response(200, json={"memories": memories})
    agent = registry.create(
        "Ops", "Watch.", "Terse.", [], tools=["Local memory"], suite="ops", memory_namespace="ops"
    )

    asyncio.run(server._execute_agent(agent["id"], "Inspect"))

    memory_request = next(r for r in upstream.requests if r.url.path == "/v1/aiia/memory")
    assert memory_request.url.params["limit"] == "200"
    for header, value in server.AIIA_HEADERS.items():
        assert memory_request.headers.get(header) == value
    chat = next(r for r in upstream.requests if r.url.path == "/v1/chat")
    system = json.loads(chat.content)["system"]
    injected = [line for line in system.splitlines() if line.startswith("- [m")]
    assert len(injected) == 6
    assert "[m6]" not in system


def test_local_memory_skips_entries_without_id_or_fact(studio):
    server, registry, _events, upstream = studio
    upstream.memory = httpx.Response(
        200,
        json={
            "memories": [
                {"id": "", "fact": "No id."},
                {"id": "m1", "fact": "   "},
                "not a dict",
                {"id": "m2", "fact": "Kept\nacross lines.", "category": "lessons"},
            ]
        },
    )
    agent = registry.create("A", "Mission.", "Persona.", [], tools=["Local memory"])

    asyncio.run(server._execute_agent(agent["id"], "Inspect"))

    chat = next(r for r in upstream.requests if r.url.path == "/v1/chat")
    system = json.loads(chat.content)["system"]
    assert "- [m2] (lessons) Kept across lines." in system
    assert "No id." not in system
    assert "(1 entries" in system


def test_suite_patch_rereads_membership_after_the_model_check(studio, monkeypatch):
    server, registry, events, _upstream = studio
    leaver, stayer = (
        registry.create(f"Ops {n}", "Watch.", "Terse.", [], suite="ops") for n in range(2)
    )

    async def move_member_while_checking() -> list[dict[str, Any]]:
        registry.update(leaver["id"], suite="other")
        return [{"id": "gemma3:4b"}]

    monkeypatch.setattr(server, "_installed_chat_models", move_member_while_checking)

    response = _call(server, "PATCH", "/api/agent-suites/ops/agents", json={"model": "gemma3:4b"})

    assert response.status_code == 200
    assert [agent["id"] for agent in response.json()["agents"]] == [stayer["id"]]
    restored = AgentRegistry(registry.data_file)
    assert restored.get(leaver["id"])["model"] == ""
    assert restored.get(stayer["id"])["model"] == "gemma3:4b"
    assert [item["id"] for _, _, item in events] == [stayer["id"]]


def test_create_and_put_refuse_a_model_while_ollama_is_down(studio):
    server, registry, events, upstream = studio
    agent = registry.create("A", "Mission.", "Persona.", [])
    upstream.tags = None
    disk = registry.data_file.read_bytes()
    body = {"name": "A", "mission": "Mission.", "model": "qwen3:8b"}

    created = _call(server, "POST", "/api/agents", json=body)
    replaced = _call(server, "PUT", f"/api/agents/{agent['id']}", json=body)

    for response in (created, replaced):
        assert response.status_code == 503
        assert response.json()["detail"] == "models_unavailable"
    assert len(registry.agents) == 1
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_agent_list_returns_model_on_every_record_including_backfilled(studio, monkeypatch):
    server, registry, _events, _upstream = studio
    registry.create("A", "Mission.", "Persona.", [], model="gemma3:4b")
    registry.create("B", "Mission.", "Persona.", [])
    stored = json.loads(registry.data_file.read_text())
    del stored["agents"][1]["model"]
    registry.data_file.write_text(json.dumps(stored))
    monkeypatch.setattr(server, "agent_registry", AgentRegistry(registry.data_file))

    agents = _call(server, "GET", "/api/agents").json()["agents"]

    assert sorted(agent["model"] for agent in agents) == ["", "gemma3:4b"]
