"""Voice Conductor: tool allowlist, fail-closed without a key, no secrets in fixtures."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from local_brain.command_center.voice_conductor import (
    ALLOWED_TOOLS,
    FORBIDDEN_TOOLS,
    VoiceConductorDeps,
    classify_tool,
    execute_tool,
    grok_tool_definitions,
    load_xai_api_key,
    session_config,
    status_payload,
    voice_configured,
)
from local_brain.command_center.voice_routes import build_voice_router


def _agent(agent_id: str = "agent_research") -> dict[str, Any]:
    return {
        "id": agent_id,
        "name": "Research",
        "mission": "Find signal.",
        "status": "idle",
        "skills": ["Research"],
        "tools": ["Local memory"],
        "last_run_at": None,
    }


def _assignment(assignment_id: str = "asg_queued", status: str = "queued") -> dict[str, Any]:
    return {
        "id": assignment_id,
        "title": "Map the surface",
        "objective": "Return three recommendations.",
        "agent_id": "agent_research",
        "priority": "normal",
        "status": status,
        "error": "",
        "created_at": "2026-09-07T00:00:00+00:00",
        "updated_at": "2026-09-07T00:00:00+00:00",
    }


def _deps(
    *,
    busy: bool = False,
    agents: list[dict[str, Any]] | None = None,
    assignments: list[dict[str, Any]] | None = None,
    created: list[dict[str, Any]] | None = None,
    run_payload: dict[str, Any] | None = None,
    run_error: Exception | None = None,
) -> VoiceConductorDeps:
    store_agents = agents if agents is not None else [_agent()]
    store_assignments = assignments if assignments is not None else [_assignment()]
    created = created if created is not None else []

    def create_assignment(**kwargs: Any) -> dict[str, Any]:
        item = _assignment("asg_new")
        item.update(kwargs)
        item["status"] = "queued"
        created.append(item)
        store_assignments.insert(0, item)
        return item

    async def run_assignment(assignment_id: str) -> dict[str, Any]:
        if run_error:
            raise run_error
        return run_payload or {
            "assignment": _assignment(assignment_id, "completed"),
            "agent": _agent(),
            "model": "qwen3:8b",
            "latency_ms": 12,
        }

    return VoiceConductorDeps(
        list_agents=lambda: store_agents,
        get_agent=lambda agent_id: next((a for a in store_agents if a["id"] == agent_id), None),
        list_assignments=lambda: store_assignments,
        get_assignment=lambda assignment_id: next(
            (item for item in store_assignments if item["id"] == assignment_id), None
        ),
        create_assignment=create_assignment,
        list_handoffs=lambda: [],
        github_status=lambda: {
            "status": "disconnected",
            "mode": "read_only",
            "provider": "github_cli",
            "account": "",
            "reason": "github_cli_missing",
        },
        available_repos=lambda: [{"id": "aiia", "name": "AIIA", "branch": "main", "dirty": False}],
        mini_busy=lambda: busy,
        run_assignment=run_assignment,
    )


# ---------------------------------------------------------------------------
# Key loading — never put a live key in fixtures
# ---------------------------------------------------------------------------


def test_load_xai_key_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "  xai-test-fixture  ")
    assert load_xai_api_key(home=tmp_path) == "xai-test-fixture"


def test_load_xai_key_from_keys_json(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    aiia = tmp_path / ".aiia"
    aiia.mkdir()
    (aiia / "keys.json").write_text(json.dumps({"xai": "xai-from-file"}))
    assert load_xai_api_key(home=tmp_path) == "xai-from-file"


def test_load_xai_key_missing_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert load_xai_api_key(home=tmp_path) == ""
    assert voice_configured(home=tmp_path) is False


def test_status_not_configured_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("AIIA_AIRGAP", raising=False)
    payload = status_payload(home=tmp_path)
    assert payload["status"] == "not_configured"
    assert payload["reason"] == "missing_xai_api_key"
    assert payload["configured"] is False
    dumped = json.dumps(payload)
    assert "xai-" not in dumped
    assert "XAI_API_KEY" not in dumped


def test_status_connected_when_key_present(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-fixture")
    monkeypatch.delenv("AIIA_AIRGAP", raising=False)
    payload = status_payload(home=tmp_path)
    assert payload["status"] == "connected"
    assert payload["voice"] == "eve"
    assert payload["model"] == "grok-voice-latest"
    assert "XAI_API_KEY" not in json.dumps(payload)


def test_status_connected_under_airgap_when_key_present(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-fixture")
    monkeypatch.setenv("AIIA_AIRGAP", "1")
    payload = status_payload(home=tmp_path)
    assert payload["status"] == "connected"
    assert payload["configured"] is True
    assert payload["reason"] == ""
    assert "xai-test-fixture" not in json.dumps(payload)


def test_status_not_configured_under_airgap_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("AIIA_AIRGAP", "1")
    payload = status_payload(home=tmp_path)
    assert payload["status"] == "not_configured"
    assert payload["reason"] == "missing_xai_api_key"
    assert payload["configured"] is False


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALLOWED_TOOLS))
def test_allowlist_classifies_known_tools(name):
    assert classify_tool(name) == "allowed"


@pytest.mark.parametrize("name", sorted(FORBIDDEN_TOOLS))
def test_forbidden_tools_are_classified_forbidden(name):
    assert classify_tool(name) == "forbidden"


def test_unknown_tool_is_not_allowed():
    assert classify_tool("call_agent") == "unknown"
    assert classify_tool("") == "unknown"


def test_session_config_only_advertises_allowlisted_function_tools():
    config = session_config()
    names = {tool["name"] for tool in config["tools"]}
    assert names == ALLOWED_TOOLS
    assert config["voice"] == "eve"
    assert config["turn_detection"] is None
    for tool in config["tools"]:
        assert tool["type"] == "function"
    advertised = {item["name"] for item in grok_tool_definitions()}
    assert advertised.isdisjoint(FORBIDDEN_TOOLS)


@pytest.mark.asyncio
async def test_execute_forbidden_tool_never_runs():
    deps = _deps()
    result = await execute_tool("git_push", {"branch": "main"}, deps)
    assert result.ok is False
    assert result.error == "tool_forbidden"
    assert result.status == 403


@pytest.mark.asyncio
async def test_execute_unknown_tool_rejected():
    result = await execute_tool("run_shell", {"cmd": "ls"}, _deps())
    assert result.ok is False
    assert result.error == "tool_forbidden"
    result = await execute_tool("deploy_production", {}, _deps())
    assert result.ok is False
    assert result.error == "tool_not_allowlisted"
    assert result.status == 403


@pytest.mark.asyncio
async def test_create_assignment_requires_existing_agent():
    result = await execute_tool(
        "create_assignment",
        {"title": "Do it", "objective": "Do the thing", "agent_id": "missing"},
        _deps(),
    )
    assert result.ok is False
    assert result.error == "agent_not_found"


@pytest.mark.asyncio
async def test_create_assignment_for_allowlisted_agent():
    created: list[dict[str, Any]] = []
    result = await execute_tool(
        "create_assignment",
        {
            "title": "Scout the repo",
            "objective": "List the three highest-leverage docs.",
            "agent_id": "agent_research",
            "priority": "high",
        },
        _deps(created=created),
    )
    assert result.ok is True
    assert result.result["assignment"]["agent_id"] == "agent_research"
    assert result.result["assignment"]["status"] == "queued"
    assert created[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_run_assignment_respects_mini_busy():
    result = await execute_tool("run_assignment", {"assignment_id": "asg_queued"}, _deps(busy=True))
    assert result.ok is False
    assert result.error == "mini_busy"
    assert result.status == 409


@pytest.mark.asyncio
async def test_run_assignment_rejects_completed():
    result = await execute_tool(
        "run_assignment",
        {"assignment_id": "asg_done"},
        _deps(assignments=[_assignment("asg_done", "completed")]),
    )
    assert result.ok is False
    assert result.error == "assignment_not_runnable"


@pytest.mark.asyncio
async def test_run_assignment_maps_http_mini_busy():
    result = await execute_tool(
        "run_assignment",
        {"assignment_id": "asg_queued"},
        _deps(run_error=HTTPException(status_code=409, detail="mini_busy")),
    )
    assert result.ok is False
    assert result.error == "mini_busy"
    assert result.status == 409


@pytest.mark.asyncio
async def test_read_tools_return_bounded_payloads():
    agents = await execute_tool("list_agents", {}, _deps())
    assert agents.ok
    assert agents.result["count"] == 1
    assert agents.result["agents"][0]["id"] == "agent_research"

    resources = await execute_tool("list_resources", {}, _deps())
    assert resources.result["github"]["mode"] == "read_only"
    assert "token" not in json.dumps(resources.result)

    mini = await execute_tool("mini_status", {}, _deps(busy=False))
    assert mini.result == {"mini_busy": False, "status": "idle"}


# ---------------------------------------------------------------------------
# HTTP routes — fail-closed without a key
# ---------------------------------------------------------------------------


def _client(deps: VoiceConductorDeps | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(build_voice_router(deps or _deps()))
    return TestClient(app)


def test_http_status_connected_under_airgap_with_key(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-fixture")
    monkeypatch.setenv("AIIA_AIRGAP", "1")
    monkeypatch.setattr("local_brain.command_center.voice_conductor.Path.home", lambda: tmp_path)
    response = _client().get("/api/voice/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "connected"
    assert body["reason"] == ""
    assert "xai-test-fixture" not in response.text


def test_http_status_not_configured_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("AIIA_AIRGAP", raising=False)
    monkeypatch.setattr("local_brain.command_center.voice_conductor.Path.home", lambda: tmp_path)
    response = _client().get("/api/voice/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_configured"
    assert "xai-test" not in response.text
    assert "Bearer" not in response.text


def test_http_session_fails_closed_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("AIIA_AIRGAP", raising=False)
    monkeypatch.setattr("local_brain.command_center.voice_conductor.Path.home", lambda: tmp_path)
    response = _client().post("/api/voice/session")
    assert response.status_code == 503
    assert response.json()["detail"] == "missing_xai_api_key"


def test_http_forbidden_tool_rejected():
    response = _client().post("/api/voice/tools", json={"name": "open_pr", "arguments": {}})
    assert response.status_code == 403
    assert response.json()["detail"] == "tool_forbidden"


def test_http_forbidden_tools_still_forbidden_under_airgap(monkeypatch):
    monkeypatch.setenv("AIIA_AIRGAP", "1")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-fixture")
    for name in ("git_push", "open_pr", "run_shell", "sanction_spend"):
        response = _client().post("/api/voice/tools", json={"name": name, "arguments": {}})
        assert response.status_code == 403, name
        assert response.json()["detail"] == "tool_forbidden"


def test_http_unknown_tool_rejected():
    response = _client().post("/api/voice/tools", json={"name": "call_agent", "arguments": {}})
    assert response.status_code == 403
    assert response.json()["detail"] == "tool_not_allowlisted"


def test_http_create_assignment_ok():
    response = _client().post(
        "/api/voice/tools",
        json={
            "name": "create_assignment",
            "arguments": {
                "title": "Draft a brief",
                "objective": "One page on Voice Conductor bounds.",
                "agent_id": "agent_research",
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["result"]["assignment"]["status"] == "queued"


@pytest.mark.asyncio
async def test_session_mint_uses_mocked_xai(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-fixture")
    monkeypatch.delenv("AIIA_AIRGAP", raising=False)

    async def allow(_tool: str, server: str | None = None):
        return type("D", (), {"allowed": True, "reason": "test"})()

    posted: list[dict[str, Any]] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"value": "ephem-test-token", "expires_at": 1_778_000_000}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: object):
            return False

        async def post(
            self, url: str, headers: dict[str, str] | None = None, json: dict | None = None
        ):
            posted.append({"url": url, "headers": headers, "json": json})
            return _Resp()

    monkeypatch.setattr("local_brain.command_center.voice_routes.authorize_egress", allow)
    monkeypatch.setattr("local_brain.command_center.voice_routes.httpx.AsyncClient", _Client)

    response = _client().post("/api/voice/session")
    assert response.status_code == 200
    body = response.json()
    assert body["token"] == "ephem-test-token"
    assert body["session"]["voice"] == "eve"
    assert {tool["name"] for tool in body["session"]["tools"]} == ALLOWED_TOOLS
    assert "xai-test-fixture" not in response.text
    assert posted[0]["headers"]["Authorization"] == "Bearer xai-test-fixture"
    assert posted[0]["url"].endswith("/v1/realtime/client_secrets")


@pytest.mark.asyncio
async def test_session_mint_allowed_under_airgap_with_mocked_xai(monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-fixture")
    monkeypatch.setenv("AIIA_AIRGAP", "1")

    posted: list[dict[str, Any]] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"value": "ephem-airgap-token", "expires_at": 1_778_000_000}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: object):
            return False

        async def post(
            self, url: str, headers: dict[str, str] | None = None, json: dict | None = None
        ):
            posted.append({"url": url, "headers": headers, "json": json})
            return _Resp()

    monkeypatch.setattr(
        "local_brain.egress.get_config", lambda: type("C", (), {"airgap_enabled": True})()
    )
    monkeypatch.setattr("local_brain.command_center.voice_routes.httpx.AsyncClient", _Client)

    response = _client().post("/api/voice/session")
    assert response.status_code == 200
    body = response.json()
    assert body["token"] == "ephem-airgap-token"
    assert "xai-test-fixture" not in response.text
    assert posted[0]["url"].endswith("/v1/realtime/client_secrets")
