"""
Tests for aiia/a2a/client.py.

Uses httpx.MockTransport to stub the HTTP layer so we can assert envelope
construction, discovery filters, success paths, and error mapping without
touching a live A2A server.

Run: pytest aiia/local_brain/tests/test_a2a_client.py -v

Note: A2A tests currently live under local_brain/tests because pytest
testpaths in pyproject.toml is [tests, local_brain/tests]. Future work
should add aiia/a2a/tests/ as a third testpath so module-local
tests can co-locate with the code they exercise.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from local_brain.a2a.client import (
    A2AClient,
    A2AClientError,
    AgentSummary,
    TaskResult,
)


def _ok_task_response(answer_text: str = "hello back") -> Dict[str, Any]:
    """Build a JSON-RPC success envelope with a completed Task carrying one text artifact."""
    return {
        "jsonrpc": "2.0",
        "id": "req-test",
        "result": {
            "task": {
                "id": "task-1",
                "contextId": "ctx-1",
                "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": [
                    {
                        "artifactId": "artifact-1",
                        "name": "result",
                        "parts": [{"text": answer_text, "kind": "text"}],
                    }
                ],
                "history": [],
            }
        },
    }


def _err_response(code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "req-test",
        "error": {"code": code, "message": message},
    }


def _mock_client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


# ─── health ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_server_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/a2a/health"
        return httpx.Response(
            200,
            json={
                "status": "online",
                "agents_registered": 7,
                "protocol": "A2A JSONRPC 1.0",
            },
        )

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        result = await c.health()

    assert result["status"] == "online"
    assert result["agents_registered"] == 7


# ─── list_agents ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_agents_no_filter_returns_all():
    captured_params: List[List[tuple]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.append(list(request.url.params.multi_items()))
        return httpx.Response(
            200,
            json={
                "count": 2,
                "agents": [
                    {
                        "agent_id": "brain-status",
                        "name": "Brain",
                        "description": "",
                        "tags": ["layer:dev-tools"],
                        "card_url": "/a2a/agents/brain-status/.well-known/agent.json",
                        "rpc_url": "/a2a/agents/brain-status/rpc",
                    },
                    {
                        "agent_id": "aiia-ask",
                        "name": "AIIA",
                        "description": "",
                        "tags": ["layer:aiia"],
                        "card_url": "/a2a/agents/aiia-ask/.well-known/agent.json",
                        "rpc_url": "/a2a/agents/aiia-ask/rpc",
                    },
                ],
            },
        )

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        agents = await c.list_agents()

    assert len(agents) == 2
    assert all(isinstance(a, AgentSummary) for a in agents)
    assert agents[0].agent_id == "brain-status"
    assert captured_params == [[]]  # no params when tags=None and require_all=False


@pytest.mark.asyncio
async def test_list_agents_with_tags_sends_repeated_query_params():
    captured_params: List[List[tuple]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.append(list(request.url.params.multi_items()))
        return httpx.Response(200, json={"count": 0, "agents": []})

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        await c.list_agents(tags=["scope:global", "layer:aiia"], require_all=True)

    params = captured_params[0]
    assert ("tag", "scope:global") in params
    assert ("tag", "layer:aiia") in params
    assert ("require_all", "true") in params


@pytest.mark.asyncio
async def test_list_agents_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.list_agents()
    assert exc_info.value.status_code == 500


# ─── get_agent_card ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_agent_card_parses_camelcase_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/.well-known/agent.json")
        return httpx.Response(
            200,
            json={
                "name": "AIIA",
                "description": "test",
                "version": "0.1.0",
                "supportedInterfaces": [
                    {
                        "url": "http://mini.test:8100/a2a/agents/aiia-ask/rpc",
                        "protocolBinding": "JSONRPC",
                        "protocolVersion": "1.0",
                    }
                ],
                "skills": [],
            },
        )

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        card = await c.get_agent_card("aiia-ask")

    assert card.name == "AIIA"
    assert len(card.supported_interfaces) == 1
    assert card.supported_interfaces[0].protocol_binding == "JSONRPC"


@pytest.mark.asyncio
async def test_get_agent_card_unknown_agent_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown agent: nope"})

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.get_agent_card("nope")
    assert exc_info.value.status_code == 404


# ─── send_message ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_message_constructs_jsonrpc_envelope():
    captured_payload: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        assert request.url.path == "/a2a/agents/aiia-ask/rpc"
        return httpx.Response(200, json=_ok_task_response("hello back"))

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        result = await c.send_message(
            agent_id="aiia-ask",
            text="hello",
            request_id="req-test",
        )

    # Wire envelope
    assert captured_payload["jsonrpc"] == "2.0"
    assert captured_payload["id"] == "req-test"
    assert captured_payload["method"] == "SendMessage"
    assert captured_payload["params"]["message"]["role"] == "user"
    assert captured_payload["params"]["message"]["parts"][0]["text"] == "hello"

    # Result
    assert isinstance(result, TaskResult)
    assert result.is_completed
    assert result.state == "TASK_STATE_COMPLETED"
    assert result.answer_text() == "hello back"


@pytest.mark.asyncio
async def test_send_message_rejects_empty_text():
    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(lambda r: httpx.Response(500))
    ) as c:
        with pytest.raises(ValueError):
            await c.send_message(agent_id="aiia-ask", text="   ")


@pytest.mark.asyncio
async def test_send_message_maps_jsonrpc_error_to_client_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_err_response(-32601, "unsupported method"))

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.send_message(agent_id="aiia-ask", text="hi")

    assert exc_info.value.code == -32601
    assert "unsupported method" in str(exc_info.value)


@pytest.mark.asyncio
async def test_send_message_maps_4xx_jsonrpc_error_envelope():
    """Router returns 200 for JSON-RPC errors, but 4xx for unknown agent (404)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=_err_response(-32601, "unknown agent: ghost"))

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.send_message(agent_id="ghost", text="hi")
    assert exc_info.value.code == -32601
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_send_message_5xx_raises_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="server overloaded")

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.send_message(agent_id="aiia-ask", text="hi")
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_send_message_network_error_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.send_message(agent_id="aiia-ask", text="hi")
    assert "network error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_send_message_malformed_task_payload_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": "x", "result": {}},  # missing 'task'
        )

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        with pytest.raises(A2AClientError) as exc_info:
            await c.send_message(agent_id="aiia-ask", text="hi")
    assert "missing result.task" in str(exc_info.value)


# ─── TaskResult helpers ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_result_concatenates_multi_artifact_text():
    payload = _ok_task_response("first")
    payload["result"]["task"]["artifacts"].append(
        {
            "artifactId": "artifact-2",
            "name": "extra",
            "parts": [{"text": "second", "kind": "text"}],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with A2AClient(
        "http://mini.test:8100", http_client=_mock_client(handler)
    ) as c:
        result = await c.send_message(agent_id="aiia-ask", text="hi")

    assert result.answer_text() == "first\n\nsecond"
    assert result.answer_text(separator=" | ") == "first | second"
