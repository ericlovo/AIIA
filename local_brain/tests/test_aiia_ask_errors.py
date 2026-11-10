"""POST /v1/aiia/ask must not leak unhandled 500s from the local model."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from local_brain import local_api


async def _noop_metrics(**kwargs):
    return None


def _fake_brain(*, side_effect=None, result=None):
    fake = MagicMock()
    if result is not None:
        fake.ask = AsyncMock(side_effect=side_effect, return_value=result)
    else:
        fake.ask = AsyncMock(side_effect=side_effect)
    fake._model = "qwen3:8b"
    fake._ollama = MagicMock(timeout=300.0)
    return fake


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(status, request=request, text="model blew up")
    return httpx.HTTPStatusError("model blew up", request=request, response=response)


@pytest.fixture
def patch_ask(monkeypatch):
    monkeypatch.setattr(local_api, "_report_metrics", _noop_metrics)
    monkeypatch.setattr(local_api.asyncio, "sleep", AsyncMock())

    def _install(brain):
        monkeypatch.setattr(local_api, "_require_aiia", AsyncMock(return_value=brain))
        return brain

    return _install


@pytest.mark.asyncio
async def test_ask_retries_timeout_then_returns_structured_503(patch_ask):
    brain = patch_ask(_fake_brain(side_effect=httpx.ReadTimeout("")))
    with pytest.raises(HTTPException) as error:
        await local_api.aiia_ask(local_api.AIIAAskRequest(question="brief"))
    assert error.value.status_code == 503
    assert error.value.detail == {
        "error": "local_model_timeout",
        "reason": "Local model generation timed out after 300s (model=qwen3:8b)",
        "model": "qwen3:8b",
    }
    assert brain.ask.await_count == 2
    local_api.asyncio.sleep.assert_awaited_once_with(local_api.ASK_RETRY_BACKOFF_SECONDS)


@pytest.mark.asyncio
async def test_ask_retries_connect_error_then_returns_structured_503(patch_ask):
    brain = patch_ask(_fake_brain(side_effect=httpx.ConnectError("connection refused")))
    with pytest.raises(HTTPException) as error:
        await local_api.aiia_ask(local_api.AIIAAskRequest(question="brief"))
    assert error.value.status_code == 503
    assert error.value.detail["error"] == "local_model_unavailable"
    assert "ConnectError" in error.value.detail["reason"]
    assert "connection refused" in error.value.detail["reason"]
    assert error.value.detail["model"] == "qwen3:8b"
    assert brain.ask.await_count == 2


@pytest.mark.asyncio
async def test_ask_retries_ollama_500_then_returns_structured_503(patch_ask):
    brain = patch_ask(_fake_brain(side_effect=_http_status_error(500)))
    with pytest.raises(HTTPException) as error:
        await local_api.aiia_ask(local_api.AIIAAskRequest(question="brief"))
    assert error.value.status_code == 503
    assert error.value.detail["error"] == "local_model_unavailable"
    assert "HTTP 500" in error.value.detail["reason"]
    assert brain.ask.await_count == 2


@pytest.mark.asyncio
async def test_ask_retries_ollama_400_then_returns_structured_400(patch_ask):
    brain = patch_ask(_fake_brain(side_effect=_http_status_error(400)))
    with pytest.raises(HTTPException) as error:
        await local_api.aiia_ask(local_api.AIIAAskRequest(question="brief"))
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "local_model_rejected"
    assert "HTTP 400" in error.value.detail["reason"]
    assert brain.ask.await_count == 2


@pytest.mark.asyncio
async def test_ask_succeeds_on_retry_after_local_model_failure(patch_ask):
    brain = patch_ask(
        _fake_brain(side_effect=[httpx.ConnectError("refused"), {"answer": "recovered brief"}])
    )
    result = await local_api.aiia_ask(local_api.AIIAAskRequest(question="brief"))
    assert result == {"answer": "recovered brief"}
    assert brain.ask.await_count == 2
    local_api.asyncio.sleep.assert_awaited_once_with(local_api.ASK_RETRY_BACKOFF_SECONDS)


@pytest.mark.asyncio
async def test_ask_does_not_mask_non_model_errors(patch_ask):
    brain = patch_ask(_fake_brain(side_effect=ValueError("chroma exploded")))
    with pytest.raises(ValueError, match="chroma exploded"):
        await local_api.aiia_ask(local_api.AIIAAskRequest(question="brief"))
    assert brain.ask.await_count == 1
    local_api.asyncio.sleep.assert_not_awaited()
