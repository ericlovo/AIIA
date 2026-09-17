"""
Tests for the `think` passthrough on /v1/chat.

qwen3 thinks by default under Ollama. Its hidden reasoning counts against
num_predict, so a capped call can return done_reason="length" with empty
content (the 2026-09-17 standup failure). Callers must be able to turn it off
and see why a response came back empty.

Run: pytest local_brain/tests/test_chat_think.py -v
"""

import pytest

from local_brain import local_api, ollama_client
from local_brain.config import LocalBrainConfig


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _patch_httpx(monkeypatch, data):
    sent = {}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json):
            sent["payload"] = json
            return _FakeResponse(data)

    monkeypatch.setattr(ollama_client.httpx, "AsyncClient", _FakeAsyncClient)
    return sent


def _client():
    config = LocalBrainConfig.__new__(LocalBrainConfig)
    config.ollama_url = "http://localhost:11434"
    config.ollama_timeout = 120.0
    return ollama_client.OllamaClient(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [False, True])
async def test_ollama_chat_sends_think_when_set(monkeypatch, think):
    sent = _patch_httpx(monkeypatch, {"message": {"content": "ok"}})

    await _client().chat("qwen3:8b", [{"role": "user", "content": "hi"}], think=think)

    assert sent["payload"]["think"] is think


@pytest.mark.asyncio
async def test_ollama_chat_omits_think_by_default(monkeypatch):
    sent = _patch_httpx(monkeypatch, {"message": {"content": "ok"}})

    await _client().chat("llama3.1:8b", [{"role": "user", "content": "hi"}])

    # Ollama rejects think=true on non-thinking models; None must send nothing.
    assert "think" not in sent["payload"]


class _RecordingOllama:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return self.response


async def _noop_metrics(**kwargs):
    return None


@pytest.mark.asyncio
async def test_local_chat_forwards_think_and_content(monkeypatch):
    fake = _RecordingOllama(
        {
            "message": {"content": "### Yesterday\n- shipped"},
            "done_reason": "stop",
            "eval_count": 134,
            "prompt_eval_count": 474,
            "_latency_ms": 8000.0,
        }
    )
    monkeypatch.setattr(local_api, "_ollama", fake)
    monkeypatch.setattr(local_api, "_report_metrics", _noop_metrics)

    resp = await local_api.local_chat(
        local_api.ChatRequest(
            messages=[{"role": "user", "content": "brief"}],
            model="qwen3:8b",
            max_tokens=2048,
            think=False,
        )
    )

    assert fake.kwargs["think"] is False
    assert resp.content == "### Yesterday\n- shipped"
    assert resp.done_reason == "stop"
    assert resp.usage == {"output_tokens": 134, "input_tokens": 474}


@pytest.mark.asyncio
async def test_local_chat_reports_length_when_budget_exhausted(monkeypatch):
    # Shape Ollama returns when thinking eats all of num_predict.
    fake = _RecordingOllama(
        {
            "message": {"content": "", "thinking": "Okay, let me..."},
            "done_reason": "length",
            "eval_count": 2048,
            "prompt_eval_count": 474,
            "_latency_ms": 120028.9,
        }
    )
    monkeypatch.setattr(local_api, "_ollama", fake)
    monkeypatch.setattr(local_api, "_report_metrics", _noop_metrics)

    resp = await local_api.local_chat(
        local_api.ChatRequest(
            messages=[{"role": "user", "content": "brief"}],
            model="qwen3:8b",
            max_tokens=2048,
        )
    )

    assert fake.kwargs["think"] is None
    assert resp.content == ""
    assert resp.done_reason == "length"
