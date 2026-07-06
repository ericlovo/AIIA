"""Air-gap mode: config force-overrides, egress guard fail-closed matrix,
and call-site degradation (distiller, whisper).

Everything runs without a live Sanction — HTTP is stubbed at the httpx
client layer, matching the repo's monkeypatch-not-respx convention.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from local_brain import egress
from local_brain.config import LocalBrainConfig
from local_brain.egress import EgressDecision, airgap_status, authorize_egress

# ----------------------------------------------------------------------------
# Config force-overrides
# ----------------------------------------------------------------------------


def test_airgap_forces_execution_and_research_off(monkeypatch):
    monkeypatch.setenv("AIIA_AIRGAP", "1")
    monkeypatch.setenv("EXECUTION_ENABLED", "true")
    monkeypatch.setenv("AIIA_RESEARCH_ENABLED", "true")
    cfg = LocalBrainConfig()
    assert cfg.airgap_enabled is True
    assert cfg.execution_enabled is False
    assert cfg.autonomy.research_enabled is False


def test_no_airgap_leaves_flags_alone(monkeypatch):
    monkeypatch.delenv("AIIA_AIRGAP", raising=False)
    monkeypatch.setenv("EXECUTION_ENABLED", "true")
    cfg = LocalBrainConfig()
    assert cfg.airgap_enabled is False
    assert cfg.execution_enabled is True


def test_airgap_status_shape(monkeypatch):
    on = airgap_status(SimpleNamespace(airgap_enabled=True))
    assert on["enabled"] is True
    assert on["egress"] and all(v == "disabled" for v in on["egress"].values())
    off = airgap_status(SimpleNamespace(airgap_enabled=False))
    assert all(v == "sanction-governed" for v in off["egress"].values())


# ----------------------------------------------------------------------------
# Egress guard — fail-closed matrix
# ----------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"content-type": "application/json"} if payload is not None else {}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stub for httpx.AsyncClient(...).post(...)."""

    def __init__(self, *, response: _FakeResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.posts: list[dict] = []

    def __call__(self, *args, **kwargs):  # httpx.AsyncClient(timeout=...) call
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "json": json})
        if self._error:
            raise self._error
        return self._response


def _set_mode(monkeypatch, *, airgap: bool, configured: bool = True):
    monkeypatch.setattr(egress, "get_config", lambda: SimpleNamespace(airgap_enabled=airgap))
    client = ("https://sanction.test/api/v1", "pxy_test") if configured else None
    monkeypatch.setattr(egress, "_client", lambda: client)


def test_airgap_denies_locally_and_reports(monkeypatch):
    _set_mode(monkeypatch, airgap=True)
    reported: list[tuple] = []
    monkeypatch.setattr(egress, "report_denied_bg", lambda t, s=None: reported.append((t, s)))
    decision = asyncio.run(authorize_egress("anthropic.messages"))
    assert decision.allowed is False
    assert "air-gap" in decision.reason
    assert reported == [("anthropic.messages", None)]


def test_airgap_denies_even_with_all_cloud_keys_set(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "configured-but-inert")
    _set_mode(monkeypatch, airgap=True)
    monkeypatch.setattr(egress, "report_denied_bg", lambda t, s=None: None)
    assert asyncio.run(authorize_egress("groq.whisper")).allowed is False


def test_hybrid_unconfigured_allows(monkeypatch):
    _set_mode(monkeypatch, airgap=False, configured=False)
    decision = asyncio.run(authorize_egress("slack.post"))
    assert decision.allowed is True


def test_hybrid_allows_only_on_authorized_true(monkeypatch):
    _set_mode(monkeypatch, airgap=False)
    fake = _FakeAsyncClient(response=_FakeResponse(200, {"authorized": True, "status": "allowed"}))
    monkeypatch.setattr(egress.httpx, "AsyncClient", fake)
    assert asyncio.run(authorize_egress("slack.post")).allowed is True


def test_hybrid_denies_on_403(monkeypatch):
    _set_mode(monkeypatch, airgap=False)
    fake = _FakeAsyncClient(
        response=_FakeResponse(403, {"authorized": False, "code": "TOOL_NOT_ALLOWED"})
    )
    monkeypatch.setattr(egress.httpx, "AsyncClient", fake)
    decision = asyncio.run(authorize_egress("anthropic.messages"))
    assert decision.allowed is False
    assert "TOOL_NOT_ALLOWED" in decision.reason


def test_hybrid_fails_closed_on_transport_error(monkeypatch):
    _set_mode(monkeypatch, airgap=False)
    fake = _FakeAsyncClient(error=ConnectionError("boom"))
    monkeypatch.setattr(egress.httpx, "AsyncClient", fake)
    decision = asyncio.run(authorize_egress("anthropic.messages"))
    assert decision.allowed is False
    assert "unreachable" in decision.reason


def test_hybrid_fails_closed_on_non_json_200(monkeypatch):
    _set_mode(monkeypatch, airgap=False)
    fake = _FakeAsyncClient(response=_FakeResponse(200, None))  # no JSON body
    monkeypatch.setattr(egress.httpx, "AsyncClient", fake)
    assert asyncio.run(authorize_egress("web.fetch")).allowed is False


# ----------------------------------------------------------------------------
# Call-site degradation
# ----------------------------------------------------------------------------


def _deny(tool, server=None):
    async def _inner(*a, **k):
        return EgressDecision(False, "denied: air-gap mode (AIIA_AIRGAP)")

    return _inner


def test_distiller_raises_distillation_error_on_deny(monkeypatch):
    from local_brain.journal import distiller

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(distiller, "authorize_egress", _deny("anthropic.messages"))
    inp = distiller.DistillationInput(
        transcript="test", started_at=datetime(2026, 7, 6), duration_seconds=1
    )
    with pytest.raises(distiller.DistillationError, match="air-gap"):
        asyncio.run(distiller.distill(inp))


def test_whisper_raises_transcription_error_on_deny(monkeypatch):
    from local_brain.journal import whisper

    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(whisper, "authorize_egress", _deny("groq.whisper"))
    with pytest.raises(whisper.TranscriptionError, match="air-gap"):
        asyncio.run(whisper.transcribe_audio(b"fake-bytes"))
