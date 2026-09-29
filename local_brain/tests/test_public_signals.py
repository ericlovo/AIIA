import asyncio
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from defusedxml.common import EntitiesForbidden
from fastapi import FastAPI

from local_brain.command_center import public_signals as signals
from local_brain.command_center import signal_routes
from local_brain.command_center.memory_inbox import MemoryInbox
from local_brain.egress import EgressDecision, airgap_allows_tool


@pytest.fixture
def service(tmp_path, monkeypatch):
    for flag in ("AIIA_NEWS_ENABLED", "AIIA_SIGNALS_ENABLED"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    return signals.SignalJobs(tmp_path / "signals.sqlite3", lambda: inbox)


def item(index=0):
    return {
        "key": f"key{index}",
        "title": f"Iowa manufacturer {index} expands",
        "publisher": "Public source",
        "url": f"https://news.google.com/rss/articles/{index}",
        "published_at": datetime.now(timezone.utc).isoformat(),
    }


def response(count=1, choice="lead"):
    return {
        "model": "jev-test",
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "answers": {
            f"item_{i}": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.8,
                "probabilities": {key: 0.85 if key == choice else 0.05 for key in signals.CRITERIA},
            }
            for i in range(count)
        },
    }


def feed(url="https://news.google.com/rss/articles/abc", date=None):
    date = date or format_datetime(datetime.now(timezone.utc))
    return f"<rss><channel><item><title>Expansion</title><link>{url}</link><pubDate>{date}</pubDate><source>News</source></item></channel></rss>".encode()


def test_parse_feed():
    assert len(signals.parse_feed(feed(), time.time())) == 1
    assert signals.parse_feed(feed(date="bad date"), time.time()) == []
    assert signals.parse_feed(feed(date="Mon, 01 Jan 2024 00:00:00 GMT"), time.time()) == []
    for url in (
        "http://localhost/private",
        "https://evil.test/rss/articles/x",
        "https://news.google.com@evil.test/rss/articles/x",
        "javascript:alert(1)",
    ):
        assert signals.parse_feed(feed(url), time.time()) == []
    with pytest.raises(ValueError, match="feed_too_large"):
        signals.parse_feed(b"x" * (signals.MAX_BYTES + 1), time.time())
    with pytest.raises(EntitiesForbidden):
        signals.parse_feed(
            b'<!DOCTYPE rss [<!ENTITY x SYSTEM "file:///etc/passwd">]><rss>&x;</rss>', time.time()
        )


def test_flags_are_separate(monkeypatch):
    monkeypatch.delenv("AIIA_NEWS_ENABLED", raising=False)
    monkeypatch.delenv("AIIA_SIGNALS_ENABLED", raising=False)
    monkeypatch.setenv("AIIA_TYPESAFE_ENABLED", "1")
    assert not airgap_allows_tool("news.fetch")
    assert not airgap_allows_tool("typesafe.signals")
    monkeypatch.setenv("AIIA_NEWS_ENABLED", "1")
    assert airgap_allows_tool("news.fetch")
    assert not airgap_allows_tool("typesafe.signals")
    assert not airgap_allows_tool("web.fetch")


@pytest.mark.asyncio
async def test_bounded_idempotent_jobs(service, monkeypatch):
    monkeypatch.setattr(signals, "retrieve", AsyncMock(return_value=[item(i) for i in range(8)]))
    monkeypatch.setattr(signals, "screen", AsyncMock(return_value=response(8)))
    result = await service.run("lead_signals")
    assert result["created"] == 3
    assert result["usage"]["input_tokens"] == 100
    with service.inbox_factory().connect() as db:
        rows = db.execute("SELECT * FROM ideas").fetchall()
    assert len(rows) == 3
    assert all("No outreach authorized" in row["text"] for row in rows)
    restarted = signals.SignalJobs(service.path, service.inbox_factory)
    with pytest.raises(ValueError, match="job_cooldown"):
        await restarted.run("lead_signals")
    assert restarted.status()["jobs"][1]["last_run"]["status"] == "review_ready"


@pytest.mark.asyncio
async def test_duplicates_across_jobs(service, monkeypatch):
    monkeypatch.setattr(signals, "retrieve", AsyncMock(return_value=[item()]))
    screen = AsyncMock(return_value=response())
    monkeypatch.setattr(signals, "screen", screen)
    assert (await service.run("lead_signals"))["created"] == 1
    assert (await service.run("market_news"))["status"] == "no_change"
    assert screen.await_count == 1


@pytest.mark.asyncio
async def test_failure_keeps_reservation_and_hides_error(service, monkeypatch):
    retrieve = AsyncMock(side_effect=RuntimeError("SECRET provider payload"))
    monkeypatch.setattr(signals, "retrieve", retrieve)
    result = await service.run("market_news")
    assert result["status"] == "failed"
    assert "SECRET" not in str(service.status())
    with pytest.raises(ValueError, match="job_cooldown"):
        await service.run("market_news")
    assert retrieve.await_count == 1


@pytest.mark.asyncio
async def test_backpressure_prevents_network(service, monkeypatch):
    for i in range(6):
        service.inbox_factory().ingest(
            text="Review", source="public_signals", source_key=str(i), project="test"
        )
    retrieve = AsyncMock()
    monkeypatch.setattr(signals, "retrieve", retrieve)
    assert (await service.run("market_news"))["status"] == "review_backlog"
    retrieve.assert_not_awaited()


def test_cross_instance_lock_and_stale_recovery(service):
    service.reserve("market_news")
    other = signals.SignalJobs(service.path, service.inbox_factory)
    with pytest.raises(ValueError, match="signals_busy"):
        other.reserve("lead_signals")
    with service.connect() as db:
        db.execute("UPDATE runs SET started=?", (time.time() - 200,))
    other.reserve("lead_signals")
    assert service.status()["jobs"][0]["last_run"]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_cancellation_recorded(service, monkeypatch):
    monkeypatch.setattr(signals, "retrieve", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await service.run("lead_signals")
    assert service.status()["jobs"][1]["last_run"]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_screen_projection_and_contract(service, monkeypatch):
    original = httpx.AsyncClient

    def handler(request):
        import json

        body = json.loads(request.content)
        assert set(body["state"][0]) == {"title", "publisher", "published_at"}
        assert "PRIVATE" not in request.content.decode()
        assert request.url.host == "api.typesafe.ai"
        return httpx.Response(200, json=response())

    monkeypatch.setattr(
        signals, "authorize_egress", AsyncMock(return_value=EgressDecision(True, "test"))
    )
    monkeypatch.setattr(
        signals.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = await signals.screen([{**item(), "private_memory": "PRIVATE"}])
    assert result["usage"]["output_tokens"] == 10


@pytest.mark.asyncio
async def test_denied_before_network(service, monkeypatch):
    monkeypatch.setattr(
        signals, "authorize_egress", AsyncMock(return_value=EgressDecision(False, "denied"))
    )
    with pytest.raises(ValueError, match="egress_denied"):
        await signals.retrieve("lead_signals")
    with pytest.raises(ValueError, match="egress_denied"):
        await signals.screen([item()])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["type", "choice", "confidence", "probabilities", "usage", "answers"]
)
async def test_malformed_screening_response(service, monkeypatch, fault):
    payload = response()
    if fault == "usage":
        payload["usage"]["input_tokens"] = True
    elif fault == "answers":
        payload["answers"] = {}
    elif fault == "probabilities":
        payload["answers"]["item_0"][fault] = {"lead": 1}
    else:
        payload["answers"]["item_0"][fault] = "invalid"
    original = httpx.AsyncClient
    monkeypatch.setattr(
        signals, "authorize_egress", AsyncMock(return_value=EgressDecision(True, "test"))
    )
    monkeypatch.setattr(
        signals.httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            **kwargs,
        ),
    )
    with pytest.raises(ValueError):
        await signals.screen([item()])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 500])
async def test_retrieval_rejects_redirects_and_failure(service, monkeypatch, status):
    original = httpx.AsyncClient
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(status, headers={"location": "http://localhost/private"})

    monkeypatch.setattr(
        signals, "authorize_egress", AsyncMock(return_value=EgressDecision(True, "test"))
    )
    monkeypatch.setattr(
        signals.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await signals.retrieve("lead_signals")
    assert calls == ["news.google.com"]


@pytest.mark.asyncio
async def test_no_news_spends_no_tokens(service, monkeypatch):
    monkeypatch.setattr(signals, "retrieve", AsyncMock(return_value=[]))
    screening = AsyncMock()
    monkeypatch.setattr(signals, "screen", screening)
    assert (await service.run("market_news"))["status"] == "no_change"
    screening.assert_not_awaited()


@pytest.mark.asyncio
async def test_uncertain_does_not_refile_or_block_later_items(service, monkeypatch):
    monkeypatch.setattr(signals, "retrieve", AsyncMock(return_value=[item()]))
    screening = AsyncMock(return_value=response(choice="uncertain"))
    monkeypatch.setattr(signals, "screen", screening)
    assert (await service.run("market_news"))["created"] == 0
    with service.connect() as db:
        db.execute("UPDATE runs SET started=?", (time.time() - signals.INTERVAL - 1,))
    assert (await service.run("market_news"))["status"] == "no_change"
    assert screening.await_count == 1


def test_public_signal_review_outcome_and_recovery(service):
    idea, _ = service.inbox_factory().ingest(
        text="Public signal", source="public_signals", source_key="review", project="test"
    )
    service.inbox_factory().attach_assignment(idea["id"], "assignment")
    assert service.inbox_factory().get(idea["id"])["review_outcome"] == "needs_work"
    assert service.inbox_factory().review_health(days=14)["totals"]["needs_work"] == 1
    service.reserve("market_news")
    with service.connect() as db:
        db.execute("UPDATE runs SET started=?", (time.time() - 200,))
    assert service.status()["jobs"][0]["last_run"]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_routes(service, monkeypatch):
    monkeypatch.setattr(signal_routes, "jobs", lambda: service)
    app = FastAPI()
    app.include_router(signal_routes.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (await client.get("/api/signal-jobs")).status_code == 200
        assert (
            await client.put("/api/signal-jobs/lead_signals", json={"enabled": True})
        ).status_code == 200
        assert (
            await client.put("/api/signal-jobs/lead_signals", json={"enabled": "yes"})
        ).status_code == 422
        assert (await client.post("/api/signal-jobs/unknown/run")).status_code == 404
        monkeypatch.delenv("AIIA_NEWS_ENABLED")
        assert (await client.post("/api/signal-jobs/lead_signals/run")).status_code == 503
        assert (
            await client.put("/api/signal-jobs/lead_signals", json={"enabled": False})
        ).status_code == 200
