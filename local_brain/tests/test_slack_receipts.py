import asyncio
import json
import sqlite3
import time
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from local_brain.command_center import slack_capture, slack_receipts
from local_brain.command_center.memory_inbox import MemoryInbox
from local_brain.egress import EgressDecision, airgap_allows_tool


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_TEST")
    monkeypatch.setenv("AIIA_SLACK_CHANNEL_IDS", "C_TEST")
    monkeypatch.setattr(
        slack_receipts, "authorize_egress", AsyncMock(return_value=EgressDecision(True, "test"))
    )
    return MemoryInbox(tmp_path / "inbox.sqlite3")


def capture(inbox):
    return inbox.capture(
        text="private idea",
        source_key="event:1",
        source="slack",
        project="mindmoor",
        workspace_id="T_TEST",
        channel_id="C_TEST",
        receipt_thread_ts="1789260567.123456",
    )


def deliver(inbox, handler):
    asyncio.run(slack_receipts.deliver_one(inbox, transport=httpx.MockTransport(handler)))


def test_receipt_only_after_durable_save_and_deduplicated(receipts):
    idea = capture(receipts)
    capture(receipts)
    assert receipts.receipt_status() == {"pending": 1}

    def handler(request):
        assert receipts.list()["total"] == 1
        data = json.loads(request.content)
        assert "private idea" not in request.content.decode()
        assert data["thread_ts"] == "1789260567.123456"
        assert data["channel"] == "C_TEST"
        assert idea["id"] in data["text"]
        assert "for review" in data["text"]
        assert str(request.url) == "https://slack.com/api/chat.postMessage"
        return httpx.Response(200, json={"ok": True, "ts": "1789260568.123456"})

    deliver(receipts, handler)
    assert receipts.receipt_status() == {"sent": 1}
    deliver(receipts, lambda r: pytest.fail("duplicate receipt"))


def test_transaction_rollback_includes_idea(receipts):
    with receipts.connect() as db:
        db.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON capture_receipts "
            "BEGIN SELECT RAISE(ABORT,'synthetic'); END"
        )
    with pytest.raises(sqlite3.Error):
        capture(receipts)
    assert receipts.list()["total"] == 0


def test_claim_lease_and_restart_recovery(receipts):
    capture(receipts)
    first = receipts.claim_receipt()
    restarted = MemoryInbox(receipts.path)
    assert restarted.claim_receipt() is None
    with restarted.connect() as db:
        db.execute("UPDATE capture_receipts SET next_attempt=0")
    second = restarted.claim_receipt()
    assert second["lease"] != first["lease"]
    restarted.finish_receipt(first, status="sent")
    assert restarted.receipt_status() == {"sending": 1}
    restarted.finish_receipt(second, status="sent")
    assert restarted.receipt_status() == {"sent": 1}


@pytest.mark.parametrize(
    "env,value", [("AIIA_SLACK_ACK_ENABLED", "0"), ("AIIA_SLACK_BOT_TOKEN", "")]
)
def test_disabled_never_dials(receipts, monkeypatch, env, value):
    capture(receipts)
    monkeypatch.setenv(env, value)
    deliver(receipts, lambda r: pytest.fail("disabled receipt dialed"))
    assert receipts.receipt_status() == {"pending": 1}


def test_source_rechecked_before_delivery(receipts, monkeypatch):
    capture(receipts)
    monkeypatch.setenv("AIIA_SLACK_CHANNEL_IDS", "C_OTHER")
    deliver(receipts, lambda r: pytest.fail("disallowed channel dialed"))
    assert receipts.receipt_status() == {"failed": 1}


def test_egress_denied_does_not_post(receipts, monkeypatch):
    capture(receipts)
    monkeypatch.setattr(
        slack_receipts, "authorize_egress", AsyncMock(return_value=EgressDecision(False, "deny"))
    )
    deliver(receipts, lambda r: pytest.fail("egress denied"))
    assert receipts.receipt_status() == {"pending": 1}


def test_rate_limit_persists_retry(receipts):
    capture(receipts)
    deliver(receipts, lambda r: httpx.Response(429, headers={"Retry-After": "120"}))
    assert receipts.receipt_status() == {"pending": 1}
    with receipts.connect() as db:
        row = db.execute("SELECT * FROM capture_receipts").fetchone()
        assert row["next_attempt"] >= time.time() + 115
        assert row["error"] == "rate_limited"
    assert receipts.claim_receipt() is None


@pytest.mark.parametrize("code", ["invalid_auth", "missing_scope", "not_in_channel"])
def test_actionable_failure_retains_memory(receipts, code):
    capture(receipts)
    deliver(receipts, lambda r: httpx.Response(200, json={"ok": False, "error": code}))
    assert receipts.receipt_status() == {"failed": 1}
    assert receipts.list()["total"] == 1


def test_transport_failure_is_bounded_and_sanitized(receipts):
    capture(receipts)

    def handler(request):
        raise httpx.ReadTimeout("synthetic-bot-token")

    with receipts.connect() as db:
        db.execute("UPDATE capture_receipts SET attempts=7")
    deliver(receipts, handler)
    assert receipts.receipt_status() == {"failed": 1}
    with receipts.connect() as db:
        assert (
            db.execute("SELECT error FROM capture_receipts").fetchone()[0] == "delivery_unavailable"
        )


def test_ack_exception_does_not_enable_general_slack(monkeypatch):
    monkeypatch.delenv("AIIA_SLACK_ACK_ENABLED", raising=False)
    assert not airgap_allows_tool("slack.capture_ack")
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    assert airgap_allows_tool("slack.capture_ack")
    assert not airgap_allows_tool("slack.post")


def test_router_lifespan_starts_and_stops_worker(monkeypatch):
    started = asyncio.Event()
    stopped = []

    async def worker(factory):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.append(True)

    monkeypatch.setattr(slack_receipts, "run_worker", worker)

    async def exercise():
        app = FastAPI()
        app.include_router(slack_capture.router)
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(started.wait(), 1)

    asyncio.run(exercise())
    assert stopped == [True]


def test_failed_receipt_can_be_retried_without_resaving(receipts):
    idea = capture(receipts)
    deliver(receipts, lambda r: httpx.Response(200, json={"ok": False, "error": "missing_scope"}))
    row = receipts.list()["ideas"][0]
    assert row["acknowledgement_status"] == "failed"
    assert row["acknowledgement_error"] == "missing_scope"
    assert receipts.retry_receipt(idea["id"])
    assert not receipts.retry_receipt(idea["id"])
    deliver(receipts, lambda r: httpx.Response(200, json={"ok": True, "ts": "123.456"}))
    assert receipts.list()["total"] == 1
    assert receipts.list()["ideas"][0]["acknowledgement_status"] == "sent"
