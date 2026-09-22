import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from local_brain.command_center import slack_capture, slack_receipts
from local_brain.command_center.memory_inbox import MemoryInbox
from local_brain.egress import EgressDecision


@pytest.fixture
def inbox(tmp_path):
    return MemoryInbox(tmp_path / "inbox.sqlite3")


def capture(inbox, *, thread="1789260567.123456", key="event:1"):
    return inbox.capture(
        text="<@U0BOT1> log this EPIC for LNS",
        source_key=key,
        source="slack",
        project="mindmoor",
        workspace_id="T_TEST",
        channel_id="C_TEST",
        author_id="U_AUTHOR",
        receipt_thread_ts=thread,
    )


def test_migration_keeps_old_rows_and_is_repeatable(tmp_path):
    path = tmp_path / "inbox.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE ideas (
            id TEXT PRIMARY KEY, source_key TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL, source TEXT NOT NULL, project TEXT NOT NULL,
            workspace_id TEXT NOT NULL, channel_id TEXT NOT NULL,
            author_id TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unreviewed'
        )""")
        db.execute("""CREATE TABLE capture_receipts (
            idea_id TEXT PRIMARY KEY, thread_ts TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt REAL NOT NULL DEFAULT 0, lease TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '', slack_ts TEXT NOT NULL DEFAULT ''
        )""")
        db.execute(
            "INSERT INTO ideas VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                "k",
                "kept",
                "slack",
                "mindmoor",
                "T",
                "C",
                "U",
                "2026-09-14T00:00:00",
                "unreviewed",
            ),
        )
        db.execute(
            "INSERT INTO capture_receipts (idea_id,thread_ts,status) VALUES ('old','1.2','sent')"
        )
    for _ in range(2):
        listing = MemoryInbox(path).list()
        assert listing["total"] == 1
        row = listing["ideas"][0]
        assert row["text"] == "kept"
        assert row["memory_id"] == "" and row["reviewed_at"] == ""
        assert row["acknowledgement_status"] == "sent"
        assert row["promotion_status"] is None
        assert listing["counts"] == {"unreviewed": 1, "promoted": 0, "dismissed": 0}


def test_priority_migration_is_additive_and_repeatable(tmp_path):
    path = tmp_path / "inbox.sqlite3"
    idea = capture(MemoryInbox(path))
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE ideas DROP COLUMN priority")
        db.execute("ALTER TABLE ideas DROP COLUMN post_requested")
        columns = {row[1] for row in db.execute("PRAGMA table_info(ideas)")}
        assert "priority" not in columns and "post_requested" not in columns
    for _ in range(2):
        row = MemoryInbox(path).get(idea["id"])
        assert row["text"] == idea["text"]
        assert row["priority"] == "normal" and row["post_requested"] == 0
    with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE ideas SET priority=NULL")


def test_promote_records_memory_and_queues_one_receipt(inbox):
    idea = capture(inbox)
    updated = inbox.promote(idea["id"], memory_id="project_1_1", category="project", note=" why ")
    assert updated["status"] == "promoted"
    assert updated["memory_id"] == "project_1_1"
    assert updated["memory_category"] == "project"
    assert updated["review_note"] == "why"
    assert updated["reviewed_at"]
    assert updated["promotion_status"] == "pending"
    assert inbox.receipt_status("promotion") == {"pending": 1}
    with pytest.raises(ValueError, match="idea_already_promoted"):
        inbox.promote(idea["id"], memory_id="project_1_2", category="project")
    assert inbox.receipt_status("promotion") == {"pending": 1}


def test_promote_records_priority_and_rejects_unknown(inbox):
    idea = capture(inbox)
    assert idea["priority"] == "normal"
    with pytest.raises(ValueError, match="invalid_priority"):
        inbox.promote(idea["id"], memory_id="m", category="project", priority="critical")
    assert inbox.get(idea["id"])["status"] == "unreviewed"
    updated = inbox.promote(idea["id"], memory_id="m", category="project", priority="urgent")
    assert updated["priority"] == "urgent"
    assert inbox.get(idea["id"])["priority"] == "urgent"


def test_list_filters_and_sorts_by_priority(inbox):
    low = capture(inbox, key="event:1")
    urgent = capture(inbox, key="event:2")
    plain = capture(inbox, key="event:3")
    inbox.promote(low["id"], memory_id="m1", category="project", priority="low")
    inbox.promote(urgent["id"], memory_id="m2", category="project", priority="urgent")
    newest = [row["id"] for row in inbox.list()["ideas"]]
    ranked = [row["id"] for row in inbox.list(sort="priority")["ideas"]]
    assert ranked[0] == urgent["id"] and ranked[-1] == low["id"]
    assert ranked[1] == plain["id"]
    assert sorted(newest) == sorted(ranked)
    only = inbox.list(priority="urgent")
    assert [row["id"] for row in only["ideas"]] == [urgent["id"]] and only["total"] == 1
    assert only["counts"] == {"unreviewed": 0, "promoted": 1, "dismissed": 0}
    for bad in ({"priority": "critical"}, {"sort": "oldest"}):
        with pytest.raises(ValueError, match="invalid_idea_query"):
            inbox.list(**bad)


def test_promote_without_thread_queues_nothing(inbox):
    idea = capture(inbox, thread="")
    updated = inbox.promote(idea["id"], memory_id="m", category="lessons")
    assert updated["status"] == "promoted" and updated["promotion_status"] is None
    assert inbox.receipt_status("promotion") == {}


def test_promote_validation(inbox):
    with pytest.raises(ValueError, match="idea_not_found"):
        inbox.promote("missing", memory_id="m", category="project")
    idea = capture(inbox)
    with pytest.raises(ValueError, match="invalid_promotion"):
        inbox.promote(idea["id"], memory_id="", category="project")
    assert inbox.get(idea["id"])["status"] == "unreviewed"


def test_dismiss_and_restore_transitions(inbox):
    idea = capture(inbox)
    dismissed = inbox.dismiss(idea["id"], note="test only")
    assert dismissed["status"] == "dismissed" and dismissed["review_note"] == "test only"
    assert dismissed["reviewed_at"]
    with pytest.raises(ValueError, match="idea_not_dismissable"):
        inbox.dismiss(idea["id"])
    restored = inbox.restore(idea["id"])
    assert restored["status"] == "unreviewed"
    assert restored["review_note"] == "" and restored["reviewed_at"] == ""
    assert restored["review_outcome"] == ""
    with pytest.raises(ValueError, match="idea_not_restorable"):
        inbox.restore(idea["id"])
    inbox.promote(idea["id"], memory_id="m", category="project")
    with pytest.raises(ValueError, match="idea_not_dismissable"):
        inbox.dismiss(idea["id"])
    with pytest.raises(ValueError, match="idea_not_restorable"):
        inbox.restore(idea["id"])
    with pytest.raises(ValueError, match="idea_not_found"):
        inbox.dismiss("missing")


def test_list_status_filter_and_counts(inbox):
    first = capture(inbox, key="event:1")
    second = capture(inbox, key="event:2")
    capture(inbox, key="event:3")
    inbox.promote(first["id"], memory_id="m", category="project")
    inbox.dismiss(second["id"])
    assert inbox.list()["counts"] == {"unreviewed": 1, "promoted": 1, "dismissed": 1}
    promoted = inbox.list(status="promoted")
    assert promoted["total"] == 1 and promoted["ideas"][0]["id"] == first["id"]
    assert promoted["counts"] == {"unreviewed": 1, "promoted": 1, "dismissed": 1}
    assert inbox.list(status="unreviewed", query="epic")["total"] == 1
    assert inbox.list(project="other")["counts"] == {"unreviewed": 0, "promoted": 0, "dismissed": 0}
    with pytest.raises(ValueError, match="invalid_idea_status"):
        inbox.list(status="weird")


@pytest.fixture
def delivering(inbox, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_TEST")
    monkeypatch.setenv("AIIA_SLACK_CHANNEL_IDS", "C_TEST")
    monkeypatch.setattr(
        slack_receipts, "authorize_egress", AsyncMock(return_value=EgressDecision(True, "test"))
    )
    return inbox


def deliver(inbox, handler):
    asyncio.run(slack_receipts.deliver_one(inbox, transport=httpx.MockTransport(handler)))


def test_promotion_receipt_is_fixed_text_in_same_thread(delivering):
    idea = capture(delivering)
    deliver(delivering, lambda r: httpx.Response(200, json={"ok": True, "ts": "1.1"}))
    delivering.promote(idea["id"], memory_id="project_7_1", category="project")
    seen = []

    def handler(request):
        data = json.loads(request.content)
        seen.append(data)
        assert "EPIC" not in request.content.decode()
        assert data["thread_ts"] == "1789260567.123456"
        assert data["channel"] == "C_TEST"
        assert "Logged to AIIA memory" in data["text"]
        assert idea["id"] in data["text"] and "project_7_1" in data["text"]
        return httpx.Response(200, json={"ok": True, "ts": "2.2"})

    deliver(delivering, handler)
    assert len(seen) == 1
    row = delivering.get(idea["id"])
    assert row["acknowledgement_status"] == "sent" and row["acknowledgement_ts"] == "1.1"
    assert row["promotion_status"] == "sent" and row["promotion_ts"] == "2.2"
    deliver(delivering, lambda r: pytest.fail("duplicate promotion receipt"))


def test_capture_and_promotion_receipts_use_distinct_message_ids(delivering):
    idea = capture(delivering)
    delivering.promote(idea["id"], memory_id="m", category="project")
    ids = []

    def handler(request):
        ids.append(json.loads(request.content)["client_msg_id"])
        return httpx.Response(200, json={"ok": True, "ts": "1.1"})

    deliver(delivering, handler)
    deliver(delivering, handler)
    assert len(ids) == 2 and ids[0] != ids[1]
    assert delivering.receipt_status() == {"sent": 1}
    assert delivering.receipt_status("promotion") == {"sent": 1}


def test_failed_promotion_receipt_can_be_retried(delivering):
    idea = capture(delivering, thread="9.9")
    deliver(delivering, lambda r: httpx.Response(200, json={"ok": True, "ts": "1.1"}))
    delivering.promote(idea["id"], memory_id="m", category="project")
    deliver(delivering, lambda r: httpx.Response(200, json={"ok": False, "error": "missing_scope"}))
    assert delivering.get(idea["id"])["promotion_status"] == "failed"
    assert not delivering.retry_receipt(idea["id"])
    assert delivering.retry_receipt(idea["id"], "promotion")
    deliver(delivering, lambda r: httpx.Response(200, json={"ok": True, "ts": "3.3"}))
    assert delivering.get(idea["id"])["promotion_status"] == "sent"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    monkeypatch.setenv("AIIA_MEMORY_INBOX_PATH", str(tmp_path / "inbox.sqlite3"))
    monkeypatch.setenv("LOCAL_BRAIN_API_KEY", "synthetic-brain-key")
    app = FastAPI()
    app.include_router(slack_capture.router)
    return app


def brain(monkeypatch, handler):
    monkeypatch.setattr(slack_capture, "BRAIN_TRANSPORT", httpx.MockTransport(handler))


def call(app, method, path, body=None):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, json=body)

    return asyncio.run(exercise())


def test_capture_text_strips_only_mentions():
    assert slack_capture.capture_text("<@U0C1DCQFMRC>  log this EPIC") == "log this EPIC"
    assert slack_capture.capture_text("<@U1>") == ""
    assert slack_capture.capture_text("keep <@U1> middle") == "keep  middle"


def test_promote_route_stores_fact_with_provenance_and_queues_receipt(app, monkeypatch):
    idea = capture(slack_capture.inbox())
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["x-api-key"] == "synthetic-brain-key"
        data = json.loads(request.content)
        assert data["fact"] == "log this EPIC for LNS"
        assert data["category"] == "decisions"
        assert data["source"] == "slack:mindmoor"
        assert data["metadata"]["capture_id"] == idea["id"]
        assert data["metadata"]["channel_id"] == "C_TEST"
        assert data["metadata"]["author_id"] == "U_AUTHOR"
        assert data["metadata"]["review_note"] == "ship it"
        return httpx.Response(200, json={"id": "decisions_3_1789", "fact": data["fact"]})

    brain(monkeypatch, handler)
    response = call(
        app,
        "POST",
        f"/api/memory-inbox/{idea['id']}/promote",
        {"category": "decisions", "note": "ship it"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["memory_id"] == "decisions_3_1789"
    assert body["idea"]["status"] == "promoted"
    assert body["idea"]["promotion_status"] == "pending"
    assert len(calls) == 1
    assert call(app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {}).status_code == 409
    assert len(calls) == 1
    assert call(app, "GET", "/api/memory-inbox?status=promoted").json()["total"] == 1
    status = call(app, "GET", "/api/integrations/slack/status").json()
    assert status["promotion_acknowledgements"] == {"pending": 1}


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (lambda r: httpx.Response(422, json={"detail": "too short"}), 422),
        (lambda r: httpx.Response(500), 503),
        (lambda r: httpx.Response(200, json={}), 503),
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")), 503),
    ],
)
def test_promote_route_leaves_idea_unreviewed_when_brain_refuses(app, monkeypatch, handler, code):
    idea = capture(slack_capture.inbox())
    brain(monkeypatch, handler)
    response = call(app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {"category": "project"})
    assert response.status_code == code
    row = slack_capture.inbox().get(idea["id"])
    assert row["status"] == "unreviewed" and row["memory_id"] == ""
    assert slack_capture.inbox().receipt_status("promotion") == {}


def test_promote_route_validation(app, monkeypatch):
    brain(monkeypatch, lambda r: pytest.fail("brain must not be called"))
    assert call(app, "POST", "/api/memory-inbox/missing/promote", {}).status_code == 404
    idea = capture(slack_capture.inbox())
    bad = call(app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {"category": "sessions"})
    assert bad.status_code == 422
    empty = slack_capture.inbox().capture(
        text="<@U0BOT1>", source_key="event:empty", source="slack", project="mindmoor"
    )
    assert call(app, "POST", f"/api/memory-inbox/{empty['id']}/promote", {}).status_code == 422
    assert call(app, "GET", "/api/memory-inbox?status=weird").status_code == 422


def test_promote_route_records_priority(app, monkeypatch):
    idea = capture(slack_capture.inbox())
    brain(monkeypatch, lambda r: pytest.fail("brain must not be called"))
    bad = call(app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {"priority": "asap"})
    assert bad.status_code == 422 and bad.json()["detail"] == "invalid_priority"
    brain(monkeypatch, lambda r: httpx.Response(200, json={"id": "project_1_1"}))
    response = call(app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {"priority": "high"})
    assert response.status_code == 200, response.text
    assert response.json()["idea"]["priority"] == "high"
    assert response.json()["idea"]["post_requested"] == 0
    listing = call(app, "GET", "/api/memory-inbox?priority=high&sort=priority").json()
    assert listing["total"] == 1
    assert call(app, "GET", "/api/memory-inbox?priority=asap").status_code == 422
    assert call(app, "GET", "/api/memory-inbox?sort=oldest").status_code == 422


def test_dismiss_and_restore_routes(app):
    idea = capture(slack_capture.inbox())
    dismissed = call(app, "POST", f"/api/memory-inbox/{idea['id']}/dismiss", {"note": "test noise"})
    assert dismissed.status_code == 200
    assert dismissed.json()["idea"]["status"] == "dismissed"
    assert call(app, "POST", f"/api/memory-inbox/{idea['id']}/dismiss", {}).status_code == 409
    assert call(app, "GET", "/api/memory-inbox?status=dismissed").json()["total"] == 1
    restored = call(app, "POST", f"/api/memory-inbox/{idea['id']}/restore")
    assert restored.status_code == 200 and restored.json()["idea"]["status"] == "unreviewed"
    assert call(app, "POST", f"/api/memory-inbox/{idea['id']}/restore").status_code == 409
    assert call(app, "POST", "/api/memory-inbox/missing/dismiss", {}).status_code == 404
    assert call(app, "POST", "/api/memory-inbox/missing/restore").status_code == 404


def test_promotion_receipt_retry_route(app, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    idea = capture(slack_capture.inbox())
    assert (
        call(
            app, "POST", f"/api/memory-inbox/{idea['id']}/acknowledgement/retry?kind=weird"
        ).status_code
        == 422
    )
    assert (
        call(
            app, "POST", f"/api/memory-inbox/{idea['id']}/acknowledgement/retry?kind=promotion"
        ).status_code
        == 409
    )
