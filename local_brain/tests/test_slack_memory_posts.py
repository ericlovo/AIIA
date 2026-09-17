import asyncio
import json
import sqlite3
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from local_brain import egress
from local_brain.command_center import slack_capture, slack_memory_posts, slack_receipts
from local_brain.command_center.memory_inbox import MemoryInbox
from local_brain.egress import (
    AIRGAP_ALLOWED_EGRESS,
    EGRESS_POINTS,
    EgressDecision,
    airgap_allows_tool,
    airgap_status,
    authorize_egress,
)

CHANNEL = "C0MEMORY01"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("AIIA_SLACK_ACK_ENABLED", raising=False)
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_ENABLED", "1")
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", CHANNEL)
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_TEST")
    monkeypatch.setenv("AIIA_SLACK_CHANNEL_IDS", "C_TEST")
    monkeypatch.setenv("AIIA_MEMORY_INBOX_PATH", str(tmp_path / "inbox.sqlite3"))
    monkeypatch.setenv("LOCAL_BRAIN_API_KEY", "synthetic-brain-key")
    egress_check = AsyncMock(return_value=EgressDecision(True, "test"))
    monkeypatch.setattr(slack_memory_posts, "authorize_egress", egress_check)
    monkeypatch.setattr(
        slack_receipts, "authorize_egress", AsyncMock(return_value=EgressDecision(True, "test"))
    )
    return SimpleNamespace(inbox=slack_capture.inbox(), egress=egress_check)


@pytest.fixture
def app(env):
    app = FastAPI()
    app.include_router(slack_capture.router)
    return app


def capture(inbox, *, key="event:1", text="<@U0BOT1> ship the cron contract", thread=""):
    return inbox.capture(
        text=text,
        source_key=key,
        source="slack",
        project="mindmoor",
        workspace_id="T_TEST",
        channel_id="C_TEST",
        author_id="U_AUTHOR",
        receipt_thread_ts=thread,
    )


def queue(inbox, *, key, priority="normal", memory_id=None):
    idea = capture(inbox, key=key)
    memory_id = memory_id or f"project_{key}"
    inbox.promote(
        idea["id"],
        memory_id=memory_id,
        category="project",
        priority=priority,
        post_channel_id=CHANNEL,
        post_body=f"body for {memory_id}",
    )
    return idea


def brain(monkeypatch, handler):
    monkeypatch.setattr(slack_capture, "BRAIN_TRANSPORT", httpx.MockTransport(handler))


def call(app, method, path, body=None):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, json=body)

    return asyncio.run(exercise())


def deliver(inbox, handler):
    asyncio.run(slack_memory_posts.deliver_one(inbox, transport=httpx.MockTransport(handler)))


def ok(request):
    return httpx.Response(200, json={"ok": True, "ts": "1789260570.000100"})


def test_promote_with_post_enqueues_exactly_one_post(app, env, monkeypatch):
    idea = capture(env.inbox)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "decisions_4_1789"})

    brain(monkeypatch, handler)
    response = call(
        app,
        "POST",
        f"/api/memory-inbox/{idea['id']}/promote",
        {"category": "decisions", "priority": "high", "post_to_slack": True},
    )
    assert response.status_code == 200, response.text
    row = response.json()["idea"]
    assert row["post_requested"] == 1 and row["priority"] == "high"
    assert row["memory_post_status"] == "pending"
    with env.inbox.connect() as db:
        posts = [dict(r) for r in db.execute("SELECT * FROM memory_posts")]
    assert len(posts) == 1
    assert posts[0]["memory_id"] == "decisions_4_1789"
    assert posts[0]["idea_id"] == idea["id"]
    assert posts[0]["channel_id"] == CHANNEL and posts[0]["priority"] == "high"
    assert "ship the cron contract" in posts[0]["body"]
    again = call(
        app,
        "POST",
        f"/api/memory-inbox/{idea['id']}/promote",
        {"category": "decisions", "post_to_slack": True},
    )
    assert again.status_code == 409 and len(calls) == 1
    retry = call(
        app, "POST", f"/api/memory-inbox/{idea['id']}/acknowledgement/retry?kind=memory_post"
    )
    assert retry.status_code == 409 and retry.json()["detail"] == "no_failed_receipt"
    assert env.inbox.memory_post_status() == {"pending": 1}
    status = call(app, "GET", "/api/integrations/slack/status").json()
    assert status["memory_posts_enabled"] is True
    assert status["memory_posts_configured"] is True
    assert status["memory_post_channel_id"] == CHANNEL
    assert status["memory_posts"] == {"pending": 1}


def test_promote_without_post_queues_nothing(app, env, monkeypatch):
    idea = capture(env.inbox)
    brain(monkeypatch, lambda r: httpx.Response(200, json={"id": "project_1_1"}))
    response = call(app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {})
    assert response.status_code == 200
    assert response.json()["idea"]["post_requested"] == 0
    assert response.json()["idea"]["memory_post_status"] is None
    assert env.inbox.memory_post_status() == {}


@pytest.mark.parametrize(
    "name,value",
    [
        ("AIIA_SLACK_MEMORY_POST_ENABLED", "0"),
        ("AIIA_SLACK_BOT_TOKEN", ""),
        ("AIIA_SLACK_TEAM_ID", ""),
        ("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", ""),
        ("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "#aiia-memory"),
        ("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "C0short"),
        ("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "D0MEMORY01"),
    ],
)
def test_posting_disabled_refuses_before_brain(app, env, monkeypatch, name, value):
    idea = capture(env.inbox)
    monkeypatch.setenv(name, value)
    brain(monkeypatch, lambda r: pytest.fail("brain must not be called"))
    response = call(
        app,
        "POST",
        f"/api/memory-inbox/{idea['id']}/promote",
        {"category": "project", "post_to_slack": True},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "memory_posting_disabled"
    row = env.inbox.get(idea["id"])
    assert row["status"] == "unreviewed" and row["memory_id"] == ""
    assert row["post_requested"] == 0 and row["priority"] == "normal"
    assert env.inbox.memory_post_status() == {}
    status = call(app, "GET", "/api/integrations/slack/status").json()
    assert status["memory_posts_configured"] is False


def test_config_change_during_brain_call_still_records_promotion(app, env, monkeypatch):
    idea = capture(env.inbox)

    def handler(request):
        monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "")
        return httpx.Response(200, json={"id": "project_7_7"})

    brain(monkeypatch, handler)
    response = call(
        app,
        "POST",
        f"/api/memory-inbox/{idea['id']}/promote",
        {"post_to_slack": True},
    )
    # The Brain fact exists, so the capture must not be left unreviewed.
    assert response.status_code == 200, response.text
    assert env.inbox.get(idea["id"])["status"] == "promoted"
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "C0ELSEWHERE")
    deliver(env.inbox, lambda r: pytest.fail("changed destination must not be dialed"))
    assert env.inbox.get(idea["id"])["memory_post_error"] == "destination_not_allowed"


def test_post_rolls_back_with_the_promotion(env):
    idea = capture(env.inbox, thread="1789260567.123456")
    with env.inbox.connect() as db:
        db.execute(
            "CREATE TRIGGER reject_post BEFORE INSERT ON memory_posts "
            "BEGIN SELECT RAISE(ABORT,'synthetic'); END"
        )
    with pytest.raises(sqlite3.Error):
        queue(env.inbox, key="event:1")
    row = env.inbox.get(idea["id"])
    assert row["status"] == "unreviewed" and row["post_requested"] == 0
    assert env.inbox.receipt_status("promotion") == {}


def test_post_needs_both_channel_and_body(env):
    idea = capture(env.inbox)
    with pytest.raises(ValueError, match="invalid_promotion"):
        env.inbox.promote(idea["id"], memory_id="m", category="project", post_body="text")
    with pytest.raises(ValueError, match="invalid_promotion"):
        env.inbox.promote(idea["id"], memory_id="m", category="project", post_channel_id=CHANNEL)
    assert env.inbox.get(idea["id"])["status"] == "unreviewed"


def test_delivery_posts_body_to_configured_channel_once(env):
    queue(env.inbox, key="event:1", memory_id="project_9_9")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert str(request.url) == "https://slack.com/api/chat.postMessage"
        assert request.headers["authorization"] == "Bearer synthetic-bot-token"
        return ok(request)

    deliver(env.inbox, handler)
    assert seen == [
        {
            "channel": CHANNEL,
            "text": "body for project_9_9",
            "reply_broadcast": False,
            "unfurl_links": False,
            "unfurl_media": False,
            "mrkdwn": False,
            "client_msg_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "project_9_9:memory_post")),
        }
    ]
    env.egress.assert_awaited_once_with("slack.memory_post", server="slack.com")
    assert env.inbox.memory_post_status() == {"sent": 1}
    row = env.inbox.list()["ideas"][0]
    assert row["memory_post_status"] == "sent" and row["memory_post_ts"] == "1789260570.000100"
    deliver(env.inbox, lambda r: pytest.fail("duplicate memory post"))


def test_receipts_stay_textless_beside_memory_posts(env, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    idea = capture(env.inbox, thread="1789260567.123456")
    env.inbox.promote(
        idea["id"],
        memory_id="m1",
        category="project",
        post_channel_id=CHANNEL,
        post_body="ship the cron contract",
    )
    receipts = []

    def receipt_handler(request):
        receipts.append(request.content.decode())
        return ok(request)

    for _ in range(2):
        asyncio.run(
            slack_receipts.deliver_one(env.inbox, transport=httpx.MockTransport(receipt_handler))
        )
    assert len(receipts) == 2
    assert all("cron contract" not in body for body in receipts)
    deliver(env.inbox, lambda r: ok(r) if b"cron contract" in r.content else pytest.fail())
    assert env.inbox.memory_post_status() == {"sent": 1}


def test_team_is_rechecked_at_delivery(env, monkeypatch):
    queue(env.inbox, key="event:1")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_OTHER")
    deliver(env.inbox, lambda r: pytest.fail("other workspace dialed"))
    assert env.inbox.memory_post_status() == {"failed": 1}
    assert env.inbox.list()["ideas"][0]["memory_post_error"] == "source_not_allowed"


def test_changed_destination_is_never_dialed(env, monkeypatch):
    queue(env.inbox, key="event:1")
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "C0ELSEWHERE")
    deliver(env.inbox, lambda r: pytest.fail("unapproved channel dialed"))
    assert env.inbox.memory_post_status() == {"failed": 1}
    assert env.inbox.list()["ideas"][0]["memory_post_error"] == "destination_not_allowed"


def test_claims_urgent_before_low_then_oldest(env):
    queue(env.inbox, key="event:1", priority="low", memory_id="low")
    queue(env.inbox, key="event:2", priority="normal", memory_id="normal-late")
    queue(env.inbox, key="event:3", priority="urgent", memory_id="urgent")
    queue(env.inbox, key="event:4", priority="high", memory_id="high")
    queue(env.inbox, key="event:5", priority="normal", memory_id="normal-early")
    with env.inbox.connect() as db:
        db.execute("UPDATE memory_posts SET next_attempt=5 WHERE memory_id='normal-late'")
        db.execute("UPDATE memory_posts SET next_attempt=1 WHERE memory_id='normal-early'")
    order = []

    def handler(request):
        order.append(json.loads(request.content)["text"].removeprefix("body for "))
        return ok(request)

    for _ in range(5):
        deliver(env.inbox, handler)
    assert order == ["urgent", "high", "normal-early", "normal-late", "low"]


@pytest.mark.parametrize(
    "name,value",
    [
        ("AIIA_SLACK_MEMORY_POST_ENABLED", "0"),
        ("AIIA_SLACK_BOT_TOKEN", ""),
        ("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "not-a-channel"),
    ],
)
def test_disabled_worker_never_dials(env, monkeypatch, name, value):
    queue(env.inbox, key="event:1")
    monkeypatch.setenv(name, value)
    deliver(env.inbox, lambda r: pytest.fail("disabled memory post dialed"))
    assert env.inbox.memory_post_status() == {"pending": 1}


def test_egress_denied_does_not_post(env, monkeypatch):
    queue(env.inbox, key="event:1")
    monkeypatch.setattr(
        slack_memory_posts,
        "authorize_egress",
        AsyncMock(return_value=EgressDecision(False, "deny")),
    )
    deliver(env.inbox, lambda r: pytest.fail("egress denied"))
    assert env.inbox.memory_post_status() == {"pending": 1}
    assert env.inbox.list()["ideas"][0]["memory_post_error"] == "egress_denied"


def test_rate_limit_persists_retry(env):
    queue(env.inbox, key="event:1")
    deliver(env.inbox, lambda r: httpx.Response(429, headers={"Retry-After": "120"}))
    with env.inbox.connect() as db:
        row = db.execute("SELECT * FROM memory_posts").fetchone()
    assert row["status"] == "pending" and row["error"] == "rate_limited"
    assert row["next_attempt"] >= time.time() + 115
    assert env.inbox.claim_memory_post() is None


@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(200, json={"ok": False, "error": "not_in_channel"}), "not_in_channel"),
        (httpx.Response(200, json={"ok": False, "error": "invalid_auth"}), "invalid_auth"),
        (
            httpx.Response(200, json={"ok": False, "error": "channel_not_found"}),
            "channel_not_found",
        ),
        (httpx.Response(403), "request_rejected"),
    ],
)
def test_permanent_errors_fail_immediately(env, response, error):
    queue(env.inbox, key="event:1")
    deliver(env.inbox, lambda r: response)
    assert env.inbox.memory_post_status() == {"failed": 1}
    assert env.inbox.list()["ideas"][0]["memory_post_error"] == error
    assert env.inbox.list()["ideas"][0]["status"] == "promoted"


def test_transient_failure_backs_off_then_caps_at_eight(env):
    queue(env.inbox, key="event:1")

    def handler(request):
        raise httpx.ReadTimeout("synthetic-bot-token")

    deliver(env.inbox, handler)
    with env.inbox.connect() as db:
        row = db.execute("SELECT * FROM memory_posts").fetchone()
        assert row["status"] == "pending" and row["attempts"] == 1
        assert row["error"] == "delivery_unavailable"
        db.execute("UPDATE memory_posts SET attempts=7,next_attempt=0")
    deliver(env.inbox, handler)
    with env.inbox.connect() as db:
        row = db.execute("SELECT * FROM memory_posts").fetchone()
    assert row["status"] == "failed" and row["attempts"] == 8
    assert "synthetic-bot-token" not in row["error"]


def test_lease_blocks_double_claim_and_stale_finish(env):
    queue(env.inbox, key="event:1")
    first = env.inbox.claim_memory_post()
    restarted = MemoryInbox(env.inbox.path)
    assert restarted.claim_memory_post() is None
    with restarted.connect() as db:
        db.execute("UPDATE memory_posts SET next_attempt=0")
    second = restarted.claim_memory_post()
    assert second["lease"] != first["lease"] and second["attempts"] == 2
    restarted.finish_memory_post(first, status="sent")
    assert restarted.memory_post_status() == {"sending": 1}
    restarted.finish_memory_post(second, status="sent")
    assert restarted.memory_post_status() == {"sent": 1}


def test_failed_post_retry_route(app, env, monkeypatch):
    idea = queue(env.inbox, key="event:1")
    path = f"/api/memory-inbox/{idea['id']}/acknowledgement/retry?kind=memory_post"
    deliver(env.inbox, lambda r: httpx.Response(200, json={"ok": False, "error": "not_in_channel"}))
    assert env.inbox.get(idea["id"])["memory_post_status"] == "failed"
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_ENABLED", "0")
    disabled = call(app, "POST", path)
    assert disabled.status_code == 409 and disabled.json()["detail"] == "memory_posting_disabled"
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_ENABLED", "1")
    assert call(app, "POST", path).status_code == 200
    assert call(app, "POST", path).status_code == 409
    row = env.inbox.get(idea["id"])
    assert row["memory_post_status"] == "pending" and row["memory_post_error"] == ""
    deliver(env.inbox, ok)
    assert env.inbox.memory_post_status() == {"sent": 1}


def test_memory_post_egress_is_opt_in_and_slack_post_stays_denied(monkeypatch):
    assert EGRESS_POINTS["slack.memory_post"] == (
        "human-approved memory post to one allowlisted channel (opt-in)"
    )
    assert "slack.post" in EGRESS_POINTS
    assert "slack.memory_post" not in AIRGAP_ALLOWED_EGRESS
    monkeypatch.setattr(egress, "get_config", lambda: SimpleNamespace(airgap_enabled=True))
    monkeypatch.setattr(egress, "_client", lambda: None)
    monkeypatch.setattr(egress, "report_denied_bg", lambda t, s=None: None)
    monkeypatch.delenv("AIIA_SLACK_MEMORY_POST_ENABLED", raising=False)
    monkeypatch.delenv("AIIA_SLACK_ACK_ENABLED", raising=False)
    assert not airgap_allows_tool("slack.memory_post")
    assert asyncio.run(authorize_egress("slack.memory_post")).allowed is False
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    assert not airgap_allows_tool("slack.memory_post")
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_ENABLED", "1")
    assert asyncio.run(authorize_egress("slack.memory_post")).allowed is True
    assert asyncio.run(authorize_egress("slack.post")).allowed is False
    status = airgap_status(SimpleNamespace(airgap_enabled=True))
    assert status["egress"]["slack.memory_post"] == "airgap-allowlisted"
    assert status["egress"]["slack.post"] == "disabled"
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_ENABLED", "0")
    assert not airgap_allows_tool("slack.memory_post")
    assert airgap_allows_tool("slack.capture_ack")


def test_router_lifespan_runs_both_workers(monkeypatch):
    started, stopped = [], []

    def fake(name):
        async def worker(factory):
            started.append(name)
            try:
                await asyncio.Future()
            finally:
                stopped.append(name)

        return worker

    monkeypatch.setattr(slack_receipts, "run_worker", fake("receipts"))
    monkeypatch.setattr(slack_memory_posts, "run_worker", fake("memory_posts"))

    async def exercise():
        app = FastAPI()
        app.include_router(slack_capture.router)
        async with app.router.lifespan_context(app):
            for _ in range(100):
                if len(started) == 2:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(exercise())
    assert sorted(started) == sorted(stopped) == ["memory_posts", "receipts"]


def post_text(text, **kwargs):
    idea = {"id": "0123456789abcdef", "text": text}
    fields = {"memory_id": "decisions_4_1789", "category": "decisions", "priority": "urgent"}
    return slack_capture.memory_post_text(idea, **{**fields, **kwargs})


def test_body_layout_strips_bot_mention_and_carries_priority_and_category():
    assert post_text("<@U0BOT1>  ship the cron contract") == (
        "[URGENT] Memory logged to decisions\n\n"
        "ship the cron contract\n\n"
        "Capture 01234567 · Memory decisions_4_1789"
    )
    assert post_text("note", priority="low", category="lessons").startswith(
        "[LOW] Memory logged to lessons\n\nnote\n\n"
    )


def test_escaping_neutralizes_broadcasts_mentions_and_links():
    body = post_text(
        "<@U0BOT1> alert <!channel> and <!here> and <!subteam^S0TEAM|devs> "
        "ask <@U0PERSON|ada> see <https://evil.example/x|docs> & a>b"
    )
    assert "<" not in body and ">" not in body
    assert "&lt;!channel&gt;" in body and "&lt;!here&gt;" in body
    assert "&lt;!subteam^S0TEAM|devs&gt;" in body
    assert "&lt;@U0PERSON|ada&gt;" in body
    assert "&lt;https://evil.example/x|docs&gt;" in body
    assert "&amp; a&gt;b" in body
    assert "&amp;lt;" not in body
    assert post_text("x", memory_id="m<!here>").endswith("Memory m&lt;!here&gt;")


def test_escaping_covers_everyone_channel_links_entities_and_unicode():
    body = post_text("hey <!everyone> in <#C0GENERAL1|general> and <#C0GENERAL1>")
    assert "<" not in body and ">" not in body
    assert "&lt;!everyone&gt;" in body
    assert "&lt;#C0GENERAL1|general&gt; and &lt;#C0GENERAL1&gt;" in body
    # Text that already holds entities is escaped again, never passed through as syntax.
    typed = post_text("&lt;!channel&gt; &amp;")
    assert "&amp;lt;!channel&amp;gt; &amp;amp;" in typed and "<" not in typed
    unicode = post_text("caf\u00e9 \U0001f680 \uff1c!here\uff1e \u2028 \u00a0<!here>")
    assert "caf\u00e9 \U0001f680 \uff1c!here\uff1e \u2028 \u00a0&lt;!here&gt;" in unicode


def test_slack_encoded_capture_is_decoded_once_then_escaped():
    idea = {
        "id": "0123456789abcdef",
        "source": "slack",
        "text": "<@U0BOT1> R&amp;D: a &lt; b &gt; c <!channel> &lt;!here&gt; &amp;lt;",
    }
    body = slack_capture.memory_post_text(
        idea, memory_id="project_1_1", category="project", priority="normal"
    )
    text = body.split("\n\n")[1]
    # Slack shows each entity once: "R&D: a < b > c <!channel> <!here> &lt;", all inert.
    assert text == ("R&amp;D: a &lt; b &gt; c &lt;!channel&gt; &lt;!here&gt; &amp;lt;")
    assert "<" not in body and ">" not in body


def test_text_over_limit_is_truncated_and_marked():
    exact = post_text("a" * 3_000)
    assert "a" * 3_000 in exact and "Truncated" not in exact and "…" not in exact
    body = post_text("b" * 3_001 + "TAIL")
    text = body.split("\n\n")[1]
    assert len(text) == 3_000 and text.endswith("…") and "TAIL" not in body
    assert body.endswith("Capture 01234567 · Memory decisions_4_1789 · Truncated")
    escaped = post_text("<" * 3_001).split("\n\n")[1]
    assert escaped == "&lt;" * 2_999 + "…"


def test_delivered_post_is_escaped_plain_text(app, env, monkeypatch):
    idea = capture(env.inbox, text="<@U0BOT1> <!channel> deploy <https://evil.example|now>")
    brain(monkeypatch, lambda r: httpx.Response(200, json={"id": "project_5_5"}))
    response = call(
        app,
        "POST",
        f"/api/memory-inbox/{idea['id']}/promote",
        {"priority": "high", "post_to_slack": True},
    )
    assert response.status_code == 200, response.text
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return ok(request)

    deliver(env.inbox, handler)
    assert len(seen) == 1 and seen[0]["mrkdwn"] is False
    assert seen[0]["text"] == (
        "[HIGH] Memory logged to project\n\n"
        "&lt;!channel&gt; deploy &lt;https://evil.example|now&gt;\n\n"
        f"Capture {idea['id'][:8]} · Memory project_5_5"
    )
    assert "thread_ts" not in seen[0]
