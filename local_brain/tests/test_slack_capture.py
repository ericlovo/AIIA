import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
from unittest.mock import Mock
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI

from local_brain.command_center import slack_capture, slack_receipts
from local_brain.command_center.memory_inbox import MemoryInbox


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.delenv("AIIA_SLACK_ACK_ENABLED", raising=False)
    monkeypatch.setenv("AIIA_SLACK_SIGNING_SECRET", "synthetic-secret")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_TEST")
    monkeypatch.setenv("AIIA_SLACK_CHANNEL_IDS", "C_TEST")
    monkeypatch.setenv("AIIA_MEMORY_INBOX_PATH", str(tmp_path / "inbox.sqlite3"))
    app = FastAPI()
    app.include_router(slack_capture.router)
    return app


def send(app, *, changes=None, age=0, bad_signature=False):
    fields = {
        "team_id": "T_TEST",
        "channel_id": "C_TEST",
        "user_id": "U_TEST",
        "command": "/aiia-capture",
        "text": "Original idea\nPreserve the exact wording.",
        "trigger_id": "synthetic-trigger",
        "response_url": "https://example.invalid/secret-do-not-save",
        "token": "do-not-save",
    }
    fields.update(changes or {})
    body = urlencode(fields).encode()
    timestamp = str(int(time.time()) - age)
    signature = (
        "v0="
        + hmac.new(
            b"synthetic-secret", b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/integrations/slack/commands",
                content=body,
                headers={
                    "x-slack-request-timestamp": timestamp,
                    "x-slack-signature": "bad" if bad_signature else signature,
                    "content-type": "application/x-www-form-urlencoded",
                },
            )

    return asyncio.run(exercise())


def test_capture_is_durable_deduplicated_and_drops_credentials(configured):
    first, second = send(configured), send(configured)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["response_type"] == "ephemeral"
    data = slack_capture.inbox().list()
    assert data["total"] == 1
    idea = data["ideas"][0]
    assert idea["text"] == "Original idea\nPreserve the exact wording."
    assert idea["status"] == "unreviewed"
    assert idea["author_id"] == "U_TEST"
    assert "do-not-save" not in str(data)
    assert slack_capture.inbox().list(query="exact wording")["total"] == 1
    assert slack_capture.inbox().list(project="other")["total"] == 0
    assert slack_capture.inbox().path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"bad_signature": True}, 401),
        ({"age": 301}, 401),
        ({"age": -301}, 401),
        ({"changes": {"team_id": "T_OTHER"}}, 403),
        ({"changes": {"channel_id": "C_OTHER"}}, 403),
        ({"changes": {"user_id": ""}}, 400),
        ({"changes": {"command": "/other"}}, 400),
    ],
)
def test_invalid_request_never_writes(configured, kwargs, code):
    assert send(configured, **kwargs).status_code == code
    assert not slack_capture.inbox().path.exists()


def test_unconfigured_fails_closed(configured, monkeypatch):
    monkeypatch.delenv("AIIA_SLACK_TEAM_ID")
    assert send(configured).status_code == 503
    assert slack_capture.slack_status()["configured"] is False
    assert not slack_capture.inbox().path.exists()


def test_storage_failure_never_acknowledges_capture(configured, monkeypatch):
    monkeypatch.setattr(
        MemoryInbox, "capture", Mock(side_effect=sqlite3.OperationalError("private-path"))
    )
    response = send(configured)
    assert response.status_code == 503
    assert response.json()["detail"] == "memory_inbox_unavailable"
    assert "private-path" not in response.text


def test_oversize_or_empty_idea_is_not_saved(configured):
    assert "not saved" in send(configured, changes={"text": "x" * 8001}).json()["text"]
    assert "Use /aiia-capture" in send(configured, changes={"text": ""}).json()["text"]
    assert not slack_capture.inbox().path.exists()


@pytest.mark.parametrize("text", ["<@U123>", " \n<@U123> <@U456>\t", " \n\t"])
def test_mention_only_command_returns_guidance_without_saving(configured, text):
    first = send(configured, changes={"text": text})
    second = send(configured, changes={"text": text})
    assert first.status_code == second.status_code == 200
    assert (
        first.json()
        == second.json()
        == {
            "response_type": "ephemeral",
            "text": "Use /aiia-capture followed by your idea.",
        }
    )
    assert not slack_capture.inbox().path.exists()


def test_command_with_mention_and_idea_keeps_original_wording(configured):
    text = "<@U123> Preserve this idea for the next sprint."
    assert "Saved idea" in send(configured, changes={"text": text}).json()["text"]
    assert slack_capture.inbox().list()["ideas"][0]["text"] == text


def send_event(app, payload, *, bad_signature=False, retry_num=None, age=0):
    body = json.dumps(payload).encode()
    timestamp = str(int(time.time()) - age)
    signature = (
        "v0="
        + hmac.new(
            b"synthetic-secret", b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/integrations/slack/events",
                content=body,
                headers={
                    "x-slack-request-timestamp": timestamp,
                    "x-slack-signature": "bad" if bad_signature else signature,
                    **(
                        {
                            "x-slack-retry-num": str(retry_num),
                            "x-slack-retry-reason": "http_timeout",
                        }
                        if retry_num is not None
                        else {}
                    ),
                },
            )

    return asyncio.run(exercise())


def mention():
    return {
        "type": "event_callback",
        "team_id": "T_TEST",
        "event_id": "Ev_TEST",
        "event": {
            "type": "app_mention",
            "channel": "C_TEST",
            "user": "U_TEST",
            "text": "<@U_AIIA> Remember this Mindmoor idea.",
        },
    }


@pytest.mark.parametrize(
    "text",
    [
        "<@U123> Remember this Mindmoor idea.",
        " \n<@U123> <@U456> Remember this Mindmoor idea.\t",
        "Keep <@U456> in the original wording. <@U123>",
    ],
)
def test_mention_durable_and_deduplicated(configured, text):
    payload = mention()
    payload["event"]["text"] = text
    assert send_event(configured, payload).status_code == 200
    assert send_event(configured, payload, retry_num=1).status_code == 200
    data = slack_capture.inbox().list(project="mindmoor")
    assert data["total"] == 1
    assert data["ideas"][0]["text"] == text
    assert data["ideas"][0]["status"] == "unreviewed"


@pytest.mark.parametrize("ack_enabled", ["0", "1"])
@pytest.mark.parametrize(
    "text",
    [
        "<@U123>",
        " \t<@U123>\n ",
        "<@U123> <@U456>",
        "<@U123><@U456>",
        "\n<@U123>\t\n<@U456>\n",
        "\u00a0<@U123>\u00a0",
    ],
)
def test_empty_mentions_never_save_or_send_receipts(configured, monkeypatch, ack_enabled, text):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", ack_enabled)
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    payload = mention()
    payload["event"].update(text=text, ts="1789260567.123456")
    first = send_event(configured, payload)
    retry = send_event(configured, payload, retry_num=1)
    assert first.status_code == retry.status_code == 200
    assert first.json() == retry.json() == {"ok": True}
    assert not slack_capture.inbox().path.exists()

    restarted = MemoryInbox(slack_capture.inbox().path)
    monkeypatch.setattr(slack_capture, "inbox", lambda: restarted)
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    assert send_event(configured, payload, retry_num=2).status_code == 200
    assert not restarted.path.exists()
    assert restarted.list()["total"] == 0
    assert restarted.receipt_status() == {}
    assert restarted.receipt_status("promotion") == {}
    asyncio.run(
        slack_receipts.deliver_one(
            restarted,
            transport=httpx.MockTransport(lambda r: pytest.fail("empty capture receipt")),
        )
    )


@pytest.mark.parametrize("text", ["", " ", "\t\n", "\u00a0"])
@pytest.mark.parametrize("ack_enabled", ["0", "1"])
def test_blank_event_text_remains_invalid(configured, monkeypatch, text, ack_enabled):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", ack_enabled)
    payload = mention()
    payload["event"].update(text=text, ts="1789260567.123456")
    for retry_num in (None, 1):
        result = send_event(configured, payload, retry_num=retry_num)
        assert result.status_code == 400
        assert result.json()["detail"] == "invalid_slack_payload"
    assert not slack_capture.inbox().path.exists()


@pytest.mark.parametrize(
    "changes,event_changes,kwargs,code",
    [
        ({}, {}, {"bad_signature": True}, 401),
        ({}, {}, {"age": 301}, 401),
        ({}, {}, {"age": -301}, 401),
        ({"team_id": "T_OTHER"}, {}, {}, 403),
        ({}, {"channel": "C_OTHER"}, {}, 403),
        ({"event_id": ""}, {}, {}, 400),
        ({}, {"user": ""}, {}, 400),
        ({}, {"ts": "not-a-timestamp"}, {}, 400),
    ],
)
def test_empty_mentions_still_require_valid_source(
    configured, monkeypatch, changes, event_changes, kwargs, code
):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    payload = mention()
    payload.update(changes)
    payload["event"].update(text="<@U123>", ts="1789260567.123456")
    payload["event"].update(event_changes)
    assert send_event(configured, payload, **kwargs).status_code == code
    assert not slack_capture.inbox().path.exists()


def test_challenge_requires_signature(configured):
    payload = {"type": "url_verification", "challenge": "test-challenge"}
    assert send_event(configured, payload, bad_signature=True).status_code == 401
    assert send_event(configured, payload).json() == {"challenge": "test-challenge"}
    assert not slack_capture.inbox().path.exists()


@pytest.mark.parametrize(
    "change,code",
    [
        ({"channel": "C_OTHER"}, 403),
        ({"channel": []}, 403),
        ({"user": ""}, 400),
        ({"text": 123}, 400),
        ({"text": "x" * 8001}, 422),
        ({"bot_id": "B_TEST"}, 200),
        ({"subtype": "bot_message"}, 200),
        ({"type": "message"}, 200),
    ],
)
def test_mention_restrictions(configured, change, code):
    payload = mention()
    payload["event"].update(change)
    assert send_event(configured, payload).status_code == code
    assert not slack_capture.inbox().path.exists()


def test_mention_wrong_team_and_missing_id(configured):
    payload = mention()
    payload["team_id"] = "T_OTHER"
    assert send_event(configured, payload).status_code == 403
    payload = mention()
    del payload["event_id"]
    assert send_event(configured, payload).status_code == 400
    assert not slack_capture.inbox().path.exists()


def test_mention_storage_failure(configured, monkeypatch):
    monkeypatch.setattr(
        MemoryInbox, "capture", Mock(side_effect=sqlite3.OperationalError("private-path"))
    )
    result = send_event(configured, mention())
    assert result.status_code == 503
    assert "private-path" not in result.text


def test_mention_queues_single_thread_receipt(configured, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    payload = mention()
    payload["event"].update(
        text="<@U123> <@U456> Remember this Mindmoor idea.",
        ts="1789260567.123456",
        thread_ts="1789260566.123456",
    )
    assert send_event(configured, payload).status_code == 200
    assert send_event(configured, payload, retry_num=1).status_code == 200
    assert slack_capture.inbox().list()["total"] == 1
    assert slack_capture.inbox().list()["ideas"][0]["text"] == payload["event"]["text"]
    assert slack_capture.inbox().receipt_status() == {"pending": 1}
    assert slack_capture.inbox().claim_receipt()["thread_ts"] == "1789260566.123456"


def test_invalid_receipt_destination_never_saves(configured, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    assert send_event(configured, mention()).status_code == 400
    assert not slack_capture.inbox().path.exists()
