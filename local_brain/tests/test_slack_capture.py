import asyncio
import hashlib
import hmac
import sqlite3
import time
from unittest.mock import Mock
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI

from local_brain.command_center import slack_capture
from local_brain.command_center.memory_inbox import MemoryInbox


@pytest.fixture
def configured(tmp_path, monkeypatch):
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
