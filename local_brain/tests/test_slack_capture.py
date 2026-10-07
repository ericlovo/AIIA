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
from local_brain.egress import (
    AIRGAP_ALLOWED_EGRESS,
    EGRESS_POINTS,
    EgressDecision,
    airgap_allows_tool,
)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.delenv("AIIA_SLACK_ACK_ENABLED", raising=False)
    monkeypatch.delenv("AIIA_SLACK_FILE_FETCH_ENABLED", raising=False)
    monkeypatch.setattr(slack_capture, "SLACK_FILE_TRANSPORT", None)
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


def make_pdf(text: str) -> bytes:
    content = b"BT /F1 24 Tf 72 700 Td (%s) Tj ET" % text.encode()
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(content), content),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj" % i + body + b"endobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += b"xref\n0 %d\n" % n
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (n, xref_pos)
    return bytes(out)


def attached_file(**overrides):
    file = {
        "id": "F_TEST",
        "name": "notes.md",
        "mimetype": "text/markdown",
        "filetype": "markdown",
        "size": 24,
        "permalink": "https://slack.test/files/F_TEST",
    }
    file.update(overrides)
    return file


def mention_with_files(files, *, text="<@U123>", subtype=None):
    payload = mention()
    payload["event"]["text"] = text
    payload["event"]["files"] = files
    if subtype is not None:
        payload["event"]["subtype"] = subtype
    return payload


def enable_file_fetch(monkeypatch, transport):
    async def allow(tool, server=None):
        assert tool == "slack.file_fetch"
        assert server == "files.slack.com"
        return EgressDecision(True, "allowed: test")

    monkeypatch.setenv("AIIA_SLACK_FILE_FETCH_ENABLED", "1")
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    monkeypatch.setattr(slack_capture, "authorize_egress", allow)
    monkeypatch.setattr(slack_capture, "SLACK_FILE_TRANSPORT", transport)


def slack_file_transport(*, body=b"fetched markdown", fail=False, info_ok=True):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        assert request.headers.get("Authorization") == "Bearer synthetic-bot-token"
        if fail:
            raise httpx.ConnectError("synthetic-download-failure")
        if request.url.host == "slack.com":
            if not info_ok:
                return httpx.Response(200, json={"ok": False, "error": "file_not_found"})
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "url_private_download": (
                            "https://files.slack.com/files-pri/T_TEST/F_TEST/download/notes.md"
                        )
                    },
                },
            )
        if request.url.host == "files.slack.com":
            return httpx.Response(200, content=body)
        raise AssertionError(f"unexpected host {request.url.host}")

    return httpx.MockTransport(handler), calls


def saved_idea():
    return slack_capture.inbox().list()["ideas"][0]


@pytest.mark.parametrize("subtype", [None, "file_share"])
def test_file_only_mention_saves_file_header(configured, subtype):
    payload = mention_with_files(
        [attached_file()],
        text="<@U123>",
        subtype=subtype,
    )
    assert send_event(configured, payload).status_code == 200
    assert send_event(configured, payload, retry_num=1).status_code == 200
    data = slack_capture.inbox().list()
    assert data["total"] == 1
    idea = data["ideas"][0]
    assert idea["text"].startswith("<@U123>")
    assert (
        "[file] notes.md | text/markdown | 24 bytes | https://slack.test/files/F_TEST"
        in idea["text"]
    )
    assert "fetched markdown" not in idea["text"]
    assert idea["source"] == "slack"
    assert slack_capture.inbox().list()["ideas"][0]["author_id"] == "U_TEST"


def test_mention_with_text_and_file_keeps_both(configured):
    payload = mention_with_files(
        [attached_file(name="brief.csv", mimetype="text/csv", size=80)],
        text="<@U123> Remember this Mindmoor idea.",
    )
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert "Remember this Mindmoor idea." in idea["text"]
    assert (
        "[file] brief.csv | text/csv | 80 bytes | https://slack.test/files/F_TEST" in idea["text"]
    )


def test_snippet_and_pdf_files_are_fetched(configured, monkeypatch):
    bodies = {
        "F_SNIP": b"print('captured snippet')",
        "F_PDF": make_pdf("PDF capture body"),
    }
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        assert request.headers.get("Authorization") == "Bearer synthetic-bot-token"
        file_id = request.url.params.get("file") or (
            "F_PDF" if request.url.path.endswith(".pdf") else "F_SNIP"
        )
        if request.url.host == "slack.com":
            name = "note.py" if file_id == "F_SNIP" else "brief.pdf"
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "url_private_download": (
                            f"https://files.slack.com/files-pri/T_TEST/{file_id}/download/{name}"
                        )
                    },
                },
            )
        return httpx.Response(
            200, content=bodies["F_PDF" if "F_PDF" in str(request.url) else "F_SNIP"]
        )

    enable_file_fetch(monkeypatch, httpx.MockTransport(handler))
    payload = mention_with_files(
        [
            attached_file(
                id="F_SNIP",
                name="note.py",
                mimetype="text/plain",
                filetype="python",
                mode="snippet",
                permalink="https://slack.test/files/F_SNIP",
            ),
            attached_file(
                id="F_PDF",
                name="brief.pdf",
                mimetype="application/pdf",
                filetype="pdf",
                size=80,
                permalink="https://slack.test/files/F_PDF",
            ),
        ]
    )
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert "print('captured snippet')" in idea["text"]
    assert "PDF capture body" in idea["text"]
    assert "[file] note.py |" in idea["text"]
    assert "[file] brief.pdf | application/pdf |" in idea["text"]
    assert calls


def test_file_only_mention_fetches_supported_content(configured, monkeypatch):
    transport, calls = slack_file_transport(body=b"# captured notes")
    enable_file_fetch(monkeypatch, transport)
    payload = mention_with_files([attached_file()])
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert "[file] notes.md | text/markdown | 24 bytes |" in idea["text"]
    assert "# captured notes" in idea["text"]
    assert any(request.url.host == "slack.com" for request in calls)
    assert any(request.url.host == "files.slack.com" for request in calls)
    assert all("synthetic-bot-token" not in str(request.url) for request in calls)


def test_oversized_file_is_truncated_within_inbox_limit(configured, monkeypatch):
    transport, _calls = slack_file_transport(body=("A" * 20_000).encode())
    enable_file_fetch(monkeypatch, transport)
    payload = mention_with_files([attached_file(size=20_000)])
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert len(idea["text"]) <= 8_000
    assert "[truncated]" in idea["text"]
    assert "[file] notes.md | text/markdown | 20000 bytes |" in idea["text"]


def test_unsupported_file_stays_header_only(configured, monkeypatch):
    transport, calls = slack_file_transport()
    enable_file_fetch(monkeypatch, transport)
    payload = mention_with_files(
        [
            attached_file(
                name="diagram.png",
                mimetype="image/png",
                filetype="png",
                size=440,
            )
        ]
    )
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert (
        "[file] diagram.png | image/png | 440 bytes | https://slack.test/files/F_TEST"
        in idea["text"]
    )
    assert "fetched markdown" not in idea["text"]
    assert calls == []


def test_file_download_failure_falls_back_to_header(configured, monkeypatch, caplog):
    transport, _calls = slack_file_transport(fail=True)
    enable_file_fetch(monkeypatch, transport)
    payload = mention_with_files([attached_file()])
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert (
        "[file] notes.md | text/markdown | 24 bytes | https://slack.test/files/F_TEST"
        in idea["text"]
    )
    assert "fetched markdown" not in idea["text"]
    assert "synthetic-bot-token" not in caplog.text


def test_file_info_failure_falls_back_to_header(configured, monkeypatch):
    transport, _calls = slack_file_transport(info_ok=False)
    enable_file_fetch(monkeypatch, transport)
    payload = mention_with_files([attached_file()])
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert "[file] notes.md |" in idea["text"]
    assert "fetched markdown" not in idea["text"]


def test_file_fetch_airgap_denial_falls_back_to_header(configured, monkeypatch):
    transport, calls = slack_file_transport()

    async def deny(tool, server=None):
        assert tool == "slack.file_fetch"
        assert server == "files.slack.com"
        return EgressDecision(False, "denied: air-gap mode (AIIA_AIRGAP)")

    enable_file_fetch(monkeypatch, transport)
    monkeypatch.setattr(slack_capture, "authorize_egress", deny)
    payload = mention_with_files([attached_file()])
    assert send_event(configured, payload).status_code == 200
    idea = saved_idea()
    assert "[file] notes.md | text/markdown | 24 bytes |" in idea["text"]
    assert "fetched markdown" not in idea["text"]
    assert calls == []


def test_file_only_mention_still_queues_receipt(configured, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    payload = mention_with_files([attached_file()])
    payload["event"]["ts"] = "1789260567.123456"
    assert send_event(configured, payload).status_code == 200
    assert send_event(configured, payload, retry_num=1).status_code == 200
    assert slack_capture.inbox().list()["total"] == 1
    assert slack_capture.inbox().receipt_status() == {"pending": 1}


@pytest.mark.parametrize(
    "file,kind",
    [
        (attached_file(), "text"),
        (attached_file(name="notes.txt", mimetype="text/plain", filetype="text"), "text"),
        (attached_file(name="rows.csv", mimetype="text/csv", filetype="csv"), "text"),
        (attached_file(name="data.json", mimetype="application/json", filetype="json"), "text"),
        (
            attached_file(name="clip", mimetype="text/plain", filetype="python", mode="snippet"),
            "text",
        ),
        (attached_file(name="brief.pdf", mimetype="application/pdf", filetype="pdf"), "pdf"),
        (attached_file(name="diagram.png", mimetype="image/png", filetype="png"), ""),
    ],
)
def test_file_kind_accepts_text_like_and_pdf(file, kind):
    assert slack_capture.file_kind(file) == kind


def test_file_fetch_egress_is_opt_in_and_not_on_airgap_allowlist(monkeypatch):
    assert EGRESS_POINTS["slack.file_fetch"] == (
        "Slack file content fetch for mention captures (opt-in)"
    )
    assert "slack.file_fetch" not in AIRGAP_ALLOWED_EGRESS
    monkeypatch.delenv("AIIA_SLACK_FILE_FETCH_ENABLED", raising=False)
    monkeypatch.delenv("AIIA_SLACK_ACK_ENABLED", raising=False)
    assert not airgap_allows_tool("slack.file_fetch")
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    assert not airgap_allows_tool("slack.file_fetch")
    monkeypatch.setenv("AIIA_SLACK_FILE_FETCH_ENABLED", "1")
    assert airgap_allows_tool("slack.file_fetch")
    assert airgap_allows_tool("slack.capture_ack")
    monkeypatch.setenv("AIIA_SLACK_FILE_FETCH_ENABLED", "0")
    assert not airgap_allows_tool("slack.file_fetch")
