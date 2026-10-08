"""Files attached to a capture mention: saved, fetched, read, and indexed on promote."""

import asyncio
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from local_brain.command_center import slack_capture, slack_files
from local_brain.command_center.memory_inbox import (
    FILE_EXCERPT_OMITTED,
    FILE_EXCERPT_TRUNCATED,
    IDEA_TEXT_LIMIT,
    MemoryInbox,
)
from local_brain.egress import EgressDecision, airgap_allows_tool

FILE = {
    "id": "F0C6HNU79J8",
    "name": "alumni-nations-project-memory.md",
    "mimetype": "text/plain",
    "size": 8800,
    "url_private_download": "https://files.slack.com/files-pri/T_TEST-F0C6HNU79J8/download/memory.md",
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_SIGNING_SECRET", "synthetic-secret")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_TEST")
    monkeypatch.setenv("AIIA_SLACK_CHANNEL_IDS", "C_TEST")
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic-bot-token")
    monkeypatch.setenv("AIIA_SLACK_FILE_CAPTURE_ENABLED", "1")
    monkeypatch.setenv("AIIA_SLACK_ACK_ENABLED", "1")
    monkeypatch.setenv("AIIA_MEMORY_INBOX_PATH", str(tmp_path / "inbox.sqlite3"))
    monkeypatch.setenv("AIIA_CAPTURE_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("LOCAL_BRAIN_API_KEY", "synthetic-brain-key")
    egress_check = AsyncMock(return_value=EgressDecision(True, "test"))
    monkeypatch.setattr(slack_files, "authorize_egress", egress_check)
    app = FastAPI()
    app.include_router(slack_capture.router)
    return SimpleNamespace(app=app, inbox=slack_capture.inbox(), egress=egress_check, tmp=tmp_path)


def signed(app, payload: dict) -> httpx.Response:
    body = json.dumps(payload).encode()
    timestamp = str(int(time.time()))
    signature = (
        "v0="
        + hmac.new(
            b"synthetic-secret", b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/integrations/slack/events",
                content=body,
                headers={
                    "x-slack-request-timestamp": timestamp,
                    "x-slack-signature": signature,
                    "content-type": "application/json",
                },
            )

    return asyncio.run(go())


def mention(text="<@U0BOT1>", files=None, event_id="Ev1"):
    event = {
        "type": "app_mention",
        "user": "U_AUTHOR",
        "text": text,
        "channel": "C_TEST",
        "ts": "1790978922.116909",
    }
    if files is not None:
        event["files"] = files
    return {"type": "event_callback", "team_id": "T_TEST", "event_id": event_id, "event": event}


def call(app, method, path, body=None):
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, json=body)

    return asyncio.run(go())


def deliver(inbox, handler):
    asyncio.run(slack_files.deliver_one(inbox, transport=httpx.MockTransport(handler)))


def test_mention_with_only_a_file_is_a_capture(env):
    response = signed(env.app, mention(files=[FILE]))
    assert response.status_code == 200
    data = env.inbox.list()
    assert data["total"] == 1
    idea = data["ideas"][0]
    assert slack_capture.capture_text(idea["text"]) == (
        "[attached: alumni-nations-project-memory.md (text/plain, 9 KB)]"
    )
    assert [f["file_id"] for f in idea["files"]] == ["F0C6HNU79J8"]
    assert idea["files"][0]["status"] == "pending"
    assert env.inbox.receipt_status() == {"pending": 1}
    # The same event again changes nothing.
    assert signed(env.app, mention(files=[FILE])).status_code == 200
    assert env.inbox.list()["total"] == 1
    assert env.inbox.file_status() == {"pending": 1}


def test_mention_with_neither_words_nor_files_is_still_dropped(env):
    assert signed(env.app, mention()).status_code == 200
    assert signed(env.app, mention(files=[])).status_code == 200
    assert (
        signed(
            env.app, mention(files=[{"id": "bad", "url_private": "https://evil.example/x"}])
        ).status_code
        == 200
    )
    assert env.inbox.list()["total"] == 0


def test_file_records_keep_only_slack_hosted_well_formed_entries():
    records = slack_files.file_records(
        [
            FILE,
            {"id": "F0C5X4W3T63", "name": "a.pdf", "url_private": "https://files.slack.com/x"},
            {"id": "F0C5X4W3T64", "url_private": "https://attacker.example/slack.com/x"},
            {"id": "not-an-id", "url_private": "https://files.slack.com/x"},
            "garbage",
        ]
    )
    assert [r["id"] for r in records] == ["F0C6HNU79J8", "F0C5X4W3T63"]
    assert records[1]["mimetype"] == "" and records[1]["size"] == 0


def test_worker_saves_the_file_and_appends_an_excerpt(env):
    signed(env.app, mention(files=[FILE]))
    idea_id = env.inbox.list()["ideas"][0]["id"]
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer synthetic-bot-token"
        return httpx.Response(
            200,
            content=b"# Alumni Nations project memory\n\nEngagement before giving.\n" * 3,
            headers={"content-type": "text/plain"},
        )

    deliver(env.inbox, handler)
    assert len(seen) == 1 and seen[0].url.host == "files.slack.com"
    idea = env.inbox.get(idea_id)
    file = idea["files"][0]
    assert file["status"] == "done" and file["chars"] > 0
    assert "[file: alumni-nations-project-memory.md]" in idea["text"]
    assert "Engagement before giving." in idea["text"]
    saved = list((env.tmp / "files" / idea_id).iterdir())
    assert len(saved) == 1 and saved[0].name.startswith("F0C6HNU79J8-")
    assert saved[0].stat().st_mode & 0o777 == 0o600
    # A second delivery pass has nothing to do and does not append twice.
    deliver(env.inbox, lambda r: pytest.fail("nothing left to fetch"))
    assert env.inbox.get(idea_id)["text"].count("[file:") == 1
    assert env.egress.await_args.args == ("slack.file_fetch",)


def test_worker_reads_pdf_text(env):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    import io

    buffer = io.BytesIO()
    writer.write(buffer)
    pdf = dict(FILE, id="F0C5X4W3T63", name="workflow.pdf", mimetype="application/pdf")
    signed(env.app, mention(files=[pdf]))
    deliver(
        env.inbox,
        lambda r: httpx.Response(
            200, content=buffer.getvalue(), headers={"content-type": "application/pdf"}
        ),
    )
    file = env.inbox.list()["ideas"][0]["files"][0]
    # A blank page has no text; the file is still saved and the capture says so.
    assert file["status"] == "done" and file["chars"] == 0
    assert "(file has no text)" in env.inbox.list()["ideas"][0]["text"]


@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(403), "missing_scope"),
        (httpx.Response(404), "file_not_found"),
        (
            httpx.Response(
                200, content=b"<html>login</html>", headers={"content-type": "text/html"}
            ),
            "missing_scope",
        ),
        (httpx.Response(413), "request_rejected"),
    ],
)
def test_permanent_fetch_failures_are_recorded_and_retryable(env, response, error):
    signed(env.app, mention(files=[FILE]))
    deliver(env.inbox, lambda r: response)
    idea = env.inbox.list()["ideas"][0]
    file = idea["files"][0]
    assert file["status"] == "failed" and file["error"] == error
    assert "[file:" not in idea["text"]
    retry = call(env.app, "POST", f"/api/memory-inbox/{idea['id']}/files/{file['file_id']}/retry")
    assert retry.status_code == 200
    assert env.inbox.file_status() == {"pending": 1}
    again = call(env.app, "POST", f"/api/memory-inbox/{idea['id']}/files/{file['file_id']}/retry")
    assert again.status_code == 409


def test_transient_failure_backs_off_and_oversize_is_refused(env):
    signed(env.app, mention(files=[FILE]))
    deliver(env.inbox, lambda r: httpx.Response(503))
    file = env.inbox.list()["ideas"][0]["files"][0]
    assert file["status"] == "pending" and file["error"] == "delivery_unavailable"
    record = env.inbox.file_record(file["file_id"])
    assert record["next_attempt"] > time.time()
    env.inbox.retry_file(record["idea_id"], file["file_id"]) if False else None
    with env.inbox.connect() as db:
        db.execute("UPDATE capture_files SET next_attempt=0")
    deliver(
        env.inbox, lambda r: httpx.Response(200, content=b"x" * (slack_files.MAX_FILE_BYTES + 1))
    )
    assert env.inbox.list()["ideas"][0]["files"][0]["error"] == "file_too_large"


def test_disabled_or_denied_worker_never_dials(env, monkeypatch):
    signed(env.app, mention(files=[FILE]))
    monkeypatch.setenv("AIIA_SLACK_FILE_CAPTURE_ENABLED", "0")
    deliver(env.inbox, lambda r: pytest.fail("disabled worker must not dial"))
    assert env.inbox.file_status() == {"pending": 1}
    assert not airgap_allows_tool("slack.file_fetch")
    monkeypatch.setenv("AIIA_SLACK_FILE_CAPTURE_ENABLED", "1")
    assert airgap_allows_tool("slack.file_fetch")
    env.egress.return_value = EgressDecision(False, "airgap")
    deliver(env.inbox, lambda r: pytest.fail("denied egress must not dial"))
    file = env.inbox.list()["ideas"][0]["files"][0]
    assert file["status"] == "pending" and file["error"] == "egress_denied"


def test_other_workspace_files_are_never_fetched(env):
    env.inbox.capture(
        text="<@U0BOT1>\n[attached: x]",
        source_key="event:other",
        source="slack",
        project="mindmoor",
        workspace_id="T_OTHER",
        channel_id="C_TEST",
        author_id="U_X",
        files=[dict(FILE, url="https://files.slack.com/other")],
    )
    deliver(env.inbox, lambda r: pytest.fail("other workspace must not be dialed"))
    assert env.inbox.list()["ideas"][0]["files"][0]["error"] == "source_not_allowed"


def test_backfill_attaches_a_file_by_id(env, monkeypatch):
    env.inbox.capture(
        text="<@U0BOT1> earlier capture",
        source_key="event:old",
        source="slack",
        project="mindmoor",
        workspace_id="T_TEST",
        channel_id="C_TEST",
        author_id="U_AUTHOR",
    )
    idea = env.inbox.list()["ideas"][0]

    def info(request):
        assert request.url.path == "/api/files.info"
        assert request.url.params["file"] == "F0C6HNU79J8"
        return httpx.Response(200, json={"ok": True, "file": FILE})

    monkeypatch.setattr(slack_files, "TRANSPORT", httpx.MockTransport(info))
    response = call(
        env.app, "POST", f"/api/memory-inbox/{idea['id']}/files", {"file_id": "F0C6HNU79J8"}
    )
    assert response.status_code == 200, response.text
    assert [f["name"] for f in response.json()["idea"]["files"]] == [FILE["name"]]
    assert env.inbox.file_status() == {"pending": 1}
    monkeypatch.setattr(
        slack_files,
        "TRANSPORT",
        httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": False})),
    )
    missing = call(
        env.app, "POST", f"/api/memory-inbox/{idea['id']}/files", {"file_id": "F0C6HNU79J9"}
    )
    assert missing.status_code == 404
    bad = call(
        env.app, "POST", f"/api/memory-inbox/{idea['id']}/files", {"file_id": "not-a-file-id"}
    )
    assert bad.status_code == 422


def _attachment(n: int, name: str) -> dict:
    return {
        "id": f"F0C6HNU79J{n}",
        "name": name,
        "mimetype": "text/plain",
        "size": 8800,
        "url_private_download": f"https://files.slack.com/files-pri/T_TEST-F0C6HNU79J{n}/{name}",
    }


def test_every_attachment_excerpt_is_stored_within_the_idea_budget(env):
    files = [
        _attachment(1, "one.md"),
        _attachment(2, "two.md"),
        _attachment(3, "three.md"),
    ]
    signed(env.app, mention(files=files))

    def handler(request):
        url = str(request.url)
        for spec in files:
            if spec["id"] in url:
                body = (spec["name"] + " " + "body " * 2000).encode()
                return httpx.Response(200, content=body, headers={"content-type": "text/plain"})
        return httpx.Response(404)

    for _ in files:
        deliver(env.inbox, handler)
    idea = env.inbox.list()["ideas"][0]
    text = idea["text"]
    assert len(text) <= IDEA_TEXT_LIMIT
    for name in ("one.md", "two.md", "three.md"):
        assert f"[file: {name}]" in text
        assert name.split(".")[0] in text or FILE_EXCERPT_TRUNCATED in text
    assert FILE_EXCERPT_TRUNCATED in text
    assert all(row["status"] == "done" for row in idea["files"])
    assert text.count("[file:") == 3


def test_finish_file_shares_the_budget_across_three_attachments(tmp_path):
    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    files = [
        {
            "id": f"F0C6HNU79J{n}",
            "name": name,
            "mimetype": "text/plain",
            "size": 100,
            "url": f"https://files.slack.com/{name}",
        }
        for n, name in enumerate(("one.md", "two.md", "three.md"), start=1)
    ]
    idea = inbox.capture(
        text="<@U0BOT1>\n[attached: one.md]\n[attached: two.md]\n[attached: three.md]",
        source_key="event:triple",
        source="slack",
        project="mindmoor",
        workspace_id="T_TEST",
        channel_id="C_TEST",
        author_id="U_AUTHOR",
        files=files,
    )
    long = "word " * 3000
    for _ in files:
        claimed = inbox.claim_file()
        assert claimed is not None
        inbox.finish_file(claimed, status="done", excerpt=long, path="/tmp/x", chars=len(long))
    stored = inbox.get(idea["id"])
    text = stored["text"]
    assert len(text) <= IDEA_TEXT_LIMIT
    for name in ("one.md", "two.md", "three.md"):
        assert f"[file: {name}]" in text
    assert FILE_EXCERPT_TRUNCATED in text
    assert all(row["status"] == "done" for row in stored["files"])


def test_a_file_is_not_marked_done_until_its_excerpt_is_stored(tmp_path):
    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    idea = inbox.capture(
        text="x" * (IDEA_TEXT_LIMIT - 5),
        source_key="event:full",
        source="slack",
        project="mindmoor",
        workspace_id="T_TEST",
        channel_id="C_TEST",
        author_id="U_AUTHOR",
        files=[
            {
                "id": "F0C6HNU79J8",
                "name": "late.md",
                "mimetype": "text/plain",
                "size": 100,
                "url": "https://files.slack.com/late.md",
            }
        ],
    )
    claimed = inbox.claim_file()
    assert claimed is not None
    inbox.finish_file(
        claimed, status="done", excerpt="this excerpt should not fit", path="/tmp/late.md", chars=10
    )
    row = inbox.file_record("F0C6HNU79J8")
    stored = inbox.get(idea["id"])["text"]
    assert row["status"] == "failed" and row["error"] == "excerpt_does_not_fit"
    assert "[file: late.md]" not in stored
    assert FILE_EXCERPT_OMITTED not in stored


def test_promote_indexes_fetched_file_text_in_the_brain(env, monkeypatch):
    signed(env.app, mention(files=[FILE]))
    deliver(
        env.inbox,
        lambda r: httpx.Response(
            200, content=b"Full document text " * 500, headers={"content-type": "text/plain"}
        ),
    )
    idea = env.inbox.list()["ideas"][0]
    brain_calls = []

    def brain(request):
        brain_calls.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/aiia/remember":
            return httpx.Response(200, json={"id": "project_9_9"})
        return httpx.Response(200, json={"status": "indexed"})

    monkeypatch.setattr(slack_capture, "BRAIN_TRANSPORT", httpx.MockTransport(brain))
    response = call(
        env.app, "POST", f"/api/memory-inbox/{idea['id']}/promote", {"post_to_slack": False}
    )
    assert response.status_code == 200, response.text
    assert response.json()["files_indexed"] == 1
    paths = [path for path, _ in brain_calls]
    assert paths == ["/v1/aiia/remember", "/v1/aiia/ingest"]
    ingest = brain_calls[1][1]
    assert ingest["doc_type"] == "capture_file"
    assert ingest["metadata"]["file_id"] == "F0C6HNU79J8"
    assert len(ingest["text"]) > slack_files.EXCERPT_CHARS
