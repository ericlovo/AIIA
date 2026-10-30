from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center import slack_capture
from local_brain.command_center.memory_inbox import MemoryInbox


def test_new_routes_preserve_status_and_existing_human_work(tmp_path):
    inbox = MemoryInbox(tmp_path / "ideas.db")
    for source, lane in [
        ("digest", "activity"),
        ("code_review", "verification"),
        ("standup", "verification"),
        ("backlog_steward", "verification"),
        ("slack", "needs_you"),
        ("unknown", "needs_you"),
    ]:
        idea = inbox.capture(
            text="Original evidence", source_key=source, source=source, project="aiia"
        )
        assert idea["attention_lane"] == lane
        assert idea["status"] == "unreviewed"
    review, created = inbox.ingest(
        text="Different evidence", source_key="review2", source="code_review", project="aiia"
    )
    assert created and review["attention_lane"] == "verification"
    with inbox.connect() as db:
        db.execute(
            "UPDATE ideas SET assignment_id='protected',review_outcome='needs_work' WHERE id=?",
            (review["id"],),
        )
    assert inbox.list(attention="needs_you")["total"] == 3
    assert inbox.list(attention="verification")["total"] == 3
    assert inbox.get(review["id"])["text"] == "Different evidence"
    assert inbox.list()["total"] == 7
    assert inbox.attention_preview()["counts"]["linked_work"] == 1


def test_preview_all_174_without_reclassifying_or_touching_originals(tmp_path, monkeypatch):
    inbox = MemoryInbox(tmp_path / "ideas.db")
    for i in range(174):
        inbox.ingest(text=f"Original {i}", source_key=str(i), source="code_review", project="aiia")
    with inbox.connect() as db:
        db.execute("UPDATE ideas SET attention_lane='legacy'")
        before = [tuple(row) for row in db.execute("SELECT * FROM ideas ORDER BY id")]
    app = FastAPI()
    app.include_router(slack_capture.router)
    monkeypatch.setattr(slack_capture, "inbox", lambda: inbox)
    client = TestClient(app)
    rows = []
    for offset in (0, 50, 100, 150):
        data = client.get(f"/api/memory-inbox/attention-preview?offset={offset}").json()
        assert data["preview_only"] and data["total"] == 174
        rows.extend(data["ideas"])
    assert len({row["id"] for row in rows}) == 174
    assert inbox.list(attention="needs_you")["total"] == 174
    assert inbox.list(attention="verification")["total"] == 0
    with inbox.connect() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM ideas ORDER BY id")] == before
    assert client.get("/api/memory-inbox?attention=made_up").status_code == 422
    assert client.get("/api/memory-inbox/attention-preview?offset=-1").status_code == 422
    assert client.post("/api/memory-inbox/attention-preview").status_code == 405


def test_old_schema_migration_and_reingest_never_reclassify(tmp_path):
    inbox = MemoryInbox(tmp_path / "ideas.db")
    with inbox.connect() as db:
        db.execute("ALTER TABLE ideas DROP COLUMN attention_lane")
        db.execute(
            "INSERT INTO ideas(id,source_key,text,source,project,workspace_id,channel_id,author_id,created_at) VALUES('old','old','keep','digest','aiia','','','','then')"
        )
    original = inbox.get("old")
    same, created = inbox.ingest(
        text="replacement", source_key="old", source="digest", project="aiia"
    )
    assert not created and same["text"] == "keep"
    assert same["attention_lane"] == original["attention_lane"] == "legacy"
    assert inbox.list(attention="needs_you")["total"] == 1
    assert inbox.attention_preview()["ideas"][0]["proposed_lane"] == "activity"


def test_human_provenance_never_silently_routed(tmp_path):
    inbox = MemoryInbox(tmp_path / "ideas.db")
    idea = inbox.capture(
        text="Please decide",
        source_key="person",
        source="code_review",
        project="aiia",
        author_id="human",
    )
    assert idea["attention_lane"] == "needs_you"
    assert inbox.attention_preview()["ideas"][0]["proposed_lane"] == "needs_you"
