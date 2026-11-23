from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center import client_meetings as meetings


def test_shared_calls_and_concurrent_note_imports(tmp_path, monkeypatch):
    path = tmp_path / "meetings.db"
    monkeypatch.setattr(meetings, "store", meetings.MeetingStore(path))
    app = FastAPI()
    app.include_router(meetings.router)
    client = TestClient(app)
    base = "/api/clients/smart-medical/meetings"
    draft = {"title": "Weekly review", "starts_at": "2026-11-12T10:00:00-06:00"}
    call = client.post(base, json=draft).json()
    assert client.post(base, json={**draft, "starts_at": "2026-11-12T16:00:00Z"}).json() == call
    assert client.post(base, json={**draft, "starts_at": "2026-11-19T16:00:00Z"}).json() != call
    assert client.post(base, json={**draft, "starts_at": "2026-11-12T10:00:00"}).status_code == 422
    url = f"{base}/{call['id']}"
    zoom = {
        "source": "zoom",
        "author": "Contributor A",
        "text": "Zoom evidence",
        "source_ref": "z1",
    }
    morrow = {"source": "morrow", "author": "Contributor B", "text": "Morrow evidence"}
    with ThreadPoolExecutor(4) as pool:
        receipts = list(
            pool.map(
                lambda n: client.post(url + "/notes", json=n).json(), [zoom, morrow, zoom, morrow]
            )
        )
    assert receipts[:2] == receipts[2:]
    assert receipts[0] != receipts[1]
    assert client.post(url + "/notes", json={**zoom, "text": "Different text"}).status_code == 409
    monkeypatch.setattr(meetings, "store", meetings.MeetingStore(path))
    notes = client.get(url).json()["notes"]
    assert {n["text"] for n in notes} == {"Zoom evidence", "Morrow evidence"}
    wrong = url.replace("smart-medical", "gci")
    assert client.get(wrong).status_code == 404
    assert client.post(wrong + "/notes", json=zoom).status_code == 404
    assert (
        client.get("/api/clients/gci/context", params={"meeting_id": call["id"]}).status_code == 404
    )
    assert client.get("/api/clients/gci/context").json()["meetings"] == []
    context = client.get("/api/clients/smart-medical/context").json()
    assert len(context["meetings"]) == 1
    assert {n["note_id"] for n in context["meetings"][0]["notes"]} == {n["id"] for n in notes}


def test_large_notes_share_agent_budget_and_keep_originals(tmp_path):
    store = meetings.MeetingStore(tmp_path / "meetings.db")
    call = store.create("client", meetings.NewMeeting(title="Call", starts_at=datetime.now(UTC)))
    for i in range(10):
        store.append(
            "client", call["id"], meetings.NewNote(author="A", source="zoom", text=str(i) * 10000)
        )
    store.append(
        "client", call["id"], meetings.NewNote(author="B", source="morrow", text="B" * 9000)
    )
    rows = store.agent_context("client", call["id"])["meetings"][0]["notes"]
    assert {row["author"] for row in rows} == {"A", "B"}
    assert sum(len(row["text"]) for row in rows) <= 8000
    assert all(row["excerpted"] for row in rows)
    assert len(store.get("client", call["id"])["notes"]) == 11
