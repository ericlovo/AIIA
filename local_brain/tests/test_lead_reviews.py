import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center import lead_reviews
from local_brain.command_center.memory_inbox import MemoryInbox


@pytest.fixture
def setup(tmp_path, monkeypatch):
    inbox = MemoryInbox(tmp_path / "inbox.db")
    monkeypatch.setattr(lead_reviews, "inbox", lambda: inbox)
    idea, _ = inbox.ingest(
        text="Public expansion report", source_key="test", source="public_signals", project="pl"
    )
    app = FastAPI()
    app.include_router(lead_reviews.router)
    return TestClient(app), inbox, f"/api/public-signals/{idea['id']}/qualification"


def test_review_history_conflict_and_inbox_unchanged(setup):
    client, inbox, path = setup
    assert client.get(path).json() == {"review": None, "history": []}
    body = {"expected_version": 0, "status": "research", "note": "Confirm ownership"}
    assert client.put(path, json=body).json()["review"]["version"] == 1
    assert client.put(path, json=body).status_code == 409
    body.update(
        expected_version=1,
        status="qualified",
        company="Example",
        evidence_url="https://example.com/news",
        account_fit="Family owned, Iowa",
        observed_change="New facility",
        note="Reviewed company announcement",
    )
    assert client.put(path, json=body).status_code == 200
    data = client.get(path).json()
    assert [row["version"] for row in data["history"]] == [2, 1]
    assert data["review"]["status"] == "qualified"
    with inbox.connect() as db:
        row = db.execute("SELECT status,assignment_id FROM ideas").fetchone()
        assert row["status"] == "unreviewed"
        assert not row["assignment_id"]


@pytest.mark.parametrize(
    "extra",
    [
        {"note": "   "},
        {"status": "qualified"},
        {"expected_version": True},
        {"evidence_url": "javascript:alert(1)"},
        {"evidence_url": "http://example.com"},
        {"evidence_url": "https://user:pass@example.com"},
        {"status": "contacted"},
        {"unexpected": "field"},
        {"company": "x" * 201},
    ],
)
def test_invalid_reviews_do_not_write(setup, extra):
    client, _, path = setup
    assert (
        client.put(
            path, json={"expected_version": 0, "status": "research", "note": "Review", **extra}
        ).status_code
        == 422
    )
    assert client.get(path).json()["history"] == []


def test_only_public_signals(setup):
    client, inbox, _ = setup
    idea, _ = inbox.ingest(
        text="Private capture", source_key="private", source="slack", project="pl"
    )
    for identifier in [idea["id"], "missing"]:
        path = f"/api/public-signals/{identifier}/qualification"
        assert client.get(path).status_code == 404
        assert (
            client.put(
                path, json={"expected_version": 0, "status": "watch", "note": "Wait"}
            ).status_code
            == 404
        )


def test_history_failure_rolls_back_current_review(setup):
    client, inbox, path = setup
    assert client.get(path).status_code == 200
    with inbox.connect() as db:
        db.execute(
            "CREATE TRIGGER fail_history BEFORE INSERT ON lead_review_history BEGIN SELECT RAISE(ABORT, 'test'); END"
        )
    response = client.put(
        path, json={"expected_version": 0, "status": "research", "note": "Not saved"}
    )
    assert response.status_code == 503
    assert client.get(path).json() == {"review": None, "history": []}


def test_storage_failure(setup, monkeypatch):
    client, inbox, path = setup

    def broken():
        raise sqlite3.OperationalError("offline")

    monkeypatch.setattr(inbox, "connect", broken)
    assert client.get(path).status_code == 503
    assert client.get("/api/public-signals/leads").status_code == 503
    assert (
        client.put(
            path, json={"expected_version": 0, "status": "watch", "note": "Wait"}
        ).status_code
        == 503
    )


def test_queue_before_reviews_and_source_isolation(setup):
    client, inbox, _ = setup
    inbox.ingest(text="Private", source_key="private", source="slack", project="pl")
    data = client.get("/api/public-signals/leads").json()
    assert data["total"] == 1
    assert data["leads"][0]["decision"] == "unreviewed"
    assert data["leads"][0]["review"] is None
    assert client.get("/api/public-signals/leads?status=qualified").json()["total"] == 0


@pytest.mark.parametrize("status", ["research", "watch", "qualified", "rejected"])
def test_queue_filters_saved_decision_and_literal_company(setup, status):
    client, _, path = setup
    body = dict(
        expected_version=0,
        status=status,
        company="Example 100% Iowa",
        evidence_url="https://example.com",
        account_fit="Family owned",
        observed_change="Expansion",
        note="Reviewed",
    )
    assert client.put(path, json=body).status_code == 200
    data = client.get(
        "/api/public-signals/leads", params={"status": status, "company": "IOWA"}
    ).json()
    assert data["total"] == 1
    assert data["leads"][0]["review"]["version"] == 1
    assert data["leads"][0]["review"]["note"] == "Reviewed"
    assert client.get("/api/public-signals/leads", params={"company": "%"}).json()["total"] == 1
    assert client.get("/api/public-signals/leads", params={"company": "_"}).json()["total"] == 0
    assert client.get("/api/public-signals/leads?status=unreviewed").json()["total"] == 0


def test_queue_pagination_and_disposition(setup):
    client, inbox, path = setup
    for i in range(30):
        inbox.ingest(
            text=f"Signal {i}", source_key=f"key{i}", source="public_signals", project="pl"
        )
    idea_id = path.split("/")[3]
    with inbox.connect() as db:
        db.execute(
            "UPDATE ideas SET status='dismissed',assignment_id='work-1' WHERE id=?", (idea_id,)
        )
    first = client.get("/api/public-signals/leads").json()
    second = client.get("/api/public-signals/leads?offset=25").json()
    assert first["total"] == second["total"] == 31
    assert len(first["leads"]) == 25
    assert len(second["leads"]) == 6
    assert not {row["id"] for row in first["leads"]} & {row["id"] for row in second["leads"]}
    all_rows = first["leads"] + second["leads"]
    archived = next(row for row in all_rows if row["id"] == idea_id)
    assert archived["inbox_status"] == "dismissed"
    assert archived["assignment_id"] == "work-1"


@pytest.mark.parametrize(
    "query", ["status=invalid", "offset=-1", "limit=101", "limit=0", "company=" + "x" * 201]
)
def test_queue_invalid_query(setup, query):
    assert setup[0].get("/api/public-signals/leads?" + query).status_code == 422


@pytest.mark.parametrize("payload", ["not json", "{}", '{"status":"qualified"}', "[]"])
def test_queue_corrupt_json_is_not_an_empty_queue(setup, payload):
    client, inbox, path = setup
    client.put(path, json=dict(expected_version=0, status="watch", note="Wait"))
    with inbox.connect() as db:
        db.execute("UPDATE lead_reviews SET payload=?", (payload,))
    assert client.get("/api/public-signals/leads").status_code == 503
