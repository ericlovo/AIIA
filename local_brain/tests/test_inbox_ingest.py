"""Proposals filed by local loops: dedupe on rerun, no receipts, no forged sources."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.memory_inbox import MemoryInbox


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    agents = AgentRegistry(tmp_path / "agents.json")
    agents.create("Steward Hand", "Work the backlog", "Precise", [])
    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", AssignmentRegistry(tmp_path / "a.json"))
    monkeypatch.setattr(server, "memory_capture_inbox", lambda: inbox)
    monkeypatch.setattr(server, "broadcast_assignment_event", AsyncMock())
    monkeypatch.setattr("local_brain.command_center.slack_capture.inbox", lambda: inbox)
    return server, agents, inbox


def ingest(server, **body):
    payload = {
        "text": "P2 Publish the release checklist — no story covers it",
        "source": "backlog_steward",
        "source_key": "steward:proposal:release-checklist",
        "project": "aiia",
        **body,
    }
    return TestClient(server.app).post("/api/memory-inbox/ingest", json=payload)


def test_a_proposal_lands_unreviewed_with_no_slack_receipt(studio):
    server, _, inbox = studio

    response = ingest(server)

    assert response.status_code == 200
    body = response.json()
    assert body["created"] is True
    idea = body["idea"]
    assert idea["status"] == "unreviewed"
    assert idea["source"] == "backlog_steward"
    assert idea["project"] == "aiia"
    # Nothing outbound: no thread to reply to, so no receipt was queued.
    assert idea["acknowledgement_status"] is None
    assert inbox.receipt_status("capture") == {}


def test_a_rerun_does_not_refile_the_same_finding(studio):
    server, _, inbox = studio
    first = ingest(server)

    second = ingest(server, text="P2 Publish the release checklist — still no story")

    assert first.json()["created"] is True
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["idea"]["id"] == first.json()["idea"]["id"]
    # The row a human may already have reviewed is left exactly as it was.
    assert second.json()["idea"]["text"] == first.json()["idea"]["text"]
    assert inbox.list()["total"] == 1


def test_a_reviewed_proposal_stays_reviewed_when_the_loop_reruns(studio):
    server, _, inbox = studio
    idea = ingest(server).json()["idea"]
    inbox.dismiss(idea["id"], note="not now")

    ingest(server)

    assert inbox.get(idea["id"])["status"] == "dismissed"


@pytest.mark.parametrize("outcome", ["already_fixed", "declined", "external_failure"])
def test_a_proposal_can_be_classified_with_a_rationale(studio, outcome):
    server, _, inbox = studio
    idea = ingest(server).json()["idea"]

    response = TestClient(server.app).post(
        f"/api/memory-inbox/{idea['id']}/triage",
        json={"outcome": outcome, "note": "Reviewed against the current branch."},
    )

    assert response.status_code == 200
    reviewed = response.json()["idea"]
    assert reviewed["status"] == "dismissed"
    assert reviewed["review_outcome"] == outcome
    assert reviewed["review_note"] == "Reviewed against the current branch."
    assert reviewed["reviewed_at"]
    ingest(server)
    assert inbox.get(idea["id"])["review_outcome"] == outcome


def test_triage_refuses_slack_missing_rationale_and_needs_work_without_an_assignment(studio):
    server, _, inbox = studio
    proposal = ingest(server).json()["idea"]
    slack = inbox.capture(
        text="human capture",
        source_key="event:1",
        source="slack",
        project="mindmoor",
    )
    client = TestClient(server.app)

    assert (
        client.post(
            f"/api/memory-inbox/{proposal['id']}/triage",
            json={"outcome": "declined", "note": ""},
        ).status_code
        == 422
    )
    invalid = client.post(
        f"/api/memory-inbox/{proposal['id']}/triage",
        json={"outcome": "needs_work", "note": "Should be assigned."},
    )
    assert invalid.status_code == 422
    refused = client.post(
        f"/api/memory-inbox/{slack['id']}/triage",
        json={"outcome": "declined", "note": "Not a local proposal."},
    )
    assert refused.status_code == 409
    assert refused.json()["detail"] == "idea_not_triageable"


def test_a_loop_cannot_file_as_slack(studio):
    server, _, inbox = studio

    response = ingest(server, source="slack")

    assert response.status_code == 422
    assert response.json()["detail"] == "unknown_ingest_source"
    assert inbox.list()["total"] == 0


def test_empty_text_and_missing_key_are_refused(studio):
    server, _, inbox = studio

    assert ingest(server, text="   ").status_code == 422
    assert ingest(server, source_key="   ").status_code == 422
    assert inbox.list()["total"] == 0


def test_a_filed_proposal_can_be_queued_as_work(studio):
    """The point of filing it: the existing review path already knows what to do."""
    server, agents, _ = studio
    idea = ingest(server).json()["idea"]

    response = TestClient(server.app).post(
        f"/api/memory-inbox/{idea['id']}/assign",
        json={
            "agent_id": agents.agents[0]["id"],
            "review_note": "Valid defect; send it to delivery.",
        },
    )

    assert response.status_code == 200
    assignment = response.json()["assignment"]
    assert assignment["status"] == "queued"
    assert assignment["source_kind"] == "memory_capture"
    assert assignment["source_ref"] == idea["id"]
    reviewed = response.json()["idea"]
    assert reviewed["review_outcome"] == "needs_work"
    assert reviewed["review_note"] == "Valid defect; send it to delivery."
    assert reviewed["reviewed_at"]


def test_releasing_a_failed_assignment_claim_rolls_back_needs_work(studio):
    _, _, inbox = studio
    idea, _ = inbox.ingest(
        text="Review this finding",
        source_key="code-review:rollback",
        source="code_review",
        project="mindmoor",
    )
    claimed = inbox.attach_assignment(idea["id"], "assignment-1", note="Valid defect.")
    assert claimed["review_outcome"] == "needs_work"

    released = inbox.attach_assignment(idea["id"], "", replace=True)

    assert released["assignment_id"] == ""
    assert released["review_outcome"] == ""
    assert released["review_note"] == ""
    assert released["reviewed_at"] == ""


def test_proposals_can_be_listed_apart_from_slack_captures(studio):
    server, _, inbox = studio
    ingest(server)
    ingest(
        server,
        text="Review bot found an optional dependency outage",
        source="code_review",
        source_key="code-review:51:optional-seed",
        project="mindmoor",
    )
    ingest(
        server,
        text="Standup found a blocked deployment",
        source="standup",
        source_key="standup:2026-09-21:deploy",
        project="sanction",
    )
    inbox.capture(
        text="a person said this",
        source_key="event:1",
        source="slack",
        project="mindmoor",
    )

    client = TestClient(server.app)
    stewarded = client.get("/api/memory-inbox?source=backlog_steward").json()
    proposals = client.get("/api/memory-inbox?source=local_proposals").json()
    slack = client.get("/api/memory-inbox?source=slack").json()
    everything = client.get("/api/memory-inbox").json()

    assert [item["source"] for item in stewarded["ideas"]] == ["backlog_steward"]
    assert {item["source"] for item in proposals["ideas"]} == {
        "backlog_steward",
        "code_review",
        "standup",
    }
    assert [item["source"] for item in slack["ideas"]] == ["slack"]
    assert stewarded["total"] == 1
    assert proposals["total"] == 3
    # Counts follow the same scope as the rows, or the tab lies about its backlog.
    assert proposals["counts"] == {"unreviewed": 3, "promoted": 0, "dismissed": 0}
    assert everything["total"] == 4
