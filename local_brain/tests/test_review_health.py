"""Review health: what the console reads to see whether a loop earns its slot.

The numbers come from the persisted outcome column, never from proposal text,
and every one of them has to be reachable as a filter on the inbox itself.
"""

from datetime import datetime, timedelta, timezone
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
    agents.create("Delivery Watch", "Watch delivery", "Precise", [])
    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", AssignmentRegistry(tmp_path / "a.json"))
    monkeypatch.setattr(server, "memory_capture_inbox", lambda: inbox)
    monkeypatch.setattr(server, "broadcast_assignment_event", AsyncMock())
    monkeypatch.setattr("local_brain.command_center.slack_capture.inbox", lambda: inbox)
    return server, agents, inbox


def proposal(inbox, key, *, source="code_review", project="mindmoor", text="a finding"):
    idea, _ = inbox.ingest(text=text, source_key=key, source=source, project=project)
    return idea


def health(server, **params):
    query = "&".join(f"{name}={value}" for name, value in params.items())
    return TestClient(server.app).get(
        f"/api/memory-inbox/review-health?{query}" if query else "/api/memory-inbox/review-health"
    )


def test_an_empty_inbox_reports_zeroes_not_an_error(studio):
    server, _, _ = studio

    body = health(server).json()

    assert body["filed"] == 0
    assert body["reviewed"] == 0
    assert body["totals"] == {
        "open": 0,
        "needs_work": 0,
        "already_fixed": 0,
        "declined": 0,
        "external_failure": 0,
        "unclassified": 0,
    }
    assert body["by_source"] == []
    assert body["by_project"] == []
    assert body["window_days"] == 14


def test_every_outcome_is_counted_under_its_own_name(studio):
    server, agents, inbox = studio
    client = TestClient(server.app)
    proposal(inbox, "k-open")
    for key, outcome in (
        ("k-fixed", "already_fixed"),
        ("k-declined", "declined"),
        ("k-external", "external_failure"),
    ):
        idea = proposal(inbox, key)
        inbox.triage(idea["id"], outcome=outcome, note="reviewed")
    accepted = proposal(inbox, "k-work")
    client.post(
        f"/api/memory-inbox/{accepted['id']}/assign",
        json={"agent_id": agents.agents[0]["id"], "review_note": "real defect"},
    )

    totals = health(server).json()["totals"]

    assert totals == {
        "open": 1,
        "needs_work": 1,
        "already_fixed": 1,
        "declined": 1,
        "external_failure": 1,
        "unclassified": 0,
    }


def test_a_row_closed_before_outcomes_existed_is_unclassified_not_declined(studio):
    server, _, inbox = studio
    legacy = proposal(inbox, "k-legacy")
    # What a pre-outcome dismissal left behind: terminal, with no verdict.
    inbox.dismiss(legacy["id"], note="old noise")

    body = health(server).json()

    assert body["totals"]["unclassified"] == 1
    assert body["totals"]["declined"] == 0
    assert body["reviewed"] == 1


def test_a_proposal_promoted_to_memory_without_a_verdict_is_unclassified(studio):
    server, _, inbox = studio
    idea = proposal(inbox, "k-promoted")
    inbox.promote(idea["id"], memory_id="mem-1", category="project")

    assert health(server).json()["totals"]["unclassified"] == 1


def test_accepted_work_is_not_counted_as_still_open(studio):
    """An accepted row keeps status `unreviewed`, so `open` cannot key on status."""
    server, agents, inbox = studio
    accepted = proposal(inbox, "k-accepted")
    TestClient(server.app).post(
        f"/api/memory-inbox/{accepted['id']}/assign",
        json={"agent_id": agents.agents[0]["id"], "review_note": "queue it"},
    )

    totals = health(server).json()["totals"]

    assert totals["open"] == 0
    assert totals["needs_work"] == 1


def test_sources_and_projects_are_broken_out(studio):
    server, _, inbox = studio
    proposal(inbox, "k-1", source="code_review", project="mindmoor")
    proposal(inbox, "k-2", source="code_review", project="mindmoor")
    standup = proposal(inbox, "k-3", source="standup", project="sanction")
    inbox.triage(standup["id"], outcome="declined", note="not now")
    proposal(inbox, "k-4", source="backlog_steward", project="")

    body = health(server).json()

    assert body["filed"] == 4
    by_source = {row["source"]: row for row in body["by_source"]}
    assert by_source["code_review"]["open"] == 2
    assert by_source["standup"]["declined"] == 1
    by_project = {row["project"]: row for row in body["by_project"]}
    assert by_project["mindmoor"]["open"] == 2
    # A proposal with no project is named, not dropped from the breakdown.
    assert by_project["unassigned"]["open"] == 1


def test_slack_captures_are_not_loop_health(studio):
    server, _, inbox = studio
    inbox.capture(
        text="a person said this", source_key="event:1", source="slack", project="mindmoor"
    )

    assert health(server).json()["filed"] == 0


def test_the_window_is_bounded_and_excludes_older_rows(studio):
    server, _, inbox = studio
    proposal(inbox, "k-new")
    old = proposal(inbox, "k-old")
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with inbox.connect() as db:
        db.execute("UPDATE ideas SET created_at=? WHERE id=?", (stale, old["id"]))

    assert health(server, days=14).json()["filed"] == 1
    assert health(server, days=60).json()["filed"] == 2
    assert health(server, days=0).status_code == 422
    assert health(server, days=91).status_code == 422


def test_a_project_can_be_scoped(studio):
    server, _, inbox = studio
    proposal(inbox, "k-a", project="mindmoor")
    proposal(inbox, "k-b", project="sanction")

    assert health(server, project="sanction").json()["filed"] == 1


@pytest.mark.parametrize(
    "outcome,expected_key",
    [
        ("open", "k-open"),
        ("already_fixed", "k-fixed"),
        ("unclassified", "k-legacy"),
    ],
)
def test_each_metric_opens_the_inbox_filtered_to_itself(studio, outcome, expected_key):
    """The point of the view: a number is a doorway, not a dead end."""
    server, _, inbox = studio
    open_row = proposal(inbox, "k-open")
    fixed = proposal(inbox, "k-fixed")
    inbox.triage(fixed["id"], outcome="already_fixed", note="landed on main")
    legacy = proposal(inbox, "k-legacy")
    inbox.dismiss(legacy["id"], note="old noise")
    keys = {"k-open": open_row["id"], "k-fixed": fixed["id"], "k-legacy": legacy["id"]}

    listed = (
        TestClient(server.app)
        .get(f"/api/memory-inbox?source=local_proposals&outcome={outcome}")
        .json()
    )

    assert [item["id"] for item in listed["ideas"]] == [keys[expected_key]]
    assert listed["total"] == 1


def test_an_unknown_outcome_filter_is_refused(studio):
    server, _, _ = studio

    assert TestClient(server.app).get("/api/memory-inbox?outcome=made_up").status_code == 422
