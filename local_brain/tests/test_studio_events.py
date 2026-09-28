import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from local_brain.command_center.studio_events import studio_event, studio_snapshot

# Deliberately independent of the production constants: additions require review.
CONTRACT_FIELDS = {
    "agent": ("id", "name", "status", "updated_at"),
    "assignment": (
        "id",
        "title",
        "agent_id",
        "status",
        "source_handoff_id",
        "updated_at",
        "started_at",
        "completed_at",
    ),
    "handoff": (
        "id",
        "source_assignment_id",
        "target_assignment_id",
        "from_agent_id",
        "to_agent_id",
        "artifact_type",
        "status",
        "updated_at",
    ),
}


@pytest.mark.parametrize("entity", CONTRACT_FIELDS)
def test_exact_event_and_snapshot_allowlists(entity):
    expected = {field: f"fixture-{field}" for field in CONTRACT_FIELDS[entity]}
    record = {
        **expected,
        "api_key": "PRIVATE-KEY",
        "token": "PRIVATE-TOKEN",
        "future_field": {"token": "PRIVATE-NESTED"},
        "mission": "PRIVATE-MISSION",
        "context": "PRIVATE-CONTEXT",
        "artifact": "PRIVATE-ARTIFACT",
        "result": "PRIVATE-RESULT",
    }
    before = deepcopy(record)
    event = studio_event(entity, "updated", record)
    assert event == {"entity": entity, "event": "updated", "item": expected}
    inputs = {"agents": [], "assignments": [], "handoffs": []}
    inputs[f"{entity}s"] = [record]
    snapshot = studio_snapshot(**inputs)
    assert snapshot == {**{key: [] for key in inputs}, f"{entity}s": [expected]}
    assert "PRIVATE" not in json.dumps([event, snapshot])
    assert record == before


@pytest.mark.parametrize("entity", CONTRACT_FIELDS)
def test_missing_projected_fields_are_explicit_nulls(entity):
    assert studio_event(entity, "deleted", {})["item"] == dict.fromkeys(CONTRACT_FIELDS[entity])


def test_unknown_entity_fails_instead_of_returning_unprojected_data():
    with pytest.raises(KeyError):
        studio_event("unknown", "created", {"api_key": "PRIVATE"})


def test_empty_snapshot_keeps_all_collections():
    assert studio_snapshot([], [], []) == {"agents": [], "assignments": [], "handoffs": []}


async def test_broadcast_wrapper_uses_projected_wire_envelope(monkeypatch):
    from local_brain.command_center import server

    socket = AsyncMock()
    manager = server.ConnectionManager()
    manager.connections.append(socket)
    monkeypatch.setattr(server, "manager", manager)
    await server.broadcast_studio_event("agent", "created", {"id": "a", "token": "PRIVATE"})
    payload = json.loads(socket.send_text.await_args.args[0])
    assert payload == {
        "type": "agent_studio_update",
        "data": {
            "entity": "agent",
            "event": "created",
            "item": {"id": "a", "name": None, "status": None, "updated_at": None},
        },
    }


@pytest.mark.parametrize("status", ["queued", "running", "failed", "completed"])
async def test_linked_handoff_emits_projected_status_after_assignment(monkeypatch, status):
    from local_brain.command_center import server

    broadcast = AsyncMock()
    monkeypatch.setattr(server, "manager", SimpleNamespace(broadcast=broadcast))
    handoff = {"id": "h", "status": status, "api_key": "PRIVATE"}
    monkeypatch.setattr(
        server, "assignment_registry", SimpleNamespace(get_handoff=lambda _: handoff)
    )
    await server.broadcast_assignment_event(
        "updated", {"id": "a", "source_handoff_id": "h", "token": "PRIVATE"}
    )
    assert broadcast.await_count == 2
    first, second = [call.args for call in broadcast.await_args_list]
    assert first[0] == second[0] == "agent_studio_update"
    assert first[1]["entity"] == "assignment"
    assert second[1] == {
        "entity": "handoff",
        "event": status,
        "item": {**dict.fromkeys(CONTRACT_FIELDS["handoff"]), "id": "h", "status": status},
    }
    assert "PRIVATE" not in json.dumps([first, second])


def test_studio_event_excludes_private_agent_and_assignment_content():
    agent = {
        "id": "agent_1",
        "name": "Researcher",
        "status": "running",
        "updated_at": "2026-09-06T12:00:00+00:00",
        "mission": "Private mission",
        "last_result": "Private result",
    }
    assignment = {
        "id": "asg_1",
        "title": "Map the system",
        "agent_id": "agent_1",
        "status": "running",
        "source_handoff_id": "",
        "updated_at": "2026-09-06T12:00:00+00:00",
        "started_at": "2026-09-06T12:00:00+00:00",
        "completed_at": None,
        "objective": "Private objective",
        "context": "Private context",
        "result": "Private result",
    }

    agent_payload = studio_event("agent", "running", agent)
    assignment_payload = studio_event("assignment", "running", assignment)

    assert agent_payload["item"] == {
        "id": "agent_1",
        "name": "Researcher",
        "status": "running",
        "updated_at": "2026-09-06T12:00:00+00:00",
    }
    assert "objective" not in assignment_payload["item"]
    assert "context" not in assignment_payload["item"]
    assert "result" not in assignment_payload["item"]


def test_studio_snapshot_preserves_handoff_topology():
    snapshot = studio_snapshot(
        agents=[
            {
                "id": "agent_1",
                "name": "Researcher",
                "status": "idle",
                "updated_at": "now",
            }
        ],
        assignments=[],
        handoffs=[
            {
                "id": "hof_1",
                "source_assignment_id": "asg_1",
                "target_assignment_id": "asg_2",
                "from_agent_id": "agent_1",
                "to_agent_id": "agent_2",
                "artifact_type": "analysis",
                "status": "queued",
                "updated_at": "now",
                "artifact": "Private artifact",
                "instructions": "Private instructions",
            }
        ],
    )

    assert snapshot["handoffs"][0]["from_agent_id"] == "agent_1"
    assert snapshot["handoffs"][0]["to_agent_id"] == "agent_2"
    assert "artifact" not in snapshot["handoffs"][0]
    assert "instructions" not in snapshot["handoffs"][0]
