from local_brain.command_center.studio_events import studio_event, studio_snapshot


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
