from local_brain.command_center.assignment_registry import AssignmentRegistry


def test_assignment_lifecycle_persists(tmp_path):
    data_file = tmp_path / "assignments.json"
    registry = AssignmentRegistry(data_file)
    assignment = registry.create_assignment(
        title="Map the authorization surface",
        objective="Return the five highest-leverage integration points.",
        agent_id="agent_strategy",
        priority="high",
        success_criteria="Five ranked recommendations with evidence.",
    )

    registry.set_running(assignment["id"])
    completed = registry.finish_assignment(
        assignment["id"], result="1. Authorization policy boundary"
    )

    assert completed["status"] == "completed"
    assert completed["result"].startswith("1. Authorization")
    restored = AssignmentRegistry(data_file)
    assert restored.get_assignment(assignment["id"])["priority"] == "high"


def test_handoff_creates_a_runnable_downstream_assignment(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    source = registry.create_assignment(
        title="Research enterprise controls",
        objective="Find the pattern.",
        agent_id="agent_research",
        context="Sanction product context",
    )
    registry.set_running(source["id"])
    registry.finish_assignment(source["id"], result="Use a scoped policy envelope.")

    handoff, target = registry.create_handoff(
        source_assignment_id=source["id"],
        to_agent_id="agent_architecture",
        artifact_type="analysis",
        instructions="Turn this analysis into an implementation plan.",
    )

    assert handoff["status"] == "queued"
    assert target["source_handoff_id"] == handoff["id"]
    assert "Use a scoped policy envelope" in target["context"]

    registry.set_running(target["id"])
    assert registry.get_handoff(handoff["id"])["status"] == "running"
    registry.finish_assignment(target["id"], result="Implementation plan")
    assert registry.get_handoff(handoff["id"])["status"] == "completed"
    assert not registry.delete_assignment(source["id"])
    assert registry.delete_handoff(handoff["id"])
    assert registry.delete_assignment(source["id"])
    assert registry.get_assignment(target["id"])["source_handoff_id"] == ""


def test_handoff_requires_completed_work_and_a_different_agent(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    source = registry.create_assignment(
        title="Draft a brief",
        objective="Draft it.",
        agent_id="agent_writer",
    )

    try:
        registry.create_handoff(
            source_assignment_id=source["id"],
            to_agent_id="agent_reviewer",
            artifact_type="brief",
            instructions="Review it.",
        )
    except ValueError as exc:
        assert str(exc) == "source_assignment_not_completed"
    else:
        raise AssertionError("incomplete work should not be handed off")

    registry.set_running(source["id"])
    registry.finish_assignment(source["id"], result="A completed brief")
    try:
        registry.create_handoff(
            source_assignment_id=source["id"],
            to_agent_id="agent_writer",
            artifact_type="brief",
            instructions="Review your own work.",
        )
    except ValueError as exc:
        assert str(exc) == "handoff_requires_different_agent"
    else:
        raise AssertionError("handoff should require a second agent")


def test_running_work_is_reconciled_after_restart(tmp_path):
    data_file = tmp_path / "assignments.json"
    registry = AssignmentRegistry(data_file)
    source = registry.create_assignment(
        title="Long-running research",
        objective="Finish the report.",
        agent_id="agent_research",
    )
    registry.set_running(source["id"])

    restored = AssignmentRegistry(data_file)
    interrupted = restored.get_assignment(source["id"])

    assert interrupted["status"] == "failed"
    assert interrupted["error"] == "interrupted_by_restart"
