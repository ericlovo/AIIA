from local_brain.command_center.assignment_registry import AssignmentRegistry


def test_pending_loop_reviews_are_scoped_and_survive_reload(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    for agent_id, trigger, status in [
        ("one", "interval", "completed"),
        ("two", "interval", "completed"),
        ("one", "manual", "completed"),
        ("one", "interval", "failed"),
        ("one", "interval", "queued"),
        ("one", "interval", "running"),
    ]:
        work = registry.create_assignment(
            title="Evidence", objective="Inspect", agent_id=agent_id, trigger=trigger
        )
        if status != "queued":
            registry.set_running(work["id"])
        if status in {"completed", "failed"}:
            registry.finish_assignment(
                work["id"], result="Evidence", error="failure" if status == "failed" else ""
            )
    assert registry.pending_loop_reviews("one") == 1
    assert registry.pending_loop_reviews("two") == 1
    assert registry.pending_loop_reviews("unknown") == 0
    restored = AssignmentRegistry(registry.data_file)
    assert restored.pending_loop_reviews("one") == 1


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
    assert target["trigger"] == "handoff"
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


def test_scheduled_assignments_deduplicate_and_apply_backpressure(tmp_path):
    data_file = tmp_path / "assignments.json"
    registry = AssignmentRegistry(data_file)

    first, created = registry.create_scheduled_assignment(
        agent_id="agent_ci",
        agent_name="CI Signal Officer",
        objective="Inspect current CI state.",
        schedule_key="agent_ci:initial:30",
        interval_minutes=30,
    )
    duplicate, duplicate_created = registry.create_scheduled_assignment(
        agent_id="agent_ci",
        agent_name="CI Signal Officer",
        objective="Inspect current CI state.",
        schedule_key="agent_ci:initial:30",
        interval_minutes=30,
    )
    blocked, blocked_created = registry.create_scheduled_assignment(
        agent_id="agent_ci",
        agent_name="CI Signal Officer",
        objective="Inspect current CI state.",
        schedule_key="agent_ci:later:30",
        interval_minutes=30,
    )

    assert created is True
    assert duplicate_created is False
    assert blocked_created is False
    assert duplicate["id"] == blocked["id"] == first["id"]
    assert first["trigger"] == "interval"
    assert first["schedule_key"] == "agent_ci:initial:30"
    assert len(registry.list_assignments()) == 1

    registry.set_running(first["id"])
    registry.finish_assignment(first["id"], result="CI is green.")
    second, second_created = registry.create_scheduled_assignment(
        agent_id="agent_ci",
        agent_name="CI Signal Officer",
        objective="Inspect current CI state.",
        schedule_key="agent_ci:later:30",
        interval_minutes=30,
    )

    assert second_created is True
    assert second["id"] != first["id"]
    restored = AssignmentRegistry(data_file)
    assert restored.get_assignment(second["id"])["trigger"] == "interval"
