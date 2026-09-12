import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from local_brain.command_center import assignment_registry as module
from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.persistence import PersistenceError


@pytest.fixture
def interrupted(tmp_path):
    agents = AgentRegistry(tmp_path / "agents.json")
    agent = agents.create(name="Synthetic", mission="Review", persona="Careful", skills=[])
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    source = assignments.create_assignment(title="Source", objective="Review", agent_id="upstream")
    assignments.finish_assignment(source["id"], result="Source evidence")
    link, work = assignments.create_handoff(
        source_assignment_id=source["id"],
        to_agent_id=agent["id"],
        artifact_type="analysis",
        instructions="Verify",
    )
    assignments.set_running(work["id"])
    agents.finish_run(
        agent["id"],
        "Verify",
        result="Recovered evidence",
        trigger="assignment",
        assignment_id=work["id"],
        run_id=work["attempt_id"],
    )
    return agents, assignments, agent, work, link


def test_reload_recovers_exact_attempt_and_preserves_review_on_repeat(interrupted):
    agents, assignments, _, work, link = interrupted
    restored = AssignmentRegistry(assignments.data_file)
    run = AgentRegistry(agents.data_file).persisted_run(work["attempt_id"])
    result = restored.recover_output(work["id"], run)
    assert result["result"] == "Recovered evidence"
    assert result["status"] == restored.get_handoff(link["id"])["status"] == "completed"
    assert result["completed_at"] == run["at"]
    restored.review_assignment(
        work["id"], decision="accepted", expected_version=result["review_version"]
    )
    before = restored.data_file.read_bytes()
    restored.recover_output(work["id"], run)
    assert restored.data_file.read_bytes() == before
    assert (
        AssignmentRegistry(restored.data_file).get_assignment(work["id"])["review_status"]
        == "accepted"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "old-attempt"),
        ("agent_id", "other"),
        ("assignment_id", "other"),
        ("trigger", "manual"),
    ],
)
def test_mismatched_evidence_never_applied(interrupted, field, value):
    agents, assignments, _, work, _ = interrupted
    run = agents.persisted_run(work["attempt_id"])
    run[field] = value
    before = assignments.data_file.read_bytes()
    with pytest.raises(ValueError, match="recovery_attempt_mismatch"):
        assignments.recover_output(work["id"], run)
    assert assignments.data_file.read_bytes() == before


def test_old_attempt_cannot_replace_new_attempt(interrupted):
    agents, assignments, _, work, _ = interrupted
    run = agents.persisted_run(work["attempt_id"])
    assignments.set_running(work["id"])
    with pytest.raises(ValueError, match="recovery_attempt_mismatch"):
        assignments.recover_output(work["id"], run)


def test_failed_recovery_rolls_back_then_retries_without_execution(interrupted, monkeypatch):
    agents, assignments, _, work, _ = interrupted
    run = agents.persisted_run(work["attempt_id"])
    before = deepcopy((assignments.assignments, assignments.handoffs))
    disk = assignments.data_file.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write_json", Mock(side_effect=PersistenceError("injected")))
        with pytest.raises(PersistenceError):
            assignments.recover_output(work["id"], run)
    assert (assignments.assignments, assignments.handoffs) == before
    assert assignments.data_file.read_bytes() == disk
    assignments.recover_output(work["id"], run)
    assert work["status"] == "completed"


def test_unsaved_memory_is_not_recovery_evidence(interrupted, monkeypatch):
    from local_brain.command_center import agent_registry

    agents, assignments, agent, work, _ = interrupted
    assignments.set_running(work["id"])
    with monkeypatch.context() as patch:
        patch.setattr(
            agent_registry, "atomic_write_json", Mock(side_effect=PersistenceError("injected"))
        )
        with pytest.raises(PersistenceError):
            agents.finish_run(
                agent["id"],
                "Verify",
                result="Unsaved",
                trigger="assignment",
                assignment_id=work["id"],
                run_id=work["attempt_id"],
            )
    assert agents.persisted_run(work["attempt_id"]) is None


def test_saved_json_outbox_is_recoverable_without_sqlite(interrupted, monkeypatch):
    from local_brain.command_center.agent_registry import RunHistoryUnavailable

    agents, assignments, agent, work, _ = interrupted
    assignments.set_running(work["id"])
    monkeypatch.setattr(agents, "recover_runs", Mock(side_effect=RunHistoryUnavailable("offline")))
    agents.finish_run(
        agent["id"],
        "Verify",
        result="Outbox evidence",
        trigger="assignment",
        assignment_id=work["id"],
        run_id=work["attempt_id"],
    )
    import sqlite3

    monkeypatch.setattr(agents.ledger, "get", Mock(side_effect=sqlite3.OperationalError("offline")))
    assert agents.persisted_run(work["attempt_id"])["result"] == "Outbox evidence"


@pytest.mark.parametrize(
    "result,error,expected",
    [
        ("", "", "empty_agent_result"),
        ("partial", "model_failed", "model_failed"),
        ("x" * 40001, "", "assignment_result_too_long"),
    ],
)
def test_failed_or_oversize_evidence_stays_failed(interrupted, result, error, expected):
    agents, assignments, _, work, link = interrupted
    run = agents.persisted_run(work["attempt_id"])
    run.update(result=result, error=error)
    assignments.recover_output(work["id"], run)
    assert work["status"] == link["status"] == "failed"
    assert work["error"] == expected


def test_recovery_api_has_no_model_calls_and_blocks_reexecution(interrupted, monkeypatch):
    from local_brain.command_center import server

    agents, assignments, _, work, _ = interrupted
    assignments = AssignmentRegistry(assignments.data_file)
    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    monkeypatch.setattr(server, "agent_run_lock", asyncio.Lock())
    executor = AsyncMock()
    monkeypatch.setattr(server, "_execute_agent", executor)
    monkeypatch.setattr(server, "broadcast_assignment_event", AsyncMock())

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            response = await client.post(f"/api/assignments/{work['id']}/run")
            assert response.status_code == 409
            assert response.json()["detail"] == "saved_output_available_recover_first"
            response = await client.post(f"/api/assignments/{work['id']}/recover")
            assert response.status_code == 200
            assert response.json()["assignment"]["result"] == "Recovered evidence"
            assert (await client.post(f"/api/assignments/{work['id']}/recover")).status_code == 200
            executor.assert_not_called()

    asyncio.run(exercise())


def test_startup_recovery_retries_storage_only(interrupted, monkeypatch):
    from local_brain.command_center import server

    agents, assignments, _, work, _ = interrupted
    assignments = AssignmentRegistry(assignments.data_file)
    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    executor = AsyncMock()
    monkeypatch.setattr(server, "_execute_agent", executor)
    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write_json", Mock(side_effect=PersistenceError("offline")))
        server.recover_pending_assignment_outputs()
    assert assignments.get_assignment(work["id"])["status"] == "failed"
    server.recover_pending_assignment_outputs()
    assert assignments.get_assignment(work["id"])["status"] == "completed"
    disk = assignments.data_file.read_bytes()
    server.recover_pending_assignment_outputs()
    assert assignments.data_file.read_bytes() == disk
    executor.assert_not_called()


@pytest.mark.parametrize(
    "condition,code", [("missing", 409), ("legacy", 409), ("busy", 409), ("write", 503)]
)
def test_api_recovery_failures_leave_evidence_untouched(interrupted, monkeypatch, condition, code):
    from local_brain.command_center import server

    agents, assignments, _, work, _ = interrupted
    assignments = AssignmentRegistry(assignments.data_file)
    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    lock = asyncio.Lock()
    monkeypatch.setattr(server, "agent_run_lock", lock)
    event, executor = AsyncMock(), AsyncMock()
    monkeypatch.setattr(server, "broadcast_assignment_event", event)
    monkeypatch.setattr(server, "_execute_agent", executor)
    if condition == "missing":
        monkeypatch.setattr(agents, "persisted_run", Mock(return_value=None))
    if condition == "legacy":
        assignments.get_assignment(work["id"]).pop("attempt_id")
        assignments.save()
    if condition == "write":
        monkeypatch.setattr(
            module, "atomic_write_json", Mock(side_effect=PersistenceError("secret-path"))
        )
    before = assignments.data_file.read_bytes()

    async def exercise():
        if condition == "busy":
            await lock.acquire()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            response = await client.post(f"/api/assignments/{work['id']}/recover")
            assert response.status_code == code
            assert "secret-path" not in response.text
        event.assert_not_called()
        executor.assert_not_called()

    asyncio.run(exercise())
    assert assignments.data_file.read_bytes() == before
