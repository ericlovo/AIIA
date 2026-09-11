import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry


def test_activity_routes_and_loop_controls(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    agent = registry.create("Review", "Inspect", "Precise", [], loop_task="Review changes")
    blank = registry.create("Unconfigured", "Inspect", "Precise", [])
    registry.finish_run(agent["id"], "Inspect", result="Evidence", trigger="assignment")
    monkeypatch.setattr(server, "agent_registry", registry)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            response = await client.get("/api/studio/activity")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert "result" not in data["runs"][0]
            detail = await client.get(f"/api/studio/runs/{data['runs'][0]['id']}")
            assert detail.json()["run"]["result"] == "Evidence"
            assert (await client.get("/api/studio/runs/missing")).status_code == 404
            assert (await client.get("/api/studio/activity?day=not-a-date")).status_code == 422
            assert (await client.get("/api/studio/activity?status=unknown")).status_code == 422
            filtered = await client.get("/api/studio/activity?status=failed")
            assert filtered.json()["matching"] == 0
            for enabled in (True, False):
                response = await client.post(
                    f"/api/agents/{agent['id']}/loop", json={"enabled": enabled}
                )
                assert response.status_code == 200
                assert response.json()["agent"]["loop_enabled"] is enabled
            response = await client.post(f"/api/agents/{blank['id']}/loop", json={"enabled": True})
            assert response.status_code == 422
            response = await client.post("/api/agents/missing/loop", json={"enabled": True})
            assert response.status_code == 404
            assert registry.ledger.activity()["total"] == 1
            assert agent["status"] == "idle"

    asyncio.run(exercise())


def test_task_status_projection_does_not_mutate_runner():
    from local_brain.command_center.aiia_tasks import TaskRunner

    runner = TaskRunner(AsyncMock(), "/unused", None)
    task_id = next(iter(runner.tasks))
    task = runner.tasks[task_id]
    task["run_history"] = [{"status": "done"}]
    projected = next(row for row in runner.get_all_tasks() if row["task_id"] == task_id)
    assert projected["last_status"] == "done"
    assert projected["interval_seconds"] > 0
    assert "last_status" not in task
    assert "interval_seconds" not in task


def test_agent_execution_parameters_reach_model_and_ledger(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    agent = registry.create("Review", "Inspect", "Precise", [], temperature=0.2, max_tokens=900)
    monkeypatch.setattr(server, "agent_registry", registry)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = httpx.Response(
        200, json={"content": "Observed evidence", "model": "test-local", "latency_ms": 12}
    )
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: client)
    asyncio.run(server._execute_agent(agent["id"], "Review changes"))
    request = client.post.call_args.kwargs["json"]
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 900
    run = registry.ledger.get(agent["runs"][0]["id"])
    assert run["status"] == "completed"
    assert run["model"] == "test-local"
    assert run["temperature"] == 0.2
    assert run["max_tokens"] == 900


def test_assignment_output_survives_history_failure_without_another_model_call(
    tmp_path, monkeypatch
):
    import sqlite3
    from unittest.mock import patch

    from local_brain.command_center import server
    from local_brain.command_center.assignment_registry import AssignmentRegistry
    from local_brain.command_center.run_ledger import RunLedger

    path = tmp_path / "agents.json"
    assignment_path = tmp_path / "assignments.json"
    registry = AgentRegistry(path)
    assignments = AssignmentRegistry(assignment_path)
    agent = registry.create("Review", "Inspect", "Precise", [])
    assignment = assignments.create_assignment(
        title="Review fixture", objective="Inspect", agent_id=agent["id"]
    )
    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    monkeypatch.setattr(server, "agent_run_lock", asyncio.Lock())
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = httpx.Response(
        200, json={"content": "Completed artifact", "model": "test-local"}
    )
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: client)

    with patch.object(RunLedger, "record", side_effect=sqlite3.OperationalError("locked")):
        result = asyncio.run(server.run_assignment(assignment["id"]))
        assert result["assignment"]["status"] == "completed"
        assert result["assignment"]["result"] == "Completed artifact"
        unavailable = AgentRegistry(path)
        assert unavailable.get(agent["id"])["status"] == "idle"
        assert unavailable.pending_run_count == 1
        assert (
            AssignmentRegistry(assignment_path).get_assignment(assignment["id"])["result"]
            == "Completed artifact"
        )

    monkeypatch.setattr(server, "agent_registry", unavailable)
    activity = asyncio.run(server.studio_activity())
    assert activity["total"] == 1
    detail = asyncio.run(server.studio_run(activity["runs"][0]["id"]))
    assert detail["run"]["result"] == "Completed artifact"
    assert detail["run"]["assignment_id"] == assignment["id"]
    assert unavailable.pending_run_count == 0
    client.post.assert_awaited_once()


def test_history_outage_is_explicit_in_both_read_routes(tmp_path, monkeypatch):
    import sqlite3
    from unittest.mock import patch

    import pytest

    from local_brain.command_center import server
    from local_brain.command_center.run_ledger import RunLedger

    with patch.object(RunLedger, "__init__", side_effect=sqlite3.OperationalError("unavailable")):
        monkeypatch.setattr(server, "agent_registry", AgentRegistry(tmp_path / "agents.json"))
        for call in (server.studio_activity, lambda: server.studio_run("missing")):
            with pytest.raises(server.HTTPException) as exc:
                asyncio.run(call())
            assert exc.value.status_code == 503
            assert "saved outputs are retained" in exc.value.detail


@pytest.mark.parametrize("assignment_run", [False, True])
@pytest.mark.parametrize("network_failure", [False, True])
def test_unsaved_output_reports_storage_failure(
    tmp_path, monkeypatch, assignment_run, network_failure
):
    from local_brain.command_center import agent_registry, server
    from local_brain.command_center.assignment_registry import AssignmentRegistry
    from local_brain.command_center.persistence import PersistenceError

    registry = AgentRegistry(tmp_path / "agents.json")
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    agent = registry.create("Review", "Inspect", "Precise", [])
    work = assignments.create_assignment(title="Review", objective="Inspect", agent_id=agent["id"])
    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    monkeypatch.setattr(server, "agent_run_lock", asyncio.Lock())
    original = agent_registry.atomic_write_json

    def fail_outbox(path, payload):
        if payload["pending_runs"]:
            raise PersistenceError("synthetic storage failure")
        original(path, payload)

    monkeypatch.setattr(agent_registry, "atomic_write_json", fail_outbox)
    client = AsyncMock()
    client.__aenter__.return_value = client
    if network_failure:
        client.post.side_effect = httpx.ConnectError("synthetic offline")
    else:
        client.post.return_value = httpx.Response(200, json={"content": "Unsaved artifact"})
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: client)
    request = (
        server.run_assignment(work["id"])
        if assignment_run
        else server.run_agent(agent["id"], server.AgentRunRequest(task="Inspect"))
    )
    with pytest.raises(server.HTTPException) as exc:
        asyncio.run(request)
    assert exc.value.status_code == 503
    assert exc.value.detail == "run_output_persistence_failed"
    client.post.assert_awaited_once()
