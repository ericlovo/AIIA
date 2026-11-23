import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from local_brain.command_center.assignment_registry import AssignmentRegistry


def test_manual_run_queue_survives_restart_and_drains_in_click_order(tmp_path, monkeypatch):
    from local_brain.command_center import server

    store = AssignmentRegistry(tmp_path / "assignments.json")
    first, second, untouched = [
        store.create_assignment(title=name, agent_id="agent", objective="Check fixture")
        for name in ("retry", "next", "manual only")
    ]
    store.finish_assignment(first["id"], error="previous_failure")
    lock = asyncio.Lock()
    monkeypatch.setattr(server, "agent_run_lock", lock)
    monkeypatch.setattr(server, "_assignment_queue_task", None)
    monkeypatch.setattr(server, "assignment_registry", store)
    monkeypatch.setattr(server.agent_registry, "get", lambda _: {"id": "agent"})
    monkeypatch.setattr(server, "broadcast_assignment_event", AsyncMock())
    runs = []

    async def execute(*args, assignment_id, **kwargs):
        async with lock:
            runs.append(assignment_id)
            if assignment_id == first["id"]:
                raise HTTPException(502, "fixture_failed")
            return {"agent": {"last_result": "done"}, "model": "test", "latency_ms": 1}

    monkeypatch.setattr(server, "_execute_agent", execute)

    async def scenario():
        async with lock:
            for row in (first, second, first):
                receipt = await server.enqueue_assignment(
                    row["id"],
                    server.AssignmentEnqueueRequest(expected_version=row["review_version"]),
                )
                assert receipt["assignment"]["run_ticket"]
            await asyncio.sleep(0)
            assert not runs
            server._assignment_queue_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server._assignment_queue_task
        recovered = AssignmentRegistry(store.data_file)
        monkeypatch.setattr(server, "assignment_registry", recovered)
        assert [row["id"] for row in recovered.run_queue()] == [first["id"], second["id"]]
        server._wake_assignment_queue()
        await asyncio.wait_for(server._assignment_queue_task, timeout=3)
        assert runs == [first["id"], second["id"]]
        assert recovered.run_queue() == []
        assert recovered.get_assignment(first["id"])["error"] == "fixture_failed"
        assert recovered.get_assignment(second["id"])["result"] == "done"
        assert recovered.get_assignment(untouched["id"])["status"] == "queued"
        with pytest.raises(ValueError, match="review_changed_refresh_required"):
            recovered.enqueue_run(first["id"], first["review_version"])

    asyncio.run(scenario())
