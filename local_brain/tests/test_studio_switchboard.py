import asyncio
from unittest.mock import AsyncMock

import httpx

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
