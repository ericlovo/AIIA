import asyncio
from unittest.mock import Mock

import httpx
import pytest

from local_brain.command_center.agent_registry import AgentRegistry, RunHistoryUnavailable
from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.run_ledger import RunLedger


def test_history_scopes_paginates_and_preserves_old_attempts(tmp_path):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    agent = {"id": "agent", "name": "Synthetic"}
    for index in range(25):
        ledger.record(
            agent,
            {
                "id": f"run-{index:02}",
                "at": "2020-01-01T00:00:00Z",
                "assignment_id": "work",
                "trigger": "assignment",
                "result": "Evidence",
            },
        )
    ledger.record(
        agent,
        {
            "id": "unrelated",
            "at": "2026-09-12T00:00:00Z",
            "assignment_id": "other",
            "trigger": "assignment",
            "result": "Private",
        },
    )
    ledger.record(
        agent,
        {
            "id": "manual",
            "at": "2026-09-12T00:00:00Z",
            "assignment_id": "work",
            "trigger": "manual",
            "result": "Manual",
        },
    )
    first = ledger.assignment_history("work")
    second = ledger.assignment_history("work", offset=20)
    assert first["total"] == second["total"] == 25
    assert len(first["runs"]) == 20
    assert len(second["runs"]) == 5
    assert {r["id"] for r in first["runs"]}.isdisjoint({r["id"] for r in second["runs"]})
    assert first["runs"][0]["id"] == "run-24"
    assert "result" not in first["runs"][0]
    assert ledger.get("run-24")["result"] == "Evidence"


def test_api_distinguishes_missing_evidence_from_unavailable_history(tmp_path, monkeypatch):
    from local_brain.command_center import server

    agents = AgentRegistry(tmp_path / "agents.json")
    agent = agents.create(name="Synthetic", mission="Review", persona="Careful", skills=[])
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    work = assignments.create_assignment(title="History", objective="Inspect", agent_id=agent["id"])
    assignments.set_running(work["id"])
    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", assignments)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            path = f"/api/assignments/{work['id']}/history"
            response = await client.get(path)
            assert response.status_code == 200
            assert response.json()["total"] == 0
            assert response.json()["current_output_saved"] is False
            assert (await client.get(path + "?offset=-1")).status_code == 422
            assert (await client.get("/api/assignments/missing/history")).status_code == 404
            agents.finish_run(
                agent["id"],
                "Inspect",
                result="Output",
                trigger="assignment",
                assignment_id=work["id"],
                run_id=work["attempt_id"],
            )
            response = await client.get(path)
            assert response.json()["current_output_saved"] is True
            assert response.json()["total"] == 1
            assert work["status"] == "running"
            monkeypatch.setattr(
                agents, "recover_runs", Mock(side_effect=RunHistoryUnavailable("offline"))
            )
            response = await client.get(path)
            assert response.status_code == 503
            assert response.json()["detail"] == "assignment_history_unavailable"

    asyncio.run(exercise())
