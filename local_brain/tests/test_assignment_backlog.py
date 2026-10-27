"""Complete, compact backlog reads never change durable decisions."""

import asyncio
import json
from copy import deepcopy

import httpx

from local_brain.command_center.assignment_registry import AssignmentRegistry


def test_full_backlog_summary_and_selected_evidence(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AssignmentRegistry(tmp_path / "assignments.json")
    original = registry.create_assignment(title="Inspect", objective="Scope", agent_id="test")
    registry.finish_assignment(original["id"], result="Evidence" * 5000)
    for index in range(401):
        item = deepcopy(original)
        item["id"] = f"backlog-{index}"
        item["context"] = "Private full context " * 1000
        item["review_note"] = "Existing verdict"
        registry.assignments.append(item)
    registry.save()
    before = registry.data_file.read_bytes()
    monkeypatch.setattr(server, "assignment_registry", registry)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            response = await client.get("/api/assignments?summary=true")
            assert response.status_code == 200
            rows = response.json()["assignments"]
            assert len(rows) == 402
            assert rows[-1]["id"] == "backlog-400"
            assert rows[-1]["has_result"] is True
            assert rows[-1]["summary"] is True
            assert rows[-1]["result"] == rows[-1]["context"] == ""
            assert rows[-1]["review_version"] == original["review_version"]
            assert rows[-1]["group_scope"] == rows[-2]["group_scope"]
            assert rows[-1]["group_scope"] != rows[0]["group_scope"]
            assert len(response.content) < len(json.dumps(registry.assignments)) / 10
            detail = await client.get("/api/assignments/backlog-400")
            assert detail.json()["assignment"]["result"] == original["result"]
            assert detail.json()["assignment"]["review_note"] == "Existing verdict"
            assert (await client.get("/api/assignments/missing")).status_code == 404
            legacy = await client.get("/api/assignments")
            assert len(legacy.json()["assignments"]) == 402
            assert legacy.json()["assignments"][-1]["context"]

    asyncio.run(check())
    assert registry.data_file.read_bytes() == before
