from copy import deepcopy
from unittest.mock import Mock

import pytest

from local_brain.command_center.assignment_registry import (
    MAX_CONTEXT_LENGTH,
    AssignmentRegistry,
)
from local_brain.command_center.persistence import PersistenceError


def source(registry):
    work = registry.create_assignment(
        title="Check delivery",
        objective="Inspect evidence",
        agent_id="test",
        context="Original scope",
        success_criteria="Cite the SHA",
    )
    registry.finish_assignment(work["id"], result="Original artifact")
    registry.review_assignment(
        work["id"],
        decision="rejected",
        expected_version=work["review_version"],
        note="Missing SHA",
    )
    return work


def test_revision_is_linked_durable_idempotent_and_does_not_execute(tmp_path):
    registry = AssignmentRegistry(tmp_path / "work.json")
    original = source(registry)
    before = deepcopy(original)
    child = registry.create_revision(
        original["id"], expected_version=original["review_version"], note="Cite SHA"
    )
    assert child["status"] == "queued"
    assert child["agent_id"] == original["agent_id"]
    assert child["revision_of"] == original["id"]
    assert "Original artifact" in child["context"]
    assert "Cite SHA" in child["context"]
    assert {k: original[k] for k in before} == before
    assert (
        registry.create_revision(
            original["id"], expected_version=original["review_version"], note="Cite SHA"
        )["id"]
        == child["id"]
    )
    restored = AssignmentRegistry(registry.data_file)
    assert restored.get_assignment(child["id"])["revision_of"] == original["id"]
    for work in (original, child):
        with pytest.raises(ValueError, match="assignment_has_revisions"):
            registry.delete_assignment(work["id"])


def test_revision_refuses_stale_blank_and_oversize_context(tmp_path):
    registry = AssignmentRegistry(tmp_path / "work.json")
    original = source(registry)
    for version, note, error in [
        ("stale", "Feedback", "review_changed"),
        (original["review_version"], "   ", "feedback_required"),
    ]:
        with pytest.raises(ValueError, match=error):
            registry.create_revision(original["id"], expected_version=version, note=note)
    original["result"] = "x" * (MAX_CONTEXT_LENGTH + 1)
    with pytest.raises(ValueError, match="context_too_long"):
        registry.create_revision(
            original["id"], expected_version=original["review_version"], note="Fix"
        )
    assert len(registry.assignments) == 1


def test_revision_requires_a_rejected_output(tmp_path):
    registry = AssignmentRegistry(tmp_path / "work.json")
    original = registry.create_assignment(
        title="Check delivery", objective="Inspect evidence", agent_id="test"
    )
    registry.finish_assignment(original["id"], result="Artifact")
    with pytest.raises(ValueError, match="assignment_not_rejected"):
        registry.create_revision(
            original["id"], expected_version=original["review_version"], note="Fix"
        )


def test_revision_rolls_back_failed_storage(tmp_path, monkeypatch):
    from local_brain.command_center import assignment_registry

    registry = AssignmentRegistry(tmp_path / "work.json")
    original = source(registry)
    before = deepcopy(registry.assignments)
    monkeypatch.setattr(
        assignment_registry,
        "atomic_write_json",
        Mock(side_effect=PersistenceError("disk")),
    )
    with pytest.raises(PersistenceError):
        registry.create_revision(
            original["id"], expected_version=original["review_version"], note="Fix"
        )
    assert registry.assignments == before


def test_revision_endpoint_does_not_execute_and_rejects_missing_agent(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from local_brain.command_center import server

    registry = AssignmentRegistry(tmp_path / "work.json")
    original = source(registry)
    monkeypatch.setattr(server, "assignment_registry", registry)
    monkeypatch.setattr(server.agent_registry, "get", lambda _id: {"id": "test"})
    execute = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(server, "_execute_agent", execute)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            url = f"/api/assignments/{original['id']}/revision"
            body = {"expected_version": original["review_version"], "note": "Cite SHA"}
            first = await client.post(url, json=body)
            assert first.status_code == 200
            second = await client.post(url, json=body)
            assert first.json() == second.json()
            assert (
                await client.post(url, json={**body, "expected_version": "stale"})
            ).status_code == 409
            assert (await client.delete(f"/api/assignments/{original['id']}")).status_code == 409
            monkeypatch.setattr(server.agent_registry, "get", lambda _id: None)
            assert (await client.post(url, json=body)).status_code == 409

    asyncio.run(exercise())
    execute.assert_not_called()
