"""Review is an output-quality decision, independent of successful execution."""

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.persistence import PersistenceError


def completed(registry):
    work = registry.create_assignment(
        title="Evidence check", objective="Classify the claim", agent_id="test"
    )
    registry.finish_assignment(
        work["id"], result="INCONCLUSIVE: the premise concerns passing checks."
    )
    return work


def test_review_lifecycle_reload_and_stale_decisions(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    work = completed(registry)
    original_version = work["review_version"]
    original_result = work["result"]
    for decision in ("accepted", "rejected", "unreviewed"):
        registry.review_assignment(
            work["id"],
            decision=decision,
            expected_version=work["review_version"],
            note="Evidence inspected",
        )
        restored = AssignmentRegistry(registry.data_file).get_assignment(work["id"])
        assert restored == work
        assert restored["status"] == "completed"
        assert restored["result"] == original_result
        assert restored["review_status"] == decision
        assert bool(restored["reviewed_at"]) == (decision != "unreviewed")
        with pytest.raises(ValueError, match="review_changed_refresh_required"):
            registry.review_assignment(
                work["id"], decision="accepted", expected_version=original_version
            )
    assert registry.handoffs == []


def test_new_execution_invalidates_review_even_for_identical_output(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    work = completed(registry)
    registry.review_assignment(
        work["id"], decision="accepted", expected_version=work["review_version"]
    )
    old_version, output = work["review_version"], work["result"]
    registry.set_running(work["id"])
    assert work["review_status"] == "unreviewed"
    registry.finish_assignment(work["id"], result=output)
    assert work["review_status"] == "unreviewed"
    assert work["review_note"] == ""
    with pytest.raises(ValueError, match="review_changed_refresh_required"):
        registry.review_assignment(work["id"], decision="accepted", expected_version=old_version)


@pytest.mark.parametrize(
    "state,result", [("queued", ""), ("running", ""), ("failed", "partial"), ("completed", " ")]
)
def test_only_completed_nonempty_output_is_reviewable(tmp_path, state, result):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    work = completed(registry)
    work.update(status=state, result=result)
    with pytest.raises(ValueError, match="assignment_not_reviewable"):
        registry.review_assignment(
            work["id"], decision="accepted", expected_version=work["review_version"]
        )


def test_legacy_output_defaults_to_unreviewed_with_stable_version(tmp_path):
    path = tmp_path / "assignments.json"
    registry = AssignmentRegistry(path)
    work = completed(registry)
    for key in ("review_status", "review_version", "review_note", "reviewed_at"):
        del work[key]
    path.write_text(json.dumps({"assignments": [work], "handoffs": []}))
    first, second = AssignmentRegistry(path), AssignmentRegistry(path)
    a, b = first.get_assignment(work["id"]), second.get_assignment(work["id"])
    assert a["review_status"] == "unreviewed"
    assert a["review_version"] == b["review_version"]
    first.review_assignment(a["id"], decision="rejected", expected_version=a["review_version"])
    assert AssignmentRegistry(path).get_assignment(a["id"])["review_status"] == "rejected"


def test_atomic_review_failure_rolls_back_memory_and_disk(tmp_path, monkeypatch):
    from local_brain.command_center import persistence

    registry = AssignmentRegistry(tmp_path / "assignments.json")
    work = completed(registry)
    before, disk = work.copy(), registry.data_file.read_bytes()
    monkeypatch.setattr(persistence.os, "replace", Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(PersistenceError):
        registry.review_assignment(
            work["id"], decision="accepted", expected_version=work["review_version"]
        )
    assert work == before
    assert registry.data_file.read_bytes() == disk
    assert not list(tmp_path.glob(".*.tmp"))


def test_review_api_validation_conflicts_and_storage_failure(tmp_path, monkeypatch):
    from local_brain.command_center import assignment_registry, server

    registry = AssignmentRegistry(tmp_path / "assignments.json")
    work = completed(registry)
    monkeypatch.setattr(server, "assignment_registry", registry)
    executor = Mock(side_effect=AssertionError("review must not execute an agent"))
    monkeypatch.setattr(server, "_execute_agent", executor)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            url = f"/api/assignments/{work['id']}/review"
            body = {
                "decision": "accepted",
                "expected_version": work["review_version"],
                "note": "Checked",
            }
            assert (await client.post(url, json={**body, "decision": "done"})).status_code == 422
            assert (await client.post(url, json={**body, "note": "x" * 2001})).status_code == 422
            assert (
                await client.post("/api/assignments/missing/review", json=body)
            ).status_code == 404
            response = await client.post(url, json=body)
            assert response.status_code == 200
            assert response.json()["assignment"]["review_status"] == "accepted"
            assert (await client.post(url, json=body)).status_code == 409
            body.update(expected_version=work["review_version"], decision="rejected")
            monkeypatch.setattr(
                assignment_registry, "atomic_write_json", Mock(side_effect=PersistenceError("disk"))
            )
            response = await client.post(url, json=body)
            assert response.status_code == 503
            assert response.json()["detail"] == "review_persistence_failed"
            saved = (await client.get("/api/assignments")).json()["assignments"][0]
            assert saved["review_status"] == "accepted"
            executor.assert_not_called()

    asyncio.run(exercise())
