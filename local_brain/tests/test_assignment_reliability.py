"""Release gates for durable, linked assignment work and complete artifacts."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from local_brain.command_center import assignment_registry as module
from local_brain.command_center import persistence
from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.persistence import PersistenceError


def create(registry, title="Synthetic gate"):
    return registry.create_assignment(title=title, objective="Review evidence", agent_id="source")


def handoff(registry, source):
    return registry.create_handoff(
        source_assignment_id=source["id"],
        to_agent_id="target",
        artifact_type="analysis",
        instructions="Preserve uncertainty",
    )


@pytest.fixture
def linked(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    source = create(registry)
    registry.finish_assignment(source["id"], result="Unverified evidence")
    link, target = handoff(registry, source)
    spare = create(registry, "Unlinked")
    return registry, source, link, target, spare


@pytest.mark.parametrize(
    "operation", ["create", "start", "finish", "handoff", "delete", "delete_handoff", "review"]
)
def test_failed_mutation_restores_both_lists_and_references(linked, monkeypatch, operation):
    registry, source, link, target, spare = linked
    references = (list(registry.assignments), list(registry.handoffs))
    before = deepcopy(references)
    disk = registry.data_file.read_bytes()
    operations = {
        "create": lambda: create(registry),
        "start": lambda: registry.set_running(target["id"]),
        "finish": lambda: registry.finish_assignment(target["id"], result="Work product"),
        "handoff": lambda: handoff(registry, source),
        "delete": lambda: registry.delete_assignment(spare["id"]),
        "delete_handoff": lambda: registry.delete_handoff(link["id"]),
        "review": lambda: registry.review_assignment(
            source["id"], decision="accepted", expected_version=source["review_version"]
        ),
    }
    monkeypatch.setattr(persistence.os, "replace", Mock(side_effect=OSError("injected")))
    with pytest.raises(PersistenceError):
        operations[operation]()
    assert (registry.assignments, registry.handoffs) == before
    for records, originals in zip((registry.assignments, registry.handoffs), references):
        assert all(record is original for record, original in zip(records, originals))
    assert registry.data_file.read_bytes() == disk
    assert not list(registry.data_file.parent.glob(".*.tmp"))


def test_nested_handoff_is_one_atomic_write(linked, monkeypatch):
    registry, source, *_ = linked
    writer = Mock(wraps=module.atomic_write_json)
    monkeypatch.setattr(module, "atomic_write_json", writer)
    link, target = handoff(registry, source)
    writer.assert_called_once()
    restored = AssignmentRegistry(registry.data_file)
    assert restored.get_handoff(link["id"]) == link
    assert restored.get_assignment(target["id"]) == target


def test_recovery_failure_blocks_initialization_and_preserves_disk(linked, monkeypatch):
    registry, source, link, target, _ = linked
    registry.set_running(target["id"])
    disk = registry.data_file.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write_json", Mock(side_effect=PersistenceError("injected")))
        with pytest.raises(PersistenceError):
            AssignmentRegistry(registry.data_file)
    assert registry.data_file.read_bytes() == disk
    restored = AssignmentRegistry(registry.data_file)
    assert restored.get_assignment(target["id"])["error"] == "interrupted_by_restart"
    assert restored.get_handoff(link["id"])["status"] == "failed"
    assert restored.get_assignment(source["id"]) == source
    stable = registry.data_file.read_bytes()
    AssignmentRegistry(registry.data_file)
    assert registry.data_file.read_bytes() == stable


def test_capacity_preserves_links_and_active_work(linked, monkeypatch):
    registry, source, link, target, spare = linked
    monkeypatch.setattr(module, "MAX_ASSIGNMENTS", 3)
    before, disk = deepcopy(registry.assignments), registry.data_file.read_bytes()
    with pytest.raises(ValueError, match="assignment_capacity_reached"):
        create(registry)
    assert registry.assignments == before
    assert registry.data_file.read_bytes() == disk
    registry.finish_assignment(spare["id"], error="Synthetic failure")
    new = create(registry)
    restored = AssignmentRegistry(registry.data_file)
    assert restored.get_assignment(spare["id"]) is None
    for item in (source, target, new):
        assert restored.get_assignment(item["id"]) == item
    assert restored.get_handoff(link["id"]) == link


def test_failed_eviction_restores_exact_order(linked, monkeypatch):
    registry, _, _, _, spare = linked
    registry.finish_assignment(spare["id"], error="Synthetic")
    monkeypatch.setattr(module, "MAX_ASSIGNMENTS", 3)
    before, disk = deepcopy(registry.assignments), registry.data_file.read_bytes()
    monkeypatch.setattr(module, "atomic_write_json", Mock(side_effect=PersistenceError("injected")))
    with pytest.raises(PersistenceError):
        create(registry)
    assert registry.assignments == before
    assert registry.data_file.read_bytes() == disk


@pytest.mark.parametrize("capacity", ["MAX_ASSIGNMENTS", "MAX_HANDOFFS"])
def test_handoff_capacity_failure_leaves_no_orphans(linked, monkeypatch, capacity):
    registry, source, *_ = linked
    monkeypatch.setattr(module, capacity, 1)
    before = deepcopy((registry.assignments, registry.handoffs))
    disk = registry.data_file.read_bytes()
    with pytest.raises(ValueError, match="capacity_reached"):
        handoff(registry, source)
    assert (registry.assignments, registry.handoffs) == before
    assert registry.data_file.read_bytes() == disk


def test_load_never_truncates_existing_records(linked, monkeypatch):
    registry, *_ = linked
    monkeypatch.setattr(module, "MAX_ASSIGNMENTS", 1)
    monkeypatch.setattr(module, "MAX_HANDOFFS", 0)
    restored = AssignmentRegistry(registry.data_file)
    assert restored.assignments == registry.assignments
    assert restored.handoffs == registry.handoffs


@pytest.mark.parametrize("length", [25_013, module.MAX_RESULT_LENGTH])
def test_full_supported_artifact_survives_handoff_and_reload(tmp_path, length):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    source = create(registry, "T" * 120)
    artifact = " " + "x" * (length - 2) + "\n"
    registry.finish_assignment(source["id"], result=artifact)
    link, target = handoff(registry, source)
    restored = AssignmentRegistry(registry.data_file)
    assert restored.get_assignment(source["id"])["result"] == artifact
    assert restored.get_handoff(link["id"])["artifact"] == artifact
    assert restored.get_assignment(target["id"])["context"].endswith(artifact)


def test_oversize_result_rejected_without_mutation(linked):
    registry, source, *_ = linked
    before, disk = deepcopy(registry.assignments), registry.data_file.read_bytes()
    with pytest.raises(ValueError, match="assignment_result_too_long"):
        registry.finish_assignment(source["id"], result="x" * (module.MAX_RESULT_LENGTH + 1))
    assert registry.assignments == before
    assert registry.data_file.read_bytes() == disk


@pytest.mark.parametrize("operation", ["create", "run", "delete", "handoff", "delete_handoff"])
def test_api_storage_failure_has_no_execution_or_success_event(linked, monkeypatch, operation):
    from local_brain.command_center import server

    registry, source, link, target, spare = linked
    monkeypatch.setattr(server, "assignment_registry", registry)
    monkeypatch.setattr(server.agent_registry, "get", Mock(return_value={"id": "source"}))
    monkeypatch.setattr(server, "agent_run_lock", asyncio.Lock())
    executor, event = AsyncMock(), AsyncMock()
    monkeypatch.setattr(server, "_execute_agent", executor)
    monkeypatch.setattr(server, "broadcast_assignment_event", event)
    monkeypatch.setattr(
        module, "atomic_write_json", Mock(side_effect=PersistenceError("private-path"))
    )
    requests = {
        "create": (
            "POST",
            "/api/assignments",
            {"title": "Synthetic", "objective": "Review", "agent_id": "source"},
        ),
        "run": ("POST", f"/api/assignments/{target['id']}/run", None),
        "delete": ("DELETE", f"/api/assignments/{spare['id']}", None),
        "handoff": (
            "POST",
            "/api/handoffs",
            {
                "source_assignment_id": source["id"],
                "to_agent_id": "target",
                "artifact_type": "analysis",
                "instructions": "Review",
            },
        ),
        "delete_handoff": ("DELETE", f"/api/handoffs/{link['id']}", None),
    }

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            method, path, body = requests[operation]
            response = await client.request(method, path, json=body)
            assert response.status_code == 503
            assert "private-path" not in response.text
            executor.assert_not_called()
            event.assert_not_called()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "outcome", ["success", "empty", "oversize", "http", "storage", "exception"]
)
def test_api_completion_write_failure_never_claims_completion(linked, monkeypatch, outcome):
    from fastapi import HTTPException

    from local_brain.command_center import server

    registry, _, link, target, _ = linked
    monkeypatch.setattr(server, "assignment_registry", registry)
    monkeypatch.setattr(server.agent_registry, "get", Mock(return_value={"id": "target"}))
    monkeypatch.setattr(server, "agent_run_lock", asyncio.Lock())
    events = []

    async def capture(event, item):
        events.append((event, deepcopy(item)))

    async def execute(*args, **kwargs):
        monkeypatch.setattr(
            module, "atomic_write_json", Mock(side_effect=PersistenceError("private-path"))
        )
        errors = {
            "http": HTTPException(status_code=502, detail="synthetic_model_failure"),
            "storage": PersistenceError("synthetic_output_failure"),
            "exception": RuntimeError("synthetic_transport_failure"),
        }
        if outcome in errors:
            raise errors[outcome]
        content = {"success": "Evidence", "empty": "", "oversize": "x" * 40_001}[outcome]
        return {"agent": {"last_result": content}, "model": "synthetic", "latency_ms": 1}

    executor = AsyncMock(side_effect=execute)
    monkeypatch.setattr(server, "_execute_agent", executor)
    monkeypatch.setattr(server, "broadcast_assignment_event", capture)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            response = await client.post(f"/api/assignments/{target['id']}/run")
            assert response.status_code == 503
            assert "private-path" not in response.text
            executor.assert_awaited_once()

    asyncio.run(exercise())
    assert [event for event, _ in events] == ["running"]
    saved = json.loads(registry.data_file.read_text())
    assert saved["assignments"] == registry.assignments
    assert saved["handoffs"] == registry.handoffs
    assert target["status"] == link["status"] == "running"
