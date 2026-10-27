import copy

import pytest

from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.attention import (
    acknowledged_keys,
    evidence_key,
    fixture_preview,
)
from local_brain.command_center.persistence import PersistenceError


def completed(registry, **changes):
    item = registry.create_assignment(title="Inspect", objective="Inspect", agent_id="agent")
    registry.finish_assignment(item["id"], result="Evidence: check 123 passed")
    item.update(changes)
    return item


def test_acknowledge_exact_repeats_durable_and_reversible(tmp_path):
    registry = AssignmentRegistry(tmp_path / "work.json")
    first = completed(registry)
    second = completed(registry)
    key = evidence_key(first)
    registry.acknowledge_attention(
        first["id"], expected_version=first["review_version"], key=key, active=True
    )
    assert key in acknowledged_keys(registry.assignments)
    assert evidence_key(second) == key
    assert all(item["review_status"] == "unreviewed" for item in registry.assignments)
    loaded = AssignmentRegistry(registry.data_file)
    assert key in acknowledged_keys(loaded.assignments)
    assert evidence_key(completed(loaded)) == key
    loaded.acknowledge_attention(
        first["id"],
        expected_version=loaded.get_assignment(first["id"])["review_version"],
        key=key,
        active=False,
    )
    assert not acknowledged_keys(loaded.assignments)
    assert loaded.get_assignment(first["id"])["attention_events"][0]["restored_at"]


@pytest.mark.parametrize(
    "change",
    [
        {"result": "new evidence"},
        {"agent_id": "different"},
        {"context": "new scope"},
        {"priority": "urgent"},
        {"source_handoff_id": "handoff"},
        {"status": "failed"},
        {"review_status": "rejected"},
        {"attempt_id": "unfinished", "completed_run_id": ""},
    ],
)
def test_changes_never_inherit_acknowledgement(tmp_path, change):
    registry = AssignmentRegistry(tmp_path / "work.json")
    item = completed(registry)
    assert evidence_key(dict(item, **change)) != evidence_key(item)


def test_stale_and_failed_save_are_atomic(tmp_path, monkeypatch):
    registry = AssignmentRegistry(tmp_path / "work.json")
    item = completed(registry)
    before = copy.deepcopy(registry.assignments)
    with pytest.raises(ValueError):
        registry.acknowledge_attention(
            item["id"], expected_version="stale", key=evidence_key(item), active=True
        )
    monkeypatch.setattr(registry, "save", lambda: (_ for _ in ()).throw(PersistenceError("disk")))
    with pytest.raises(PersistenceError):
        registry.acknowledge_attention(
            item["id"], expected_version=item["review_version"], key=evidence_key(item), active=True
        )
    assert registry.assignments == before


def test_preview_never_classifies_and_protects_approvals(tmp_path):
    registry = AssignmentRegistry(tmp_path / "work.json")
    item = completed(registry, title="Inspect — Signal Officer", result="GREEN")
    before = copy.deepcopy(item)
    assert len(fixture_preview([item], set(), set())["candidates"]) == 1
    assert not fixture_preview([item], {"agent"}, set())["candidates"]
    assert fixture_preview([item], set(), {item["id"]})["blocked"][0]["id"] == item["id"]
    assert not fixture_preview([dict(item, context="real scope")], set(), set())["candidates"]
    assert item == before


@pytest.mark.asyncio
async def test_pending_approval_cannot_be_acknowledged(monkeypatch):
    from fastapi import HTTPException

    from local_brain.command_center import server

    item = completed(server.assignment_registry)
    monkeypatch.setattr(server, "_pending_attention_approvals", lambda: {item["id"]})
    with pytest.raises(HTTPException) as error:
        await server.acknowledge_attention(
            item["id"],
            server.AttentionRequest(
                expected_version=item["review_version"], key=evidence_key(item), active=True
            ),
        )
    assert error.value.status_code == 409
    assert not acknowledged_keys(server.assignment_registry.assignments)


def test_default_studio_registries_are_temporary(tmp_path):
    from local_brain.command_center import server

    for registry in (
        server.agent_registry,
        server.assignment_registry,
        server.git_workspace_registry,
        server.git_write_registry,
    ):
        assert registry.data_file.is_relative_to(tmp_path)
    agent = server.agent_registry.create("Mock", "Inspect", "Terse", [])
    item = server._open_inbox_item(agent, "Inspect", "mock-run")
    server._close_inbox_item(item["id"], "GREEN")
    assert (
        AssignmentRegistry(server.assignment_registry.data_file).get_assignment(item["id"])[
            "result"
        ]
        == "GREEN"
    )
