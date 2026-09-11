from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from local_brain.command_center import persistence
from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.persistence import PersistenceError


@pytest.fixture
def registry(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    registry.create("First", "Original", "Precise", [], suite="mindmoor")
    registry.create("Second", "Original", "Precise", [])
    return registry


@pytest.mark.parametrize(
    "operation", ["create", "update", "delete", "loop", "start", "counter", "loop_start"]
)
def test_failed_mutation_restores_memory_disk_and_references(registry, monkeypatch, operation):
    agent = registry.agents[0]
    references = list(registry.agents)
    before = deepcopy(registry.agents)
    disk = registry.data_file.read_bytes()

    def fail(*args):
        raise OSError("private-storage-location")

    monkeypatch.setattr(persistence.os, "replace", fail)
    actions = {
        "create": lambda: registry.create("Third", "New", "Precise", []),
        "update": lambda: registry.update(agent["id"], name="Changed", temperature=0.8),
        "delete": lambda: registry.delete(agent["id"]),
        "loop": lambda: registry.update(agent["id"], loop_enabled=True),
        "start": lambda: registry.set_running(agent["id"]),
        "loop_start": lambda: registry.set_running(agent["id"], loop_run=True),
        "counter": lambda: registry.record_loop_run(agent["id"]),
    }
    with pytest.raises(PersistenceError):
        actions[operation]()
    assert registry.agents == before
    assert all(a is b for a, b in zip(registry.agents, references))
    assert registry.data_file.read_bytes() == disk
    assert not list(registry.data_file.parent.glob(".*.tmp"))


def test_validation_error_does_not_partially_edit_agent(registry):
    agent = registry.agents[0]
    before = deepcopy(agent)
    with pytest.raises(ValueError):
        registry.update(agent["id"], name="Changed", suite="invalid suite")
    assert agent == before


def test_restart_pauses_interrupted_loop_without_fabricating_output(registry):
    agent = registry.agents[0]
    registry.update(agent["id"], loop_enabled=True, loop_task="Inspect")
    registry.finish_run(agent["id"], "Previous task", result="Retained evidence")
    previous_runs = deepcopy(agent["runs"])
    registry.record_loop_run(agent["id"])
    registry.set_running(agent["id"])
    recovered = AgentRegistry(registry.data_file)
    restored = recovered.get(agent["id"])
    assert restored["status"] == "error"
    assert "interrupted_agent_run" in restored["last_error"]
    assert restored["loop_enabled"] is False
    assert restored["loop_runs_today"] == 1
    assert restored["runs"] == previous_runs
    assert restored["last_result"] == "Retained evidence"
    assert restored["suite"] == restored["memory_namespace"] == "mindmoor"
    assert recovered.due_loop() is None
    assert recovered.ledger.activity()["total"] == 1
    assert AgentRegistry(registry.data_file).get(agent["id"]) == restored
    assert recovered.get(registry.agents[1]["id"])["status"] == "idle"


def test_explicit_resume_allows_never_completed_interrupted_loop(registry):
    agent = registry.agents[0]
    registry.update(agent["id"], loop_enabled=True, loop_task="Inspect")
    registry.set_running(agent["id"])
    recovered = AgentRegistry(registry.data_file)
    assert recovered.due_loop() is None
    recovered.update(agent["id"], loop_enabled=True)
    assert recovered.due_loop()["id"] == agent["id"]


def test_recovery_write_failure_prevents_registry_startup(registry, monkeypatch):
    registry.set_running(registry.agents[0]["id"])
    before = registry.data_file.read_bytes()

    def fail(*args):
        raise OSError("synthetic outage")

    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(PersistenceError):
        AgentRegistry(registry.data_file)
    assert registry.data_file.read_bytes() == before


@pytest.mark.parametrize("operation", ["create", "update", "delete", "loop", "run"])
def test_api_storage_error_is_visible_without_success_event(registry, monkeypatch, operation):
    from local_brain.command_center import server

    monkeypatch.setattr(server, "agent_registry", registry)
    broadcast = AsyncMock()
    model_client = AsyncMock()
    monkeypatch.setattr(server, "broadcast_studio_event", broadcast)
    monkeypatch.setattr(server.httpx, "AsyncClient", model_client)
    agent_id = registry.agents[0]["id"]
    body = {"name": "Changed", "mission": "New", "persona": "Precise", "skills": [], "tools": []}
    calls = {
        "create": ("POST", "/api/agents", body),
        "update": ("PUT", f"/api/agents/{agent_id}", body),
        "delete": ("DELETE", f"/api/agents/{agent_id}", None),
        "loop": ("POST", f"/api/agents/{agent_id}/loop", {"enabled": True}),
        "run": ("POST", f"/api/agents/{agent_id}/run", {"task": "Synthetic"}),
    }
    registry.update(agent_id, loop_task="Synthetic")
    before = deepcopy(registry.agents)

    def fail(*args):
        raise OSError("private-storage-location")

    monkeypatch.setattr(persistence.os, "replace", fail)
    method, path, payload = calls[operation]
    response = TestClient(server.app).request(method, path, json=payload)
    assert response.status_code == 503
    assert "not saved" in response.json()["detail"]
    assert "private-storage-location" not in response.text
    assert registry.agents == before
    broadcast.assert_not_called()
    model_client.assert_not_called()
