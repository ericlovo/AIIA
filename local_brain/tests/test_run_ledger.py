from datetime import datetime, timezone

from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.run_ledger import RunLedger


def test_ledger_survives_cache_eviction_and_agent_deletion(tmp_path):
    path = tmp_path / "agents.json"
    registry = AgentRegistry(path)
    agent = registry.create("Builder", "Inspect", "Precise", [])
    for n in range(20):
        registry.finish_run(agent["id"], f"task {n}", result="Evidence")
    assert len(agent["runs"]) == 12
    reloaded = AgentRegistry(path)
    assert reloaded.ledger.activity()["total"] == 20
    reloaded.delete(agent["id"])
    assert AgentRegistry(path).ledger.activity()["total"] == 20


def test_backfill_idempotent_empty_output_and_filters(tmp_path):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    agent = {
        "id": "one",
        "name": "One",
        "runs": [
            {"at": "2026-09-09T23:00:00Z", "result": " ", "task": "check"},
            {"at": "2026-09-10T01:00:00Z", "result": "done", "task": "check"},
        ],
    }
    ledger.backfill([agent])
    ledger.backfill([agent])
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    data = ledger.activity(day="2026-09-09", status="failed", now=now)
    assert data["total"] == 2
    assert data["matching"] == 1
    assert len(data["days"]) == 2
    assert "result" not in data["runs"][0]
    assert ledger.get(data["runs"][0]["id"])["temperature"] is None
    assert ledger.activity(agent_id="absent", now=now)["matching"] == 0
    assert ledger.get("absent") is None


def test_same_timestamp_new_runs_are_not_deduplicated(tmp_path):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    agent = {"id": "one", "name": "One"}
    run = {"at": "2026-09-10T00:00:00Z", "result": "done"}
    ledger.record(agent, {**run, "id": "attempt-1"})
    ledger.record(agent, {**run, "id": "attempt-2"})
    assert ledger.activity()["total"] == 2


def test_bad_legacy_record_does_not_block_startup(tmp_path, caplog):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    ledger.backfill(
        [
            {
                "id": "one",
                "name": "One",
                "runs": [
                    {"at": "bad-date", "result": "secret"},
                    {"at": "2026-09-10T01:00:00Z", "result": None},
                ],
            }
        ]
    )
    assert ledger.activity()["total"] == 1
    assert "Skipped invalid legacy run" in caplog.text
    assert "secret" not in caplog.text


def test_pending_history_survives_outage_cache_eviction_and_agent_deletion(tmp_path, monkeypatch):
    import sqlite3
    from unittest.mock import patch

    path = tmp_path / "agents.json"
    registry = AgentRegistry(path)
    agent = registry.create("Original", "Inspect", "Precise", [], temperature=0.2, max_tokens=900)
    with patch.object(RunLedger, "record", side_effect=sqlite3.OperationalError("locked")):
        for n in range(20):
            registry.finish_run(agent["id"], f"task {n}", result=f"artifact {n}")
        assert len(agent["runs"]) == 12
        assert registry.pending_run_count == 20
        ids = {entry["run"]["id"] for entry in registry._pending_runs}
        registry.update(agent["id"], name="Changed", temperature=0.9)
        registry.delete(agent["id"])
        unavailable = AgentRegistry(path)
        assert unavailable.list() == []
        assert unavailable.pending_run_count == 20

    reloaded = AgentRegistry(path)
    assert reloaded.pending_run_count == 0
    assert reloaded.ledger.activity()["total"] == 20
    for run_id in ids:
        run = reloaded.ledger.get(run_id)
        assert run["agent_name"] == "Original"
        assert run["temperature"] == 0.2
        assert run["max_tokens"] == 900
        assert not run["legacy"]
    assert AgentRegistry(path).ledger.activity()["total"] == 20


def test_pending_output_loads_even_when_sqlite_cannot_open(tmp_path):
    import sqlite3
    from unittest.mock import patch

    path = tmp_path / "agents.json"
    with patch.object(RunLedger, "__init__", side_effect=sqlite3.OperationalError("unavailable")):
        registry = AgentRegistry(path)
        agent = registry.create("Review", "Inspect", "Precise", [])
        registry.set_running(agent["id"])
        registry.finish_run(agent["id"], "Inspect", result="Saved artifact")
        unavailable = AgentRegistry(path)
        assert unavailable.ledger is None
        assert unavailable.pending_run_count == 1
        assert unavailable.get(agent["id"])["status"] == "idle"
        assert unavailable.get(agent["id"])["last_result"] == "Saved artifact"
    recovered = AgentRegistry(path)
    assert recovered.pending_run_count == 0
    assert recovered.ledger.activity()["total"] == 1


def test_replay_after_cleanup_write_failure_does_not_duplicate_run(tmp_path, monkeypatch):
    from local_brain.command_center import agent_registry
    from local_brain.command_center.persistence import PersistenceError

    path = tmp_path / "agents.json"
    registry = AgentRegistry(path)
    agent = registry.create("Review", "Inspect", "Precise", [])
    original = agent_registry.atomic_write_json

    def fail_cleanup(path, payload):
        if not payload["pending_runs"]:
            raise PersistenceError("synthetic cleanup failure")
        original(path, payload)

    monkeypatch.setattr(agent_registry, "atomic_write_json", fail_cleanup)
    registry.finish_run(agent["id"], "Inspect", result="Only artifact")
    assert registry.pending_run_count == 1
    assert registry.ledger.activity()["total"] == 1
    run_id = agent["runs"][0]["id"]
    monkeypatch.setattr(agent_registry, "atomic_write_json", original)
    recovered = AgentRegistry(path)
    assert recovered.pending_run_count == 0
    assert recovered.ledger.activity()["total"] == 1
    assert recovered.ledger.get(run_id)["result"] == "Only artifact"


def test_atomic_registry_failure_preserves_previous_file_and_reports_failure(tmp_path, monkeypatch):
    import pytest

    from local_brain.command_center import persistence
    from local_brain.command_center.persistence import PersistenceError

    path = tmp_path / "agents.json"
    registry = AgentRegistry(path)
    agent = registry.create("Review", "Inspect", "Precise", [])
    registry.set_running(agent["id"])
    original_bytes = path.read_bytes()

    def fail_replace(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    with pytest.raises(PersistenceError):
        registry.finish_run(agent["id"], "Inspect", result="Unsaved artifact")
    assert path.read_bytes() == original_bytes
    assert registry.pending_run_count == 1
    assert not list(tmp_path.glob(".*.tmp"))
