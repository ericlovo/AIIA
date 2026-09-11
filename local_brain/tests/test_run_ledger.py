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
