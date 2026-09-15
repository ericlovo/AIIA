import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.run_ledger import RunLedger, token_counts


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        {"input_tokens": 3},
        {"input_tokens": True, "output_tokens": 2},
        {"input_tokens": -1, "output_tokens": 2},
        {"input_tokens": "3", "output_tokens": 2},
        {"input_tokens": 3.5, "output_tokens": 2},
        {"input_tokens": 2**63, "output_tokens": 2},
    ],
)
def test_missing_or_invalid_usage_is_unknown(value):
    assert token_counts(value) == (None, None)


def test_zero_is_measured():
    assert token_counts({"input_tokens": 0, "output_tokens": 0}) == (0, 0)


def test_migration_preserves_old_rows_and_is_repeatable(tmp_path):
    path = tmp_path / "runs.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE runs (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, agent_name TEXT NOT NULL,
            repo_id TEXT NOT NULL, at TEXT NOT NULL, status TEXT NOT NULL,
            trigger TEXT NOT NULL, assignment_id TEXT NOT NULL, model TEXT NOT NULL,
            latency_ms REAL NOT NULL, legacy INTEGER NOT NULL, payload TEXT NOT NULL
        )""")
        db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                "a",
                "Agent",
                "",
                "2026-09-15T00:00:00+00:00",
                "completed",
                "manual",
                "",
                "local",
                12,
                0,
                '{"result":"existing"}',
            ),
        )
    for _ in range(2):
        ledger = RunLedger(path)
        assert ledger.get("old")["result"] == "existing"
        assert ledger.get("old")["input_tokens"] is None
        assert ledger.get("old")["output_tokens"] is None


def test_attribution_filters_and_deduplication(tmp_path):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    now = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
    a, b = {"id": "a", "name": "A"}, {"id": "b", "name": "B"}
    run = {
        "id": "one",
        "at": "2026-09-15T00:00:00Z",
        "result": "done",
        "input_tokens": 100,
        "output_tokens": 20,
        "assignment_id": "work",
        "trigger": "assignment",
    }
    ledger.record(a, run)
    ledger.record(a, run)
    ledger.record(a, {"id": "old", "at": "2026-09-14T23:00:00Z", "result": "done"}, legacy=True)
    ledger.record(
        b,
        {**run, "id": "two", "error": "empty", "result": "", "input_tokens": 0, "output_tokens": 0},
    )
    rows = ledger.activity(now=now)["usage_by_agent"]
    assert rows == [
        {
            "agent_id": "a",
            "agent_name": "A",
            "runs": 2,
            "measured_runs": 1,
            "input_tokens": 100,
            "output_tokens": 20,
        },
        {
            "agent_id": "b",
            "agent_name": "B",
            "runs": 1,
            "measured_runs": 1,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    ]
    today = ledger.activity(now=now, agent_id="a", day="2026-09-15")["usage_by_agent"]
    assert today[0]["runs"] == 1
    assert ledger.activity(now=now, status="failed")["usage_by_agent"] == rows[1:]
    assert ledger.activity(now=now, day="2026-09-14")["usage_by_agent"][0]["input_tokens"] is None
    assert ledger.get("one")["output_tokens"] == 20


def test_usage_survives_outbox_recovery_and_deletion(tmp_path):
    path = tmp_path / "agents.json"
    registry = AgentRegistry(path)
    agent = registry.create("Usage agent", "Inspect", "Precise", [])
    with patch.object(RunLedger, "record", side_effect=sqlite3.OperationalError("locked")):
        registry.finish_run(
            agent["id"], "task", result="done", usage={"input_tokens": 120, "output_tokens": 40}
        )
        run_id = agent["runs"][0]["id"]
        registry.delete(agent["id"])
    recovered = AgentRegistry(path)
    assert recovered.ledger.get(run_id)["input_tokens"] == 120
    assert recovered.ledger.get(run_id)["output_tokens"] == 40
    assert AgentRegistry(path).ledger.activity()["usage_by_agent"][0]["measured_runs"] == 1


def test_aggregates_all_matches_not_just_visible_ledger(tmp_path):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    for i in range(205):
        ledger.record(
            {"id": "a", "name": "A"},
            {
                "id": str(i),
                "at": "2026-09-15T00:00:00Z",
                "result": "done",
                "input_tokens": 1,
                "output_tokens": 2,
            },
        )
    data = ledger.activity(now=datetime(2026, 9, 15, tzinfo=timezone.utc))
    assert len(data["runs"]) == 200
    assert data["usage_by_agent"][0]["input_tokens"] == 205
    assert data["usage_by_agent"][0]["output_tokens"] == 410
