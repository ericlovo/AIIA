"""The daily digest: records in, one body out, delivered once per day."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from local_brain.command_center import daily_digest
from local_brain.command_center.aiia_tasks import (
    TASK_DEFINITIONS,
    TaskRunner,
    cron_label,
    cron_target,
)
from local_brain.command_center.memory_inbox import MemoryInbox

DATE = "2026-10-05"


def agents():
    return [
        {
            "id": "a1",
            "name": "CI Signal Officer",
            "loop_enabled": True,
            "last_run_at": f"{DATE}T12:00:00+00:00",
            "last_result": "GREEN\n\n**Material drift:** none",
            "last_error": "",
        },
        {"id": "a2", "name": "Scribe Scout", "loop_enabled": False, "last_run_at": None},
    ]


def assignments():
    return [
        {
            "agent_id": "a1",
            "status": "completed",
            "review_status": "unreviewed",
            "result": "GREEN",
            "completed_at": f"{DATE}T12:00:00+00:00",
        },
        {
            "agent_id": "a1",
            "status": "completed",
            "review_status": "accepted",
            "result": "x",
            "completed_at": f"{DATE}T09:00:00+00:00",
        },
        {
            "agent_id": "a1",
            "status": "failed",
            "result": "",
            "completed_at": f"{DATE}T08:00:00+00:00",
        },
        {
            "agent_id": "a1",
            "status": "failed",
            "result": "",
            "completed_at": "2026-09-30T08:00:00+00:00",
        },
    ]


LOOPS = {
    "standup": {
        "last_run": f"{DATE}T07:30:19-05:00",
        "last_status": "ok",
        "last_note": "17 commits, 5 active stories",
    },
    "code-review": {
        "last_run": "2026-10-04T22:44:21-05:00",
        "last_status": "ok",
        "last_note": "0 findings",
    },
}


def test_build_digest_is_one_line_per_agent_and_loop():
    body = daily_digest.build_digest(
        date=DATE,
        agents=agents(),
        assignments=assignments(),
        run_counts={"a1": 3},
        loops=LOOPS,
        tasks=[
            {
                "task_id": "test_runner",
                "name": "Test Runner",
                "last_status": "failed",
                "last_result": "FAILED: Test suite did not run: ERROR x",
            }
        ],
        inbox_counts={"code_review": 31, "standup": 38},
    )
    lines = body.splitlines()
    assert lines[0] == f"AIIA digest {DATE}"
    assert "- CI Signal Officer: 3 runs, 1 waiting review, 1 failed — GREEN" in lines
    assert "- Scribe Scout: 0 runs, no schedule" in lines
    assert "- standup: ok 2026-10-05 07:30 — 17 commits, 5 active stories" in lines
    assert "- Test Runner: Test suite did not run: ERROR x" in lines
    assert lines[-1] == "Inbox waiting review: 69 (31 code_review, 38 standup)"
    assert len(body) <= daily_digest.MAX_BODY


def test_build_digest_with_nothing_still_reads():
    body = daily_digest.build_digest(
        date=DATE, agents=[], assignments=[], run_counts={}, loops={}, tasks=[], inbox_counts={}
    )
    assert "- no agents" in body and "- no loop registry found" in body
    assert body.endswith("Inbox waiting review: 0")


def test_load_loops_tolerates_missing_or_broken_registry(tmp_path):
    assert daily_digest.load_loops(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert daily_digest.load_loops(broken) == {}
    good = tmp_path / "good.json"
    good.write_text(json.dumps(LOOPS))
    assert daily_digest.load_loops(good) == LOOPS


def test_cron_target_follows_the_task_timezone():
    defn = {"schedule_cron_hour": 7, "schedule_cron_minute": 40, "schedule_tz": "America/Chicago"}
    # 2026-10-05 13:00 UTC is 08:00 CDT: today's 07:40 CDT has passed.
    now = datetime(2026, 10, 5, 13, 0, tzinfo=timezone.utc)
    assert cron_target(defn, now) == datetime(2026, 10, 5, 12, 40, tzinfo=timezone.utc)
    # 12:00 UTC is 07:00 CDT: the most recent target is yesterday's.
    assert cron_target(defn, now - timedelta(hours=1)) == datetime(
        2026, 10, 4, 12, 40, tzinfo=timezone.utc
    )
    assert cron_label(defn) == "daily 07:40 America/Chicago"
    utc = {"schedule_cron_hour": 6}
    assert cron_target(utc, now) == datetime(2026, 10, 5, 6, 0, tzinfo=timezone.utc)
    assert cron_label(utc) == "daily 06:00 UTC"
    assert cron_target({"schedule_cron_hour": 6, "schedule_tz": "Not/AZone"}, now).hour == 6


def test_cron_task_is_due_after_the_target_even_if_the_minute_was_missed(monkeypatch):
    runner = TaskRunner(AsyncMock(), "/unused", None)
    for task_id in TASK_DEFINITIONS:
        runner.tasks[task_id]["last_run"] = datetime.now(timezone.utc).isoformat()
    defn = TASK_DEFINITIONS["daily_digest"]
    target = cron_target(defn, datetime.now(timezone.utc))
    runner.tasks["daily_digest"]["last_run"] = (target - timedelta(hours=2)).isoformat()
    assert runner._find_due_task() == "daily_digest"
    runner.tasks["daily_digest"]["last_run"] = (target + timedelta(minutes=1)).isoformat()
    assert runner._find_due_task() is None
    runner._update_next_run("daily_digest")
    next_run = datetime.fromisoformat(runner.tasks["daily_digest"]["next_run"])
    assert next_run > datetime.now(timezone.utc)
    assert next_run - target == timedelta(days=1)


def _runner(tmp_path, monkeypatch) -> tuple[TaskRunner, MemoryInbox]:
    inbox = MemoryInbox(tmp_path / "inbox.sqlite3")
    registry = tmp_path / "loops.json"
    registry.write_text(json.dumps(LOOPS))
    monkeypatch.setenv("AIIA_LOOPS_REGISTRY", str(registry))
    runner = TaskRunner(AsyncMock(), "/unused", None)
    runner._progress = AsyncMock()  # type: ignore[method-assign]
    runner.studio_sources = {
        "agents": agents,
        "assignments": assignments,
        "run_counts": lambda: {"a1": 3},
        "inbox": lambda: inbox,
    }
    return runner, inbox


def test_digest_task_files_one_row_and_one_post_per_day(tmp_path, monkeypatch):
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_ENABLED", "1")
    monkeypatch.setenv("AIIA_SLACK_MEMORY_POST_CHANNEL_ID", "C0MEMORY01")
    monkeypatch.setenv("AIIA_SLACK_BOT_TOKEN", "synthetic")
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "T_TEST")
    runner, inbox = _runner(tmp_path, monkeypatch)
    inbox.ingest(text="a finding", source_key="review:x", source="code_review", project="aiia")

    summary, body = asyncio.run(runner._task_daily_digest())
    assert (
        summary.startswith("Digest ")
        and "slack post queued" in summary
        and "inbox row new" in summary
    )
    assert "Inbox waiting review: 1 (1 code_review)" in body
    rows = inbox.list(source="digest")
    assert rows["total"] == 1 and rows["ideas"][0]["post_requested"] == 1
    posts = inbox.memory_post_status()
    assert posts == {"pending": 1}
    post = inbox.claim_memory_post()
    assert post["memory_id"].startswith("digest:") and post["channel_id"] == "C0MEMORY01"
    assert post["workspace_id"] == ""
    assert "&lt;" not in post["body"] or "<" not in body

    summary2, _ = asyncio.run(runner._task_daily_digest())
    assert "inbox row existing" in summary2 and "already queued" in summary2
    assert inbox.list(source="digest")["total"] == 1
    with inbox.connect() as db:
        assert db.execute("SELECT count(*) FROM memory_posts").fetchone()[0] == 1


def test_digest_task_without_slack_is_an_inbox_row_only(tmp_path, monkeypatch):
    monkeypatch.delenv("AIIA_SLACK_MEMORY_POST_ENABLED", raising=False)
    runner, inbox = _runner(tmp_path, monkeypatch)
    summary, _ = asyncio.run(runner._task_daily_digest())
    assert "slack not configured" in summary
    assert inbox.list(source="digest")["total"] == 1
    assert inbox.memory_post_status() == {}


def test_digest_task_fails_loudly_when_not_wired():
    runner = TaskRunner(AsyncMock(), "/unused", None)
    with pytest.raises(RuntimeError, match="not wired"):
        asyncio.run(runner._task_daily_digest())


def test_digest_is_a_registered_always_on_task():
    runner = TaskRunner(AsyncMock(), "/unused", None)
    row = next(r for r in runner.get_all_tasks() if r["task_id"] == "daily_digest")
    assert row["schedule"] == "daily 07:40 America/Chicago"
    brief = next(r for r in runner.get_all_tasks() if r["task_id"] == "daily_brief")
    assert brief["schedule"] == "daily 07:00 America/Chicago"
    assert Path(daily_digest.loops_registry_path()).name == "loops-registry.json"
