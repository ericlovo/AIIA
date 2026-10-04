"""Built-in tasks report what happened: a suite that never ran is a failure, not a quality signal."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from local_brain.command_center.aiia_tasks import (
    DAILY_BRIEF_TIMEOUT_SECONDS,
    TaskRunner,
    parse_pytest_summary,
)


def test_parse_pytest_summary_reads_the_summary_line_only():
    lines = [
        "ERROR local_brain/tests/test_phase2_config.py",
        "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
        "91 warnings, 1 error in 1.71s",
    ]
    assert parse_pytest_summary(lines) == (0, 0, 1)
    assert parse_pytest_summary(["993 passed, 9 skipped, 6 warnings in 27.34s"]) == (993, 0, 0)
    assert parse_pytest_summary(["3 passed, 1 failed, 2 errors in 0.5s"]) == (3, 1, 2)
    assert parse_pytest_summary(["no summary here"]) == (0, 0, 0)


def test_every_builtin_task_is_listed_as_always_on_with_a_schedule():
    runner = TaskRunner(AsyncMock(), "/unused", None)
    rows = runner.get_all_tasks()
    assert rows
    for row in rows:
        assert row["enabled"] is True
        assert row["pausable"] is False
        assert row["schedule"].startswith(("every ", "daily "))
    brief = next(row for row in rows if row["task_id"] == "daily_brief")
    assert brief["schedule"] == "daily 08:00 UTC"


class _Proc:
    def __init__(self, output: bytes, returncode: int) -> None:
        self._output = output
        self.returncode = returncode
        self.killed = False

    async def communicate(self):
        return self._output, b""

    def kill(self):
        self.killed = True


def _runner(monkeypatch, output: bytes, returncode: int) -> TaskRunner:
    runner = TaskRunner(AsyncMock(), "/unused", None)
    calls: list[tuple] = []

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return _Proc(output, returncode)

    monkeypatch.setattr(
        "local_brain.command_center.aiia_tasks.asyncio.create_subprocess_exec", fake_exec
    )
    monkeypatch.setattr(
        "local_brain.command_center.aiia_tasks.os.walk",
        lambda root: iter([(str(root), [], ["test_a.py", "test_b.py"])]),
    )
    runner._aiia_request = AsyncMock(return_value={})  # type: ignore[method-assign]
    runner._emit_insight = AsyncMock()  # type: ignore[method-assign]
    runner._progress = AsyncMock()  # type: ignore[method-assign]
    runner._calls = calls  # type: ignore[attr-defined]
    return runner


def test_test_runner_raises_when_collection_is_interrupted(monkeypatch):
    output = b"ERROR local_brain/tests/test_phase2_config.py\n!!! Interrupted: 1 error during collection !!!\n1 error in 1.6s\n"
    runner = _runner(monkeypatch, output, returncode=2)
    with pytest.raises(RuntimeError, match="did not run"):
        asyncio.run(runner._task_test_runner())
    args = runner._calls[0]  # type: ignore[attr-defined]
    # pytest is given the test root, not every file, so conftest governs collection.
    assert args[-1] == "local_brain/tests"
    assert not any(str(a).endswith("test_a.py") for a in args)
    runner._aiia_request.assert_not_called()  # type: ignore[attr-defined]


def test_test_runner_records_a_real_run(monkeypatch):
    runner = _runner(monkeypatch, b"...\n3 passed, 1 failed in 0.4s\n", returncode=1)
    summary, report = asyncio.run(runner._task_test_runner())
    assert summary == "3 passed, 1 failed, 0 errors"
    assert "3 passed" in report
    assert runner._extra["test_trends"][-1]["passed"] == 3


def test_daily_brief_timeout_is_not_shorter_than_the_brain_ollama_timeout():
    assert DAILY_BRIEF_TIMEOUT_SECONDS >= 300


def test_daily_brief_failure_names_the_exception_and_remembers_nothing(monkeypatch):
    monitor = MagicMock()
    monitor.get_full_snapshot.return_value = {"services": {}}
    runner = TaskRunner(AsyncMock(), "/unused", monitor)
    remembered: list[dict] = []

    async def fake_request(method, path, body=None, timeout=30.0):
        if path == "/v1/aiia/ask":
            raise httpx.ReadTimeout("")
        if path == "/v1/aiia/remember":
            remembered.append(body or {})
        return {"total_docs": 1, "memories": [], "stats": {}, "total_memories": 0, "items": []}

    runner._aiia_request = fake_request  # type: ignore[method-assign]
    runner._progress = AsyncMock()  # type: ignore[method-assign]
    runner._emit_insight = AsyncMock()  # type: ignore[method-assign]
    runner._run_git = AsyncMock(return_value="")  # type: ignore[method-assign]
    monkeypatch.setattr(
        "local_brain.command_center.aiia_tasks.generate_report", lambda **kwargs: {}
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(runner._task_daily_brief())
    assert "ReadTimeout" in str(excinfo.value)
    assert not any("Daily Brief" in str(item.get("fact")) for item in remembered)
