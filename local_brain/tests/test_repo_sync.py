"""repo_sync is fetch-only by default; pull is opt-in and verified."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_brain.command_center.aiia_tasks import (
    REPO_SYNC_PULL_ENV,
    TASK_DEFINITIONS,
    TaskRunFailure,
    TaskRunner,
    format_repo_sync_summary,
    repo_sync_pull_enabled,
)


def _runner(repo_path: str | Path, extra: dict | None = None) -> TaskRunner:
    runner = TaskRunner(AsyncMock(), str(repo_path), None)
    runner._progress = AsyncMock()
    runner._aiia_request = AsyncMock(return_value={"status": "online"})
    runner._emit_insight = AsyncMock()
    if extra:
        runner._extra.update(extra)
    return runner


def _git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=check,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _configure_git(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "pull.ff", "only")


def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _linked_repos(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    """Bare origin + seed working clone + Mini checkout. Returns checkout, origin, seed, sha."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "-b", "main")

    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-b", "main")
    _configure_git(seed)
    sha = _commit_file(seed, "README.md", "# AIIA\n", "initial")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "main")

    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", str(origin), str(checkout)],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    _configure_git(checkout)
    return checkout, origin, seed, sha


class FakeGit:
    """Scripted git for command-order tests (no real repository)."""

    def __init__(self, head: str, origin: str) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.head = head
        self.origin = origin
        self.dirty = ""
        self.allow_pull = False
        self.fast_forward = True
        self.diff_files = ""

    async def run(self, *args: str) -> str:
        self.calls.append(args)
        cmd = args[0]
        if cmd == "rev-parse":
            if args[1] == "HEAD":
                return self.head
            if args[1] == "origin/main":
                return self.origin
            raise RuntimeError(f"unexpected rev-parse {args}")
        if cmd == "fetch":
            assert args == ("fetch", "origin")
            return ""
        if cmd == "rev-list":
            assert args[:2] == ("rev-list", "--count")
            return "0" if self.head == self.origin else "2"
        if cmd == "status":
            return self.dirty
        if cmd == "pull":
            if not self.allow_pull:
                raise AssertionError(f"unexpected pull: {args}")
            assert args == ("pull", "--ff-only", "origin", "main")
            if not self.fast_forward:
                raise RuntimeError("Not possible to fast-forward")
            self.head = self.origin
            return "Fast-forward"
        if cmd == "cat-file":
            return "commit"
        if cmd == "diff":
            return self.diff_files
        raise AssertionError(f"unexpected git {args}")


def test_repo_sync_definition_is_not_a_silent_pull():
    assert "Pull only if AIIA_REPO_SYNC_PULL=1" in TASK_DEFINITIONS["repo_sync"]["description"]
    assert "Git pull" not in TASK_DEFINITIONS["repo_sync"]["description"]


def test_repo_sync_pull_enabled_only_for_exact_one():
    assert repo_sync_pull_enabled({}) is False
    assert repo_sync_pull_enabled({REPO_SYNC_PULL_ENV: "1"}) is True
    assert repo_sync_pull_enabled({REPO_SYNC_PULL_ENV: "true"}) is False
    assert repo_sync_pull_enabled({REPO_SYNC_PULL_ENV: "0"}) is False
    assert repo_sync_pull_enabled({REPO_SYNC_PULL_ENV: " 1 "}) is True


def test_format_repo_sync_summary_reports_head_movement_not_stored_gap():
    old = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    new = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    # HEAD already at the new commit: stored index SHA may still lag.
    text = format_repo_sync_summary(
        pull_enabled=False,
        head_before=new,
        head_after=new,
        origin_sha=new,
        behind=0,
        fetch_note="fetched origin",
        verify_result=None,
        index_msg=f"re-indexed 2 files (3 chunks) at {old[:8]}..{new[:8]}",
    )
    assert "did not move" in text
    assert f"Pulled {old[:8]}" not in text
    assert "0 commits behind" in text

    pulled = format_repo_sync_summary(
        pull_enabled=True,
        head_before=old,
        head_after=new,
        origin_sha=new,
        behind=0,
        fetch_note="fetched origin",
        verify_result="ok",
        index_msg="re-indexed 1 files (1 chunks)",
    )
    assert f"Pulled {old[:8]}→{new[:8]}" in pulled
    assert "verify=/health ok" in pulled


@pytest.mark.asyncio
async def test_fetch_only_does_not_pull_and_reports_behind(monkeypatch):
    monkeypatch.delenv(REPO_SYNC_PULL_ENV, raising=False)
    head = "a" * 40
    origin = "b" * 40
    fake = FakeGit(head, origin)
    runner = _runner("/unused", extra={"last_commit_sha": head})
    runner._run_git = fake.run  # type: ignore[method-assign]

    summary = await runner._task_repo_sync()

    assert ("pull",) not in {c[:1] for c in fake.calls}
    assert ("fetch", "origin") in fake.calls
    assert "Fetch-only" in summary
    assert "2 commit(s) behind origin/main" in summary
    assert f"HEAD {head[:8]} did not move" in summary
    assert runner._extra["last_commit_sha"] == head
    runner._aiia_request.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_only_does_not_report_stored_sha_gap_as_head_movement(monkeypatch):
    monkeypatch.delenv(REPO_SYNC_PULL_ENV, raising=False)
    stored = "a" * 40
    head = "c" * 40  # already at the new commit
    fake = FakeGit(head, head)
    fake.diff_files = ""
    runner = _runner("/unused", extra={"last_commit_sha": stored})
    runner._run_git = fake.run  # type: ignore[method-assign]

    summary = await runner._task_repo_sync()

    assert "did not move" in summary
    assert f"Pulled {stored[:8]}" not in summary
    assert "Fetch-only" in summary
    assert runner._extra["last_commit_sha"] == head


@pytest.mark.asyncio
async def test_opt_in_pull_refuses_tracked_changes(monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    head = "a" * 40
    fake = FakeGit(head, "b" * 40)
    fake.dirty = " M local_brain/command_center/aiia_tasks.py"
    runner = _runner("/unused")
    runner._run_git = fake.run  # type: ignore[method-assign]

    with pytest.raises(TaskRunFailure, match="tracked changes") as exc:
        await runner._task_repo_sync()
    assert fake.head == head
    assert not any(c[0] == "pull" for c in fake.calls)
    assert "M local_brain" in exc.value.output


@pytest.mark.asyncio
async def test_opt_in_pull_refuses_when_not_fast_forward(monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    head = "a" * 40
    fake = FakeGit(head, "b" * 40)
    fake.allow_pull = True
    fake.fast_forward = False
    runner = _runner("/unused")
    runner._run_git = fake.run  # type: ignore[method-assign]

    with pytest.raises(TaskRunFailure, match="cannot fast-forward"):
        await runner._task_repo_sync()
    assert fake.head == head


@pytest.mark.asyncio
async def test_opt_in_pull_records_shas_and_failed_verify(monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    old = "a" * 40
    new = "b" * 40
    fake = FakeGit(old, new)
    fake.allow_pull = True
    runner = _runner("/unused", extra={"last_commit_sha": old})
    runner._run_git = fake.run  # type: ignore[method-assign]

    async def boom(method, path, body=None, timeout=30.0):
        raise RuntimeError("connection refused")

    runner._aiia_request = boom  # type: ignore[method-assign]

    with pytest.raises(TaskRunFailure, match="Post-pull verify failed") as exc:
        await runner._task_repo_sync()
    assert old[:8] in str(exc.value)
    assert new[:8] in str(exc.value)
    assert fake.head == new
    assert f"old_sha={old}" in exc.value.output
    assert f"new_sha={new}" in exc.value.output
    assert runner._extra.get("last_commit_sha") == old  # unchanged on failed verify


@pytest.mark.asyncio
async def test_opt_in_pull_success_records_verify_ok(monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    old = "a" * 40
    new = "b" * 40
    fake = FakeGit(old, new)
    fake.allow_pull = True
    runner = _runner("/unused", extra={"last_commit_sha": old})
    runner._run_git = fake.run  # type: ignore[method-assign]

    summary = await runner._task_repo_sync()

    assert f"Pulled {old[:8]}→{new[:8]}" in summary
    assert "verify=/health ok" in summary
    assert runner._extra["last_commit_sha"] == new
    runner._aiia_request.assert_awaited()
    assert runner._aiia_request.await_args_list[0].args[1] == "/health"


@pytest.mark.asyncio
async def test_failed_verify_marks_execute_task_failed_without_restart(monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    old = "a" * 40
    new = "b" * 40
    fake = FakeGit(old, new)
    fake.allow_pull = True
    runner = _runner("/unused")
    runner._run_git = fake.run  # type: ignore[method-assign]
    runner.save_state = MagicMock()  # type: ignore[method-assign]
    runner._update_next_run = MagicMock()  # type: ignore[method-assign]

    async def boom(method, path, body=None, timeout=30.0):
        raise RuntimeError("connection refused")

    runner._aiia_request = boom  # type: ignore[method-assign]
    runner.restart_brain = AsyncMock()  # type: ignore[attr-defined]
    runner._restart = AsyncMock()  # type: ignore[attr-defined]

    await runner._execute_task("repo_sync")

    task = runner.tasks["repo_sync"]
    assert task["status"] == "failed"
    assert task["fail_count"] == 1
    assert "Post-pull verify failed" in task["last_result"]
    runner.restart_brain.assert_not_called()
    runner._restart.assert_not_called()


@pytest.mark.asyncio
async def test_temp_repo_fetch_only_does_not_move_head(tmp_path, monkeypatch):
    monkeypatch.delenv(REPO_SYNC_PULL_ENV, raising=False)
    checkout, _origin, seed, sha = _linked_repos(tmp_path)
    ahead = _commit_file(seed, "docs.md", "ahead\n", "origin ahead")
    _git(seed, "push", "origin", "main")
    assert _git(checkout, "rev-parse", "HEAD") == sha

    runner = _runner(checkout, extra={"last_commit_sha": sha})
    summary = await runner._task_repo_sync()

    assert _git(checkout, "rev-parse", "HEAD") == sha
    assert ahead != sha
    assert "Fetch-only" in summary
    assert "1 commit(s) behind origin/main" in summary
    assert "did not move" in summary
    assert runner._extra["last_commit_sha"] == sha


@pytest.mark.asyncio
async def test_temp_repo_stored_sha_gap_is_not_head_movement(tmp_path, monkeypatch):
    monkeypatch.delenv(REPO_SYNC_PULL_ENV, raising=False)
    checkout, _origin, seed, first = _linked_repos(tmp_path)
    second = _commit_file(seed, "notes.md", "# notes\n", "second")
    _git(seed, "push", "origin", "main")
    _git(checkout, "pull", "--ff-only", "origin", "main")
    assert _git(checkout, "rev-parse", "HEAD") == second

    runner = _runner(checkout, extra={"last_commit_sha": first})
    summary = await runner._task_repo_sync()

    assert _git(checkout, "rev-parse", "HEAD") == second
    assert "did not move" in summary
    assert f"Pulled {first[:8]}" not in summary
    assert f"{first[:8]}..{second[:8]}" in summary
    assert runner._extra["last_commit_sha"] == second
    ingest = [
        c
        for c in runner._aiia_request.await_args_list
        if c.args[0] == "POST" and c.args[1] == "/v1/aiia/ingest"
    ]
    assert ingest


@pytest.mark.asyncio
async def test_temp_repo_opt_in_pull_fast_forwards_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    checkout, _origin, seed, sha = _linked_repos(tmp_path)
    ahead = _commit_file(seed, "docs.md", "# docs\n", "origin ahead")
    _git(seed, "push", "origin", "main")

    runner = _runner(checkout, extra={"last_commit_sha": sha})
    summary = await runner._task_repo_sync()

    assert _git(checkout, "rev-parse", "HEAD") == ahead
    assert f"Pulled {sha[:8]}→{ahead[:8]}" in summary
    assert "verify=/health ok" in summary
    assert runner._extra["last_commit_sha"] == ahead
    assert runner._aiia_request.await_args_list[0].args[1] == "/health"


@pytest.mark.asyncio
async def test_temp_repo_opt_in_pull_refuses_dirty_tracked_tree(tmp_path, monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    checkout, _origin, seed, sha = _linked_repos(tmp_path)
    _commit_file(seed, "docs.md", "# docs\n", "origin ahead")
    _git(seed, "push", "origin", "main")
    (checkout / "README.md").write_text("# dirty\n")

    runner = _runner(checkout, extra={"last_commit_sha": sha})
    with pytest.raises(TaskRunFailure, match="tracked changes"):
        await runner._task_repo_sync()
    assert _git(checkout, "rev-parse", "HEAD") == sha


@pytest.mark.asyncio
async def test_temp_repo_untracked_files_do_not_block_pull(tmp_path, monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    checkout, _origin, seed, sha = _linked_repos(tmp_path)
    ahead = _commit_file(seed, "docs.md", "# docs\n", "origin ahead")
    _git(seed, "push", "origin", "main")
    (checkout / "scratch.tmp").write_text("untracked\n")

    runner = _runner(checkout, extra={"last_commit_sha": sha})
    summary = await runner._task_repo_sync()

    assert _git(checkout, "rev-parse", "HEAD") == ahead
    assert "verify=/health ok" in summary


@pytest.mark.asyncio
async def test_temp_repo_opt_in_pull_refuses_diverged_history(tmp_path, monkeypatch):
    monkeypatch.setenv(REPO_SYNC_PULL_ENV, "1")
    checkout, _origin, seed, sha = _linked_repos(tmp_path)
    _commit_file(seed, "docs.md", "# docs\n", "origin ahead")
    _git(seed, "push", "origin", "main")
    local_head = _commit_file(checkout, "local.md", "# local\n", "local diverged")

    runner = _runner(checkout, extra={"last_commit_sha": sha})
    with pytest.raises(TaskRunFailure, match="cannot fast-forward"):
        await runner._task_repo_sync()
    assert _git(checkout, "rev-parse", "HEAD") == local_head
    # Local commit remains; origin was not merged in.
    assert "docs.md" not in _git(checkout, "ls-tree", "-r", "--name-only", "HEAD")
