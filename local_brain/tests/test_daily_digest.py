"""Daily digest: mocked git/gh evidence, one-line format, quiet vs review, channel fallback."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from local_brain.command_center.agent_output import (
    DEFAULT_OUTPUT_CHANNEL,
    SLACK_NOT_CONFIGURED,
    SLACK_POSTING_PENDING,
)
from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.aiia_tasks import TASK_DEFINITIONS, TaskRunner
from local_brain.command_center.assignment_registry import (
    MAX_PENDING_LOOP_REVIEWS,
    AssignmentRegistry,
)
from local_brain.command_center.daily_digest import (
    DIGEST_AGENT_NAME,
    DIGEST_CRON_HOUR_UTC,
    DIGEST_LINE_MAX,
    DIGEST_ONE_LINER,
    DriftSignal,
    RepoEvidence,
    classify_digest,
    collect_and_deliver,
    collect_digest,
    collect_repo_evidence,
    deliver_digest,
    digest_agent_payload,
    find_digest_agent,
    format_digest_line,
    github_slug_from_remote,
)
from local_brain.scripts.ensure_daily_digest_agent import main as seed_main


def _git(path: Path, *args: str, env: dict[str, str] | None = None) -> None:
    base = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        env={**base, **(env or {})},
    )


def _commit(path: Path, name: str, text: str) -> None:
    (path / name).write_text(text)
    _git(path, "add", name)
    _git(path, "commit", "-qm", text)


def _repo_with_drift(tmp_path: Path) -> Path:
    repo = tmp_path / "mindmoor"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "base")
    _git(repo, "branch", "production")
    _git(repo, "branch", "alumni")
    _commit(repo, "app.py", "newer on main")
    _git(repo, "remote", "add", "origin", "https://github.com/ericlovo/mindmoor.git")
    return repo


def _empty_github(endpoint: str) -> object:
    if endpoint.endswith("/pulls?state=open&per_page=20") or "/pulls?" in endpoint:
        return []
    if "/actions/runs" in endpoint:
        return {"workflow_runs": []}
    if "/pulls/" in endpoint:
        return {"mergeable": True, "mergeable_state": "clean"}
    return {}


def test_daily_brief_schedule_is_unchanged_and_digest_is_separate():
    brief = TASK_DEFINITIONS["daily_brief"]
    digest = TASK_DEFINITIONS["daily_digest"]
    assert brief["schedule_cron_hour"] == 8
    assert brief["schedule_cron_minute"] == 0
    assert brief["uses_llm"] is True
    assert digest["schedule_cron_hour"] == DIGEST_CRON_HOUR_UTC == 12
    assert digest["schedule_cron_minute"] == 0
    assert digest["uses_llm"] is False
    rows = TaskRunner(AsyncMock(), "/unused", None).get_all_tasks()
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["daily_brief"]["schedule"] == "daily 08:00 UTC"
    assert by_id["daily_digest"]["schedule"] == "daily 12:00 UTC"
    assert by_id["daily_digest"]["enabled"] is True


def test_github_slug_accepts_tokenized_https_without_leaking_secrets():
    assert (
        github_slug_from_remote("https://github.com/ericlovo/mindmoor.git") == "ericlovo/mindmoor"
    )
    assert (
        github_slug_from_remote("https://x-access-token:ghs_secret@github.com/ericlovo/AIIA.git")
        == "ericlovo/AIIA"
    )
    assert github_slug_from_remote("git@github.com:ericlovo/sanction.git") == "ericlovo/sanction"
    assert github_slug_from_remote("https://gitlab.com/ericlovo/AIIA.git") == ""
    assert "ghs_secret" not in github_slug_from_remote(
        "https://x-access-token:ghs_secret@github.com/ericlovo/AIIA.git"
    )


def test_format_clear_vs_stuck_and_truncates():
    clear = [
        RepoEvidence(repo_id="aiia", mounted=True, complete=True, open_prs=0, failing_ci=0),
        RepoEvidence(repo_id="mindmoor", mounted=True, complete=True, open_prs=0, failing_ci=0),
    ]
    assert format_digest_line(clear) == "CLEAR: no material drift/CI"
    assert classify_digest(clear) == "clear"

    stuck = [
        RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            open_prs=2,
            failing_ci=1,
            moved=["AIIA 2 PRs"],
            stuck=["AIIA CI"],
        ),
        RepoEvidence(
            repo_id="mindmoor",
            mounted=True,
            complete=True,
            drift=[DriftSignal("mindmoor", "production", 12)],
        ),
    ]
    line = format_digest_line(stuck)
    assert line.startswith("Moved: AIIA 2 PRs")
    assert "Stuck: AIIA CI" in line
    assert "Drift: mindmoor production −12 behind main" in line
    assert classify_digest(stuck) == "stuck"
    assert len(line) <= DIGEST_LINE_MAX

    long_stuck = [
        RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            stuck=["X" * 300],
        )
    ]
    trimmed = format_digest_line(long_stuck)
    assert len(trimmed) == DIGEST_LINE_MAX
    assert trimmed.endswith("…")

    assert format_digest_line([]) == "INCOMPLETE: no mounted repos"
    assert classify_digest([]) == "incomplete"


def test_unmounted_repos_are_skipped_not_incomplete(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    mounts = {"aiia": missing, "mindmoor": missing}
    result = collect_digest(mounts=mounts, github_api=_empty_github)
    assert result.severity == "incomplete"
    assert result.line == "INCOMPLETE: no mounted repos"
    assert all(not row.mounted for row in result.evidence)


def test_evidence_gathering_reads_prs_ci_conflicts_and_mindmoor_drift(tmp_path):
    repo = _repo_with_drift(tmp_path)
    calls: list[str] = []

    def github(endpoint: str) -> object:
        calls.append(endpoint)
        if "/pulls?state=open" in endpoint:
            return [
                {
                    "number": 4,
                    "title": "conflicted",
                    "mergeable": False,
                    "mergeable_state": "dirty",
                },
                {"number": 5, "title": "ok", "mergeable": True, "mergeable_state": "clean"},
            ]
        if "/actions/runs" in endpoint:
            return {
                "workflow_runs": [
                    {"name": "CI", "status": "completed", "conclusion": "failure"},
                    {"name": "CI", "status": "completed", "conclusion": "success"},
                    {"name": "Deploy", "status": "completed", "conclusion": "success"},
                ]
            }
        return {}

    mounts = {"mindmoor": repo, "aiia": tmp_path / "nope"}
    row = collect_repo_evidence("mindmoor", github_api=github, mounts=mounts)
    assert row.mounted and row.complete
    assert row.open_prs == 2
    assert row.failing_ci == 1
    assert row.merge_conflicts == 1
    refs = {signal.ref: signal.behind_main for signal in row.drift}
    assert refs["production"] == 1
    assert refs["alumni"] == 1
    assert any("/pulls?state=open" in item for item in calls)
    assert any("/actions/runs" in item for item in calls)
    assert not any(item.endswith("/pulls/4") for item in calls)

    result = collect_digest(github_api=github, mounts=mounts)
    assert result.severity == "stuck"
    assert "Drift: mindmoor production −1 behind main" in result.line
    assert "mindmoor alumni −1 behind main" in result.line
    assert "Stuck:" in result.line


def test_alumni_release_branch_is_used_when_exact_ref_is_missing(tmp_path):
    repo = tmp_path / "mindmoor"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "base")
    _git(repo, "branch", "release/alumni-2026")
    _commit(repo, "app.py", "ahead")
    mounts = {"mindmoor": repo}

    def github(endpoint: str) -> object:
        if "/pulls?" in endpoint:
            return []
        if "/actions/runs" in endpoint:
            return {"workflow_runs": []}
        return {}

    row = collect_repo_evidence("mindmoor", github_api=github, mounts=mounts)
    assert [signal.ref for signal in row.drift] == ["alumni"]
    assert row.drift[0].behind_main == 1


def test_github_read_failure_is_incomplete_not_clear(tmp_path):
    repo = tmp_path / "aiia"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "base")
    _git(repo, "remote", "add", "origin", "https://github.com/ericlovo/AIIA.git")

    def boom(endpoint: str) -> object:
        raise RuntimeError("github_api_unavailable")

    result = collect_digest(github_api=boom, mounts={"aiia": repo})
    assert result.severity == "incomplete"
    assert result.line != "CLEAR: no material drift/CI"
    assert "Incomplete: aiia" in result.line


def _agent(tmp_path: Path, **overrides):
    agents = AgentRegistry(tmp_path / "agents.json")
    payload = digest_agent_payload()
    payload.update(overrides)
    return agents, agents.create(
        payload["name"],
        payload["mission"],
        payload["persona"],
        payload["skills"],
        tools=payload["tools"],
        one_liner=payload["one_liner"],
        output_channel=payload["output_channel"],
        loop_enabled=False,
    )


def test_clear_day_records_quietly_and_does_not_block_tomorrow(tmp_path):
    agents, agent = _agent(tmp_path)
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    from local_brain.command_center.daily_digest import DigestResult

    result = DigestResult(
        line="CLEAR: no material drift/CI",
        severity="clear",
        fingerprint="clear-1",
        evidence=[],
    )
    first = deliver_digest(result, agent=agent, agents=agents, assignments=assignments)
    assert first["quiet"] is True
    assert first["assignment_id"] == ""
    assert assignments.list_assignments() == []
    assert assignments.pending_loop_reviews(agent["id"]) == 0
    assert agents.get(agent["id"])["loop_skip_reason"] == "quiet_clear"
    assert agents.get(agent["id"])["last_result"] == "CLEAR: no material drift/CI"
    assert agents.get(agent["id"])["runs"][0]["delivered_channel"] == DEFAULT_OUTPUT_CHANNEL
    checks = agents.recover_runs().loop_checks()
    assert [row["outcome"] for row in checks] == ["quiet_clear"]

    second = deliver_digest(
        result, agent=agents.get(agent["id"]), agents=agents, assignments=assignments
    )
    assert second["quiet"] is True
    assert assignments.pending_loop_reviews(agent["id"]) == 0
    assert assignments.list_assignments() == []


def test_stuck_opens_one_review_item_same_fingerprint_stays_quiet(tmp_path):
    agents, agent = _agent(tmp_path)
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    from local_brain.command_center.daily_digest import DigestResult

    result = DigestResult(
        line="Stuck: AIIA CI | Drift: mindmoor production −3 behind main",
        severity="stuck",
        fingerprint="stuck-1",
        evidence=[],
    )
    now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    first = deliver_digest(result, agent=agent, agents=agents, assignments=assignments, now=now)
    assert first["quiet"] is False
    assert first["reason"] == "stuck"
    assert assignments.pending_loop_reviews(agent["id"]) == 1
    work = assignments.list_assignments()
    assert len(work) == 1
    assert work[0]["trigger"] == "interval"
    assert work[0]["result"] == result.line
    assert work[0]["review_status"] == "unreviewed"

    next_day = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
    again = deliver_digest(
        result,
        agent=agents.get(agent["id"]),
        agents=agents,
        assignments=assignments,
        now=next_day,
    )
    assert again["quiet"] is True
    assert again["reason"] == "unchanged_repository_input"
    assert len(assignments.list_assignments()) == 1
    assert assignments.pending_loop_reviews(agent["id"]) == 1


def test_awaiting_review_cap_skips_new_stuck_item_but_task_can_run_again(tmp_path):
    agents, agent = _agent(tmp_path)
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    from local_brain.command_center.daily_digest import DigestResult

    for index in range(MAX_PENDING_LOOP_REVIEWS):
        deliver_digest(
            DigestResult(
                line=f"Stuck: AIIA CI {index}",
                severity="stuck",
                fingerprint=f"fp-{index}",
                evidence=[],
            ),
            agent=agents.get(agent["id"]),
            agents=agents,
            assignments=assignments,
            now=datetime(2026, 10, 1 + index, tzinfo=timezone.utc),
        )
    assert assignments.pending_loop_reviews(agent["id"]) == MAX_PENDING_LOOP_REVIEWS
    blocked = deliver_digest(
        DigestResult(line="Stuck: new", severity="stuck", fingerprint="fp-new", evidence=[]),
        agent=agents.get(agent["id"]),
        agents=agents,
        assignments=assignments,
        now=datetime(2026, 10, 10, tzinfo=timezone.utc),
    )
    assert blocked["reason"] == "awaiting_review"
    assert blocked["quiet"] is True
    assert assignments.pending_loop_reviews(agent["id"]) == MAX_PENDING_LOOP_REVIEWS
    assert agents.get(agent["id"])["loop_skip_reason"] == "awaiting_review"


def test_incomplete_surfaces_without_blocking_the_next_day(tmp_path):
    agents, agent = _agent(tmp_path)
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    from local_brain.command_center.daily_digest import DigestResult

    result = DigestResult(
        line="Incomplete: aiia",
        severity="incomplete",
        fingerprint="inc-1",
        evidence=[],
        failures=("github_api_unavailable",),
    )
    first = deliver_digest(result, agent=agent, agents=agents, assignments=assignments)
    assert first["reason"] == "check_incomplete"
    assert assignments.pending_loop_reviews(agent["id"]) == 0
    failure = assignments.list_assignments()[0]
    assert failure["source_kind"] == "loop_check"
    assert failure["status"] == "failed"

    second = deliver_digest(
        result, agent=agents.get(agent["id"]), agents=agents, assignments=assignments
    )
    assert second["reason"] == "check_incomplete"
    assert assignments.pending_loop_reviews(agent["id"]) == 0
    assert len(assignments.list_assignments()) == 1


@pytest.mark.parametrize(
    ("ready", "note"),
    [(False, SLACK_NOT_CONFIGURED), (True, SLACK_POSTING_PENDING)],
)
def test_slack_channel_falls_back_to_inbox(tmp_path, ready, note):
    agents, agent = _agent(tmp_path, output_channel="slack")
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    from local_brain.command_center.daily_digest import DigestResult

    result = DigestResult(
        line="Stuck: AIIA CI",
        severity="stuck",
        fingerprint="slack-1",
        evidence=[],
    )
    delivery = deliver_digest(
        result,
        agent=agent,
        agents=agents,
        assignments=assignments,
        slack_ready=ready,
        now=datetime(2026, 10, 5, tzinfo=timezone.utc),
    )
    assert delivery["declared"] == "slack"
    assert delivery["delivered_channel"] == DEFAULT_OUTPUT_CHANNEL
    assert delivery["delivery_note"] == note
    assert assignments.list_assignments()[0]["result"] == "Stuck: AIIA CI"
    run = agents.get(agent["id"])["runs"][0]
    assert run["delivered_channel"] == DEFAULT_OUTPUT_CHANNEL
    assert run["delivery_note"] == note


def test_missing_agent_keeps_the_line_on_the_task_only(tmp_path):
    agents = AgentRegistry(tmp_path / "agents.json")
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    from local_brain.command_center.daily_digest import DigestResult

    result = DigestResult(
        line="CLEAR: no material drift/CI", severity="clear", fingerprint="x", evidence=[]
    )
    delivery = deliver_digest(result, agent=None, agents=agents, assignments=assignments)
    assert delivery["delivery_note"].startswith("daily digest agent not enabled")
    assert delivery["delivered_channel"] == ""
    assert assignments.list_assignments() == []


def test_find_digest_agent_and_seed_payload():
    payload = digest_agent_payload()
    assert payload["name"] == DIGEST_AGENT_NAME
    assert payload["one_liner"] == DIGEST_ONE_LINER
    assert payload["output_channel"] == "studio_inbox"
    assert payload["loop_enabled"] is False
    assert find_digest_agent([payload, {"name": "Other"}])["one_liner"] == DIGEST_ONE_LINER
    assert find_digest_agent([{"name": "Other"}]) is None


def test_seed_script_prints_and_never_writes(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert seed_main([]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["agent"]["name"] == DIGEST_AGENT_NAME
    assert list(tmp_path.glob("*.json")) == []


def test_collect_and_deliver_wires_the_named_agent(tmp_path):
    repo = tmp_path / "aiia"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "base")
    _git(repo, "remote", "add", "origin", "https://github.com/ericlovo/AIIA.git")
    agents, _agent_row = _agent(tmp_path)
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    result, delivery = collect_and_deliver(
        agents=agents,
        assignments=assignments,
        github_api=_empty_github,
        mounts={"aiia": repo, "mindmoor": tmp_path / "missing"},
    )
    assert result.severity == "clear"
    assert result.line == "CLEAR: no material drift/CI"
    assert delivery["quiet"] is True
    assert assignments.list_assignments() == []


def test_task_runner_invokes_digest_without_llm(tmp_path, monkeypatch):
    from local_brain.command_center import daily_digest as module

    runner = TaskRunner(AsyncMock(), "/unused", None)
    runner.agent_registry = AgentRegistry(tmp_path / "agents.json")
    runner.assignment_registry = AssignmentRegistry(tmp_path / "assignments.json")
    runner._progress = AsyncMock()  # type: ignore[method-assign]
    called = {}

    def fake_collect_and_deliver(**kwargs):
        called.update(kwargs)
        from local_brain.command_center.daily_digest import DigestResult

        result = DigestResult(
            line="CLEAR: no material drift/CI",
            severity="clear",
            fingerprint="t",
            evidence=[],
        )
        return result, {"line": result.line, "quiet": True, "delivered_channel": "studio_inbox"}

    monkeypatch.setattr(module, "collect_and_deliver", fake_collect_and_deliver)
    import asyncio

    summary, output = asyncio.run(runner._task_daily_digest())
    assert summary == "CLEAR: no material drift/CI"
    assert "studio_inbox" in output
    assert called["agents"] is runner.agent_registry
    assert called["assignments"] is runner.assignment_registry
    assert runner._extra["daily_digest"]["quiet"] is True
