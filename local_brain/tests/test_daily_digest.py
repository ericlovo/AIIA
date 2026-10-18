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
    cron_is_due,
    cron_label,
    cron_next,
    cron_target,
)
from local_brain.command_center.memory_inbox import MemoryInbox

DATE = "2026-10-05"
OWNER = "owner"


@pytest.fixture(autouse=True)
def _digest_owner_login(monkeypatch):
    monkeypatch.setattr(daily_digest, "load_owners", lambda path=None: frozenset({OWNER}))


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


PRODUCTS = [
    daily_digest.DigestProduct(
        id="aiia", name="AIIA", github="ericlovo/AIIA", mount="aiia", holds=(47, 52)
    ),
    daily_digest.DigestProduct(
        id="mindmoor",
        name="Mindmoor",
        github="tonybangert/mindmoor",
        mount="mindmoor",
        drift=("production", "alumni"),
    ),
    daily_digest.DigestProduct(
        id="sanction",
        name="Sanction",
        github="example/sanction",
        mount="sanction",
        asks=False,
    ),
    daily_digest.DigestProduct(
        id="mia", name="MIA", github="example/mia", mount="mia", holds=(42,)
    ),
    daily_digest.DigestProduct(id="morrow", name="Morrow", github="example/morrow", mount="morrow"),
]

CUSTOMERS = [
    daily_digest.DigestCustomer(
        id="trs",
        code="TRS",
        name="That's Right Sweetie",
        label_override="That's Right Sweetie (TRS)",
        products=("mindmoor",),
        tenant="trs",
        branch="trs",
        excluded_from_releases=True,
    ),
    daily_digest.DigestCustomer(
        id="alumni-nations",
        code="AN",
        name="Alumni Nations",
        products=("mindmoor",),
        agents=("Alumni Nations Research Scout",),
        drift=("alumni",),
        phase=daily_digest.DigestPhase(name="Phase 1", start="2026-10-15", end="2027-01-12"),
    ),
    daily_digest.DigestCustomer(
        id="smart-medical",
        code="SM",
        name="Smart Medical",
        consulting=True,
        no_repo=True,
    ),
]


def test_build_digest_leads_with_products_then_decisions_then_footer():
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
        inbox_counts={"code_review": 31, "standup": 38, "slack": 2},
        inbox_items=[
            {
                "text": "Ship the digest rewrite?",
                "source": "slack",
                "project": "aiia",
                "priority": "high",
            },
            {"text": "Alumni is behind", "source": "slack", "project": "mindmoor"},
        ],
        repo_evidence=[],
        products=PRODUCTS,
        customers=[],
    )
    lines = body.splitlines()
    assert lines[0] == f"AIIA digest {DATE}"
    assert lines[2] == "AIIA: no repo mounted"
    assert lines[6] == "Morrow: no repo mounted"
    assert "Needs a decision" in lines
    assert "- aiia · Ship the digest rewrite?" in lines
    assert "- mindmoor · Alumni is behind" in lines
    assert not any(line.startswith("- plus ") for line in lines)
    footer = lines[-1]
    assert footer.startswith("Agents: 2 (0 active, 1 waiting review)")
    assert "Loops: 2 ok" in footer
    assert "Built-ins failing: Test Runner" in footer
    assert "Repos" not in lines
    assert "Inbox waiting review" not in body
    assert len(body) <= daily_digest.MAX_BODY


def test_build_digest_with_nothing_still_reads():
    body = daily_digest.build_digest(
        date=DATE,
        agents=[],
        assignments=[],
        run_counts={},
        loops={},
        tasks=[],
        inbox_counts={},
        products=PRODUCTS,
        customers=[],
    )
    assert "AIIA: no repo mounted" in body
    assert "Morrow: no repo mounted" in body
    assert "Needs a decision" in body
    assert "- none" in body
    assert "Agents: 0 (0 active, 0 waiting review)" in body
    assert "Loops: none" in body
    assert "Built-ins failing: none" in body


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


def _idle_cron_runner(now: datetime) -> TaskRunner:
    """A runner whose interval and cron tasks all look freshly run at `now`."""
    runner = TaskRunner(AsyncMock(), "/unused", None)
    stamp = now.isoformat()
    for task_id in TASK_DEFINITIONS:
        runner.tasks[task_id]["last_run"] = stamp
    return runner


def test_cron_task_is_due_after_the_target_even_if_the_minute_was_missed(monkeypatch):
    # 07:45 CDT on 2026-10-05: five minutes after 07:40, the busy minute is over.
    now = datetime(2026, 10, 5, 12, 45, tzinfo=timezone.utc)
    monkeypatch.setattr("local_brain.command_center.aiia_tasks._utc_now", lambda: now)
    runner = _idle_cron_runner(now)
    defn = TASK_DEFINITIONS["daily_digest"]
    yesterday = datetime(2026, 10, 4, 12, 40, tzinfo=timezone.utc)
    runner.tasks["daily_digest"]["last_run"] = yesterday.isoformat()
    assert runner._find_due_task() == "daily_digest"
    runner.tasks["daily_digest"]["last_run"] = now.isoformat()
    assert runner._find_due_task() is None
    runner._update_next_run("daily_digest")
    next_run = datetime.fromisoformat(runner.tasks["daily_digest"]["next_run"])
    assert next_run == cron_next(defn, now)
    assert next_run == datetime(2026, 10, 6, 12, 40, tzinfo=timezone.utc)


def test_cron_restart_catch_up_runs_at_most_once_per_local_day(monkeypatch):
    defn = TASK_DEFINITIONS["daily_digest"]
    before = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)  # 07:00 CDT, before 07:40
    after = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)  # 10:00 CDT
    assert cron_is_due(defn, before, None) is False
    assert cron_is_due(defn, after, None) is True
    monkeypatch.setattr("local_brain.command_center.aiia_tasks._utc_now", lambda: after)
    runner = _idle_cron_runner(after)
    runner.tasks["daily_digest"]["last_run"] = None
    runner.tasks["daily_brief"]["last_run"] = after.isoformat()
    for task_id, spec in TASK_DEFINITIONS.items():
        if task_id in {"daily_digest", "daily_brief"} or "schedule_cron_hour" not in spec:
            continue
        runner.tasks[task_id]["last_run"] = after.isoformat()
    assert runner._find_due_task() == "daily_digest"
    runner.tasks["daily_digest"]["last_run"] = after.isoformat()
    assert runner._find_due_task() is None
    later = datetime(2026, 10, 5, 15, 1, tzinfo=timezone.utc)
    assert cron_is_due(defn, later, after.isoformat()) is False
    next_slot = datetime(2026, 10, 6, 12, 50, tzinfo=timezone.utc)
    assert cron_is_due(defn, next_slot, after.isoformat()) is True


def test_cron_schedule_move_does_not_double_run_daily_brief(monkeypatch):
    """Yesterday's 08:00 UTC brief still fills yesterday; the Chicago move is one run."""
    defn = TASK_DEFINITIONS["daily_brief"]
    last = datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc).isoformat()
    before = datetime(2026, 10, 5, 11, 30, tzinfo=timezone.utc)  # 06:30 CDT
    at_slot = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)  # 07:00 CDT
    assert cron_is_due(defn, before, last) is False
    assert cron_is_due(defn, at_slot, last) is True
    monkeypatch.setattr("local_brain.command_center.aiia_tasks._utc_now", lambda: at_slot)
    runner = _idle_cron_runner(at_slot)
    runner.tasks["daily_brief"]["last_run"] = last
    for task_id in TASK_DEFINITIONS:
        if task_id != "daily_brief":
            runner.tasks[task_id]["last_run"] = at_slot.isoformat()
    assert runner._find_due_task() == "daily_brief"
    runner.tasks["daily_brief"]["last_run"] = at_slot.isoformat()
    assert runner._find_due_task() is None
    assert cron_is_due(defn, at_slot + timedelta(hours=1), at_slot.isoformat()) is False


def test_cron_busy_scheduler_still_fires_once_after_the_minute(monkeypatch):
    defn = TASK_DEFINITIONS["daily_digest"]
    yesterday = datetime(2026, 10, 4, 12, 40, tzinfo=timezone.utc).isoformat()
    missed = datetime(2026, 10, 5, 12, 45, tzinfo=timezone.utc)  # 07:45 CDT
    assert cron_is_due(defn, missed, yesterday) is True
    monkeypatch.setattr("local_brain.command_center.aiia_tasks._utc_now", lambda: missed)
    runner = _idle_cron_runner(missed)
    runner.tasks["daily_digest"]["last_run"] = yesterday
    for task_id in TASK_DEFINITIONS:
        if task_id != "daily_digest":
            runner.tasks[task_id]["last_run"] = missed.isoformat()
    assert runner._find_due_task() == "daily_digest"
    runner.tasks["daily_digest"]["last_run"] = missed.isoformat()
    assert runner._find_due_task() is None
    assert cron_is_due(defn, missed + timedelta(minutes=5), missed.isoformat()) is False


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
    evidence = [
        daily_digest.RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            open_prs=2,
            failing_ci=1,
            moved=["AIIA 2 PRs"],
            stuck=["AIIA CI"],
        )
    ]
    monkeypatch.setattr(
        daily_digest,
        "collect_digest",
        lambda **kwargs: daily_digest.DigestResult(
            line="Moved: AIIA 2 PRs | Stuck: AIIA CI",
            severity="stuck",
            fingerprint="test",
            evidence=evidence,
        ),
    )
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
    assert "Needs a decision" in body
    assert "a finding" in body
    assert any(line.startswith("AIIA:") for line in body.splitlines())
    assert "CI on main" in body
    assert "Agents:" in body
    rows = inbox.list(source="digest")
    assert rows["total"] == 1 and rows["ideas"][0]["post_requested"] == 1
    posts = inbox.memory_post_status()
    assert posts == {"pending": 1}
    post = inbox.claim_memory_post()
    assert post["memory_id"].startswith("digest:") and post["channel_id"] == "C0MEMORY01"
    assert post["workspace_id"] == ""
    assert post["body"].startswith("*AIIA · ")
    assert "*waiting on you:*" in post["body"] or "no repo" in post["body"]
    assert "_Details in Studio_" in post["body"]
    assert "Agents:" not in post["body"]
    assert "Loops:" not in post["body"]
    assert "Needs a decision" not in post["body"]
    assert "Needs a decision" in body
    assert "Agents:" in body
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


def test_build_digest_includes_product_lines_from_repo_evidence():
    evidence = [
        daily_digest.RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            open_prs=2,
            failing_ci=1,
            moved=["AIIA 2 PRs"],
            stuck=["AIIA CI"],
        ),
        daily_digest.RepoEvidence(
            repo_id="mindmoor",
            mounted=True,
            complete=True,
            drift=[daily_digest.DriftSignal("mindmoor", "production", 12)],
        ),
    ]
    body = daily_digest.build_digest(
        date=DATE,
        agents=[],
        assignments=[],
        run_counts={},
        loops={},
        tasks=[],
        inbox_counts={},
        repo_evidence=evidence,
        products=PRODUCTS,
        customers=[],
    )
    assert "AIIA: shipped none | blocked CI on main | waiting on you none" in body
    assert "Mindmoor: shipped none | blocked production −12 | waiting on you none" in body
    assert "Morrow: no repo mounted" in body
    assert "Repos" not in body.splitlines()


def test_github_slug_accepts_tokenized_https_without_leaking_secrets():
    secret = "https://x-access-token:ghs_secret@github.com/ericlovo/AIIA.git"
    assert daily_digest.github_slug_from_remote("https://github.com/ericlovo/mindmoor.git") == (
        "ericlovo/mindmoor"
    )
    assert daily_digest.github_slug_from_remote(secret) == "ericlovo/AIIA"
    assert daily_digest.github_slug_from_remote("git@github.com:ericlovo/sanction.git") == (
        "ericlovo/sanction"
    )
    assert daily_digest.github_slug_from_remote("https://gitlab.com/ericlovo/AIIA.git") == ""
    assert "ghs_secret" not in daily_digest.github_slug_from_remote(secret)
    assert "x-access-token" not in daily_digest.github_slug_from_remote(secret)


def test_format_clear_vs_stuck_and_truncates():
    clear = [
        daily_digest.RepoEvidence(
            repo_id="aiia", mounted=True, complete=True, open_prs=0, failing_ci=0
        ),
        daily_digest.RepoEvidence(
            repo_id="mindmoor", mounted=True, complete=True, open_prs=0, failing_ci=0
        ),
    ]
    assert daily_digest.format_digest_line(clear) == "CLEAR: no material drift/CI"
    assert daily_digest.classify_digest(clear) == "clear"

    stuck = [
        daily_digest.RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            open_prs=2,
            failing_ci=1,
            moved=["AIIA 2 PRs"],
            stuck=["AIIA CI"],
        ),
        daily_digest.RepoEvidence(
            repo_id="mindmoor",
            mounted=True,
            complete=True,
            drift=[daily_digest.DriftSignal("mindmoor", "production", 12)],
        ),
    ]
    line = daily_digest.format_digest_line(stuck)
    assert line.startswith("Moved: AIIA 2 PRs")
    assert "Stuck: AIIA CI" in line
    assert "Drift: mindmoor production −12 behind main" in line
    assert daily_digest.classify_digest(stuck) == "stuck"
    assert len(line) <= daily_digest.DIGEST_LINE_MAX

    trimmed = daily_digest.format_digest_line(
        [daily_digest.RepoEvidence(repo_id="aiia", mounted=True, complete=True, stuck=["X" * 300])]
    )
    assert len(trimmed) == daily_digest.DIGEST_LINE_MAX
    assert trimmed.endswith("…")
    assert daily_digest.format_digest_line([]) == "INCOMPLETE: no mounted repos"
    assert daily_digest.classify_digest([]) == "incomplete"


def _git(path: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
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
    row = daily_digest.collect_repo_evidence("mindmoor", github_api=github, mounts=mounts)
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

    result = daily_digest.collect_digest(github_api=github, mounts=mounts)
    assert result.severity == "stuck"
    assert "Drift: mindmoor production −1 behind main" in result.line
    assert "mindmoor alumni −1 behind main" in result.line
    assert "Stuck:" in result.line
    assert "ghs_" not in result.line and "x-access-token" not in result.line


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

    row = daily_digest.collect_repo_evidence("mindmoor", github_api=github, mounts=mounts)
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

    result = daily_digest.collect_digest(github_api=boom, mounts={"aiia": repo})
    assert result.severity == "incomplete"
    assert result.line != "CLEAR: no material drift/CI"
    assert "Incomplete: aiia" in result.line


def test_tokenized_origin_collects_ci_without_leaking_the_token(tmp_path):
    repo = tmp_path / "aiia"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "README.md", "base")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://x-access-token:ghs_secret@github.com/ericlovo/AIIA.git",
    )

    def github(endpoint: str) -> object:
        assert "ghs_secret" not in endpoint
        if "/pulls?" in endpoint:
            return []
        if "/actions/runs" in endpoint:
            return {
                "workflow_runs": [{"name": "CI", "status": "completed", "conclusion": "failure"}]
            }
        return {}

    row = daily_digest.collect_repo_evidence("aiia", github_api=github, mounts={"aiia": repo})
    assert row.mounted and row.complete
    assert row.failing_ci == 1
    line = daily_digest.format_digest_line([row])
    assert "ghs_secret" not in line and "x-access-token" not in line
    assert "Stuck: AIIA CI" in line


def test_unmounted_repos_are_skipped_not_incomplete(tmp_path):
    missing = tmp_path / "missing"
    mounts = {"aiia": missing, "mindmoor": missing}
    result = daily_digest.collect_digest(
        mounts=mounts, github_api=lambda endpoint: [], products=PRODUCTS
    )
    assert result.severity == "incomplete"
    assert result.line == "INCOMPLETE: no mounted repos"
    assert all(not row.mounted for row in result.evidence)


def test_format_product_line_empty_segments_use_none():
    product = PRODUCTS[0]
    evidence = daily_digest.RepoEvidence(repo_id="aiia", mounted=True, complete=True)
    line = daily_digest.format_product_line(product, evidence)
    assert line == "AIIA: shipped none | blocked none | waiting on you none"


def test_format_product_line_unmounted_and_unreadable():
    assert daily_digest.format_product_line(PRODUCTS[-1], None) == "Morrow: no repo mounted"
    assert (
        daily_digest.format_product_line(
            PRODUCTS[2], daily_digest.RepoEvidence(repo_id="sanction", mounted=False, complete=True)
        )
        == "Sanction: no repo mounted"
    )
    unread = daily_digest.RepoEvidence(
        repo_id="aiia", mounted=True, complete=False, failures=("git_head_failed",)
    )
    assert daily_digest.format_product_line(PRODUCTS[0], unread) == "AIIA: repo unreadable"
    waiting = daily_digest.format_product_line(PRODUCTS[-1], None, review=2)
    assert waiting == "Morrow: no repo mounted | waiting on you 2 review"


def test_format_product_line_uses_pr_numbers_and_counts():
    evidence = daily_digest.RepoEvidence(
        repo_id="aiia",
        mounted=True,
        complete=True,
        failing_ci=1,
        merged_prs=[
            daily_digest.PullSignal(104, "feed repo CI into the daily digest", merged=True)
        ],
        open_pulls=[
            daily_digest.PullSignal(12, "conflicted", mergeable=False, mergeable_state="dirty"),
            daily_digest.PullSignal(
                18,
                "ready",
                mergeable=True,
                mergeable_state="clean",
                author=OWNER,
                head_ref="cursor/ready",
            ),
            daily_digest.PullSignal(
                21,
                "ready draft",
                draft=True,
                mergeable=True,
                mergeable_state="clean",
                author=OWNER,
                head_ref="cursor/draft",
            ),
        ],
        drift=[daily_digest.DriftSignal("mindmoor", "production", 12)],
    )
    line = daily_digest.format_product_line(
        PRODUCTS[0], evidence, paused=["Delivery Watch"], review=2
    )
    assert line.startswith("AIIA: shipped #104 feed repo CI into the daily")
    assert "blocked CI on main, #12 conflict, production −12, Delivery Watch paused" in line
    assert "waiting on you #18 merge, 2 review" in line
    assert "undraft" not in line


def test_map_agent_to_product_uses_suite_namespace_repo_handles_kind():
    assert daily_digest.map_agent_to_product({"suite": "mindmoor"}, PRODUCTS).id == "mindmoor"
    assert (
        daily_digest.map_agent_to_product({"memory_namespace": "sanction"}, PRODUCTS).id
        == "sanction"
    )
    assert daily_digest.map_agent_to_product({"repo_id": "aiia"}, PRODUCTS).id == "aiia"
    assert daily_digest.map_agent_to_product({"repo": "mia"}, PRODUCTS).id == "mia"
    assert daily_digest.map_agent_to_product({"handles": ["morrow", "inbox"]}, PRODUCTS).id == (
        "morrow"
    )
    assert daily_digest.map_agent_to_product({"kind": "morrow"}, PRODUCTS).id == "morrow"
    assert daily_digest.map_agent_to_product({"kind": "product", "handles": ["ci"]}, PRODUCTS) is (
        None
    )
    assert daily_digest.map_agent_to_product({"name": "Mindmoor Scout"}, PRODUCTS).id == "mindmoor"
    assert daily_digest.map_agent_to_product({"name": "Scribe Scout"}, PRODUCTS) is None
    assert daily_digest.map_agent_to_product({"id": "legacy"}, PRODUCTS) is None


def test_checked_in_product_config_has_the_five_defaults():
    products = daily_digest.load_products()
    assert [product.id for product in products] == ["aiia", "mindmoor", "sanction", "mia", "morrow"]
    assert products[0].github.endswith("/AIIA")
    assert products[1].github.endswith("/mindmoor")
    assert products[2].github.endswith("/sanction")
    assert products[3].github.endswith("/moral-intention-analyst")
    assert products[4].github.endswith("/morrow")
    assert products[0].holds == (47, 52)
    assert products[2].asks is False
    assert products[3].holds == (42,)
    assert products[1].drift == ("production", "alumni")
    customers = daily_digest.load_customers()
    assert [customer.id for customer in customers] == ["trs", "alumni-nations", "smart-medical"]
    assert customers[0].code == "TRS" and customers[0].branch == "trs"
    assert customers[1].agents == ("Alumni Nations Research Scout",)
    assert customers[1].phase and customers[1].phase.start == "2026-10-15"
    assert customers[2].mapped() is True
    assert customers[2].consulting is True
    assert customers[2].no_repo is True
    assert customers[2].status_note == ""
    payload = daily_digest._config_payload() or {}
    assert payload.get("owners")


def test_product_config_env_override(tmp_path, monkeypatch):
    path = tmp_path / "products.json"
    path.write_text(
        json.dumps({"products": [{"id": "solo", "name": "Solo", "github": "", "mount": "solo"}]})
    )
    monkeypatch.setenv(daily_digest.PRODUCTS_ENV, str(path))
    products = daily_digest.load_products()
    assert [product.id for product in products] == ["solo"]
    assert [customer.id for customer in daily_digest.load_customers()] == [
        "trs",
        "alumni-nations",
        "smart-medical",
    ]


def test_decisions_prefer_slack_and_product_tags_and_collapse_the_rest():
    items = [
        {"text": "Ship it?", "source": "slack", "project": "aiia", "priority": "high"},
        {"text": "Alumni drift", "source": "code_review", "project": "mindmoor"},
        {"text": "Noise", "source": "standup", "project": ""},
        {"text": "More slack", "source": "slack", "project": ""},
        {"text": "digest row", "source": "digest", "project": "aiia"},
    ]
    lines = daily_digest.format_decision_lines(
        items,
        products=PRODUCTS,
        inbox_counts={"slack": 4, "code_review": 3, "standup": 8},
    )
    assert lines[0] == "- aiia · Ship it?"
    assert lines[1] == "- mindmoor · Alumni drift"
    assert lines[2] == "- slack · More slack"
    assert lines[-1] == "- plus 1 more (1 standup)"
    assert all("digest row" not in line for line in lines)
    assert daily_digest.format_decision_lines([], products=PRODUCTS) == ["- none"]


def test_build_digest_size_cap():
    products = [
        daily_digest.DigestProduct(id=f"p{i}", name=f"Product-{i}", mount=f"p{i}")
        for i in range(80)
    ]
    evidence = [
        daily_digest.RepoEvidence(
            repo_id=f"p{i}",
            mounted=True,
            complete=True,
            failing_ci=1,
            merged_prs=[daily_digest.PullSignal(i + 1, "shipped title " + ("y" * 40), merged=True)],
            open_pulls=[
                daily_digest.PullSignal(
                    i + 100,
                    "blocked title " + ("z" * 40),
                    mergeable=False,
                    mergeable_state="dirty",
                )
            ],
        )
        for i in range(80)
    ]
    body = daily_digest.build_digest(
        date=DATE,
        agents=[],
        assignments=[],
        run_counts={},
        loops={},
        tasks=[],
        inbox_counts={"slack": 40},
        inbox_items=[
            {"text": f"Decision {index} " + ("x" * 80), "source": "slack", "project": "aiia"}
            for index in range(40)
        ],
        repo_evidence=evidence,
        products=products,
        customers=[],
    )
    assert len(body) <= daily_digest.MAX_BODY
    assert body.endswith("…")
    assert len(body) == daily_digest.MAX_BODY
    long_line = daily_digest.format_product_line(
        PRODUCTS[0],
        daily_digest.RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            stuck=["X" * 400],
            open_pulls=[
                daily_digest.PullSignal(n, "x" * 80, mergeable=False, mergeable_state="dirty")
                for n in range(1, 20)
            ],
        ),
    )
    assert len(long_line) == daily_digest.PRODUCT_LINE_MAX
    assert long_line.endswith("…")


def _status_digest_kwargs(**overrides):
    payload = {
        "date": "2026-10-08",
        "agents": [
            {
                "id": "a1",
                "name": "CI Signal Officer",
                "repo_id": "aiia",
                "status": "running",
                "loop_enabled": True,
                "loop_task": "watch CI",
            },
            {
                "id": "a2",
                "name": "Delivery Watch",
                "suite": "mindmoor",
                "loop_enabled": False,
                "loop_task": "watch delivery",
                "loop_skip_reason": "awaiting_review",
            },
            {
                "id": "a4",
                "name": "Alumni Nations Research Scout",
                "loop_enabled": True,
                "loop_task": "alumni research",
                "loop_skip_reason": "awaiting_review",
            },
            {"id": "a3", "name": "Scribe Scout", "status": "idle"},
        ],
        "assignments": [
            {
                "agent_id": "a1",
                "status": "completed",
                "review_status": "unreviewed",
                "result": "GREEN",
            }
        ],
        "run_counts": {"a1": 3},
        "loops": LOOPS,
        "tasks": [
            {
                "name": "Daily Brief",
                "last_status": "failed",
                "last_result": "FAILED: AIIA returned HTTP 500",
            }
        ],
        "inbox_counts": {"slack": 5, "code_review": 2},
        "inbox_items": [
            {"text": "Ship the digest rewrite today?", "source": "slack", "project": "aiia"},
            {"text": "Alumni is 12 commits behind main", "source": "slack", "project": "mindmoor"},
            {"text": "Morrow repo still unmounted", "source": "slack", "project": "morrow"},
        ],
        "repo_evidence": [
            daily_digest.RepoEvidence(
                repo_id="aiia",
                mounted=True,
                complete=True,
                failing_ci=1,
                merged_prs=[daily_digest.PullSignal(104, "feed repo CI", merged=True)],
                open_pulls=[
                    daily_digest.PullSignal(
                        18,
                        "ready",
                        mergeable=True,
                        mergeable_state="clean",
                        author=OWNER,
                        head_ref="cursor/ready",
                    )
                ],
            ),
            daily_digest.RepoEvidence(
                repo_id="mindmoor",
                mounted=True,
                complete=True,
                drift=[
                    daily_digest.DriftSignal("mindmoor", "production", 12),
                    daily_digest.DriftSignal("mindmoor", "alumni", 4),
                ],
            ),
            daily_digest.RepoEvidence(
                repo_id="sanction",
                mounted=True,
                complete=True,
                open_pulls=[
                    daily_digest.PullSignal(
                        12, "conflicted", mergeable=False, mergeable_state="dirty"
                    )
                ],
            ),
            daily_digest.RepoEvidence(
                repo_id="mia",
                mounted=True,
                complete=True,
                open_pulls=[
                    daily_digest.PullSignal(
                        7, "ready draft", draft=True, mergeable=True, mergeable_state="clean"
                    )
                ],
            ),
            daily_digest.RepoEvidence(
                repo_id="morrow",
                mounted=False,
                complete=True,
                remote=True,
            ),
        ],
        "products": PRODUCTS,
        "customers": CUSTOMERS,
        "customer_evidence": [
            daily_digest.CustomerEvidence(customer_id="trs", behind_main=8, ref_found=True)
        ],
    }
    payload.update(overrides)
    return payload


def test_sample_rendered_digest_matches_the_product_status_shape():
    body = daily_digest.build_digest(**_status_digest_kwargs())
    assert body.splitlines() == [
        "AIIA digest 2026-10-08",
        "",
        "AIIA: shipped #104 feed repo CI | blocked CI on main | waiting on you #18 merge, 1 review",
        "Mindmoor: shipped none | blocked production −12, alumni −4 | waiting on you 1 review",
        "Sanction: shipped none | blocked #12 conflict | waiting on you none",
        "MIA: shipped none | blocked none | waiting on you none",
        "Morrow: shipped none | blocked none | waiting on you none",
        "",
        "Customers",
        "That's Right Sweetie (TRS): shipped none | blocked main −8 not on TRS | waiting on you none",
        "Alumni Nations: shipped none | blocked alumni −4 | waiting on you 1 review · 7 days to kickoff",
        "Smart Medical: no repo, consulting",
        "",
        "Needs a decision",
        "- aiia · Ship the digest rewrite today?",
        "- mindmoor · Alumni is 12 commits behind main",
        "- morrow · Morrow repo still unmounted",
        "",
        "Agents: 4 (1 active, 3 waiting review) · Loops: 2 ok · Built-ins failing: Daily Brief (AIIA returned HTTP 500)",
    ]
    assert len(body) <= daily_digest.MAX_BODY
    assert "ghs_" not in body and "x-access-token" not in body
    assert body.index("Customers") < body.index("Needs a decision")


def test_slack_digest_leads_with_actions_and_distinguishes_merged_from_deployed():
    slack = daily_digest.format_slack_digest(**_status_digest_kwargs())
    assert slack.startswith("*AIIA · Thu Oct 8*\n\n*Needs attention*")
    assert "review for merge: #18" in slack
    assert "resolve conflicts: #12" in slack
    assert "review 1 agent output" in slack
    assert "*Merged — deployment not verified*" in slack
    assert slack.index("*Needs attention*") < slack.index("*Merged")
    assert "RuntimeError" not in slack and "HTTP 500" not in slack
    assert "Daily Brief*: failed; inspect the run in Studio before retrying" in slack
    assert "*Decisions*" in slack
    assert len(slack) <= daily_digest.SLACK_BODY_MAX


def test_slack_digest_omits_empty_needs_you_and_quiet_footer():
    slack = daily_digest.format_slack_digest(
        **_status_digest_kwargs(
            tasks=[],
            assignments=[],
            inbox_counts={},
            inbox_items=[],
            agents=[{"id": "a3", "name": "Scribe Scout", "status": "idle"}],
        )
    )
    assert "*Needs you" not in slack
    assert "failed" not in slack
    assert "Agents:" not in slack
    assert slack.endswith("_Details in Studio_")
    clean = daily_digest.format_slack_digest(
        date="2026-10-08",
        agents=[],
        assignments=[],
        loops={},
        tasks=[],
        inbox_counts={},
        repo_evidence=[
            daily_digest.RepoEvidence(
                repo_id="aiia",
                mounted=True,
                complete=True,
                merged_prs=[daily_digest.PullSignal(131, "shipped", merged=True)],
            )
        ],
        products=[PRODUCTS[0]],
        customers=[],
    )
    assert "*Needs attention*" not in clean
    assert "*Merged — deployment not verified*\n- *AIIA*: #131" in clean
    assert "No active alerts: AIIA" in clean
    assert clean.endswith("_Details in Studio_")


def test_morning_note_payload_uses_the_same_status_lines():
    payload = daily_digest.build_morning_note(
        **{
            key: value
            for key, value in _status_digest_kwargs().items()
            if key not in {"run_counts", "loops"}
        }
    )
    assert payload["date"] == "2026-10-08"
    assert [row["name"] for row in payload["products"]] == [
        "AIIA",
        "Mindmoor",
        "Sanction",
        "MIA",
        "Morrow",
    ]
    assert payload["products"][0]["state"] == "blocked"
    assert payload["customers"][-1]["state"] == "consulting"
    assert payload["customers"][-1]["note"] == "no repo, consulting"
    assert payload["customers"][1]["target"] == "2026-10-15T09:00:00"
    assert payload["needs_you_total"] >= 3
    assert payload["decisions"][0]["title"] == "Ship the digest rewrite today?"
    assert payload["decisions"][0]["kind"] == "inbox"
    assert payload["failure"] == "Daily Brief failed: AIIA returned HTTP 500"
    assert all("none" not in row["note"] for row in payload["products"])


def test_alumni_nations_is_a_countdown_when_nothing_is_waiting():
    line = daily_digest.morning_customer_line(CUSTOMERS[1], date="2026-10-08")
    assert line.state == "countdown"
    assert line.target == "2026-10-15T09:00:00"
    assert line.note == "7 days to kickoff"
    assert daily_digest._slack_line(line) == "🟢 *Alumni Nations*: 7 days to kickoff"
    waiting = daily_digest.morning_customer_line(
        CUSTOMERS[1],
        product_evidence=[
            daily_digest.RepoEvidence(
                repo_id="mindmoor",
                mounted=True,
                complete=True,
                drift=[daily_digest.DriftSignal("mindmoor", "alumni", 4)],
            )
        ],
        date="2026-10-08",
    )
    assert waiting.state == "waiting"
    assert waiting.target == "2026-10-15T09:00:00"
    assert "alumni 4 behind main" in waiting.note


def test_slack_header_date_drops_leading_zero():
    assert daily_digest.slack_header_date("2026-10-08") == "Thu Oct 8"
    assert daily_digest.slack_header_date("2026-10-15") == "Thu Oct 15"


def test_format_customer_line_unmapped_and_empty_segments():
    ghost = daily_digest.DigestCustomer(id="ghost", name="Ghost")
    assert daily_digest.format_customer_line(ghost) == "Ghost: not mapped yet"
    assert daily_digest.format_customer_line(CUSTOMERS[2]) == "Smart Medical: no repo, consulting"
    line = daily_digest.format_customer_line(CUSTOMERS[0], date=DATE)
    assert line == ("That's Right Sweetie (TRS): shipped none | blocked none | waiting on you none")


def test_format_customer_line_missing_branch_does_not_invent_drift():
    line = daily_digest.format_customer_line(
        CUSTOMERS[0],
        customer_evidence=daily_digest.CustomerEvidence(customer_id="trs", ref_found=False),
        date=DATE,
    )
    assert "not on TRS" not in line
    assert "blocked none" in line


def test_phase_note_days_to_kickoff_and_days_into():
    phase = CUSTOMERS[1].phase
    assert daily_digest.phase_note(phase, "2026-10-08") == "7 days to kickoff"
    assert daily_digest.phase_note(phase, "2026-10-14") == "1 day to kickoff"
    assert daily_digest.phase_note(phase, "2026-10-15") == "kickoff today"
    assert daily_digest.phase_note(phase, "2026-10-16") == "1 day into Phase 1"
    assert daily_digest.phase_note(phase, "2026-10-22") == "7 days into Phase 1"
    assert daily_digest.phase_note(phase, "2027-01-13") == "Phase 1 ended"
    assert daily_digest.phase_note(None, "2026-10-08") == ""


def test_map_agent_to_customers_uses_name_handles_namespace_not_suite():
    assert [
        customer.id
        for customer in daily_digest.map_agent_to_customers(
            {"name": "Alumni Nations Research Scout"}, CUSTOMERS
        )
    ] == ["alumni-nations"]
    assert [
        customer.id
        for customer in daily_digest.map_agent_to_customers({"handles": ["trs"]}, CUSTOMERS)
    ] == ["trs"]
    assert [
        customer.id
        for customer in daily_digest.map_agent_to_customers(
            {"memory_namespace": "alumni-nations", "name": "Other"},
            [
                daily_digest.DigestCustomer(
                    id="alumni-nations",
                    name="Alumni Nations",
                    namespaces=("alumni-nations",),
                )
            ],
        )
    ] == ["alumni-nations"]
    assert (
        daily_digest.map_agent_to_customers(
            {"name": "Mindmoor Scout", "suite": "mindmoor"}, CUSTOMERS
        )
        == []
    )
    assert daily_digest.map_agent_to_customers({"name": "Scribe Scout"}, CUSTOMERS) == []


def test_trs_behind_main_is_collected_only_when_the_branch_exists(tmp_path):
    repo = _repo_with_drift(tmp_path)
    mounts = {"mindmoor": repo}

    def github(endpoint: str) -> object:
        if "/pulls?" in endpoint:
            return []
        if "/actions/runs" in endpoint:
            return {"workflow_runs": []}
        return {}

    missing = daily_digest.collect_customer_evidence(CUSTOMERS, products=PRODUCTS, mounts=mounts)
    trs_missing = next(row for row in missing if row.customer_id == "trs")
    assert trs_missing.ref_found is False
    assert trs_missing.behind_main is None

    _git(repo, "branch", "trs", "HEAD~1")
    found = daily_digest.collect_customer_evidence(CUSTOMERS, products=PRODUCTS, mounts=mounts)
    trs = next(row for row in found if row.customer_id == "trs")
    assert trs.ref_found is True
    assert trs.behind_main == 1
    line = daily_digest.format_customer_line(CUSTOMERS[0], customer_evidence=trs, date=DATE)
    assert "main −1 not on TRS" in line


def _pull(
    number: int,
    *,
    title: str = "x",
    draft: bool = False,
    author: str = "",
    head_ref: str = "feature",
    labels: tuple[str, ...] = (),
    mergeable: bool = True,
    mergeable_state: str = "clean",
) -> daily_digest.PullSignal:
    return daily_digest.PullSignal(
        number,
        title,
        draft=draft,
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        author=author,
        head_ref=head_ref,
        labels=labels,
    )


def test_held_prs_never_appear_in_blocked_or_waiting():
    evidence = daily_digest.RepoEvidence(
        repo_id="aiia",
        mounted=True,
        complete=True,
        open_pulls=[
            _pull(47, title="hold list", mergeable=False, mergeable_state="dirty"),
            _pull(52, title="hold list ready", author=OWNER),
            _pull(18, title="eric ready", author=OWNER),
        ],
    )
    slack = daily_digest.morning_product_line(PRODUCTS[0], evidence)
    inbox = daily_digest.format_product_line(PRODUCTS[0], evidence)
    assert "#47" not in slack.note and "#52" not in slack.note
    assert "#47" not in inbox and "#52" not in inbox
    assert slack.waiting == "merge"
    assert "#18" in slack.note
    labeled = daily_digest.RepoEvidence(
        repo_id="mia",
        mounted=True,
        complete=True,
        open_pulls=[
            _pull(
                99,
                title="labeled hold",
                mergeable=False,
                mergeable_state="dirty",
                labels=("hold",),
            )
        ],
    )
    mia = daily_digest.morning_product_line(PRODUCTS[3], labeled)
    assert "#99" not in mia.note
    assert mia.state == "clean"


def test_no_asks_from_codex_drafts_or_sanction():
    sanction = daily_digest.RepoEvidence(
        repo_id="sanction",
        mounted=True,
        complete=True,
        open_pulls=[_pull(12, title="ready", author=OWNER)],
    )
    line = daily_digest.morning_product_line(PRODUCTS[2], sanction)
    assert line.waiting == ""
    assert "merge" not in line.note
    aiia = daily_digest.RepoEvidence(
        repo_id="aiia",
        mounted=True,
        complete=True,
        open_pulls=[
            _pull(
                111,
                title="codex draft",
                draft=True,
                author=OWNER,
                head_ref="codex/fix-x",
            ),
            _pull(114, title="codex other", draft=True, head_ref="codex/other"),
            _pull(88, title="other author", author="someone-else"),
        ],
    )
    note = daily_digest.morning_product_line(PRODUCTS[0], aiia)
    assert "undraft" not in note.note
    assert "merge" not in note.waiting
    assert "waiting on you" not in note.note


def test_unmounted_repo_falls_back_to_github_slug():
    calls: list[str] = []

    def github(endpoint: str) -> object:
        calls.append(endpoint)
        if "/pulls?state=open" in endpoint:
            return [
                {
                    "number": 3,
                    "title": "ok",
                    "draft": False,
                    "mergeable": True,
                    "mergeable_state": "clean",
                    "user": {"login": OWNER},
                    "head": {"ref": "cursor/ok", "sha": "abc"},
                }
            ]
        if "/actions/runs" in endpoint:
            return {"workflow_runs": []}
        if "/pulls?state=closed" in endpoint:
            return []
        return {}

    slug = "example/moral-intention-analyst"
    row = daily_digest.collect_repo_evidence("mia", github_api=github, mounts={}, github_slug=slug)
    assert row.mounted is False
    assert row.remote is True
    assert row.open_prs == 1
    assert row.open_pulls[0].author == OWNER
    assert any(f"repos/{slug}/pulls" in item for item in calls)
    line = daily_digest.morning_product_line(PRODUCTS[3], row)
    assert line.state != "unmounted"
    assert "no repo" not in line.note
    morrow = daily_digest.load_products()[4]
    assert morrow.github.endswith("/morrow")
    payload = daily_digest._config_payload() or {}
    assert payload.get("owners")


def test_checked_in_mia_and_morrow_use_github_when_unmounted():
    calls: list[str] = []

    def github(endpoint: str) -> object:
        calls.append(endpoint)
        if "/pulls?state=open" in endpoint:
            return []
        if "/actions/runs" in endpoint:
            return {"workflow_runs": []}
        if "/pulls?state=closed" in endpoint:
            return []
        return {}

    products = daily_digest.load_products()
    by_id = {product.id: product for product in products}
    assert by_id["mia"].github.endswith("/moral-intention-analyst")
    assert by_id["morrow"].github.endswith("/morrow")
    result = daily_digest.collect_digest(github_api=github, mounts={}, products=products)
    evidence = {row.repo_id: row for row in result.evidence}
    assert evidence["mia"].mounted is False and evidence["mia"].remote is True
    assert evidence["morrow"].mounted is False and evidence["morrow"].remote is True
    assert any("moral-intention-analyst" in item for item in calls)
    assert any(
        "/morrow/" in item or item.endswith("/morrow/pulls?state=open&per_page=20")
        for item in calls
    )
    slack = daily_digest.format_slack_digest(
        date="2026-10-16",
        agents=[],
        assignments=[],
        loops={},
        tasks=[],
        inbox_counts={},
        repo_evidence=result.evidence,
        products=products,
        customers=[],
    )
    assert "⚪ *MIA*: no repo" not in slack
    assert "⚪ *Morrow*: no repo" not in slack
    assert "No active alerts: AIIA, Mindmoor, Sanction, MIA, Morrow" in slack


def test_skipped_dependabot_and_codex_drafts_are_accounted_for():
    evidence = daily_digest.RepoEvidence(
        repo_id="aiia",
        mounted=True,
        complete=True,
        open_pulls=[
            _pull(109, author="dependabot[bot]", head_ref="dependabot/pip/x"),
            _pull(110, author="dependabot[bot]", head_ref="dependabot/npm/y"),
            _pull(111, draft=True, head_ref="codex/one"),
            _pull(114, draft=True, head_ref="codex/two"),
            _pull(18, author=OWNER, head_ref="cursor/ready"),
        ],
    )
    line = daily_digest.morning_product_line(PRODUCTS[0], evidence)
    assert line.extra == "skipped: 2 dependabot (#109, #110), 2 codex drafts (#111, #114)"
    slack = daily_digest._slack_line(line)
    assert "skipped: 2 dependabot (#109, #110), 2 codex drafts (#111, #114)" in slack
    assert "#109" in slack and "#111" in slack


def test_needs_you_dedupes_and_expires_stale_inbox_items():
    now = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
    items = [
        {
            "text": "Next best action: merge AIIA #18",
            "source": "slack",
            "project": "aiia",
            "created_at": "2026-10-16T10:00:00+00:00",
        },
        {
            "text": "Next best action: merge AIIA #18",
            "source": "slack",
            "project": "aiia",
            "created_at": "2026-10-15T10:00:00+00:00",
        },
        {
            "text": "Confirm shipped #104",
            "source": "slack",
            "project": "aiia",
            "created_at": "2026-10-16T09:00:00+00:00",
        },
        {
            "text": "Next best action: old leftover",
            "source": "slack",
            "created_at": "2026-10-10T12:00:00+00:00",
        },
        {
            "text": "Ship the digest rewrite today?",
            "source": "slack",
            "project": "aiia",
        },
    ]
    evidence = [
        daily_digest.RepoEvidence(
            repo_id="aiia",
            mounted=True,
            complete=True,
            open_pulls=[_pull(18, author=OWNER)],
            merged_prs=[daily_digest.PullSignal(104, "feed repo CI", merged=True)],
        )
    ]
    slack = daily_digest.format_slack_digest(
        date="2026-10-16",
        agents=[],
        assignments=[],
        loops={},
        tasks=[],
        inbox_counts={"slack": 117, "code_review": 40},
        inbox_items=items,
        repo_evidence=evidence,
        products=[PRODUCTS[0]],
        customers=[],
        now=now,
    )
    assert "*Decisions*" in slack
    assert slack.count("merge AIIA #18") == 1
    assert "Confirm shipped" not in slack
    assert "old leftover" not in slack
    assert "117" not in slack


def test_daily_brief_failure_names_the_task_and_error():
    line = daily_digest.slack_failure_line(
        tasks=[
            {
                "name": "Daily Brief",
                "last_status": "failed",
                "last_result": "FAILED: AIIA returned HTTP 500",
            }
        ],
        assignments=[],
        date=DATE,
    )
    assert line == "Daily Brief failed: AIIA returned HTTP 500"


def test_smart_medical_is_consulting_with_optional_note():
    blank = daily_digest.DigestCustomer(
        id="smart-medical",
        name="Smart Medical",
        code="SM",
        consulting=True,
        no_repo=True,
    )
    assert daily_digest.format_customer_line(blank) == "Smart Medical: no repo, consulting"
    slack = daily_digest._slack_line(daily_digest.morning_customer_line(blank))
    assert slack == "⚪ *Smart Medical*: no repo, consulting"
    noted = daily_digest.DigestCustomer(
        id="smart-medical",
        name="Smart Medical",
        consulting=True,
        no_repo=True,
        status_note="Tony reviewing the PerformanceLabs brief",
    )
    assert "PerformanceLabs" in daily_digest.format_customer_line(noted)
    assert "not mapped yet" not in daily_digest.format_customer_line(noted)


def test_slack_condenses_stale_stories_without_mutating_inbox():
    items = [
        {"text": f"Stale story: private_internal_slug_{n}", "source": "slack", "project": "aiia"}
        for n in range(77)
    ]
    body = daily_digest.format_slack_digest(**_status_digest_kwargs(inbox_items=items))
    assert "77 backlog items in Studio" in body
    assert "private_internal_slug" not in body
    assert "*Decisions*" not in body
    assert len(daily_digest.current_inbox_items(items)) == 77


def test_slack_separates_draft_conflicts_and_maintenance_from_main_failure():
    ev = daily_digest.RepoEvidence(
        repo_id="aiia",
        mounted=True,
        complete=True,
        failing_ci=1,
        open_pulls=[
            _pull(201, draft=True, mergeable=False, mergeable_state="dirty"),
            daily_digest.PullSignal(
                202, "Synthetic update", author="dependabot[bot]", checks="failure"
            ),
            _pull(203, author=OWNER),
            _pull(204, author=OWNER),
        ],
    )
    body = daily_digest.format_slack_digest(
        **_status_digest_kwargs(
            repo_evidence=[ev],
            products=[PRODUCTS[0]],
            customers=[],
            assignments=[],
            tasks=[],
            inbox_items=[],
        )
    )
    assert "investigate failing CI on main" in body
    assert "review for merge: #203, #204" in body
    assert "1 draft PR" in body and "1 dependency PR" in body
    assert "#201" not in body and "#202" not in body


def test_slack_incomplete_evidence_is_not_an_all_clear():
    ev = daily_digest.RepoEvidence(
        repo_id="aiia",
        mounted=False,
        remote=True,
        complete=False,
        open_pulls=[_pull(203, author=OWNER)],
    )
    body = daily_digest.format_slack_digest(
        **_status_digest_kwargs(
            repo_evidence=[ev],
            products=[PRODUCTS[0]],
            customers=[],
            assignments=[],
            tasks=[],
            inbox_items=[],
        )
    )
    assert "status unavailable or incomplete" in body
    assert "No active alerts" not in body
    assert "review for merge" not in body


def test_slack_review_marker_does_not_double_count_an_output():
    roster = [{"id": "one", "repo_id": "aiia", "loop_skip_reason": "awaiting_review"}]
    rows = [
        {
            "agent_id": "one",
            "status": "completed",
            "review_status": "unreviewed",
            "result": "synthetic",
        }
    ]
    body = daily_digest.format_slack_digest(
        **_status_digest_kwargs(
            agents=roster,
            assignments=rows,
            products=[PRODUCTS[0]],
            customers=[],
            inbox_items=[],
            tasks=[],
        )
    )
    assert "review 1 agent output" in body
    assert "review 2 agent outputs" not in body


def test_slack_preserves_footer_and_hides_raw_failure_payload_under_size_pressure():
    products = [
        daily_digest.DigestProduct(id=f"p{n}", name=f"Project {n}", mount=f"p{n}")
        for n in range(80)
    ]
    body = daily_digest.format_slack_digest(
        **_status_digest_kwargs(
            products=products,
            customers=[],
            tasks=[
                {
                    "name": "Daily Brief",
                    "last_status": "failed",
                    "last_error": "secret=synthetic-sensitive-value\nRuntimeError",
                }
            ],
        )
    )
    assert len(body) <= daily_digest.SLACK_BODY_MAX
    assert body.endswith("_Details in Studio_")
    assert "_More items in Studio_" in body
    assert "synthetic-sensitive-value" not in body and "RuntimeError" not in body
    assert "Daily Brief*: failed" in body


def test_slack_keeps_failures_for_agents_missing_from_roster():
    body = daily_digest.format_slack_digest(
        **_status_digest_kwargs(
            agents=[],
            assignments=[
                {"agent_id": "missing", "status": "failed", "completed_at": "2026-10-08T10:00:00Z"}
            ],
            products=[],
            customers=[],
            tasks=[],
            inbox_items=[],
        )
    )
    assert "Inspect 1 other failed agent runs in Studio" in body
