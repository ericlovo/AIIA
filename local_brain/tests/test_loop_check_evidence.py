"""Scheduled loops classify on evidence the check produced, never on model prose.

- Verified no change: every read succeeded and saw the last run's inputs. History
  only; no attention item, no model run, no review verdict.
- Changed inputs: one reviewable item per observed state.
- Incomplete check: one deduplicated failure item; never an all-clear.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from local_brain.command_center import repository_tools as tools
from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.command_center.repository_tools import Observation
from local_brain.tests.test_agent_execute import _create_agent, _studio


def _repo_agent(agents, **overrides):
    payload = {
        "tools": ["Repository read"],
        "repo_id": "test",
        "loop_enabled": True,
        "loop_task": "Inspect current regression risk.",
        **overrides,
    }
    return _create_agent(agents, **payload)


def _checks(agents):
    return agents.recover_runs().loop_checks()


async def test_incomplete_check_surfaces_one_failure_and_runs_no_model(tmp_path, monkeypatch):
    cc, agents, assignments, events, fake = _studio(tmp_path, monkeypatch, content="All clear.")
    monkeypatch.setattr(
        cc,
        "observe_repository",
        lambda repo_id: Observation("unreadable", False, ("git_status_failed",)),
    )
    agent = _repo_agent(agents)
    agents.record_loop_input(agent["id"], "last-verified-state")

    first = await cc._run_scheduled_agent(agent)
    second = await cc._run_scheduled_agent(agent)

    assert first["reason"] == second["reason"] == "check_incomplete"
    assert fake.posts == []  # no model ran, so nothing could report an all-clear
    items = assignments.list_assignments()
    assert len(items) == 1  # deduplicated across repeats
    failure = items[0]
    assert failure["source_kind"] == "loop_check"
    assert failure["status"] == "failed"
    assert "git_status_failed" in failure["error"]
    assert failure["occurrences"] == 2
    assert failure["review_status"] == "unreviewed"
    assert assignments.pending_loop_reviews(agent["id"]) == 0
    updated = agents.get(agent["id"])
    assert updated["loop_skip_reason"] == "check_incomplete"
    assert updated["loop_input_hash"] == "last-verified-state"  # not marked verified
    assert updated["loop_runs_today"] == 0
    assert [c["outcome"] for c in _checks(agents)] == ["check_incomplete", "check_incomplete"]
    assert _checks(agents)[0]["failures"] == ["git_status_failed"]
    assert any(e == ("assignment", "created") for e in [(x, y) for x, y, _ in events])


async def test_incomplete_check_never_runs_queued_work(tmp_path, monkeypatch):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="Fine.")
    monkeypatch.setattr(
        cc, "observe_repository", lambda repo_id: Observation("x", False, ("git_log_failed",))
    )
    agent = _repo_agent(agents)
    queued, _ = assignments.create_scheduled_assignment(
        agent_id=agent["id"],
        agent_name=agent["name"],
        objective=agent["loop_task"],
        schedule_key="survived-restart",
        interval_minutes=60,
    )

    result = await cc._run_scheduled_agent(agent)

    assert result["reason"] == "check_incomplete"
    assert assignments.get_assignment(queued["id"])["status"] == "queued"
    assert fake.posts == []


async def test_failure_item_reopens_only_after_dismissal(tmp_path, monkeypatch):
    cc, agents, assignments, _events, _fake = _studio(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cc, "observe_repository", lambda repo_id: Observation("x", False, ("git_diff_failed",))
    )
    agent = _repo_agent(agents)
    await cc._run_scheduled_agent(agent)
    first = assignments.list_assignments()[0]
    assignments.dismiss_assignment(
        first["id"], dismissed=True, expected_version=first["review_version"]
    )

    await cc._run_scheduled_agent(agent)

    failures = [a for a in assignments.list_assignments() if a["source_kind"] == "loop_check"]
    assert len(failures) == 2
    assert sum(1 for a in failures if not a["dismissed_at"]) == 1


async def test_verified_unchanged_is_history_only(tmp_path, monkeypatch):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="Report.")
    monkeypatch.setattr(
        cc, "observe_repository", lambda repo_id: Observation("commit abc123, clean", True)
    )
    agent = _repo_agent(agents)

    await cc._run_scheduled_agent(agent)
    skipped = await cc._run_scheduled_agent(agent)

    assert skipped["reason"] == "unchanged_repository_input"
    items = assignments.list_assignments()
    assert len(items) == 1
    assert len(fake.posts) == 1
    # The earlier output still awaits a person: a quiet check is not a verdict.
    assert items[0]["review_status"] == "unreviewed"
    assert items[0]["reviewed_at"] is None
    history = _checks(agents)
    assert [c["outcome"] for c in history] == ["verified_unchanged"]
    assert history[0]["fingerprint"] == agents.get(agent["id"])["loop_input_hash"]
    # Checks are not runs: run counts and token usage describe inference only.
    assert agents.recover_runs().activity()["total"] == 1


@pytest.mark.parametrize("observation_kind", ["unchanged", "changed", "incomplete"])
async def test_full_review_queue_preserves_check_evidence(tmp_path, monkeypatch, observation_kind):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="Report.")
    observed = {"value": Observation("tree", True)}
    monkeypatch.setattr(cc, "observe_repository", lambda repo_id: observed["value"])
    monkeypatch.setattr(cc, "MAX_PENDING_LOOP_REVIEWS", 1)
    agent = _repo_agent(agents)
    await cc._run_scheduled_agent(agent)
    original = assignments.list_assignments()[0].copy()
    baseline = agents.get(agent["id"])["loop_input_hash"]
    if observation_kind == "changed":
        observed["value"] = Observation("new commit", True)
    elif observation_kind == "incomplete":
        observed["value"] = Observation("unreadable", False, ("git_status_failed",))

    result = await cc._run_scheduled_agent(agents.get(agent["id"]))

    expected = {
        "unchanged": "unchanged_repository_input",
        "changed": "awaiting_review",
        "incomplete": "check_incomplete",
    }
    assert result["reason"] == expected[observation_kind]
    assert len(fake.posts) == 1
    assert assignments.get_assignment(original["id"]) == original
    assert agents.get(agent["id"])["loop_input_hash"] == baseline
    checks = _checks(agents)
    if observation_kind == "unchanged":
        assert [check["outcome"] for check in checks] == ["verified_unchanged"]
        assert len(assignments.list_assignments()) == 1
    elif observation_kind == "changed":
        assert checks == []
        assert result["pending_reviews"] == 1
        assert len(assignments.list_assignments()) == 1
    else:
        assert [check["outcome"] for check in checks] == ["check_incomplete"]
        assert result["assignment"]["status"] == "failed"
        assert len(assignments.list_assignments()) == 2


async def test_github_reads_join_the_fingerprint(tmp_path, monkeypatch):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="CI report.")
    remote = {"text": "runs: ci success"}
    monkeypatch.setattr(cc, "observe_repository", lambda repo_id: Observation("tree", True))
    monkeypatch.setattr(cc, "observe_github", lambda repo_id: Observation(remote["text"], True))
    agent = _repo_agent(agents, tools=["Repository read", "GitHub read"])

    await cc._run_scheduled_agent(agent)
    await cc._run_scheduled_agent(agent)
    assert len(fake.posts) == 1  # unchanged CI state: verified, no second item

    remote["text"] = "runs: ci failure"
    await cc._run_scheduled_agent(agent)
    assert len(fake.posts) == 2
    assert len(assignments.list_assignments()) == 2


async def test_unobservable_agents_keep_running_and_producing_reviews(tmp_path, monkeypatch):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="Notes.")
    agent = _create_agent(agents, loop_enabled=True, loop_task="Think.")

    assert cc._observe_scheduled_inputs(agent) is None
    await cc._run_scheduled_agent(agent)
    assert len(fake.posts) == 1
    assert assignments.list_assignments()[0]["review_status"] == "unreviewed"


def test_one_observed_state_yields_one_item(tmp_path):
    registry = AssignmentRegistry(tmp_path / "assignments.json")
    kwargs = dict(agent_id="agt", agent_name="Watch", objective="x", interval_minutes=60)
    first, created = registry.create_scheduled_assignment(
        schedule_key="w1", observed_fingerprint="state-a", **kwargs
    )
    registry.set_running(first["id"])
    registry.finish_assignment(first["id"], result="report")
    again, created_again = registry.create_scheduled_assignment(
        schedule_key="w2", observed_fingerprint="state-a", **kwargs
    )
    other, created_other = registry.create_scheduled_assignment(
        schedule_key="w3", observed_fingerprint="state-b", **kwargs
    )

    assert created and not created_again and created_other
    assert again["id"] == first["id"]
    assert other["id"] != first["id"]


async def test_full_registry_is_reported_not_silently_evicted(tmp_path, monkeypatch):
    from local_brain.command_center import assignment_registry as module

    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="Report.")
    monkeypatch.setattr(module, "MAX_ASSIGNMENTS", 2)
    for index in range(2):
        item = assignments.create_assignment(title=f"t{index}", objective="o", agent_id="agt")
        assignments.set_running(item["id"])
        assignments.finish_assignment(item["id"], result="awaiting a person")
    agent = _create_agent(agents, loop_enabled=True, loop_task="Think.")

    result = await cc._run_scheduled_agent(agent)

    assert result["reason"] == "assignment_capacity_reached"
    assert len(assignments.list_assignments()) == 2  # both unreviewed items kept
    assert fake.posts == []
    assert agents.get(agent["id"])["loop_skip_reason"] == "assignment_capacity_reached"


@pytest.fixture()
def mounted_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *a],
        check=True,
        capture_output=True,
        env={**env, "PATH": "/usr/bin:/bin"},
    )
    run("init", "-q")
    (repo / "README.md").write_text("hello")
    run("add", ".")
    run("commit", "-qm", "init")
    monkeypatch.setitem(tools.REPO_MOUNTS, "test", repo)
    return repo


def test_complete_repository_observation(mounted_repo: Path):
    observation = tools.observe_repository("test")
    assert observation.complete
    assert observation.failures == ()
    assert "init" in observation.text


def test_failed_git_read_is_incomplete_not_clean(mounted_repo: Path, monkeypatch):
    real = tools._run

    def failing_status(args, timeout=5.0):
        if "status" in args:
            return subprocess.CompletedProcess(args, 128, "", "fatal: unsafe repository")
        return real(args, timeout=timeout)

    monkeypatch.setattr(tools, "_run", failing_status)
    observation = tools.observe_repository("test")

    assert not observation.complete
    assert observation.failures == ("git_status_failed",)
    assert "Git status:\nunavailable (read failed)" in observation.text
    assert "Git status:\nclean" not in observation.text


@pytest.mark.parametrize(
    "api",
    [
        lambda endpoint, timeout=8.0: (_ for _ in ()).throw(RuntimeError("github_api_unavailable")),
        lambda endpoint, timeout=8.0: {"not": "a list"},
    ],
)
def test_github_failures_are_incomplete(mounted_repo: Path, monkeypatch, api):
    monkeypatch.setattr(tools, "_origin_slug", lambda path: "owner/repo")
    monkeypatch.setattr(tools, "_github_api", api)
    observation = tools.observe_github("test")
    assert not observation.complete
    assert observation.failures
    assert "disconnected" in observation.text


def test_unmounted_repository_is_incomplete():
    assert not tools.observe_repository("no-such-repo").complete
    assert not tools.observe_github("no-such-repo").complete


async def test_loop_checks_endpoint_lists_history(tmp_path, monkeypatch):
    cc, agents, _assignments, _events, _fake = _studio(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cc, "observe_repository", lambda repo_id: Observation("x", False, ("git_log_failed",))
    )
    agent = _repo_agent(agents)
    await cc._run_scheduled_agent(agent)

    body = await cc.studio_loop_checks(agent_id=agent["id"])
    assert [c["outcome"] for c in body["checks"]] == ["check_incomplete"]
    assert body["checks"][0]["failures"] == ["git_log_failed"]
    with pytest.raises(cc.HTTPException):
        await cc.studio_loop_checks(limit=0)


def test_audit_script_is_read_only(tmp_path, capsys, monkeypatch):
    import sys

    from local_brain.command_center.agent_registry import AgentRegistry
    from local_brain.scripts import loop_review_audit

    agents = AgentRegistry(tmp_path / "agent_data.json")
    registry = AssignmentRegistry(tmp_path / "assignment_data.json")
    agent = _create_agent(agents, loop_enabled=True, loop_task="Watch.")
    item, _ = registry.create_scheduled_assignment(
        agent_id=agent["id"],
        agent_name=agent["name"],
        objective="Watch.",
        schedule_key="w1",
        interval_minutes=60,
    )
    registry.set_running(item["id"])
    registry.finish_assignment(item["id"], result="private run output")
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}

    monkeypatch.setattr(sys, "argv", ["audit", "--data-dir", str(tmp_path)])
    assert loop_review_audit.main() == 0

    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.json")} == before
    out = capsys.readouterr().out
    assert "pending review (guard's count) = 1" in out
    assert "private run output" not in out  # counts and hashes only, never output


MINI_LOOP_TOOLS = ["GitHub read", "Local memory", "Repository read"]  # both live loops


def _observe_all(cc, monkeypatch, memory):
    monkeypatch.setattr(cc, "observe_repository", lambda repo_id: Observation("tree", True))
    monkeypatch.setattr(cc, "observe_github", lambda repo_id: Observation("ci green", True))

    async def fetch(agent):
        return memory["text"]

    monkeypatch.setattr(cc, "_local_memory_context", fetch)


async def test_live_loop_tool_set_is_observable(tmp_path, monkeypatch):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="Report.")
    memory = {"text": "Local memory retrieved for this run (1 entries). - [m1] fact"}
    _observe_all(cc, monkeypatch, memory)
    agent = _repo_agent(agents, tools=MINI_LOOP_TOOLS)

    assert cc._observe_scheduled_inputs(agent, memory["text"]) is not None
    await cc._run_scheduled_agent(agent)
    skipped = await cc._run_scheduled_agent(agent)
    assert skipped["reason"] == "unchanged_repository_input"
    assert len(fake.posts) == 1

    memory["text"] += "\n- [m2] new fact"
    await cc._run_scheduled_agent(agent)
    assert len(fake.posts) == 2  # a memory change is an input change
    assert len(assignments.list_assignments()) == 2


async def test_failed_memory_fetch_is_an_incomplete_check(tmp_path, monkeypatch):
    cc, agents, assignments, _events, fake = _studio(tmp_path, monkeypatch, content="All good.")
    _observe_all(cc, monkeypatch, {"text": cc.LOCAL_MEMORY_UNAVAILABLE})
    agent = _repo_agent(agents, tools=MINI_LOOP_TOOLS)

    result = await cc._run_scheduled_agent(agent)

    assert result["reason"] == "check_incomplete"
    assert result["failures"] == ["local_memory_unavailable"]
    assert fake.posts == []
    assert assignments.list_assignments()[0]["source_kind"] == "loop_check"
