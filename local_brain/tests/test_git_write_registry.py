import subprocess

import pytest

from local_brain.command_center import git_write_registry as write_mod
from local_brain.command_center.git_workspace_registry import GitWorkspaceRegistry
from local_brain.command_center.git_write_registry import GitWriteRegistry, safe_worktree_path


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("# Workspace test\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _assignment():
    return {
        "id": "asg_123",
        "title": "Implement audit view",
        "status": "completed",
    }


def _agent():
    return {
        "id": "agent_123",
        "tools": ["Repository read", "Git workspace"],
        "repo_id": "test",
    }


def _ready_pair(tmp_path, monkeypatch):
    repo = _source_repo(tmp_path)
    monkeypatch.setattr(write_mod, "repo_mount", lambda repo_id: repo)
    from local_brain.command_center import git_workspace_registry as ws_mod

    monkeypatch.setattr(ws_mod, "repo_mount", lambda repo_id: repo)
    monkeypatch.setattr(ws_mod, "repo_write_eligibility", lambda repo_id: (True, ""))
    workspaces = GitWorkspaceRegistry(tmp_path / "ws.json", tmp_path / "worktrees")
    proposal = workspaces.propose(_assignment(), _agent())
    workspace = workspaces.approve(proposal["id"])
    writes = GitWriteRegistry(tmp_path / "writes.json", workspace_registry=workspaces)
    return repo, workspace, writes


def test_propose_requires_ready_workspace(tmp_path, monkeypatch):
    repo = _source_repo(tmp_path)
    monkeypatch.setattr(write_mod, "repo_mount", lambda repo_id: repo)
    from local_brain.command_center import git_workspace_registry as ws_mod

    monkeypatch.setattr(ws_mod, "repo_mount", lambda repo_id: repo)
    monkeypatch.setattr(ws_mod, "repo_write_eligibility", lambda repo_id: (True, ""))
    workspaces = GitWorkspaceRegistry(tmp_path / "ws.json", tmp_path / "worktrees")
    pending = workspaces.propose(_assignment(), _agent())
    writes = GitWriteRegistry(tmp_path / "writes.json", workspace_registry=workspaces)

    with pytest.raises(ValueError, match="workspace_not_ready"):
        writes.propose(
            pending["id"],
            "write_file",
            {"path": "notes.md", "content": "too soon\n"},
        )


def test_path_escape_rejection(tmp_path, monkeypatch):
    _repo, workspace, writes = _ready_pair(tmp_path, monkeypatch)
    worktree = tmp_path / "worktrees" / "test" / workspace["id"]

    with pytest.raises(ValueError, match="path_escape"):
        writes.propose(workspace["id"], "write_file", {"path": "../secret.txt", "content": "x"})
    with pytest.raises(ValueError, match="path_escape"):
        writes.propose(workspace["id"], "write_file", {"path": "/tmp/x", "content": "x"})
    with pytest.raises(ValueError, match="path_escape"):
        writes.propose(
            workspace["id"],
            "write_file",
            {"path": "notes/../../outside.txt", "content": "x"},
        )
    with pytest.raises(ValueError, match="path_escape"):
        safe_worktree_path(worktree, ".git/config")
    assert not (worktree / "secret.txt").exists()


def test_refuse_operating_on_source_checkout(tmp_path, monkeypatch):
    repo = _source_repo(tmp_path)
    monkeypatch.setattr(write_mod, "repo_mount", lambda repo_id: repo)
    workspaces = GitWorkspaceRegistry(tmp_path / "ws.json", tmp_path / "worktrees")
    workspaces.workspaces.append(
        {
            "id": "gws_source",
            "assignment_id": "asg_123",
            "agent_id": "agent_123",
            "repo_id": "test",
            "status": "ready",
            "path": str(repo),
            "events": [],
        }
    )
    writes = GitWriteRegistry(tmp_path / "writes.json", workspace_registry=workspaces)

    with pytest.raises(ValueError, match="refuses_source_checkout"):
        writes.propose(
            "gws_source",
            "write_file",
            {"path": "README.md", "content": "hijacked\n"},
        )
    assert (repo / "README.md").read_text() == "# Workspace test\n"


def test_write_file_is_behind_approve(tmp_path, monkeypatch):
    repo, workspace, writes = _ready_pair(tmp_path, monkeypatch)
    worktree = tmp_path / "worktrees" / "test" / workspace["id"]

    proposal = writes.propose(
        workspace["id"],
        "write_file",
        {"path": "notes.md", "content": "hello from worktree\n"},
    )

    assert proposal["status"] == "pending"
    assert not (worktree / "notes.md").exists()
    assert not (repo / "notes.md").exists()

    completed = writes.approve(proposal["id"])

    assert completed["status"] == "completed"
    assert (worktree / "notes.md").read_text() == "hello from worktree\n"
    assert not (repo / "notes.md").exists()


def test_approve_creates_commit_only_in_worktree(tmp_path, monkeypatch):
    repo, workspace, writes = _ready_pair(tmp_path, monkeypatch)
    worktree = tmp_path / "worktrees" / "test" / workspace["id"]
    source_head = _git(repo, "rev-parse", "HEAD")

    writes.approve(
        writes.propose(
            workspace["id"],
            "write_file",
            {"path": "notes.md", "content": "commit me\n"},
        )["id"]
    )
    proposal = writes.propose(
        workspace["id"],
        "commit",
        {"message": "Add notes", "paths": ["notes.md"]},
    )
    assert proposal["status"] == "pending"
    assert _git(worktree, "rev-parse", "HEAD") == source_head

    completed = writes.approve(proposal["id"])

    assert completed["status"] == "completed"
    assert completed["result"]["sha"]
    assert _git(worktree, "rev-parse", "HEAD") != source_head
    assert _git(worktree, "log", "-1", "--pretty=%s") == "Add notes"
    assert _git(repo, "rev-parse", "HEAD") == source_head
    assert _git(repo, "branch", "--show-current") == "main"
    assert not (repo / "notes.md").exists()


def test_deferred_push_and_open_pr_rejected(tmp_path, monkeypatch):
    _repo, workspace, writes = _ready_pair(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="op_deferred:push"):
        writes.propose(workspace["id"], "push", {})
    with pytest.raises(ValueError, match="op_deferred:open_pr"):
        writes.propose(workspace["id"], "open_pr", {"title": "nope"})
    assert writes.list() == []


def test_run_tests_allowlist_and_execute(tmp_path, monkeypatch):
    _repo, workspace, writes = _ready_pair(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="test_command_not_allowlisted"):
        writes.propose(workspace["id"], "run_tests", {"command": "rm -rf /"})

    writes.approve(
        writes.propose(
            workspace["id"],
            "write_file",
            {
                "path": "test_ok.py",
                "content": "def test_ok():\n    assert True\n",
            },
        )["id"]
    )
    proposal = writes.propose(
        workspace["id"],
        "run_tests",
        {"command": "pytest -q"},
    )
    completed = writes.approve(proposal["id"])

    assert completed["status"] == "completed"
    assert completed["result"]["command"] == "pytest -q"
    assert completed["result"]["ok"] is True
    assert completed["result"]["returncode"] == 0
