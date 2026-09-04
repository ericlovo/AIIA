import subprocess

import pytest

from local_brain.command_center import git_workspace_registry
from local_brain.command_center.git_workspace_registry import GitWorkspaceRegistry


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


def test_proposal_does_not_create_branch_or_worktree(tmp_path, monkeypatch):
    repo = _source_repo(tmp_path)
    monkeypatch.setattr(git_workspace_registry, "repo_mount", lambda repo_id: repo)
    monkeypatch.setattr(
        git_workspace_registry, "repo_write_eligibility", lambda repo_id: (True, "")
    )
    registry = GitWorkspaceRegistry(tmp_path / "data.json", tmp_path / "worktrees")

    workspace = registry.propose(_assignment(), _agent())

    assert workspace["status"] == "pending"
    assert workspace["path"] == ""
    assert _git(repo, "branch", "--list", workspace["branch"]) == ""
    assert not (tmp_path / "worktrees").exists()


def test_approval_creates_isolated_worktree_and_preserves_source(tmp_path, monkeypatch):
    repo = _source_repo(tmp_path)
    (repo / "README.md").write_text("local source change\n")
    monkeypatch.setattr(git_workspace_registry, "repo_mount", lambda repo_id: repo)
    monkeypatch.setattr(
        git_workspace_registry, "repo_write_eligibility", lambda repo_id: (True, "")
    )
    registry = GitWorkspaceRegistry(tmp_path / "data.json", tmp_path / "worktrees")
    proposal = registry.propose(_assignment(), _agent())

    workspace = registry.approve(proposal["id"])

    worktree = tmp_path / "worktrees" / "test" / proposal["id"]
    assert workspace["status"] == "ready"
    assert workspace["path"] == str(worktree)
    assert _git(worktree, "branch", "--show-current") == workspace["branch"]
    assert _git(repo, "branch", "--show-current") == "main"
    assert (repo / "README.md").read_text() == "local source change\n"
    assert "README.md" in _git(repo, "status", "--short")
    assert workspace["git_status"].startswith(f"## {workspace['branch']}")


def test_proposal_requires_git_workspace_tool(tmp_path, monkeypatch):
    repo = _source_repo(tmp_path)
    monkeypatch.setattr(git_workspace_registry, "repo_mount", lambda repo_id: repo)
    monkeypatch.setattr(
        git_workspace_registry, "repo_write_eligibility", lambda repo_id: (True, "")
    )
    registry = GitWorkspaceRegistry(tmp_path / "data.json", tmp_path / "worktrees")
    agent = {**_agent(), "tools": ["Repository read"]}

    with pytest.raises(ValueError, match="git_workspace_tool_required"):
        registry.propose(_assignment(), agent)
