import json
import subprocess

from local_brain.command_center import repository_tools


def _git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def test_repo_snapshot_contains_bounded_git_context(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("# Test repository\n")
    (repo / "app.py").write_text("print('one')\n")
    _git(repo, "add", "README.md", "app.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "remote", "add", "origin", "https://github.com/example/project.git")
    (repo / "app.py").write_text("print('two')\n")

    monkeypatch.setattr(repository_tools, "REPO_MOUNTS", {"test": repo})
    monkeypatch.setattr(repository_tools, "REPO_NAMES", {"test": "Test"})

    snapshot = repository_tools.repo_snapshot("test")

    assert "Access mode: read only" in snapshot
    assert "GitHub remote: example/project" in snapshot
    assert "app.py" in snapshot
    assert "README context:\n# Test repository" in snapshot


def test_github_snapshot_uses_only_get_requests(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(repository_tools, "REPO_MOUNTS", {"test": repo})
    monkeypatch.setattr(repository_tools, "_origin_slug", lambda path: "example/project")
    calls = []

    def fake_api(endpoint):
        calls.append(endpoint)
        if endpoint == "repos/example/project":
            return {"private": True, "default_branch": "main"}
        if "/pulls?" in endpoint:
            return [{"number": 7, "title": "Ready to review"}]
        if "/issues?" in endpoint:
            return [{"number": 9, "title": "Known issue"}]
        return {"workflow_runs": [{"name": "CI", "status": "completed", "conclusion": "success"}]}

    monkeypatch.setattr(repository_tools, "_github_api", fake_api)

    snapshot = repository_tools.github_snapshot("test")

    assert calls == [
        "repos/example/project",
        "repos/example/project/pulls?state=open&per_page=5",
        "repos/example/project/issues?state=open&per_page=10",
        "repos/example/project/actions/runs?per_page=5",
    ]
    assert "Access mode: read-only GET adapter" in snapshot
    assert "#7 Ready to review" in snapshot
    assert "CI: completed/success" in snapshot


def test_github_api_command_is_hard_coded_to_get(monkeypatch):
    captured = []

    def fake_run(args, timeout=5.0):
        captured.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps({"login": "eric"}), "")

    monkeypatch.setattr(repository_tools.shutil, "which", lambda name: "/opt/homebrew/bin/gh")
    monkeypatch.setattr(repository_tools, "_run", fake_run)

    assert repository_tools.github_status()["status"] == "connected"
    assert captured[0][1:4] == ["api", "--method", "GET"]


def test_duplicate_remote_is_not_git_workspace_eligible(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for repo in (first, second):
        repo.mkdir()
        _git(repo, "init", "-b", "main")
        _git(repo, "remote", "add", "origin", "https://github.com/example/shared.git")

    monkeypatch.setattr(repository_tools, "REPO_MOUNTS", {"first": first, "second": second})

    assert repository_tools.repo_write_eligibility("first") == (
        False,
        "ambiguous_github_remote",
    )
