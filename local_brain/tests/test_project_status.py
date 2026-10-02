import copy
import subprocess
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center import project_status, repository_tools

SHA = "a" * 40
DATE = "2026-10-02T12:00:00Z"


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    path = tmp_path / "project"
    path.mkdir()
    git(path, "init", "-b", "feature/current")
    git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "fixture",
    )
    git(path, "remote", "add", "origin", "https://github.com/example/project.git")
    monkeypatch.setattr(repository_tools, "REPO_MOUNTS", {"test": path})
    monkeypatch.setattr(repository_tools, "REPO_NAMES", {"test": "Test project"})
    monkeypatch.setattr(
        repository_tools,
        "_github_api",
        lambda *args, **kwargs: pytest.fail("Unexpected GitHub request"),
    )
    return path


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(project_status.router)

    @app.get("/event-loop-thread")
    async def event_loop_thread():
        return threading.get_ident()

    with TestClient(app) as client:
        yield client


@pytest.fixture
def run():
    return {
        "id": 123,
        "name": "CI",
        "html_url": "https://github.com/example/project/actions/runs/123",
        "head_sha": SHA,
        "head_branch": "feature/remote",
        "status": "completed",
        "conclusion": "success",
        "created_at": DATE,
        "updated_at": DATE,
    }


def mock_runs(monkeypatch, runs, total=None):
    calls = []

    def api(endpoint, timeout):
        calls.append((endpoint, timeout))
        return {
            "workflow_runs": runs,
            "total_count": len(runs) if total is None else total,
        }

    monkeypatch.setattr(repository_tools, "_github_api", api)
    return calls


def test_metadata_uses_checkout_without_github(client, mounted):
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    project = response.json()["projects"][0]
    assert project["id"] == "test"
    assert project["branch"] == "feature/current"
    assert project["head_sha"] == git(mounted, "rev-parse", "HEAD")
    assert project["checkout_status"] == "available"
    assert project["detached"] is False
    assert project["checked_at"]
    assert project["errors"] == []
    assert project["deployment"] == {"status": "unknown", "connection": "not_connected"}


def test_detached_checkout_is_not_a_branch(client, mounted):
    git(mounted, "checkout", "--detach")
    project = client.get("/api/projects").json()["projects"][0]
    assert project["detached"] is True
    assert project["branch"] is None
    assert project["head_sha"] == git(mounted, "rev-parse", "HEAD")


def test_unborn_checkout_does_not_claim_revision(client, mounted):
    git(mounted, "checkout", "--orphan", "unborn")
    project = client.get("/api/projects").json()["projects"][0]
    assert project["checkout_status"] == "unavailable"
    assert project["head_sha"] is None
    assert project["branch"] is None
    assert project["errors"][0]["code"] == "checkout_unavailable"


@pytest.mark.parametrize("revision", [None, "", "unavailable", "bad\nmain", SHA, f"{SHA}\n"])
def test_failed_or_malformed_git_read_is_unavailable(client, mounted, monkeypatch, revision):
    monkeypatch.setattr(repository_tools, "_git_checked", lambda *args: revision)
    project = client.get("/api/projects").json()["projects"][0]
    assert project["checkout_status"] == "unavailable"
    assert project["head_sha"] is None


def test_missing_mount_is_explicit(client, mounted, monkeypatch, tmp_path):
    monkeypatch.setattr(repository_tools, "REPO_MOUNTS", {"missing": tmp_path / "absent"})
    project = client.get("/api/projects").json()["projects"][0]
    assert project["mounted"] is False
    assert project["errors"][0]["code"] == "repository_not_mounted"
    ci = client.get("/api/projects/missing/ci").json()
    assert ci["status"] == "unavailable"
    assert ci["fetched_at"] is None
    assert ci["total_count"] is None


@pytest.mark.parametrize("repo_id", ["unknown", "https:evil", "%2e%2e"])
def test_unknown_repo_never_reads_or_fetches(client, mounted, monkeypatch, repo_id):
    monkeypatch.setattr(
        repository_tools,
        "_git_checked",
        lambda *args: pytest.fail("Unexpected git read"),
    )
    assert client.get(f"/api/projects/{repo_id}/ci").status_code == 404


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "https://evil.test/repo",
        "../project",
        "example/..",
        "example/repo?redirect=bad",
    ],
)
def test_origin_allowlist(client, mounted, monkeypatch, origin):
    monkeypatch.setattr(repository_tools, "_origin_slug", lambda path: origin)
    ci = client.get("/api/projects/test/ci").json()
    assert ci["status"] == "unavailable"
    assert ci["source_url"] is None
    assert ci["errors"][0]["code"] == "github_origin_unavailable"


def test_ci_retains_evidence_without_replacing_checkout(client, mounted, monkeypatch, run):
    calls = mock_runs(monkeypatch, [run])
    response = client.get("/api/projects/test/ci")
    ci = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert ci["status"] == "available"
    assert ci["scope"] == "repository_recent_runs"
    assert ci["project"]["branch"] == "feature/current"
    assert ci["project"]["head_sha"] != SHA
    assert ci["runs"][0] == {**run, "fetched_at": ci["fetched_at"]}
    assert ci["source_url"] == "https://github.com/example/project/actions"
    assert ci["errors"] == []
    assert calls == [("repos/example/project/actions/runs?per_page=20&page=1", 8.0)]
    assert ci["project"]["deployment"]["status"] == "unknown"


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("queued", None),
        ("in_progress", None),
        ("waiting", None),
        ("pending", None),
        ("completed", "success"),
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("completed", "skipped"),
        ("completed", "neutral"),
        ("completed", "timed_out"),
        ("completed", None),
        ("future_status", None),
    ],
)
def test_run_states_are_evidence_not_aggregate_health(
    client, mounted, monkeypatch, run, status, conclusion
):
    run.update(status=status, conclusion=conclusion)
    mock_runs(monkeypatch, [run])
    ci = client.get("/api/projects/test/ci").json()
    assert ci["status"] == "available"
    assert ci["runs"][0]["status"] == status
    assert ci["runs"][0]["conclusion"] == conclusion
    assert "passing" not in ci and "health" not in ci


def test_empty_ci_is_available_but_not_passing(client, mounted, monkeypatch):
    mock_runs(monkeypatch, [])
    ci = client.get("/api/projects/test/ci").json()
    assert ci["status"] == "available"
    assert ci["runs"] == []
    assert ci["total_count"] == 0
    assert ci["has_more"] is False


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"workflow_runs": None},
        {"workflow_runs": []},
        {"workflow_runs": [], "total_count": -1},
        {"workflow_runs": [], "total_count": True},
        {"workflow_runs": [], "total_count": 1},
    ],
)
def test_malformed_envelope_is_unavailable(client, mounted, monkeypatch, payload):
    monkeypatch.setattr(repository_tools, "_github_api", lambda *args, **kwargs: payload)
    ci = client.get("/api/projects/test/ci").json()
    assert ci["status"] == "unavailable"
    assert ci["fetched_at"] is None
    assert ci["errors"][0]["code"] == "github_api_invalid_response"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", True),
        ("id", "123"),
        ("head_sha", "bad"),
        ("status", None),
        ("status", "queued"),
        ("conclusion", {}),
        ("head_branch", []),
        ("html_url", "javascript:alert(1)"),
        ("html_url", "https://github.com/other/repo/actions/runs/123"),
        ("html_url", "https://github.com/example/project/actions/runs/999"),
        ("created_at", "yesterday"),
        ("updated_at", "2026-10-02T12:00:00"),
    ],
)
def test_malformed_run_invalidates_page_not_silently_dropped(
    client, mounted, monkeypatch, run, key, value
):
    broken = {**run, key: value}
    mock_runs(monkeypatch, [run, broken])
    ci = client.get("/api/projects/test/ci").json()
    assert ci["status"] == "unavailable"
    assert ci["runs"] == []
    assert ci["errors"][0]["code"] == "github_api_invalid_response"


def test_missing_field_and_duplicate_run_are_unavailable(client, mounted, monkeypatch, run):
    for runs in [
        [{k: v for k, v in run.items() if k != "conclusion"}],
        [run, run],
        [None],
    ]:
        mock_runs(monkeypatch, runs)
        ci = client.get("/api/projects/test/ci").json()
        assert ci["status"] == "unavailable"


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (RuntimeError("github_cli_missing"), "github_cli_missing"),
        (RuntimeError("github_api_unavailable"), "github_api_unavailable"),
        (RuntimeError("github_api_invalid_response"), "github_api_invalid_response"),
        (RuntimeError("private stderr"), "github_api_unavailable"),
        (OSError("private path"), "github_api_unavailable"),
        (subprocess.TimeoutExpired("gh", 8), "github_api_timeout"),
    ],
)
def test_api_failures_are_actionable_and_sanitized(client, mounted, monkeypatch, failure, code):
    def api(*args, **kwargs):
        raise failure

    monkeypatch.setattr(repository_tools, "_github_api", api)
    ci = client.get("/api/projects/test/ci").json()
    assert ci["status"] == "unavailable"
    assert ci["attempted_at"]
    assert ci["fetched_at"] is None
    assert ci["errors"][0]["code"] == code
    assert "refresh" in ci["errors"][0]["message"].lower()
    assert "private" not in str(ci["errors"])


def test_only_one_bounded_page_and_no_logs(client, mounted, monkeypatch, run):
    runs = []
    for number in range(1, 30):
        item = copy.deepcopy(run)
        item.update(
            id=number,
            html_url=f"https://github.com/example/project/actions/runs/{number}",
        )
        runs.append(item)
    calls = mock_runs(monkeypatch, runs, total=100)
    ci = client.get("/api/projects/test/ci").json()
    assert len(ci["runs"]) == 20
    assert ci["has_more"] is True
    assert len(calls) == 1


def test_blocking_reads_run_off_event_loop(client, mounted, monkeypatch):
    thread_ids = []
    original = repository_tools._git_checked

    def read(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        return original(*args, **kwargs)

    def api(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        return {"workflow_runs": [], "total_count": 0}

    monkeypatch.setattr(repository_tools, "_git_checked", read)
    monkeypatch.setattr(repository_tools, "_github_api", api)
    loop_thread = client.get("/event-loop-thread").json()
    client.get("/api/projects")
    client.get("/api/projects/test/ci")
    assert len(thread_ids) == 3
    assert all(thread != loop_thread for thread in thread_ids)


def test_routes_do_not_accept_mutations(client, mounted):
    for method in ["post", "put", "patch", "delete"]:
        for path in ["/api/projects", "/api/projects/test/ci"]:
            assert getattr(client, method)(path).status_code == 405
