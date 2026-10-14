from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center import project_delivery as delivery
from local_brain.scripts.roadmap_store import RoadmapStore

SHA = "a" * 40


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = RoadmapStore(str(tmp_path))
    story = store.create(title="Feature", product="one")
    other = store.create(title="Elsewhere", product="two", dedup=False)
    items = {
        "asg_1": {
            "id": "asg_1",
            "agent_id": "agent",
            "status": "completed",
            "review_status": "accepted",
        }
    }
    assignments = SimpleNamespace(
        get_assignment=items.get, list_assignments=lambda: list(items.values())
    )
    agents = SimpleNamespace(get=lambda _: {"name": "Worker", "status": "idle", "repo_id": "two"})
    monkeypatch.setattr(delivery.repos, "REPO_MOUNTS", {"one": tmp_path, "two": tmp_path})
    monkeypatch.setattr(delivery, "_project", lambda _: {"github_repo": "owner/repo"})
    monkeypatch.setattr(
        delivery, "collect_deployments", lambda _: {"status": "available", "environments": []}
    )
    app = FastAPI()
    app.include_router(delivery.create_router(store, assignments, agents))
    return TestClient(app), store, story, other, items


def test_scope_links_and_agent_reassignment(setup):
    client, store, story, other, _ = setup
    url = f"/api/projects/one/features/{story['id']}/links"
    assert (
        client.put(
            url,
            json={
                "expected_updated_at": story["updated_at"],
                "assignment_ids": ["asg_1"],
                "pr_numbers": [],
            },
        ).status_code
        == 200
    )
    feature = client.get("/api/projects/one/delivery").json()["features"][0]
    assert feature["assignments"][0]["agent_name"] == "Worker"
    assert feature["assignments"][0]["review_status"] == "accepted"
    assert feature["product"] == "one"
    assert (
        client.put(
            f"/api/projects/one/features/{other['id']}/links",
            json={
                "expected_updated_at": story["updated_at"],
                "assignment_ids": [],
                "pr_numbers": [],
            },
        ).status_code
        == 404
    )
    assert store.get(other["id"]).get("assignment_ids") is None


@pytest.mark.parametrize(
    "body",
    [
        {"assignment_ids": ["missing"], "pr_numbers": []},
        {"assignment_ids": [], "pr_numbers": [True]},
        {"assignment_ids": [], "pr_numbers": [-1]},
        {"assignment_ids": [123], "pr_numbers": []},
    ],
)
def test_invalid_links(setup, body):
    client, _, story, _, _ = setup
    assert (
        client.put(
            f"/api/projects/one/features/{story['id']}/links",
            json={**body, "expected_updated_at": story["updated_at"]},
        ).status_code
        == 422
    )


def test_missing_reference_and_wrong_project_assignment(setup):
    client, store, story, _, items = setup
    store.update_delivery_links(story["id"], "one", ["missing"], [])
    data = client.get("/api/projects/one/delivery").json()
    assert data["status"] == "partial"
    assert data["features"][0]["assignments"] == [{"id": "missing", "status": "missing"}]
    items["asg_1"]["repo_id"] = "two"
    assert (
        client.put(
            f"/api/projects/one/features/{story['id']}/links",
            json={
                "expected_updated_at": story["updated_at"],
                "assignment_ids": ["asg_1"],
                "pr_numbers": [],
            },
        ).status_code
        == 422
    )


def test_atomic_failure_rolls_back(setup, monkeypatch):
    client, store, story, _, _ = setup
    before = store._file.read_text()
    monkeypatch.setattr(
        "local_brain.scripts.roadmap_store.os.replace", Mock(side_effect=OSError("disk"))
    )
    response = client.put(
        f"/api/projects/one/features/{story['id']}/links",
        json={
            "expected_updated_at": story["updated_at"],
            "assignment_ids": ["asg_1"],
            "pr_numbers": [1],
        },
    )
    assert response.status_code == 503
    assert "assignment_ids" not in store.get(story["id"])
    assert store._file.read_text() == before


def pr():
    return {
        "number": 1,
        "state": "open",
        "merged": False,
        "draft": False,
        "title": "PR",
        "head": {"sha": SHA},
        "html_url": "https://github.com/owner/repo/pull/1",
    }


def run(sha=SHA):
    return {
        "id": 2,
        "name": "CI",
        "html_url": "https://github.com/owner/repo/actions/runs/2",
        "head_sha": sha,
        "head_branch": "topic",
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def test_exact_sha_evidence(setup, monkeypatch):
    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", [], [1])
    api = Mock(side_effect=[pr(), {"workflow_runs": [run()], "total_count": 1}])
    monkeypatch.setattr(delivery.repos, "_github_api", api)
    result = client.get("/api/projects/one/delivery").json()
    ci = result["features"][0]["pull_requests"][0]["ci"]
    assert ci["head_sha"] == SHA and ci["status"] == "available"
    assert f"head_sha={SHA}" in api.call_args_list[1].args[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"workflow_runs": [run("b" * 40)], "total_count": 1},
        None,
        {"workflow_runs": [], "total_count": "bad"},
    ],
)
def test_malformed_or_wrong_sha_keeps_local_data(setup, monkeypatch, payload):
    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", ["asg_1"], [1])
    monkeypatch.setattr(delivery.repos, "_github_api", Mock(side_effect=[pr(), payload]))
    result = client.get("/api/projects/one/delivery").json()
    assert result["status"] == "partial"
    assert result["features"][0]["assignments"][0]["id"] == "asg_1"
    assert result["features"][0]["pull_requests"][0]["ci"]["status"] == "unavailable"


def test_remote_budget_and_dedup(setup, monkeypatch):
    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", [], list(range(1, 12)))
    api = Mock(side_effect=RuntimeError("github_api_unavailable"))
    monkeypatch.setattr(delivery.repos, "_github_api", api)
    result = client.get("/api/projects/one/delivery").json()
    assert result["status"] == "partial"
    assert api.call_count == delivery.MAX_PRS
    assert len(result["features"][0]["pull_requests"]) == 11


def test_timeout_retains_local_evidence(setup, monkeypatch):
    import subprocess

    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", ["asg_1"], [1])
    monkeypatch.setattr(
        delivery.repos, "_github_api", Mock(side_effect=subprocess.TimeoutExpired("gh", 4))
    )
    result = client.get("/api/projects/one/delivery").json()
    assert result["status"] == "partial"
    assert result["features"][0]["assignments"][0]["id"] == "asg_1"


def test_stale_link_write_is_rejected(setup):
    client, store, story, _, _ = setup
    url = f"/api/projects/one/features/{story['id']}/links"
    body = {
        "expected_updated_at": story["updated_at"],
        "assignment_ids": ["asg_1"],
        "pr_numbers": [],
    }
    assert client.put(url, json=body).status_code == 200
    assert client.put(url, json={**body, "assignment_ids": []}).status_code == 409
    assert store.get(story["id"])["assignment_ids"] == ["asg_1"]


def test_corrupt_store_is_unavailable(tmp_path, monkeypatch):
    (tmp_path / "stories.json").write_text("{broken")
    store = RoadmapStore(str(tmp_path))
    assert store.load_error
    monkeypatch.setattr(delivery.repos, "REPO_MOUNTS", {"one": tmp_path})
    monkeypatch.setattr(delivery, "_project", lambda _: {"github_repo": None})
    monkeypatch.setattr(delivery, "collect_deployments", lambda _: {"status": "unavailable"})
    app = FastAPI()
    app.include_router(
        delivery.create_router(store, SimpleNamespace(list_assignments=lambda: []), None)
    )
    client = TestClient(app)
    data = client.get("/api/projects/one/delivery").json()
    assert data["sources"]["roadmap"]["status"] == "unavailable"
    assert data["status"] == "partial"
    assert (
        client.put(
            "/api/projects/one/features/missing/links",
            json={"expected_updated_at": "old", "assignment_ids": [], "pr_numbers": []},
        ).status_code
        == 503
    )
    assert (tmp_path / "stories.json").read_text() == "{broken"


def test_corrupt_assignment_store_reports_unavailable(tmp_path, monkeypatch):
    from local_brain.command_center.assignment_registry import AssignmentRegistry

    file = tmp_path / "assignments.json"
    file.write_text("{broken")
    assignments = AssignmentRegistry(file)
    assert assignments.load_error
    store = RoadmapStore(str(tmp_path / "roadmap"))
    story = store.create(title="Feature", product="one")
    monkeypatch.setattr(delivery.repos, "REPO_MOUNTS", {"one": tmp_path})
    monkeypatch.setattr(delivery, "_project", lambda _: {"github_repo": None})
    monkeypatch.setattr(delivery, "collect_deployments", lambda _: {"status": "unavailable"})
    app = FastAPI()
    app.include_router(delivery.create_router(store, assignments, None))
    client = TestClient(app)
    assert (
        client.get("/api/projects/one/delivery").json()["sources"]["assignments"]["status"]
        == "unavailable"
    )
    assert (
        client.put(
            f"/api/projects/one/features/{story['id']}/links",
            json={
                "expected_updated_at": story["updated_at"],
                "assignment_ids": [],
                "pr_numbers": [],
            },
        ).status_code
        == 503
    )


def test_mergeability_and_workflow_identity(setup, monkeypatch):
    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", [], [1])
    payload = {**pr(), "mergeable": False, "mergeable_state": "dirty"}
    latest = {**run(), "workflow_id": 77}
    older = {
        **run(),
        "id": 1,
        "html_url": "https://github.com/owner/repo/actions/runs/1",
        "workflow_id": 77,
        "conclusion": "failure",
        "created_at": "2025-12-31T00:00:00Z",
    }
    monkeypatch.setattr(
        delivery.repos,
        "_github_api",
        Mock(side_effect=[payload, {"workflow_runs": [latest, older], "total_count": 2}]),
    )
    result = client.get("/api/projects/one/delivery").json()["features"][0]["pull_requests"][0]
    assert result["mergeable"] is False and result["mergeable_state"] == "dirty"
    assert [r["workflow_id"] for r in result["ci"]["runs"]] == [77, 77]
    assert [r["conclusion"] for r in result["ci"]["runs"]] == ["success", "failure"]


@pytest.mark.parametrize(
    "bad",
    [
        {"mergeable": "false"},
        {"mergeable": 0},
        {"mergeable_state": []},
        {"mergeable_state": "x" * 81},
    ],
)
def test_invalid_mergeability_is_unavailable(setup, monkeypatch, bad):
    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", [], [1])
    monkeypatch.setattr(delivery.repos, "_github_api", Mock(return_value={**pr(), **bad}))
    result = client.get("/api/projects/one/delivery").json()["features"][0]["pull_requests"][0]
    assert result["status"] == "unavailable"


@pytest.mark.parametrize("workflow_id", [True, 0, "77"])
def test_invalid_workflow_identity_is_unavailable(setup, monkeypatch, workflow_id):
    client, store, story, _, _ = setup
    store.update_delivery_links(story["id"], "one", [], [1])
    monkeypatch.setattr(
        delivery.repos,
        "_github_api",
        Mock(
            side_effect=[
                pr(),
                {"workflow_runs": [{**run(), "workflow_id": workflow_id}], "total_count": 1},
            ]
        ),
    )
    result = client.get("/api/projects/one/delivery").json()["features"][0]["pull_requests"][0]
    assert result["ci"]["status"] == "unavailable"
    assert result["mergeable"] is None
