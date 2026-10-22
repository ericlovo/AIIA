import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from local_brain.command_center.research_shadow_routes import shadow_router
from local_brain.command_center.research_shadow_store import ResearchShadowStore


@pytest.fixture
def setup(tmp_path):
    store = ResearchShadowStore(tmp_path / "shadow.sqlite3")
    principal = {"tenant": "synthetic-a"}

    def authenticated():
        if principal["tenant"] is None:
            raise HTTPException(401, "unauthenticated")
        return principal["tenant"]

    app = FastAPI()
    app.include_router(shadow_router(store, authenticated))
    job = store.enqueue(
        "synthetic-a",
        "assignment",
        "hash",
        "v1",
        {
            "secret_refs": ["private-reference"],
            "fixture": {"text": "private source"},
            "schedule": False,
            "budgets": {},
        },
    )
    return TestClient(app), store, principal, job


def test_bound_scope_redaction_and_status(setup):
    client, store, principal, job = setup
    result = client.get("/api/research-shadow/jobs", headers={"X-Tenant-ID": "synthetic-b"})
    assert result.status_code == 200
    assert result.json()["jobs"][0]["job_id"] == job["id"]
    assert "private" not in result.text
    assert "lease_token" not in result.text
    principal["tenant"] = "synthetic-b"
    assert client.get("/api/research-shadow/jobs").json() == {"jobs": []}
    assert client.get(f"/api/research-shadow/jobs/{job['id']}").status_code == 404
    assert client.post(f"/api/research-shadow/jobs/{job['id']}/cancel").status_code == 404
    assert store.get("synthetic-a", job["id"])["state"] == "queued"


def test_unauthenticated_and_production_rejected(setup):
    client, _, principal, _ = setup
    principal["tenant"] = None
    assert client.get("/api/research-shadow/jobs").status_code == 401
    principal["tenant"] = "production"
    assert client.get("/api/research-shadow/jobs").status_code == 403


def test_pause_kill_cancel_and_body_scope(setup):
    client, store, _, job = setup
    root = "/api/research-shadow"
    assert (
        client.post(
            root + "/assignments/assignment/pause",
            json={"paused": True, "tenant_id": "synthetic-b"},
        ).status_code
        == 422
    )
    assert (
        client.post(root + "/assignments/assignment/pause", json={"paused": True}).status_code
        == 200
    )
    assert store.claim("synthetic-a", job["id"], "worker") is None
    client.post(root + "/assignments/assignment/pause", json={"paused": False})
    assert client.post(root + "/kill-switch", json={"killed": True}).status_code == 200
    assert store.claim("synthetic-a", job["id"], "worker") is None
    client.post(root + "/kill-switch", json={"killed": False})
    assert client.post(f"{root}/jobs/{job['id']}/cancel").json()["state"] == "cancelled"


def test_receipt_and_controls_are_visible_without_source_text(setup):
    from local_brain.command_center.research_shadow import (
        enqueue_replay,
        run_once,
        synthetic_example,
    )

    client, store, principal, _ = setup
    config, fixture = synthetic_example()
    principal["tenant"] = config.tenant_id
    job = enqueue_replay(store, config, fixture)
    run_once(store, config.tenant_id, job["id"], "worker")
    store.set_paused(config.tenant_id, config.assignment_id, True)
    result = client.get(f"/api/research-shadow/jobs/{job['id']}")
    body = result.json()
    assert body["receipt"]["disposition"] == "mock-staged"
    assert body["receipt"]["approved"] is False
    assert body["controls"] == {"paused": True, "killed": False}
    assert fixture.source_text not in result.text
