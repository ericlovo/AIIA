import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center.bd_conductor import CONFIG_PATH, BDConductor
from local_brain.command_center.bd_routes import create_router
from local_brain.command_center.bd_service import BDService


@pytest.fixture
def route_app(tmp_path):
    store = BDConductor(tmp_path / "api.sqlite")

    async def pipeline(job, emit):
        return {"packets": []}

    service = BDService(store, pipeline=pipeline)
    app = FastAPI()
    app.include_router(create_router(factory=lambda: service))
    return app, service


def test_lifespan_starts_worker_without_browser_request_and_stops(route_app):
    app, service = route_app
    with TestClient(app):
        assert service._worker is not None
        assert not service._worker.done()
    assert service._worker is None


def test_shadow_dedup_snapshot_and_kill_resume(route_app):
    app, service = route_app
    with TestClient(app) as client:
        first = client.post("/api/bd/shadow", json={})
        assert first.status_code == 200
        assert len(first.json()["job_ids"]) == 4
        assert client.post("/api/bd/shadow", json={}).json() == first.json()
        snapshot = client.get("/api/bd/snapshot").json()
        assert snapshot["mode"] == "replay"
        assert snapshot["schedule_enabled"] is False
        assert len(snapshot["bots"]) == 4
        assert "lease_token" not in json.dumps(snapshot)
        assert client.post("/api/bd/kill-switch", json={"enabled": True}).json()["kill_switch"]
        assert client.post("/api/bd/shadow", json={}).status_code == 409
        assert (
            client.post("/api/bd/kill-switch", json={"enabled": False}).json()["kill_switch"]
            is False
        )
        assert client.post("/api/bd/shadow", json={}).status_code == 200
        assert not service._worker.done()


def test_schedule_toggle_is_the_activation_flag(route_app):
    app, service = route_app
    with TestClient(app) as client:
        snapshot = client.get("/api/bd/snapshot").json()
        assert snapshot["schedule_enabled"] is False
        assert snapshot["status_banner"]["label"] == "Paused: schedule off"
        enabled = client.post("/api/bd/schedule", json={"enabled": True}).json()
        assert enabled["schedule_enabled"] is True
        assert enabled["status_banner"]["label"] == "Waiting for next run"
        assert service.store.schedule_enabled() is True
        paused = client.post("/api/bd/schedule", json={"enabled": False}).json()
        assert paused["status_banner"]["label"] == "Paused: schedule off"
        assert client.post("/api/bd/schedule", json={"enabled": "true"}).status_code == 422
        assert (
            client.post(
                "/api/bd/schedule", json={"enabled": True, "tenant_id": "victim"}
            ).status_code
            == 422
        )


@pytest.mark.parametrize(
    "headers", [{"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}]
)
def test_cross_origin_mutations_rejected(route_app, headers):
    app, service = route_app
    with TestClient(app) as client:
        assert client.post("/api/bd/shadow", json={}, headers=headers).status_code == 403
        assert (
            client.post("/api/bd/kill-switch", json={"enabled": True}, headers=headers).status_code
            == 403
        )
        assert (
            client.post("/api/bd/schedule", json={"enabled": True}, headers=headers).status_code
            == 403
        )
        assert service.store.snapshot()["jobs"] == []
        assert service.store.snapshot()["kill_switch"] is False


@pytest.mark.parametrize("body", [{"tenant_id": "victim"}, {"mode": "live"}, {"bot_id": "srg"}])
def test_shadow_scope_injection_rejected(route_app, body):
    app, service = route_app
    with TestClient(app) as client:
        assert client.post("/api/bd/shadow", json=body).status_code == 422
        assert service.store.snapshot()["jobs"] == []


@pytest.mark.parametrize(
    "body", [{"enabled": "true"}, {"enabled": 1}, {"enabled": True, "tenant_id": "victim"}]
)
def test_kill_requires_exact_boolean_and_no_extra_fields(route_app, body):
    app, service = route_app
    with TestClient(app) as client:
        assert client.post("/api/bd/kill-switch", json=body).status_code == 422
        assert service.store.snapshot()["kill_switch"] is False


def test_same_origin_allowed(route_app):
    app, _ = route_app
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/bd/shadow", json={}, headers={"origin": "http://testserver"}
            ).status_code
            == 200
        )


def test_packet_triage_persists_without_model_approval(route_app):
    app, service = route_app
    job = service.store.enqueue("healthcare", "synthetic-healthcare", "triage", "research")
    lease = service.store.lease("test-worker")
    service.store.start(job["id"], lease["lease_token"])
    service.store.complete(
        job["id"],
        lease["lease_token"],
        {"packets": [{"organization": "Example", "approved": False}]},
    )
    path = f"/api/bd/packets/{job['id']}/0/review"
    with TestClient(app) as client:
        response = client.post(path, json={"status": "research_next"})
        assert response.status_code == 200
        snapshot = client.get("/api/bd/snapshot").json()
        assert snapshot["packets"][0]["review_status"] == "research_next"
        assert service.store.snapshot()["jobs"][0]["result"]["packets"][0]["approved"] is False
        assert client.post(path, json={"status": "approved"}).status_code == 422
        assert client.post(path, json={"status": "watch", "tenant_id": "victim"}).status_code == 422
        assert (
            client.post(
                path, json={"status": "watch"}, headers={"origin": "https://evil.example"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/api/bd/packets/{job['id']}/1/review", json={"status": "watch"}
            ).status_code
            == 409
        )
        assert (
            client.post("/api/bd/packets/missing/0/review", json={"status": "watch"}).status_code
            == 404
        )


@pytest.fixture
def live_route_app(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["mode"] = "live"
    config["bots"][0]["tenant_id"] = "segment-healthcare"
    config["bots"][0]["source_allowlist"] = ["example.org"]
    store = BDConductor(tmp_path / "live-api.sqlite", config=config)

    async def pipeline(job, emit):
        return {"packets": []}

    service = BDService(store, pipeline=pipeline)
    app = FastAPI()
    app.include_router(create_router(factory=lambda: service))
    return app, service


def import_body(**changes):
    import hashlib

    text = "Orchard Clinic opens a clinic. Public office research@example.org."
    return {
        "bot_id": "healthcare",
        "source": {
            "url": "https://example.org/news",
            "text": text,
            "source_hash": hashlib.sha256(text.encode()).hexdigest(),
            "captured_at": "2026-01-01T12:00:00Z",
            "public": True,
        },
        **changes,
    }


def test_source_import_scope_dedup_and_no_model_calls(live_route_app):
    app, service = live_route_app
    with TestClient(app) as client:
        first = client.post("/api/bd/sources", json=import_body())
        assert first.status_code == 202
        assert first.json()["tenant_id"] == "segment-healthcare"
        assert client.post("/api/bd/sources", json=import_body()).json() == first.json()
        jobs = service.store.snapshot()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["tenant_id"] == "segment-healthcare"
        assert jobs[0]["task"] == "public-source-research"
        assert jobs[0]["payload"]["public_source"] == import_body()["source"]
        assert "source_allowlist" not in jobs[0]["payload"]


def test_source_import_requires_live_mode(route_app):
    app, service = route_app
    with TestClient(app) as client:
        assert client.post("/api/bd/sources", json=import_body()).status_code == 409
        assert service.store.snapshot()["jobs"] == []


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "victim"},
        {"source_allowlist": ["attacker.org"]},
        {"mode": "live"},
        {"bot_id": 1},
    ],
)
def test_source_import_rejects_scope_overrides(live_route_app, changes):
    app, service = live_route_app
    with TestClient(app) as client:
        assert client.post("/api/bd/sources", json=import_body(**changes)).status_code == 422
        assert service.store.snapshot()["jobs"] == []


def test_source_import_requires_enabled_segment_allowlist(live_route_app):
    app, service = live_route_app
    with TestClient(app) as client:
        assert client.post("/api/bd/sources", json=import_body(bot_id="srg")).status_code == 409
        assert (
            client.post("/api/bd/sources", json=import_body(bot_id="pe-diligence")).status_code
            == 422
        )
        assert client.post("/api/bd/sources", json=import_body(bot_id="unknown")).status_code == 404
        service.store.bots["healthcare"]["enabled"] = False
        assert client.post("/api/bd/sources", json=import_body()).status_code == 404
        assert service.store.snapshot()["jobs"] == []


def test_source_import_rejects_killed_cross_origin_and_bad_hash(live_route_app):
    app, service = live_route_app
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/bd/sources", json=import_body(), headers={"origin": "https://evil.example"}
            ).status_code
            == 403
        )
        body = import_body()
        body["source"]["source_hash"] = "0" * 64
        assert client.post("/api/bd/sources", json=body).status_code == 422
        client.post("/api/bd/kill-switch", json={"enabled": True})
        assert client.post("/api/bd/sources", json=import_body()).status_code == 409
        assert service.store.snapshot()["jobs"] == []
