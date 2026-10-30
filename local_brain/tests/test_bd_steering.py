import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center.bd_conductor import CONFIG_PATH, BDConductor
from local_brain.command_center.bd_routes import create_router
from local_brain.command_center.bd_service import BDService

CHICAGO = ZoneInfo("America/Chicago")
LIVE_WORKSPACE = "bec83090-fa8a-4b55-97c8-e90a54c7f663"


@pytest.fixture
def route_app(tmp_path):
    now = [datetime(2026, 10, 27, 10, tzinfo=CHICAGO).timestamp()]
    store = BDConductor(tmp_path / "steer.sqlite", clock=lambda: now[0])

    async def pipeline(job, emit):
        return {"packets": []}

    service = BDService(store, pipeline=pipeline)
    service.now = now
    app = FastAPI()
    app.include_router(create_router(factory=lambda: service))
    return app, service


def steer(client, **body):
    return client.post("/api/bd/steer", json=body)


def test_comcast_bot_is_synthetic_single_tenant_and_not_live_bound(route_app):
    app, service = route_app
    roster = json.loads(CONFIG_PATH.read_text())
    comcast = next(bot for bot in roster["bots"] if bot["id"] == "comcast")
    assert comcast["kind"] == "client"
    assert comcast["tenant_id"] == "synthetic-comcast"
    assert comcast["research_segment"] == "commercial-fit-out"
    assert "managed_leads" not in comcast
    assert comcast["tenant_id"] != LIVE_WORKSPACE
    assert LIVE_WORKSPACE not in json.dumps(roster)
    with pytest.raises(ValueError):
        service.store.enqueue("comcast", "synthetic-srg", "cross", "research")
    with pytest.raises(ValueError):
        service.store.enqueue("comcast", LIVE_WORKSPACE, "live", "research")
    job = service.store.enqueue("comcast", "synthetic-comcast", "own", "research")
    assert job["tenant_id"] == "synthetic-comcast"
    with TestClient(app) as client:
        snapshot = client.get("/api/bd/snapshot").json()
        bot = next(item for item in snapshot["bots"] if item["id"] == "comcast")
        assert bot["tenant_id"] == "synthetic-comcast"
        assert bot["research_segment"] == "commercial-fit-out"
        assert "managed_leads" not in bot
        assert all("managed_leads" not in item for item in snapshot["bots"])


def test_every_steer_action_audits_and_shows_in_snapshot(route_app):
    app, service = route_app
    with TestClient(app) as client:
        paused = steer(client, action="pause")
        assert paused.status_code == 200
        assert paused.json()["schedule_enabled"] is False
        resumed = steer(client, action="resume")
        assert resumed.json()["schedule_enabled"] is True
        assert steer(client, action="pause_bot", bot_id="comcast").json()["steering"][
            "paused_bots"
        ] == ["comcast"]
        service.now[0] = datetime(2026, 10, 27, 10, tzinfo=CHICAGO).timestamp()
        before = set(service.enqueue_cycle())
        assert before
        assert not any(
            job["bot_id"] == "comcast" and job["id"] in before
            for job in service.store.snapshot()["jobs"]
        )
        assert (
            steer(client, action="resume_bot", bot_id="comcast").json()["steering"]["paused_bots"]
            == []
        )
        focused = steer(client, action="focus", bot_id="comcast")
        assert focused.json()["steering"]["focus"] == {"kind": "bot", "id": "comcast"}
        service.now[0] += 1800
        focused_jobs = service.enqueue_cycle()
        assert focused_jobs
        assert all(
            job["bot_id"] == "comcast"
            for job in service.store.snapshot()["jobs"]
            if job["id"] in set(focused_jobs)
        )
        cleared = steer(client, action="clear_focus")
        assert cleared.json()["steering"]["focus"] is None
        queued = steer(
            client,
            action="queue",
            bot_id="comcast",
            query="Minneapolis commercial building permits issued this week",
        )
        assert queued.status_code == 200
        job_id = next(
            job["id"]
            for job in queued.json()["jobs"]
            if job["bot_id"] == "comcast" and job["state"] == "queued"
        )
        bumped = steer(client, action="bump", job_id=job_id)
        assert bumped.status_code == 200
        overnight = steer(client, action="set_cadence", overnight_enabled=True)
        assert overnight.json()["steering"]["overnight_enabled"] is True
        clock = steer(client, action="set_cadence", bot_id="comcast", around_the_clock=True)
        assert clock.json()["steering"]["around_the_clock"]["comcast"] is True
        cadence = steer(client, action="set_cadence", overnight_interval_seconds=1800)
        assert cadence.json()["cadence"]["overnight"]["interval_seconds"] == 1800
        budget = steer(client, action="set_budget", tokens=1000, calls=5, seconds=60)
        assert budget.json()["budget"]["tokens"]["limit"] == 1000
        killed = steer(client, action="kill_switch", enabled=True)
        assert killed.json()["kill_switch"] is True
        audit = killed.json()["steering"]["audit"]
        actions = {item["action"] for item in audit}
        assert {
            "pause",
            "resume",
            "pause_bot",
            "resume_bot",
            "focus",
            "clear_focus",
            "queue",
            "bump",
            "set_cadence",
            "set_budget",
            "kill_switch",
        } <= actions
        assert all("tenant_id" not in item for item in audit)
        live = client.get("/api/bd/snapshot").json()
        assert live["steering"]["audit"][0]["action"] == "kill_switch"
        assert live["assignments"]["qwen"]["status"] in {
            "replay",
            "idle",
            "data_unavailable",
            "running",
        }


def test_steer_rejects_cross_origin_and_scope_injection(route_app):
    app, service = route_app
    with TestClient(app) as client:
        assert steer(client, action="pause").status_code == 200
        assert (
            client.post(
                "/api/bd/steer",
                json={"action": "pause"},
                headers={"origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/bd/steer",
                json={"action": "kill_switch", "enabled": True},
                headers={"sec-fetch-site": "cross-site"},
            ).status_code
            == 403
        )
        assert service.store.snapshot()["kill_switch"] is False
        for body in (
            {"action": "queue", "bot_id": "comcast", "query": "permits", "tenant_id": "victim"},
            {"action": "queue", "bot_id": "comcast", "query": "permits", "mode": "live"},
            {"action": "focus", "bot_id": "comcast", "segment": "healthcare"},
            {"action": "pause", "enabled": True},
            {"action": "kill_switch", "enabled": "true"},
            {"action": "set_budget", "tokens": 1, "tenant_id": LIVE_WORKSPACE},
        ):
            assert client.post("/api/bd/steer", json=body).status_code == 422
        assert (
            client.post("/api/bd/steer", json={"action": "queue", "bot_id": "comcast"}).status_code
            == 422
        )
        assert client.post(
            "/api/bd/steer", json={"action": "bump", "job_id": "missing"}
        ).status_code in {404, 409}


def test_steer_budget_cannot_exceed_roster_ceiling(route_app):
    app, service = route_app
    with TestClient(app) as client:
        ceiling = service.store.config["global_daily_budget"]["tokens"]
        assert steer(client, action="set_budget", tokens=ceiling + 1).status_code == 409
        assert service.snapshot()["budget"]["tokens"]["limit"] == ceiling


def test_overnight_steer_takes_effect_on_next_tick_without_restart(route_app):
    app, service = route_app
    service.now[0] = datetime(2026, 10, 27, 22, tzinfo=CHICAGO).timestamp()
    with TestClient(app) as client:
        client.post("/api/bd/schedule", json={"enabled": True})
        assert service.enqueue_cycle() == []
        steer(client, action="set_cadence", overnight_enabled=True)
        assert service.enqueue_cycle() == []
        steer(client, action="set_cadence", bot_id="comcast", around_the_clock=True)
        restarted = BDService(service.store, pipeline=service.pipeline)
        restarted.now = service.now
        queued = restarted.enqueue_cycle()
        assert queued
        assert restarted.store.snapshot()["jobs"][0]["bot_id"] == "comcast"
        assert restarted.snapshot()["steering"]["overnight_enabled"] is True


def test_queue_cancel_undo_does_not_write_crm(route_app):
    app, service = route_app
    with TestClient(app) as client:
        queued = steer(
            client,
            action="queue",
            bot_id="comcast",
            query="Uptown restaurant build-outs",
        )
        assert queued.status_code == 200
        job = next(
            item
            for item in queued.json()["jobs"]
            if item["bot_id"] == "comcast" and item.get("operator_query")
        )
        assert job["operator_query"] == "Uptown restaurant build-outs"
        cancelled = steer(client, action="cancel", job_id=job["id"])
        assert cancelled.status_code == 200
        assert (
            next(item for item in cancelled.json()["jobs"] if item["id"] == job["id"])["state"]
            == "cancelled"
        )


def test_live_research_tenants_are_valid_operator_config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["mode"] = "live"
    config["schedule_enabled"] = True
    tenants = {
        "healthcare": "research-healthcare",
        "pe-diligence": "research-pe",
        "alumni-associations": "research-alumni",
    }
    for bot in config["bots"]:
        if bot["id"] in tenants:
            bot["tenant_id"] = tenants[bot["id"]]
            bot["enabled"] = True
            bot["schedule"]["enabled"] = True
    now = [datetime(2026, 10, 27, 10, tzinfo=CHICAGO).timestamp()]
    store = BDConductor(tmp_path / "live-research.sqlite", config=config, clock=lambda: now[0])
    assert store.config["mode"] == "live"
    assert store.bots["healthcare"]["tenant_id"] == "research-healthcare"
    job = store.enqueue("healthcare", "research-healthcare", "account", "research")
    assert job["mode"] == "live"
    assert job["tenant_id"] == "research-healthcare"
    service = BDService(store)
    queued = service.enqueue_cycle()
    assert queued
    allowed = set(tenants.values()) | {"synthetic-comcast", "synthetic-srg"}
    for item in store.snapshot()["jobs"]:
        assert item["mode"] == "live"
        assert item["tenant_id"] == store.bots[item["bot_id"]]["tenant_id"]
        assert item["tenant_id"] in allowed
    research_jobs = [
        item
        for item in store.snapshot()["jobs"]
        if item["bot_id"] in tenants and item["id"] in set(queued)
    ]
    if research_jobs:
        assert all(item["tenant_id"].startswith("research-") for item in research_jobs)


def test_assignments_use_data_unavailable_not_zeros_when_live_unobserved(route_app):
    app, service = route_app
    service.store.config["mode"] = "live"
    service.store.bots["healthcare"]["tenant_id"] = "research-healthcare"
    with TestClient(app) as client:
        snapshot = client.get("/api/bd/snapshot").json()
        for provider in ("qwen", "jev"):
            block = snapshot["assignments"][provider]
            assert block["status"] == "data_unavailable"
            assert block["calls"] is None
            assert block["errors"] is None
            assert block["latency_ms"] is None
            assert block["calls"] != 0
            assert snapshot["model_activity"][provider]["attempts"] is None
