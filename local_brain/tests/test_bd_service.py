import asyncio
import json

import pytest

from local_brain.command_center.bd_conductor import CONFIG_PATH, BDConductor
from local_brain.command_center.bd_service import BDService


@pytest.fixture
def service(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    now = [1800000000.0]
    store = BDConductor(tmp_path / "jobs.sqlite", config=config, clock=lambda: now[0])
    instance = BDService(store)
    instance.now = now
    return instance


def test_shadow_manual_ignores_disabled_schedule_and_dedups(service):
    first = service.enqueue_cycle(shadow=True)
    assert len(first) == 4
    assert first == service.enqueue_cycle(shadow=True)
    assert len(service.store.snapshot()["jobs"]) == 4
    assert all(j["tenant_id"].startswith("synthetic-") for j in service.store.snapshot()["jobs"])


def test_schedule_respects_bot_enabled_flags(service):
    service.store.config["schedule_enabled"] = True
    assert service.enqueue_cycle() == []
    bot = service.store.bots["healthcare"]
    bot["schedule"]["enabled"] = True
    assert len(service.enqueue_cycle()) == 1


def test_schedule_uses_bot_interval_and_survives_service_restart(service):
    bot = service.store.bots["healthcare"]
    bot["schedule"].update(enabled=True, interval_seconds=30)
    service.store.config["schedule_enabled"] = True
    first = service.enqueue_cycle()
    restarted = BDService(service.store)
    assert restarted.enqueue_cycle() == first
    service.now[0] += 31
    assert restarted.enqueue_cycle() != first
    assert len(service.store.snapshot()["jobs"]) == 2


def test_kill_blocks_manual_enqueue(service):
    service.set_kill_switch(True)
    with pytest.raises(ValueError):
        service.enqueue_cycle(shadow=True)


def test_run_once_budget_events_result_and_public_projection(service):
    async def pipeline(job, emit):
        await emit("jev_called", {"reserve_calls": 1, "reserve_tokens": 20})
        await emit("jev_verdict", {"primitive": "Choice", "latency_ms": 2})
        return {
            "packets": [
                {
                    "organization": "Example Co",
                    "emails": ["info@example.invalid"],
                    "signals": ["Synthetic signal"],
                }
            ]
        }

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    assert asyncio.run(service.run_once()) is True
    snapshot = service.snapshot()
    assert len(snapshot["packets"]) == 1
    assert snapshot["packets"][0]["review_status"] == "human_review_required"
    assert snapshot["packets"][0]["mode"] == "replay"
    assert snapshot["jev"]["calls_per_minute"] == 1
    assert snapshot["jev"]["verdict_mix"]["Choice"] == 1
    assert snapshot["health"]["managed_leads"]["status"] == "data_unavailable"
    assert all("lease_token" not in job and "payload" not in job for job in snapshot["jobs"])
    completed = next(j for j in service.store.snapshot()["jobs"] if j["state"] == "completed")
    assert completed["used_budget"]["tokens"] == 20
    assert completed["used_budget"]["calls"] == 1
    assert snapshot["budget"]["tokens"]["used"] == service.store.config["job_budget"]["tokens"]


def test_over_budget_emit_prevents_adapter_work(service):
    performed = []

    async def pipeline(job, emit):
        await emit("qwen_called", {"reserve_calls": 5})
        performed.append(True)

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    assert performed == []
    failed_attempt = next(j for j in service.store.snapshot()["jobs"] if j["attempts"])
    assert failed_attempt["state"] == "queued"
    assert failed_attempt["error"] == "ValueError"


def test_exception_details_are_not_exposed(service):
    async def pipeline(job, emit):
        raise RuntimeError("SECRET_TOKEN_IN_URL")

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    assert "SECRET_TOKEN_IN_URL" not in json.dumps(service.snapshot())


def test_kill_cancels_current_attempt_without_killing_worker(service):
    async def scenario():
        started = asyncio.Event()

        async def pipeline(job, emit):
            started.set()
            await asyncio.Event().wait()

        service.pipeline = pipeline
        service.enqueue_cycle(shadow=True)
        await service.start()
        await asyncio.wait_for(started.wait(), timeout=1)
        service.set_kill_switch(True)
        await asyncio.sleep(0.05)
        survived = not service._worker.done()
        await service.stop()
        assert survived
        assert service._active is None
        assert any(j["state"] == "cancelled" for j in service.store.snapshot()["jobs"])

    asyncio.run(scenario())


def test_live_shadow_rejected_and_configured_tenant_used(service):
    service.store.config["mode"] = "live"
    bot = service.store.bots["healthcare"]
    bot["tenant_id"] = "configured-tenant"
    bot["schedule"]["enabled"] = True
    with pytest.raises(ValueError):
        service.enqueue_cycle(shadow=True)
    service.enqueue_cycle()
    job = next(j for j in service.store.snapshot()["jobs"] if j["bot_id"] == "healthcare")
    assert job["tenant_id"] == "configured-tenant"
    assert job["mode"] == "live"


def test_historical_packet_survives_removed_bot(service):
    async def pipeline(job, emit):
        return {"packets": [{"organization": "Historical example"}]}

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    completed = next(j for j in service.store.snapshot()["jobs"] if j["state"] == "completed")
    del service.store.bots[completed["bot_id"]]
    assert service.snapshot()["packets"][0]["organization"] == "Historical example"


def test_service_stop_leaves_durable_recoverable_lease(service):
    async def scenario():
        started = asyncio.Event()

        async def pipeline(job, emit):
            started.set()
            await asyncio.Event().wait()

        service.pipeline = pipeline
        service.enqueue_cycle(shadow=True)
        await service.start()
        await asyncio.wait_for(started.wait(), 1)
        await service.stop()
        assert service._active is None
        job = next(j for j in service.store.snapshot()["jobs"] if j["state"] == "running")
        service.now[0] = job["lease_until"] + 1
        service.store.recover()
        recovered = next(j for j in service.store.snapshot()["jobs"] if j["id"] == job["id"])
        assert recovered["state"] == "queued"
        assert recovered["attempts"] == 1

    asyncio.run(scenario())


def test_jev_latency_uses_completed_call_event(service):
    async def pipeline(job, emit):
        await emit("jev_called", {"reserve_calls": 1})
        await emit("jev_completed", {"latency_ms": 125})
        await emit("jev_verdict", {"primitive": "choice"})
        return {"packets": []}

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    assert service.snapshot()["jev"]["latency_ms"] == 125


def test_empty_source_result_never_becomes_fake_handoff(service):
    async def pipeline(job, emit):
        return {"status": "no_observations", "packets": [], "usage": {"tokens": 0}}

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    assert service.snapshot()["packets"] == []


def test_provider_health_expires_and_preserves_observation_time(service):
    async def pipeline(job, emit):
        await emit(
            "provider_health",
            {"provider": "qwen", "provider_status": "available", "message": "Completed"},
        )
        return {"packets": []}

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    fresh = service.snapshot()["health"]["qwen"]
    assert fresh["status"] == "available"
    service.now[0] += 121
    stale = service.snapshot()["health"]["qwen"]
    assert stale["status"] == "data_unavailable"
    assert stale["observed_at"] == fresh["observed_at"]
