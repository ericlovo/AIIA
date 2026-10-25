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
        await emit(
            "qwen_called", {"reserve_calls": service.store.config["job_budget"]["calls"] + 1}
        )
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
    assert stale["status"] == "stale"
    assert stale["observed_at"] == fresh["observed_at"]


def test_worker_passes_scoped_history_and_hides_integration_config(service):
    bot = next(iter(service.store.bots.values()))
    bot["managed_leads"] = {"token_env": "PRIVATE_REF"}
    bot["source_allowlist"] = ["example.org"]
    observations = []

    async def pipeline(job, emit):
        observations.append(job)
        return {"packets": [{"organization": "Synthetic", "evidence_key": "evidence-one"}]}

    service.pipeline = pipeline
    service.store.enqueue(bot["id"], bot["tenant_id"], "first", "research")
    asyncio.run(service.run_once())
    service.store.enqueue(bot["id"], bot["tenant_id"], "second", "research")
    asyncio.run(service.run_once())
    assert observations[0]["seen_evidence"] == []
    assert observations[1]["seen_evidence"] == ["evidence-one"]
    assert observations[1]["managed_leads"]["token_env"] == "PRIVATE_REF"
    assert observations[1]["source_allowlist"] == ["example.org"]
    public = service.snapshot()
    assert all(
        "managed_leads" not in row and "source_allowlist" not in row for row in public["bots"]
    )


def test_model_activity_tracks_live_attempt_without_exposing_inputs(service):
    async def scenario():
        service.store.config["mode"] = "live"
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pipeline(job, emit):
            await emit(
                "qwen_called",
                {
                    "provider": "qwen",
                    "model": "qwen3:8b",
                    "reserve_calls": 1,
                    "raw_input": "PRIVATE_SOURCE_BODY",
                },
            )
            entered.set()
            await release.wait()
            await emit(
                "qwen_completed", {"provider": "qwen", "model": "qwen3:8b", "latency_ms": 75}
            )
            await emit("provider_health", {"provider": "qwen", "provider_status": "available"})
            return {"packets": []}

        service.pipeline = pipeline
        job = service.store.enqueue("healthcare", "synthetic-healthcare", "live", "research")
        work = asyncio.create_task(service.run_once())
        await entered.wait()
        active = service.snapshot()
        assert active["model_activity"]["qwen"]["active_jobs"] == [job["id"]]
        assert active["model_activity"]["qwen"]["status"] == "running"
        assert active["model_activity"]["qwen"]["attempts"] == 1
        assert active["model_activity"]["qwen"]["completed"] == 0
        assert active["model_activity"]["jev"]["attempts"] is None
        assert "PRIVATE_SOURCE_BODY" not in json.dumps(active)
        release.set()
        await work
        completed = service.snapshot()["model_activity"]["qwen"]
        assert completed["status"] == "idle"
        assert completed["active_jobs"] == []
        assert completed["completed"] == 1
        assert completed["latency_ms"] == 75
        assert completed["replayed"] == 0
        assert completed["metrics_scope"] == "latest_200_events"

    asyncio.run(scenario())


def test_model_activity_replay_is_not_live_inference(service):
    async def pipeline(job, emit):
        await emit("qwen_replayed", {"model": "replay:saved-qwen"})
        await emit("jev_replayed", {"model": "replay:recorded-jev", "latency_ms": 0})
        return {"packets": []}

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    for provider in ("qwen", "jev"):
        activity = service.snapshot()["model_activity"][provider]
        assert activity["status"] == "replay"
        assert activity["replayed"] == 1
        assert activity["attempts"] == 0
        assert activity["completed"] == 0
        assert activity["latency_ms"] is None
        assert activity["active_jobs"] == []


def test_failed_model_attempt_has_no_completed_count(service):
    async def pipeline(job, emit):
        await emit("jev_called", {"model": "jev-latest", "reserve_calls": 1})
        await emit("jev_error", {"message": "jev_egress_denied"})
        raise ValueError("unavailable")

    service.store.config["mode"] = "live"
    service.pipeline = pipeline
    service.store.enqueue("healthcare", "synthetic-healthcare", "live", "research")
    asyncio.run(service.run_once())
    activity = service.snapshot()["model_activity"]["jev"]
    assert activity["status"] == "data_unavailable"
    assert activity["attempts"] == 1
    assert activity["completed"] == 0
    assert activity["errors"] == 1
    assert activity["latency_ms"] is None


def test_large_backlog_keeps_running_job_oldest_queue_and_completed_packets(service):
    store = service.store
    completed = store.enqueue("healthcare", "synthetic-healthcare", "completed", "research")
    lease = store.lease("worker")
    store.start(completed["id"], lease["lease_token"])
    store.complete(
        completed["id"], lease["lease_token"], {"packets": [{"organization": "Earlier research"}]}
    )
    store.config["mode"] = "live"
    active = store.enqueue("healthcare", "synthetic-healthcare", "active", "research")
    lease = store.lease("worker")
    store.start(active["id"], lease["lease_token"])
    store.heartbeat(active["id"], lease["lease_token"], step="qwen_called")
    queued_ids = []
    for index in range(205):
        queued_ids.append(
            store.enqueue("healthcare", "synthetic-healthcare", str(index), "research")["id"]
        )
    snapshot = service.snapshot()
    assert len(snapshot["jobs"]) == 200
    assert snapshot["jobs"][0]["id"] == active["id"]
    assert [job["id"] for job in snapshot["jobs"][1:]] == queued_ids[:199]
    assert snapshot["model_activity"]["qwen"]["active_jobs"] == [active["id"]]
    assert snapshot["model_activity"]["qwen"]["status"] == "running"
    assert snapshot["packets"][0]["organization"] == "Earlier research"
    assert snapshot["job_counts"] == {
        "queued": 205,
        "leased": 0,
        "running": 1,
        "completed": 1,
        "failed": 0,
        "cancelled": 0,
    }
    assert snapshot["job_window"]["queue_total"] == 205
    assert snapshot["job_window"]["queue_returned"] == 199
    assert snapshot["job_window"]["truncated"] is True
    assert snapshot["job_window"]["total"] == 207


def configure_window(service, start, end):
    from datetime import datetime, timezone

    service.store.config["schedule_enabled"] = True
    service.store.bots["healthcare"]["schedule"].update(enabled=True, interval_seconds=60)
    service.store.config["schedule_window"] = {
        "start_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "end_at": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "timezone": "America/Chicago",
    }


def test_schedule_projection_waiting_pause_boundaries_and_expiry(service):
    now = service.now[0]
    configure_window(service, now + 50, now + 130)
    service.set_kill_switch(True)
    schedule = service.snapshot()["schedule"]
    assert schedule["state"] == "waiting"
    assert schedule["next_run_at"] == schedule["start_at"]
    assert schedule["timezone"] == "America/Chicago"
    assert schedule["display_only"] is True
    assert service.snapshot()["kill_switch"] is True
    service.now[0] += 50
    assert service.snapshot()["schedule"]["state"] == "paused"
    assert service.snapshot()["schedule"]["next_run_at"] is None
    service.set_kill_switch(False)
    assert service.snapshot()["schedule"]["state"] == "scheduled"
    service.now[0] = now + 130
    assert service.snapshot()["schedule"]["state"] == "expired"
    assert service.snapshot()["schedule"]["next_run_at"] is None


def test_schedule_next_run_uses_epoch_intervals_and_existing_dedup_keys(service):
    from datetime import datetime, timezone

    now = service.now[0]
    configure_window(service, now + 50, now + 120)
    bot = service.store.bots["healthcare"]
    service.store.enqueue(
        bot["id"], bot["tenant_id"], f"healthcare:{int((now + 50) // 60)}", "public-research"
    )
    schedule = service.snapshot()["schedule"]
    assert schedule["next_run_at"] == datetime.fromtimestamp(now + 60, timezone.utc).isoformat()
    # A different task does not occupy the periodic research dedup key.
    service.store.enqueue(
        bot["id"], bot["tenant_id"], f"healthcare:{int((now + 60) // 60)}", "other-task"
    )
    assert service.snapshot()["schedule"]["next_run_at"] == schedule["next_run_at"]
    service.store.enqueue(
        bot["id"], bot["tenant_id"], f"healthcare:{int((now + 60) // 60)}", "public-research"
    )
    assert service.snapshot()["schedule"]["next_run_at"] is None  # next epoch equals exclusive end


def test_schedule_next_run_selects_earliest_eligible_bot(service):
    from datetime import datetime, timezone

    now = service.now[0]
    configure_window(service, now - 10, now + 1000)
    first = service.store.bots["healthcare"]
    service.store.enqueue(
        first["id"], first["tenant_id"], f"healthcare:{int(now // 60)}", "public-research"
    )
    assert (
        service.snapshot()["schedule"]["next_run_at"]
        == datetime.fromtimestamp(now + 60, timezone.utc).isoformat()
    )
    second = service.store.bots["pe-diligence"]
    second["schedule"].update(enabled=True, interval_seconds=120)
    assert (
        service.snapshot()["schedule"]["next_run_at"]
        == datetime.fromtimestamp(now, timezone.utc).isoformat()
    )
    second["enabled"] = False
    assert (
        service.snapshot()["schedule"]["next_run_at"]
        == datetime.fromtimestamp(now + 60, timezone.utc).isoformat()
    )


def test_schedule_disabled_and_unwindowed_pause_do_not_invent_dates(service):
    assert service.snapshot()["schedule"]["state"] == "disabled"
    assert service.snapshot()["schedule"]["start_at"] is None
    assert service.snapshot()["schedule"]["stop_at"] is None
    service.store.config["schedule_enabled"] = True
    assert service.snapshot()["schedule"]["state"] == "disabled"  # no scheduled bots
    service.store.bots["healthcare"]["schedule"]["enabled"] = True
    service.set_kill_switch(True)
    assert service.snapshot()["schedule"]["state"] == "paused"
    assert service.snapshot()["schedule"]["next_run_at"] is None


def test_aged_provider_failure_remains_failure_not_stale_success(service):
    async def pipeline(job, emit):
        await emit(
            "provider_health",
            {
                "provider": "qwen",
                "provider_status": "data_unavailable",
                "message": "qwen_unavailable",
            },
        )
        return {"packets": []}

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    service.now[0] += 121
    health = service.snapshot()["health"]["qwen"]
    assert health["status"] == "data_unavailable"
    assert health["last_status"] == "data_unavailable"
    assert health["detail"] == "qwen_unavailable"
    assert health["observed_at"]


def test_candidates_require_segment_role_and_grounded_history(service):
    from local_brain.command_center.bd_pipeline import proposal_packet, synthetic_fixture
    from local_brain.command_center.bd_service import candidate_assessment

    for segment in ("healthcare", "pe-technical-diligence", "alumni-associations"):
        source, extraction, judgments = synthetic_fixture(segment)
        packet = proposal_packet(
            {"segment": segment, "tenant_id": "synthetic-test"}, source, extraction, judgments, {}
        )
        assert candidate_assessment(packet, segment)[0] == "supported"
        packet["judgments"].pop("entity_role")
        assert candidate_assessment(packet, segment)[0] == "inconclusive"
        packet["review_status"] = "research_next"
        assert candidate_assessment(packet, segment)[0] == "inconclusive"
        packet["organization"] = ""
        assert candidate_assessment(packet, segment) == ("inconclusive", "no_named_organization")
        packet["disposition"] = "reject"
        assert candidate_assessment(packet, segment)[0] == "rejected"


def test_withheld_outcome_visible_without_exposing_raw_result(service):
    async def pipeline(job, emit):
        return {
            "status": "no_supported_prospect",
            "research_outcome": "rejected",
            "reason": "segment_entity_or_event_unconfirmed",
            "packets": [],
            "private_debug": "DO_NOT_EXPOSE",
        }

    service.pipeline = pipeline
    service.enqueue_cycle(shadow=True)
    asyncio.run(service.run_once())
    snapshot = service.snapshot()
    job = next(j for j in snapshot["jobs"] if j["state"] == "completed")
    assert job["research_outcome"] == "rejected"
    assert job["result_reason"] == "segment_entity_or_event_unconfirmed"
    assert snapshot["packets"] == []
    assert "DO_NOT_EXPOSE" not in json.dumps(snapshot)


@pytest.mark.parametrize("mutation", ["quote", "provenance", "disposition", "nan"])
def test_historical_evidence_fails_closed(mutation):
    from local_brain.command_center.bd_pipeline import proposal_packet, synthetic_fixture
    from local_brain.command_center.bd_service import candidate_assessment

    source, extraction, judgments = synthetic_fixture("pe-technical-diligence")
    packet = proposal_packet(
        {"segment": "pe-technical-diligence", "tenant_id": "synthetic-pe"},
        source,
        extraction,
        judgments,
        {},
    )
    if mutation == "quote":
        packet["market_signals"][0]["quote"] = packet["organization"] + " fabricated quote"
    elif mutation == "provenance":
        packet["provenance"] = {"url": "https://example.org"}
    elif mutation == "disposition":
        packet["disposition"] = "watch"
    else:
        packet["judgments"]["entity_role"]["confidence"] = float("nan")
    assert candidate_assessment(packet, "pe-technical-diligence")[0] == "inconclusive"


def test_client_candidate_uses_configured_research_segment(service):
    from local_brain.command_center.bd_pipeline import proposal_packet, synthetic_fixture

    bot = service.store.bots["srg"]
    bot["research_segment"] = "pe-technical-diligence"
    source, extraction, judgments = synthetic_fixture("pe-technical-diligence")
    packet = proposal_packet(
        {"segment": "client-bd", "tenant_id": bot["tenant_id"]}, source, extraction, judgments, {}
    )

    async def pipeline(job, emit):
        return {"packets": [packet]}

    service.pipeline = pipeline
    service.store.enqueue("srg", bot["tenant_id"], "client-preview", "research", mode="replay")
    asyncio.run(service.run_once())
    snapshot = service.snapshot()
    assert snapshot["packets"][0]["candidate_status"] == "supported"
    assert snapshot["jobs"][0]["research_outcome"] == "supported"
