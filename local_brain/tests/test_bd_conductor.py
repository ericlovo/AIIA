import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_brain.command_center.bd_conductor import CONFIG_PATH, BDConductor, LeaseLost


@pytest.fixture
def runtime(tmp_path):
    now = [1800000000.0]
    runtime = BDConductor(tmp_path / "jobs.sqlite", clock=lambda: now[0])
    runtime.now = now
    return runtime


def enqueue(runtime, entity="account-1", bot="healthcare", tenant="synthetic-healthcare"):
    return runtime.enqueue(bot, tenant, entity, "research")


def test_state_machine_and_durable_result(runtime):
    job = enqueue(runtime)
    assert enqueue(runtime)["id"] == job["id"]
    leased = runtime.lease("worker")
    token = leased["lease_token"]
    with pytest.raises(LeaseLost):
        runtime.complete(job["id"], token)
    runtime.start(job["id"], token)
    runtime.heartbeat(job["id"], token, step="extracting")
    runtime.consume(job["id"], token, tokens=100, calls=1)
    runtime.record_event(job["id"], token, "candidate_proposed", count=1)
    runtime.complete(job["id"], token, {"status": "awaiting_human"})
    restored = BDConductor(runtime.path, clock=runtime.clock)
    assert restored.snapshot()["jobs"][0]["result"]["status"] == "awaiting_human"
    assert "lease_token" not in restored.snapshot()["jobs"][0]
    assert restored.events()[-1]["kind"] == "job_completed"
    with pytest.raises(LeaseLost):
        runtime.fail(job["id"], token, "late failure")


def test_concurrent_dedup_and_single_lease(runtime):
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: enqueue(runtime), range(16)))
    assert len({j["id"] for j in jobs}) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = list(pool.map(lambda n: runtime.lease(str(n)), range(16)))
    assert sum(j is not None for j in leases) == 1
    enqueue(runtime, "second")
    assert runtime.lease("other") is None


def test_expiry_retries_backoff_and_stale_fencing(runtime):
    job = enqueue(runtime)
    first = runtime.lease("first")
    runtime.now[0] += 31
    restarted = BDConductor(runtime.path, clock=runtime.clock)
    restarted.recover()
    assert restarted.lease("second") is None
    with pytest.raises(LeaseLost):
        runtime.heartbeat(job["id"], first["lease_token"])
    runtime.now[0] += 5
    second = restarted.lease("second")
    assert second["attempts"] == 2
    with pytest.raises(LeaseLost):
        runtime.start(job["id"], first["lease_token"])
    restarted.fail(job["id"], second["lease_token"], "retry")
    runtime.now[0] += 10
    third = restarted.lease("third")
    restarted.fail(job["id"], third["lease_token"], "final")
    assert restarted.snapshot()["jobs"][0]["state"] == "failed"
    assert restarted.lease("fourth") is None


def test_heartbeat_cannot_extend_attempt_wall_time(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    for _ in range(5):
        runtime.now[0] += 20
        runtime.heartbeat(job["id"], lease["lease_token"])
    runtime.now[0] += 21
    with pytest.raises(LeaseLost):
        runtime.complete(job["id"], lease["lease_token"])
    runtime.recover()
    assert runtime.snapshot()["jobs"][0]["state"] == "queued"


def test_kill_switch_durable_and_cancels_active(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    runtime.set_kill_switch(True)
    restarted = BDConductor(runtime.path, clock=runtime.clock)
    assert restarted.snapshot()["kill_switch"] is True
    assert restarted.snapshot()["jobs"][0]["state"] == "cancelled"
    assert restarted.lease("worker") is None
    restarted.set_kill_switch(False)
    with pytest.raises(LeaseLost):
        restarted.start(job["id"], lease["lease_token"])


def test_cancel_idempotent_and_fenced(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    assert runtime.cancel(job["id"]) == runtime.cancel(job["id"])
    with pytest.raises(LeaseLost):
        runtime.consume(job["id"], lease["lease_token"], calls=1)


@pytest.mark.parametrize("resource", ["tokens", "calls", "seconds"])
def test_budget_atomic_and_attempts_never_refunded(tmp_path, resource):
    config = json.loads(CONFIG_PATH.read_text())
    config["global_daily_budget"][resource] = config["job_budget"][resource]
    runtime = BDConductor(tmp_path / "budget.sqlite", config=config)
    first = enqueue(runtime)
    enqueue(runtime, "second", "srg", "synthetic-srg")
    runtime.lease("worker")
    runtime.cancel(first["id"])
    assert runtime.lease("worker") is None
    reserved = [
        x
        for x in runtime.snapshot()["budgets"]
        if x["scope"] == "global" and x["resource"] == resource
    ]
    assert reserved[0]["reserved"] == config["job_budget"][resource]


def test_consumption_enforced_and_rollback(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    with pytest.raises(ValueError):
        runtime.consume(
            job["id"],
            lease["lease_token"],
            tokens=100,
            calls=runtime.config["job_budget"]["calls"] + 1,
        )
    assert runtime.snapshot()["jobs"][0]["used_budget"] == {"tokens": 0, "calls": 0}
    runtime.consume(
        job["id"],
        lease["lease_token"],
        tokens=runtime.config["job_budget"]["tokens"],
        calls=runtime.config["job_budget"]["calls"],
    )
    with pytest.raises(ValueError):
        runtime.consume(job["id"], lease["lease_token"], calls=1)


def test_scope_mode_and_config_fail_closed(runtime, tmp_path):
    with pytest.raises(ValueError):
        enqueue(runtime, tenant="synthetic-srg")
    with pytest.raises(ValueError):
        runtime.enqueue("healthcare", "synthetic-healthcare", "x", "research", mode="live")
    for invalid in (float("nan"), -1, True, 0):
        config = copy.deepcopy(runtime.config)
        config["job_budget"]["calls"] = invalid
        with pytest.raises(ValueError):
            BDConductor(tmp_path / "invalid.sqlite", config=config)


def test_day_rollover_allows_new_budget(runtime):
    runtime.config["global_daily_budget"] = runtime.config["job_budget"].copy()
    enqueue(runtime)
    lease = runtime.lease("worker")
    runtime.cancel(lease["id"])
    enqueue(runtime, "second")
    assert runtime.lease("worker") is None
    runtime.now[0] += 86400
    assert runtime.lease("worker") is not None


def test_live_requires_explicit_config_and_scoped_tenant(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["mode"] = "live"
    config["bots"][0]["tenant_id"] = "tenant-healthcare-approved"
    runtime = BDConductor(tmp_path / "live.sqlite", config=config)
    job = runtime.enqueue("healthcare", "tenant-healthcare-approved", "account", "research")
    assert job["mode"] == "live"
    assert job["replay_only"] is False
    assert runtime.snapshot()["replay_only"] is False
    assert runtime.events()[0]["mode"] == "live"
    with pytest.raises(ValueError):
        runtime.enqueue("healthcare", "synthetic-healthcare", "other", "research")
    with pytest.raises(ValueError):
        runtime.enqueue(
            "healthcare", "tenant-healthcare-approved", "other", "research", mode="replay"
        )


def test_bot_budget_does_not_block_other_bot(runtime):
    runtime.config["bot_daily_budget"] = runtime.config["job_budget"].copy()
    first = enqueue(runtime)
    enqueue(runtime, "second")
    third = enqueue(runtime, "third", "srg", "synthetic-srg")
    runtime.lease("worker")
    runtime.cancel(first["id"])
    assert runtime.lease("worker")["id"] == third["id"]


def test_reconfiguration_cannot_run_old_tenant_or_mode(runtime):
    enqueue(runtime)
    config = copy.deepcopy(runtime.config)
    config["mode"] = "live"
    updated = BDConductor(runtime.path, config=config, clock=runtime.clock)
    assert updated.lease("worker") is None
    config["mode"] = "replay"
    config["bots"][0]["tenant_id"] = "synthetic-changed"
    updated = BDConductor(runtime.path, config=config, clock=runtime.clock)
    assert updated.lease("worker") is None


def test_exact_expiry_boundary_fences_worker(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    runtime.now[0] = lease["lease_until"]
    with pytest.raises(LeaseLost):
        runtime.start(job["id"], lease["lease_token"])


def test_pagination_is_ordered_and_replay_labeled(runtime):
    for n in range(3):
        enqueue(runtime, str(n))
    first = runtime.events(limit=2)
    second = runtime.events(after_id=first[-1]["id"])
    assert len(first) == 2 and len(second) == 1
    assert all(e["replay_only"] for e in first + second)
    assert second[0]["id"] > first[-1]["id"]


def test_packet_review_persists_separate_from_model_result(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    runtime.start(job["id"], lease["lease_token"])
    packet = {"organization": "Example", "approved": False, "outreach_allowed": False}
    runtime.complete(job["id"], lease["lease_token"], {"packets": [packet]})
    for status in ("research_next", "watch", "dismissed", "human_review_required"):
        assert runtime.set_packet_review(job["id"], 0, status)["status"] == status
    restarted = BDConductor(runtime.path, clock=runtime.clock)
    persisted = restarted.snapshot()["jobs"][0]
    assert persisted["result"]["packets"][0] == packet
    assert persisted["packet_reviews"]["0"]["status"] == "human_review_required"
    assert restarted.events()[-1]["kind"] == "packet_reviewed"


@pytest.mark.parametrize(
    "index,status", [(-1, "watch"), (True, "watch"), (0, "approved"), (1, "watch")]
)
def test_packet_review_rejects_invalid_input(runtime, index, status):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    runtime.start(job["id"], lease["lease_token"])
    runtime.complete(job["id"], lease["lease_token"], {"packets": [{"organization": "Example"}]})
    with pytest.raises(ValueError):
        runtime.set_packet_review(job["id"], index, status)


def test_packet_review_rejects_noncompleted_and_empty_results(runtime):
    job = enqueue(runtime)
    with pytest.raises(ValueError):
        runtime.set_packet_review(job["id"], 0, "watch")
    lease = runtime.lease("worker")
    runtime.start(job["id"], lease["lease_token"])
    runtime.complete(job["id"], lease["lease_token"], {"packets": []})
    with pytest.raises(ValueError):
        runtime.set_packet_review(job["id"], 0, "watch")


@pytest.mark.parametrize(
    "schedule",
    [
        None,
        {},
        {"enabled": "false", "interval_seconds": 60},
        {"enabled": 0, "interval_seconds": 60},
        {"enabled": True, "interval_seconds": 0},
        {"enabled": True, "interval_seconds": True},
        {"enabled": True, "interval_seconds": float("nan")},
        {"enabled": True, "interval_seconds": 0.1},
    ],
)
def test_invalid_schedule_fails_closed(tmp_path, schedule):
    config = json.loads(CONFIG_PATH.read_text())
    config["bots"][0]["schedule"] = schedule
    with pytest.raises(ValueError):
        BDConductor(tmp_path / "bad-schedule.sqlite", config=config)


@pytest.mark.parametrize("bots", [[], None, ["bad"], [{"id": []}]])
def test_malformed_roster_fails_closed(tmp_path, bots):
    config = json.loads(CONFIG_PATH.read_text())
    config["bots"] = bots
    with pytest.raises(ValueError):
        BDConductor(tmp_path / "bad-roster.sqlite", config=config)


def test_concurrent_initialization_and_enqueue(tmp_path):
    path = tmp_path / "concurrent-init.sqlite"

    def initialize(_):
        store = BDConductor(path)
        return enqueue(store)["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(initialize, range(8)))
    assert len(set(ids)) == 1


def test_budget_blocked_reason_once_per_day_and_rollover_resumes(runtime):
    runtime.config["global_daily_budget"] = runtime.config["job_budget"].copy()
    first = enqueue(runtime)
    runtime.lease("worker")
    runtime.cancel(first["id"])
    queued = enqueue(runtime, "second")
    for _ in range(5):
        assert runtime.lease("worker") is None
    blocked = next(j for j in runtime.snapshot()["jobs"] if j["id"] == queued["id"])
    assert blocked["step"] == "daily_budget_exhausted"
    assert blocked["attempts"] == 0
    assert len([e for e in runtime.events() if e["kind"] == "daily_budget_exhausted"]) == 1
    restarted = BDConductor(runtime.path, config=runtime.config, clock=runtime.clock)
    assert restarted.lease("worker") is None
    assert len([e for e in runtime.events() if e["kind"] == "daily_budget_exhausted"]) == 1
    runtime.now[0] += 86400
    lease = restarted.lease("worker")
    assert lease["id"] == queued["id"]
    assert lease["step"] == "leased"


@pytest.mark.parametrize(
    "hosts",
    [
        "example.org",
        None,
        [123],
        ["https://example.org"],
        ["*.example.org"],
        ["Example.org"],
        ["127.0.0.1"],
        ["example.org:443"],
        ["example.local"],
        ["a..com"],
        ["-bad.example"],
        ["example.org/path"],
    ],
)
def test_source_allowlist_rejects_nonexact_or_unsafe_hosts(tmp_path, hosts):
    config = json.loads(CONFIG_PATH.read_text())
    config["bots"][0]["source_allowlist"] = hosts
    with pytest.raises(ValueError):
        BDConductor(tmp_path / "bad-hosts.sqlite", config=config)


def test_source_allowlist_accepts_exact_public_hosts(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["bots"][0]["source_allowlist"] = [
        "example.org",
        "www.example.org",
        "public-news.example.com",
    ]
    assert (
        BDConductor(tmp_path / "hosts.sqlite", config=config).bots["healthcare"]["source_allowlist"]
        == config["bots"][0]["source_allowlist"]
    )


def test_evidence_keys_durable_scoped_and_legacy_fallback(runtime):
    import hashlib

    def complete(bot, tenant, entity, packets):
        job = enqueue(runtime, entity, bot, tenant)
        lease = runtime.lease("worker")
        runtime.start(job["id"], lease["lease_token"])
        runtime.complete(job["id"], lease["lease_token"], {"packets": packets})

    complete(
        "healthcare",
        "synthetic-healthcare",
        "first",
        [
            {"evidence_key": "explicit"},
            {"source_url": "https://example.org/a", "provenance": {"source_hash": "abc"}},
            {"source_url": "https://example.org/missing-hash"},
        ],
    )
    complete("srg", "synthetic-srg", "other-tenant", [{"evidence_key": "other-tenant"}])
    # Distinct configured bots can share a tenant without sharing their evidence history.
    runtime.bots["pe-diligence"]["tenant_id"] = "synthetic-healthcare"
    complete("pe-diligence", "synthetic-healthcare", "other-bot", [{"evidence_key": "other-bot"}])
    expected = sorted(["explicit", hashlib.sha256(b"https://example.org/a\0abc").hexdigest()])
    assert runtime.evidence_keys("healthcare", "synthetic-healthcare") == expected
    restarted = BDConductor(runtime.path, clock=runtime.clock)
    assert restarted.evidence_keys("healthcare", "synthetic-healthcare") == expected
    assert restarted.evidence_keys("srg", "synthetic-srg") == ["other-tenant"]
    with pytest.raises(ValueError):
        restarted.evidence_keys("healthcare", "synthetic-srg")
    with pytest.raises(ValueError):
        restarted.evidence_keys("unknown", "synthetic-healthcare")


def test_evidence_keys_ignore_uncompleted_jobs(runtime):
    job = enqueue(runtime)
    lease = runtime.lease("worker")
    runtime.start(job["id"], lease["lease_token"])
    assert runtime.evidence_keys("healthcare", "synthetic-healthcare") == []
    runtime.fail(job["id"], lease["lease_token"], "provider_failed")
    assert runtime.evidence_keys("healthcare", "synthetic-healthcare") == []


def test_managed_leads_optional_configuration_scoped(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    bot = config["bots"][-1]
    bot["research_segment"] = "healthcare"
    bot["managed_leads"] = {
        "base_url": "https://managed-leads.vercel.app",
        "client_id_env": "BD_CLIENT_ID",
        "token_env": "BD_TOKEN",
        "workspace_id": bot["tenant_id"],
    }
    assert (
        BDConductor(tmp_path / "managed.sqlite", config=config).bots["srg"]["research_segment"]
        == "healthcare"
    )
    for invalid in (
        {},
        {"token": "literal-secret"},
        {**bot["managed_leads"], "workspace_id": "other"},
        {**bot["managed_leads"], "token_env": "literal-secret"},
        {**bot["managed_leads"], "base_url": "https://other.example"},
        {**bot["managed_leads"], "client_id_env": 5},
    ):
        modified = copy.deepcopy(config)
        modified["bots"][-1]["managed_leads"] = invalid
        with pytest.raises(ValueError):
            BDConductor(tmp_path / "invalid-managed.sqlite", config=modified)
    config["bots"][-1]["research_segment"] = "client-bd"
    with pytest.raises(ValueError):
        BDConductor(tmp_path / "invalid-segment.sqlite", config=config)


@pytest.mark.parametrize(
    "window",
    [
        {},
        None,
        {
            "start_at": "2026-10-24T06:00:00",
            "end_at": "2026-10-26T05:00:00Z",
            "timezone": "America/Chicago",
        },
        {"start_at": "2026-10-24T06:00:00Z", "end_at": "2026-10-24T06:00:00Z", "timezone": "UTC"},
        {"start_at": "2026-10-24T06:00:00Z", "end_at": "2026-10-23T06:00:00Z", "timezone": "UTC"},
        {
            "start_at": "2026-10-24T06:00:00Z",
            "end_at": "2026-10-26T05:00:00Z",
            "timezone": "Not/AZone",
        },
        {"start_at": 1, "end_at": "2026-10-26T05:00:00Z", "timezone": "UTC"},
    ],
)
def test_schedule_window_configuration_rejected_when_ambiguous(tmp_path, window):
    config = json.loads(CONFIG_PATH.read_text())
    config["schedule_window"] = window
    with pytest.raises(ValueError):
        BDConductor(tmp_path / "window.sqlite", config=config)


def test_schedule_window_accepts_aware_offsets_and_iana_zone(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["schedule_window"] = {
        "start_at": "2026-10-24T01:00:00-05:00",
        "end_at": "2026-10-26T00:00:00-05:00",
        "timezone": "America/Chicago",
    }
    assert (
        BDConductor(tmp_path / "window.sqlite", config=config).config["schedule_window"]
        == config["schedule_window"]
    )
