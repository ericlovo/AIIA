from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from local_brain.command_center.research_shadow_store import ResearchShadowStore


@pytest.fixture
def stores(tmp_path):
    now = [100.0]
    path = tmp_path / "shadow.sqlite"
    return ResearchShadowStore(path, lambda: now[0]), path, now


def enqueue(store, tenant="one", assignment="assignment", **budgets):
    return store.enqueue(tenant, assignment, "hash", "v1", {"budgets": budgets})


def running(store, tenant="one", **budgets):
    job = enqueue(store, tenant, **budgets)
    lease = store.claim(tenant, job["id"], "worker")
    return store.start(tenant, job["id"], lease["lease_token"])


def test_idempotency_and_tenant_isolation(stores):
    store, _, _ = stores
    job = enqueue(store)
    assert enqueue(store) == job
    with pytest.raises(ValueError, match="different configuration"):
        enqueue(store, max_attempts=1)
    other = enqueue(store, "two")
    assert other["id"] != job["id"]
    assert store.get("two", job["id"]) is None
    assert store.claim("two", job["id"], "worker") is None
    assert store.cancel("two", job["id"]) is None
    assert store.list_jobs("one") == [job]
    store.set_killed("one", True)
    assert store.claim("two", other["id"], "worker")


def test_distinct_connections_race_for_lease_and_concurrency(stores):
    store, path, now = stores
    first = enqueue(store)
    second = enqueue(store, assignment="other")
    barrier = Barrier(2)

    def claim(job):
        connection = ResearchShadowStore(path, lambda: now[0])
        barrier.wait()
        return connection.claim("one", job["id"], "worker")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, [first, second]))
    assert sum(result is not None for result in results) == 1
    winner = next(result for result in results if result)
    assert store.claim("one", winner["id"], "duplicate") is None


def test_reclaim_fences_old_worker_and_retains_checkpoint_on_restart(stores):
    store, path, now = stores
    job = running(store)
    token = job["lease_token"]
    assert store.checkpoint("one", job["id"], token, {"cursor": 4})
    now[0] += 31
    assert store.complete("one", job["id"], token, {}) is None
    assert store.claim("one", job["id"], "next") is None
    now[0] += 1
    fresh = ResearchShadowStore(path, lambda: now[0])
    claimed = fresh.claim("one", job["id"], "next")
    assert claimed["attempts"] == 2
    assert claimed["checkpoint"] == {"cursor": 4}
    assert claimed["lease_token"] != token
    assert fresh.start("one", job["id"], token) is None
    assert fresh.checkpoint("one", job["id"], token, {}) is None
    assert fresh.fail("one", job["id"], token) is None
    assert fresh.start("one", job["id"], claimed["lease_token"])
    completed = fresh.complete("one", job["id"], claimed["lease_token"], {"review": "pending"})
    assert completed["state"] == "completed"
    assert fresh.cancel("one", job["id"]) == completed
    assert fresh.claim("one", job["id"], "again") is None


@pytest.mark.parametrize("control", ["pause", "kill"])
def test_controls_block_all_worker_transitions_and_persist(stores, control):
    store, path, now = stores
    job = running(store)
    token = job["lease_token"]
    if control == "pause":
        store.set_paused("one", "assignment", True)
    else:
        store.set_killed("one", True)
    fresh = ResearchShadowStore(path, lambda: now[0])
    assert fresh.claim("one", job["id"], "other") is None
    assert fresh.start("one", job["id"], token) is None
    assert fresh.heartbeat("one", job["id"], token) is None
    assert fresh.checkpoint("one", job["id"], token, {}) is None
    assert fresh.complete("one", job["id"], token, {}) is None
    assert fresh.fail("one", job["id"], token) is None
    cancelled = fresh.cancel("one", job["id"])
    assert cancelled["state"] == "cancelled"
    assert fresh.complete("one", job["id"], token, {}) is None


def test_attempt_and_wall_budgets(stores):
    store, _, now = stores
    job = running(store, max_attempts=2)
    failed = store.fail("one", job["id"], job["lease_token"])
    assert failed["state"] == "queued"
    assert store.claim("one", job["id"], "next") is None
    now[0] += 1
    lease = store.claim("one", job["id"], "next")
    assert store.fail("one", job["id"], lease["lease_token"])["state"] == "failed"
    assert store.claim("one", job["id"], "third") is None
    timed = running(store, "two", max_elapsed_seconds=2)
    now[0] += 2
    assert store.heartbeat("two", timed["id"], timed["lease_token"]) is None
    assert store.get("two", timed["id"])["reason"] == "budget_exhausted"


def test_heartbeat_extends_lease_but_not_overall_deadline(stores):
    store, _, now = stores
    job = running(store, max_elapsed_seconds=40)
    now[0] += 20
    assert store.heartbeat("one", job["id"], job["lease_token"])["lease_expires_at"] == 150
    now[0] += 20
    assert store.complete("one", job["id"], job["lease_token"], {}) is None
    assert store.get("one", job["id"])["state"] == "failed"


def test_scope_and_budget_validation(stores):
    store, _, _ = stores
    with pytest.raises(ValueError):
        store.list_jobs("")
    with pytest.raises(ValueError):
        enqueue(store, max_attempts=-1)
    with pytest.raises(ValueError):
        enqueue(store, max_concurrency=1.5)
    job = enqueue(store)
    with pytest.raises(ValueError):
        store.claim("one", job["id"], "worker", lease_seconds=float("nan"))


def test_enqueue_race_is_idempotent_across_connections(stores):
    store, path, now = stores
    barrier = Barrier(2)

    def create(_):
        other = ResearchShadowStore(path, lambda: now[0])
        barrier.wait()
        return enqueue(other)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(create, range(2)))
    assert jobs[0] == jobs[1]
    assert len(store.list_jobs("one")) == 1


def test_resume_after_pause_and_kill_without_changing_assignment(stores):
    store, path, now = stores
    job = enqueue(store)
    store.set_paused("one", "assignment", True)
    assert store.claim("one", job["id"], "worker") is None
    store.set_paused("one", "assignment", False)
    store.set_killed("one", True)
    assert store.claim("one", job["id"], "worker") is None
    store.set_killed("one", False)
    fresh = ResearchShadowStore(path, lambda: now[0])
    assert fresh.claim("one", job["id"], "worker")["attempts"] == 1


def test_config_tenant_binding_and_json_safety(stores):
    store, _, _ = stores
    with pytest.raises(ValueError, match="tenant_id"):
        store.enqueue("one", "a", "hash", "v1", {"tenant_id": "two"})
    with pytest.raises(ValueError):
        store.enqueue("one", "a", "hash", "v1", {"value": float("nan")})
    with pytest.raises(ValueError):
        store.enqueue("one", "a", "hash", "v1", {"budgets": []})
    job = running(store)
    with pytest.raises(ValueError):
        store.checkpoint("one", job["id"], job["lease_token"], {"value": float("inf")})
    assert store.get("one", job["id"])["checkpoint"] is None


def test_get_controls_is_tenant_and_assignment_scoped(stores):
    store, _, _ = stores
    assert store.get_controls("one", "assignment") == {"paused": False, "killed": False}
    store.set_paused("one", "assignment", True)
    store.set_killed("one", True)
    assert store.get_controls("one", "assignment") == {"paused": True, "killed": True}
    assert store.get_controls("one", "other") == {"paused": False, "killed": True}
    assert store.get_controls("two", "assignment") == {"paused": False, "killed": False}
    store.set_paused("one", "assignment", False)
    store.set_killed("one", False)
    assert store.get_controls("one", "assignment") == {"paused": False, "killed": False}
