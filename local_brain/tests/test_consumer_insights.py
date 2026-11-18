import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_brain.command_center.bd_conductor import CONFIG_PATH, BDConductor
from local_brain.command_center.bd_pipeline import proposal_packet, synthetic_fixture
from local_brain.command_center.bd_routes import create_router
from local_brain.command_center.bd_service import BDService


@pytest.fixture
def service(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["mode"] = "live"
    original = config["bots"][0]
    config["bots"] = [
        {**original, "id": "first", "tenant_id": "one"},
        {**original, "id": "second", "tenant_id": "two"},
    ]
    store = BDConductor(tmp_path / "insights.sqlite", config=config, clock=lambda: 1795000000)
    service = BDService(store)
    app = FastAPI()
    app.include_router(create_router(factory=lambda: service))
    return service, TestClient(app)


def packet(organization="Orchard Clinic"):
    source, extraction, judgments = synthetic_fixture("healthcare")
    prior = extraction["organization"]
    extraction["organization"] = organization
    extraction["quote"] = extraction["quote"].replace(prior, organization)
    source["title"] = source["title"].replace(prior, organization)
    judgments["model"] = "recorded-public-research"
    return proposal_packet(
        {"mode": "live", "tenant_id": "one", "segment": "healthcare"},
        source,
        extraction,
        judgments,
        {},
    )


def insert(
    service,
    number,
    *,
    tenant="one",
    bot="first",
    state="completed",
    mode="live",
    packets=None,
    reviews=None,
    scope_override=None,
    outcome=None,
):
    row = {
        "id": f"job-{number}",
        "tenant_id": tenant,
        "bot_id": bot,
        "mode": mode,
        "state": state,
        "created_at": 1794800000 + number,
        "finished_at": 1794800000 + number,
        "result": {
            "packets": [packet()] if packets is None else packets,
            **({"research_outcome": outcome} if outcome is not None else {}),
        },
        "packet_reviews": reviews or {},
    }
    with sqlite3.connect(service.store.path) as db:
        db.execute(
            "INSERT INTO bd_jobs (id,tenant_id,entity_id,task,state,data) VALUES (?,?,?,?,?,?)",
            (
                row["id"],
                scope_override or tenant,
                f"entity-{number}",
                "research",
                state,
                json.dumps(row),
            ),
        )
    return row


def test_insights_counts_all_records_and_groups_exact_names_per_tenant_without_writes(
    service, monkeypatch
):
    current, client = service
    for index in range(220):
        insert(current, index)
    insert(current, 220, tenant="two", bot="second")
    insert(current, 221, packets=[packet("ORCHARD CLINIC")])
    # Earlier jobs remain counted after the existing snapshot's 200-job cap.
    before = hashlib.sha256(Path(current.store.path).read_bytes()).hexdigest()
    monkeypatch.setattr(
        current.store, "snapshot", lambda: pytest.fail("read called mutable snapshot")
    )
    response = client.get("/api/bd/insights")
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["completed_jobs"] == 222
    assert body["totals"]["packets"] == 222
    assert body["job_outcomes"] == {
        "supported": 222,
        "inconclusive": 0,
        "rejected": 0,
        "unknown": 0,
    }
    assert body["totals"]["prospects"] == 2
    assert sorted(row["packet_count"] for row in body["prospects"]) == [1, 221]
    assert sum(row["count"] for row in body["review"]) == 222
    assert body["totals"]["insights"] == 2
    assert body["source_coverage"] == {"with_url": 222, "with_quote": 222, "domains": 1}
    assert len({row["id"] for row in body["prospects"]}) == 2
    for prospect in body["prospects"]:
        assert prospect["id"] in prospect["href"]
        assert all(
            source["id"] == f"{source['job_id']}:{source['index']}"
            for source in prospect["sources"]
        )
    assert hashlib.sha256(Path(current.store.path).read_bytes()).hexdigest() == before
    assert current._worker is None


def test_insights_excludes_unconfigured_scopes_and_synthetic_evidence(service):
    current, client = service
    insert(current, 1)
    insert(current, 2, tenant="foreign")
    insert(current, 3, bot="removed")
    insert(current, 4, tenant="foreign", scope_override="one")
    insert(current, 5, mode="replay")
    synthetic = packet()
    synthetic["provenance"]["source_kind"] = "synthetic_fixture"
    insert(current, 6, packets=[synthetic])
    uncertain = packet()
    uncertain["judgments"] = {}
    rejected = packet()
    rejected["disposition"] = "reject"
    insert(current, 7, packets=[uncertain, rejected])
    insert(current, 8, state="failed", packets=[])
    body = client.get("/api/bd/insights").json()
    assert body["totals"]["packets"] == 3
    assert body["totals"]["prospects"] == 1
    assert body["totals"]["failed_jobs"] == 1
    assert body["excluded_replay_jobs"] == 1
    assert body["excluded_synthetic_packets"] == 1
    assert body["evidence"] == {"supported": 1, "inconclusive": 1, "rejected": 1, "unknown": 0}
    assert all(row["tenant_id"] == "one" for row in body["prospects"])


def test_segment_filter_and_review_details_apply_to_exact_packet(service):
    current, client = service
    second = current.store.bots["second"]
    second["segment"] = "client-bd"
    second["research_segment"] = "healthcare"
    insert(current, 1)
    latest = packet("Another Clinic")
    latest["source_url"] = "https://example.org/primary"
    insert(
        current,
        2,
        tenant="two",
        bot="second",
        packets=[packet(), latest],
        reviews={"1": {"status": "watch"}},
    )
    body = client.get("/api/bd/insights?segment=client-bd").json()
    assert body["segment"] == "client-bd"
    assert body["totals"]["packets"] == 2 and body["totals"]["prospects"] == 2
    chosen = next(row for row in body["prospects"] if row["organization"] == "Another Clinic")
    assert chosen["job_id"] == "job-2" and chosen["index"] == 1
    assert chosen["review_status"] == "watch"
    assert chosen["sources"][0]["id"] == "job-2:1"
    assert chosen["sources"][0]["source_url"] == "https://example.org/primary"
    assert chosen["insight_status"] == "unverified_hypothesis"
    assert client.get("/api/bd/insights?segment=foreign").status_code == 422


def test_unavailable_store_does_not_create_empty_database(service, tmp_path):
    current, client = service
    path = tmp_path / "absent.sqlite"
    current.store.path = str(path)
    assert client.get("/api/bd/insights").status_code == 503
    assert not path.exists()


def test_research_outcome_mix_includes_withheld_jobs_without_retained_packets(service):
    current, client = service
    insert(current, 1, packets=[], outcome="rejected")
    insert(current, 2, packets=[], outcome="inconclusive")
    insert(current, 3, packets=[])
    body = client.get("/api/bd/insights").json()
    assert body["totals"]["completed_jobs"] == 3
    assert body["totals"]["packets"] == 0 and body["totals"]["prospects"] == 0
    assert body["job_outcomes"] == {"supported": 0, "inconclusive": 1, "rejected": 1, "unknown": 1}
    assert body["source_coverage"] == {"with_quote": 0, "with_url": 0, "domains": 0}


def test_latest_review_does_not_hide_earlier_pending_evidence(service):
    current, client = service
    insert(current, 1)
    insert(current, 2, reviews={"0": {"status": "watch"}})
    body = client.get("/api/bd/insights").json()
    assert body["totals"]["awaiting_review"] == 1
    assert body["totals"]["prospects"] == 1
    prospect = body["prospects"][0]
    assert prospect["job_id"] == "job-2"
    assert prospect["review_status"] == "watch"
    assert prospect["packet_count"] == 2
    assert prospect["pending_packet_count"] == 1
    assert {source["id"]: source["review_status"] for source in prospect["sources"]} == {
        "job-2:0": "watch",
        "job-1:0": "human_review_required",
    }
