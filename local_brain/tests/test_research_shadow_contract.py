"""Offline runner-contract assertions; no HTTP server or credentials required."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from local_brain.command_center.research_shadow_contract import (
    AssignmentSnapshot,
    ContractError,
    MockRunnerTransport,
    build_submission,
    synthetic_assignment,
)

TOKEN = "fake-synthetic-token-never-use-live-0001"
OTHER_TOKEN = "fake-synthetic-token-never-use-live-0002"
FOREIGN_TOKEN = "fake-synthetic-token-never-use-live-0003"
HEADERS = {"X-Runner-Client": "replay", "Authorization": f"Bearer {TOKEN}"}
OTHER = {"X-Runner-Client": "other", "Authorization": f"Bearer {OTHER_TOKEN}"}
FOREIGN = {"X-Runner-Client": "foreign", "Authorization": f"Bearer {FOREIGN_TOKEN}"}
CANDIDATE = {"organization": "Synthetic Orchard", "location": "Madison", "signal_type": "expansion"}


@pytest.fixture
def setup():
    assignment = synthetic_assignment("synthetic-demo")
    transport = MockRunnerTransport(
        [assignment, synthetic_assignment("synthetic-other")],
        {
            "replay": ("synthetic-demo", TOKEN),
            "other": ("synthetic-demo", OTHER_TOKEN),
            "foreign": ("synthetic-other", FOREIGN_TOKEN),
        },
        now=lambda: "2026-01-01T00:00:00+00:00",
    )
    return assignment, transport


def test_minimal_wire_shape_and_snapshot_immutability(setup):
    assignment, transport = setup
    status, wire = transport.get_assignment(HEADERS)
    assert status == 200
    snapshot = AssignmentSnapshot.from_wire(wire)
    envelope = build_submission(snapshot, "job-001", CANDIDATE)
    assert set(envelope) == {
        "schemaVersion",
        "mode",
        "idempotencyKey",
        "assignmentRevision",
        "source",
    }
    assert str(UUID(envelope["idempotencyKey"])) == envelope["idempotencyKey"]
    assert envelope["source"]["brief"] == wire["brief"]
    c = envelope["source"]["candidates"][0]
    assert set(c) == {
        "id",
        "org",
        "whyFit",
        "identityStatus",
        "decision",
        "gaps",
        "claims",
        "contacts",
    }
    assert c["identityStatus"] == "unknown" and c["decision"] == "pending"
    assert c["claims"] == c["contacts"] == []
    assert c["gaps"]
    assert "probability" not in str(envelope) and "source_hash" not in str(envelope)
    wire["brief"]["exclusions"].append("org:Injected")
    assert snapshot.to_wire()["brief"]["exclusions"] == []
    with pytest.raises(FrozenInstanceError):
        snapshot.workspace_id = "synthetic-forged"
    assert TOKEN not in repr(transport) + str(envelope) + str(transport.get_assignment(HEADERS))


def test_submission_idempotency_receipt_and_revision_order(setup):
    assignment, transport = setup
    envelope = build_submission(assignment, "job-001", CANDIDATE)
    status, body = transport.submit(envelope, HEADERS)
    assert status == 201 and body["replayed"] is False
    receipt = body["receipt"]
    assert set(receipt) == {
        "schemaVersion",
        "receiptId",
        "mode",
        "status",
        "candidateCount",
        "receivedAt",
        "assignmentRevision",
        "payloadHash",
    }
    assert receipt["status"] == "shadow_received"
    assert "approved" not in receipt
    assert transport.get_receipt(receipt["receiptId"], HEADERS) == (200, receipt)
    assert transport.submit(envelope, HEADERS) == (200, {"receipt": receipt, "replayed": True})
    updated = assignment.to_wire()
    updated["assignmentRevision"] = "f" * 64
    updated["brief"]["icp"] = "New synthetic scope"
    transport.replace_assignment(AssignmentSnapshot.from_wire(updated))
    assert transport.submit(envelope, HEADERS)[0] == 200
    with pytest.raises(ContractError) as error:
        transport.submit(build_submission(assignment, "job-002", CANDIDATE), HEADERS)
    assert error.value.status_code == 409
    changed = deepcopy(envelope)
    changed["source"]["candidates"][0]["org"] = "Other Synthetic"
    with pytest.raises(ContractError) as error:
        transport.submit(changed, HEADERS)
    assert error.value.status_code == 409
    receipt["status"] = "forged"
    assert transport.get_receipt(receipt["receiptId"], HEADERS)[1]["status"] == "shadow_received"


@pytest.mark.parametrize(
    "headers,query,status",
    [
        ({}, None, 401),
        ({**HEADERS, "Cookie": "session=fake"}, None, 401),
        ({**HEADERS, "authorization": "Bearer fake-other"}, None, 401),
        ({**HEADERS, "X-Runner-Client": "other"}, None, 401),
        ({**HEADERS, "Authorization": "Bearer wrong"}, None, 401),
        (HEADERS, {"token": TOKEN}, 422),
    ],
)
def test_auth_query_cookie_binding_rejection(setup, headers, query, status):
    _, transport = setup
    with pytest.raises(ContractError) as error:
        transport.get_assignment(headers, query)
    assert error.value.status_code == status
    assert TOKEN not in str(error.value)


def test_foreign_receipt_and_scoped_idempotency(setup):
    assignment, transport = setup
    envelope = build_submission(assignment, "job-001", CANDIDATE)
    _, result = transport.submit(envelope, HEADERS)
    receipt_id = result["receipt"]["receiptId"]
    for headers in (OTHER, FOREIGN):
        with pytest.raises(ContractError) as error:
            transport.get_receipt(receipt_id, headers)
        assert error.value.status_code == 404
    _, other = transport.submit(envelope, OTHER)
    assert other["receipt"]["receiptId"] != receipt_id


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(workspaceId="synthetic-other"),
        lambda p: p.update(schemaVersion=True),
        lambda p: p.update(idempotencyKey="not-uuid"),
        lambda p: p["source"]["candidates"][0].update(identityStatus="confirmed"),
        lambda p: p["source"]["candidates"][0].update(decision="approved"),
        lambda p: p["source"].update(probability=0.99),
        lambda p: p["source"]["brief"].update(icp="Forged scope"),
    ],
)
def test_unsupported_assertions_and_wrong_brief_rejected(setup, mutate):
    assignment, transport = setup
    payload = build_submission(assignment, "job-001", CANDIDATE)
    mutate(payload)
    with pytest.raises(ContractError) as error:
        transport.submit(payload, HEADERS)
    assert error.value.status_code == 422


def test_only_explicit_fake_tokens_and_synthetic_scopes():
    with pytest.raises(ValueError):
        MockRunnerTransport(
            [synthetic_assignment("synthetic-demo")],
            {"replay": ("synthetic-demo", "real-looking-private-token-material-012345")},
        )
    with pytest.raises(ValueError):
        synthetic_assignment("production")


def test_duplicate_credential_digest_cannot_cross_runner_binding():
    assignment = synthetic_assignment("synthetic-demo")
    with pytest.raises(ValueError, match="unique"):
        MockRunnerTransport(
            [assignment], {"first": ("synthetic-demo", TOKEN), "second": ("synthetic-demo", TOKEN)}
        )
