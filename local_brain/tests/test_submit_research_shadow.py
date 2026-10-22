"""No network: live-client transport is replaced with an HTTPX mock."""

import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest

from local_brain.command_center.research_shadow_contract import (
    build_submission,
    synthetic_assignment,
)
from local_brain.scripts.submit_research_shadow import BASE_URL, RunnerClient, RunnerClientError

TOKEN = "fake-test-only-credential-not-for-production-001"


@pytest.fixture
def credential(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "credential.json"
    path.write_text(
        json.dumps(
            {
                "baseUrl": BASE_URL,
                "workspaceId": "synthetic-test",
                "runnerId": "test-runner",
                "token": TOKEN,
            }
        )
    )
    path.chmod(0o600)
    return path


def connect(credential, monkeypatch, handler):
    original = httpx.Client

    def mocked(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", mocked)
    return RunnerClient.from_file(credential, "synthetic-test")


def test_submit_and_read_bound_receipt(credential, monkeypatch):
    from local_brain.command_center.research_shadow_contract import _normalized

    assignment = synthetic_assignment("synthetic-test")
    envelope = build_submission(
        assignment,
        "job-1",
        {"organization": "Synthetic Orchard", "location": "Madison", "signal_type": "expansion"},
    )
    receipt = {
        "schemaVersion": 1,
        "receiptId": str(
            uuid5(
                NAMESPACE_URL,
                json.dumps(
                    ["synthetic-test", "runner-shadow", "test-runner", envelope["idempotencyKey"]]
                ),
            )
        ),
        "mode": "shadow",
        "status": "shadow_received",
        "candidateCount": 1,
        "receivedAt": "2026-01-01T00:00:00Z",
        "assignmentRevision": assignment.assignment_revision,
        "payloadHash": hashlib.sha256(
            json.dumps(_normalized(envelope), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.host == "managed-leads.vercel.app"
        assert request.headers["authorization"] == "Bearer " + TOKEN
        assert "cookie" not in request.headers
        if request.url.path.endswith("/assignment"):
            return httpx.Response(200, json=assignment.to_wire())
        if request.method == "POST":
            assert json.loads(request.content) == envelope
            return httpx.Response(201, json={"receipt": receipt, "replayed": False})
        return httpx.Response(200, json=receipt)

    with connect(credential, monkeypatch, handler) as client:
        assert client.submit(envelope) == (201, {"receipt": receipt, "replayed": False})
        assert client.receipt(receipt["receiptId"]) == receipt
    assert len(calls) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        {"baseUrl": "http://managed-leads.vercel.app"},
        {"baseUrl": "https://example.org"},
        {"baseUrl": BASE_URL + "/"},
        {"workspaceId": "other"},
        {"token": "short"},
        {"cookie": "invalid"},
    ],
)
def test_private_credential_validation_without_requests(credential, mutation):
    data = json.loads(credential.read_text())
    data.update(mutation)
    credential.write_text(json.dumps(data))
    with pytest.raises(RunnerClientError) as error:
        RunnerClient.from_file(credential, "synthetic-test")
    assert TOKEN not in str(error.value)


def test_modes_and_symlinks(credential):
    credential.chmod(0o644)
    with pytest.raises(RunnerClientError):
        RunnerClient.from_file(credential, "synthetic-test")
    credential.chmod(0o600)
    link = credential.parent / "link.json"
    link.symlink_to(credential)
    with pytest.raises(RunnerClientError):
        RunnerClient.from_file(link, "synthetic-test")
    credential.parent.chmod(0o755)
    with pytest.raises(RunnerClientError):
        RunnerClient.from_file(credential, "synthetic-test")


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 409, 422, 429, 500])
def test_errors_and_redirects_never_follow_or_echo(credential, monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://example.org/"}, text=TOKEN)

    with (
        connect(credential, monkeypatch, handler) as client,
        pytest.raises(RunnerClientError) as error,
    ):
        client.assignment()
    assert TOKEN not in str(error.value)
    assert len(calls) == 1


def test_wrong_server_workspace_stops_before_submission(credential, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=synthetic_assignment("synthetic-foreign").to_wire())

    with (
        connect(credential, monkeypatch, handler) as client,
        pytest.raises(RunnerClientError, match="binding"),
    ):
        client.assignment()
    assert len(calls) == 1


def test_server_cookies_rejected_and_cleared(credential, monkeypatch):
    with connect(
        credential,
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json=synthetic_assignment("synthetic-test").to_wire(),
            headers={"set-cookie": "session=not-allowed"},
        ),
    ) as client:
        with pytest.raises(RunnerClientError, match="cookies"):
            client.assignment()
        assert not client._http.cookies


def test_timeout_redacts_request_exception(credential, monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout(TOKEN, request=request)

    with (
        connect(credential, monkeypatch, handler) as client,
        pytest.raises(RunnerClientError) as error,
    ):
        client.assignment()
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("payloadHash", "0" * 64),
        ("assignmentRevision", "0" * 64),
        ("candidateCount", 2),
        ("receiptId", "00000000-0000-0000-0000-000000000001"),
    ],
)
def test_submit_rejects_uncorrelated_receipt(credential, monkeypatch, field, value):
    from local_brain.command_center.research_shadow_contract import MockRunnerTransport

    assignment = synthetic_assignment("synthetic-test")
    envelope = build_submission(
        assignment,
        "stable-job",
        {"organization": "Synthetic Orchard", "location": "Madison", "signal_type": "expansion"},
    )
    mock = MockRunnerTransport([assignment], {"test-runner": ("synthetic-test", TOKEN)})

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=assignment.to_wire())
        status, body = mock.submit(json.loads(request.content), dict(request.headers))
        body["receipt"][field] = value
        return httpx.Response(status, json=body)

    with (
        connect(credential, monkeypatch, handler) as client,
        pytest.raises(RunnerClientError, match="mismatch"),
    ):
        client.submit(envelope)


def test_exact_retry_preserves_original_key_and_payload(credential, monkeypatch):
    from local_brain.command_center.research_shadow_contract import MockRunnerTransport

    assignment = synthetic_assignment("synthetic-test")
    envelope = build_submission(
        assignment,
        "stable-job",
        {"organization": "Synthetic Orchard", "location": "Madison", "signal_type": "expansion"},
    )
    mock = MockRunnerTransport([assignment], {"test-runner": ("synthetic-test", TOKEN)})
    bodies = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=assignment.to_wire())
        bodies.append(request.content)
        status, body = mock.submit(json.loads(request.content), dict(request.headers))
        return httpx.Response(status, json=body)

    with connect(credential, monkeypatch, handler) as client:
        first = client.submit(envelope)
        second = client.submit(envelope)
    assert first[0] == 201 and second[0] == 200
    assert first[1]["receipt"] == second[1]["receipt"]
    assert bodies[0] == bodies[1]


def test_receipt_lookup_rejects_returned_other_id(credential, monkeypatch):
    assignment = synthetic_assignment("synthetic-test")
    requested = "00000000-0000-0000-0000-000000000001"
    body = {
        "schemaVersion": 1,
        "receiptId": "00000000-0000-0000-0000-000000000002",
        "mode": "shadow",
        "status": "shadow_received",
        "candidateCount": 1,
        "receivedAt": "2026-01-01T00:00:00Z",
        "assignmentRevision": assignment.assignment_revision,
        "payloadHash": "0" * 64,
    }

    def handler(request):
        return httpx.Response(
            200, json=assignment.to_wire() if request.url.path.endswith("assignment") else body
        )

    with (
        connect(credential, monkeypatch, handler) as client,
        pytest.raises(RunnerClientError, match="receipt_id_mismatch"),
    ):
        client.receipt(requested)
