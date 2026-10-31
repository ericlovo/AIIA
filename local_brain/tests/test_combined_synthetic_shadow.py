"""Combined pipeline using exclusively synthetic in-memory provider doubles."""

import json
from copy import deepcopy
from uuid import uuid4

import pytest

from local_brain.jev_judgment import JUDGMENTS, InputError
from local_brain.scripts.combined_synthetic_shadow import (
    MODEL,
    REMOTE,
    SOURCE,
    CombinedError,
    bound_request,
    jev_payload,
    run,
    run_mini_loopback,
)

AIRGAP = {"enabled": True, "egress": {"typesafe.signals": "disabled", "web.fetch": "disabled"}}
BINDING = {"model": "qwen3:8b", "host": "synthetic-host"}
EXTRACTION = {
    "organization": "Synthetic Cedar Works",
    "location": "Wisconsin",
    "event": "expanding",
    "quote": SOURCE,
}


def jev_response(probability=0.95):
    return {
        "model": MODEL,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "answers": {
            **{key: {"type": "noul", "noul": probability} for key in JUDGMENTS},
            "evidence_link": {
                "type": "choice",
                "choice": "linked",
                "confidence": 0.95,
                "probabilities": {"linked": 0.95, "unresolved": 0.03, "unrelated": 0.02},
            },
        },
    }


class Runner:
    def __init__(self, workspace):
        self.workspace = workspace
        self.submissions = []
        self.saved = None

    def assignment(self):
        return {
            "schemaVersion": 1,
            "mode": "shadow",
            "workspaceId": self.workspace,
            "assignmentRevision": "a" * 64,
            "brief": {
                "version": "synthetic-v1",
                "icp": "Synthetic only",
                "offer": "Test",
                "industry": "",
                "exclusions": [],
                "evidenceMaxAgeDays": None,
                "contactRequirement": "optional",
            },
        }

    def submit(self, payload):
        self.submissions.append(deepcopy(payload))
        self.saved = {"receiptId": str(uuid4()), "status": "shadow_received"}
        return 201, {"receipt": self.saved, "replayed": False}

    def validate_submission_receipt(self, submission, receipt):
        assert submission == self.submissions[-1]
        if receipt != self.saved:
            raise CombinedError("receipt_submission_mismatch")

    def receipt(self, receipt_id):
        assert receipt_id == self.saved["receiptId"]
        return self.saved


def test_combined_outputs_bound_extraction_and_resumes_without_provider_calls(tmp_path):
    workspace = str(uuid4())
    runner = Runner(workspace)
    calls = []

    def qwen():
        calls.append("qwen")
        assert (tmp_path / "qwen.pending.json").exists()
        return {"model": "qwen3:8b", "extraction": EXTRACTION}

    def jev(payload):
        calls.append("jev")
        assert (tmp_path / "jev.pending.json").exists()
        assert payload["state"]["extraction"]["organization"]["value"] == EXTRACTION["organization"]
        return jev_response()

    result = run(tmp_path, workspace, BINDING, lambda: AIRGAP, qwen, jev, runner)
    assert result["status"] == "shadow_received"
    assert calls == ["qwen", "jev"]
    assert runner.submissions[0]["source"]["candidates"][0]["decision"] == "pending"
    assert runner.submissions[0]["source"]["candidates"][0]["identityStatus"] == "unknown"
    assert run(tmp_path, workspace, BINDING, lambda: AIRGAP, qwen, jev, runner) == result
    assert calls == ["qwen", "jev"] and len(runner.submissions) == 1
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["qwen", "jev"])
def test_uncertain_provider_not_repeated(tmp_path, stage):
    workspace = str(uuid4())
    runner = Runner(workspace)
    calls = []

    def qwen():
        calls.append("qwen")
        if stage == "qwen":
            raise TimeoutError()
        return {"model": "qwen3:8b", "extraction": EXTRACTION}

    def jev(_):
        calls.append("jev")
        raise TimeoutError()

    with pytest.raises(TimeoutError):
        run(tmp_path, workspace, BINDING, lambda: AIRGAP, qwen, jev, runner)
    previous = calls.copy()
    with pytest.raises(CombinedError, match="uncertain"):
        run(tmp_path, workspace, BINDING, lambda: AIRGAP, qwen, jev, runner)
    assert calls == previous and not runner.submissions


@pytest.mark.parametrize(
    "answer,allowed",
    [
        (jev_response(0.7), True),
        ({"model": "jev-latest"}, False),
        ({"model": MODEL, "usage": {}, "answers": {}}, False),
        ({**jev_response(), "answers": {}}, False),
    ],
)
def test_valid_abstention_staged_with_gap_but_malformed_blocks(tmp_path, answer, allowed):
    workspace = str(uuid4())
    runner = Runner(workspace)
    args = (
        tmp_path,
        workspace,
        BINDING,
        lambda: AIRGAP,
        lambda: {"model": "qwen3:8b", "extraction": EXTRACTION},
        lambda _: answer,
        runner,
    )
    if allowed:
        result = run(*args)
        assert result["judgment"] == "abstain"
        assert "did not support" in runner.submissions[0]["source"]["candidates"][0]["gaps"][0]
    else:
        with pytest.raises(CombinedError):
            run(*args)
        assert not runner.submissions


def test_airgap_change_detected_and_disabled_required(tmp_path):
    workspace = str(uuid4())
    runner = Runner(workspace)
    health = iter([AIRGAP, {**AIRGAP, "changed": True}, {**AIRGAP, "changed": True}])
    with pytest.raises(CombinedError, match="changed"):
        run(
            tmp_path,
            workspace,
            BINDING,
            lambda: next(health),
            lambda: {"model": "qwen3:8b", "extraction": EXTRACTION},
            lambda _: jev_response(),
            runner,
        )
    assert not runner.submissions
    second = tmp_path / "second"
    with pytest.raises(CombinedError, match="airgap"):
        run(
            second,
            workspace,
            BINDING,
            lambda: {"enabled": False},
            lambda: pytest.fail("provider called"),
            lambda _: pytest.fail("provider called"),
            runner,
        )


def test_unsupported_qwen_values_cannot_get_jev_judgment():
    with pytest.raises(InputError):
        bound_request(
            {**EXTRACTION, "organization": "Invented Other"},
            "synthetic",
            "job",
            "2026-01-01T00:00:00Z",
        )
    request = bound_request(EXTRACTION, "synthetic", "job", "2026-01-01T00:00:00Z")
    assert jev_payload(request)["model"] == MODEL
    assert request.policy.min_probability == request.policy.min_link_confidence == 0.8


def test_saved_envelope_precedes_post_and_retry_is_exact(tmp_path):
    workspace = str(uuid4())
    runner = Runner(workspace)
    original = runner.submit
    seen = []

    def uncertain(payload):
        assert json.loads((tmp_path / "envelope.json").read_text()) == payload
        seen.append(deepcopy(payload))
        if len(seen) == 1:
            raise TimeoutError()
        return original(payload)

    runner.submit = uncertain
    args = (
        tmp_path,
        workspace,
        BINDING,
        lambda: AIRGAP,
        lambda: {"model": "qwen3:8b", "extraction": EXTRACTION},
        lambda _: jev_response(),
        runner,
    )
    with pytest.raises(TimeoutError):
        run(*args)
    run(*args)
    assert seen[0] == seen[1]


def test_remote_call_uses_stdin_fixed_script_and_quoted_paths(monkeypatch):
    from types import SimpleNamespace

    from local_brain.scripts import combined_synthetic_shadow as combined

    seen = []

    def ssh(argv, **kwargs):
        seen.append((argv, kwargs))
        assert argv[0] == "ssh" and "shell" not in kwargs
        assert "StrictHostKeyChecking=yes" in argv
        assert kwargs["timeout"] == 110
        assert "http://127.0.0.1:11434/api/show" in kwargs["input"]
        assert "get('format')!='gguf'" in kwargs["input"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(AIRGAP))

    monkeypatch.setattr(combined.subprocess, "run", ssh)
    assert (
        combined.remote_call(
            "synthetic-host", "/test path/python", "/test repo", "qwen3:8b", "health"
        )
        == AIRGAP
    )
    assert seen[0][0][-1] == "'/test path/python' -"
    with pytest.raises(CombinedError):
        combined.remote_call("-oProxyCommand=bad", "/python", "/repo", "qwen3:8b", "health")
    with pytest.raises(CombinedError):
        combined.remote_call("host", "/python", "/repo", "cloud-model", "qwen")
    assert len(seen) == 1


@pytest.mark.parametrize("credential_key", ["token", "TYPESAFE_API_KEY"])
def test_workstation_jev_transport_pinned_no_retry_or_redirect(
    tmp_path, monkeypatch, credential_key
):
    import httpx

    from local_brain.scripts import combined_synthetic_shadow as combined

    tmp_path.chmod(0o700)
    credential = tmp_path / "jev.json"
    credential.write_text(json.dumps({credential_key: "fake-test-only-jev-token"}))
    credential.chmod(0o600)
    original = httpx.Client
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert json.loads(request.content)["model"] == MODEL
        return httpx.Response(307, headers={"location": "https://example.org"})

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False and kwargs["trust_env"] is False
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(combined.httpx, "Client", client)
    request = bound_request(EXTRACTION, "synthetic", "job", "2026-01-01T00:00:00Z")
    with pytest.raises(CombinedError, match="jev_http_307"):
        combined.call_jev(jev_payload(request), credential)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "models,format_name,blocked",
    [([], "gguf", False), ([{"name": "busy"}], "gguf", True), ([], "remote", True)],
)
def test_remote_loopback_request_bounded_and_no_contention(
    monkeypatch, models, format_name, blocked
):
    import httpx

    original = httpx.Client
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1"
        path = request.url.path
        payloads = {
            "/health": {"airgap": AIRGAP},
            "/api/ps": {"models": models},
            "/api/show": {"details": {"format": format_name}},
        }
        if path == "/api/chat":
            body = json.loads(request.content)
            assert body["format"] == "json" and body["keep_alive"] == 0
            assert body["stream"] is False and body["think"] is False
            assert body["options"] == {"temperature": 0, "num_predict": 256, "num_ctx": 2048}
            return httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "done": True,
                    "done_reason": "stop",
                    "message": {"content": json.dumps(EXTRACTION)},
                },
            )
        return httpx.Response(200, json=payloads[path])

    def client(**kwargs):
        assert kwargs == {"timeout": 90, "trust_env": False, "follow_redirects": False}
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", client)
    # SSH still sends the standalone REMOTE payload; tests call the equivalent function
    # so Bandit does not see exec() of a code string.
    assert "127.0.0.1:11434/api/chat" in REMOTE
    assert "'think':False" in REMOTE.replace(" ", "")
    if blocked:
        with pytest.raises(ValueError):
            run_mini_loopback("qwen3:8b", "qwen", SOURCE)
        assert not any(r.url.path == "/api/chat" for r in requests)
    else:
        result = run_mini_loopback("qwen3:8b", "qwen", SOURCE)
        assert result["extraction"] == EXTRACTION


def test_saved_receipt_from_other_submission_rejected_without_provider_calls(tmp_path):
    workspace = str(uuid4())
    runner = Runner(workspace)
    args = (
        tmp_path,
        workspace,
        BINDING,
        lambda: AIRGAP,
        lambda: {"model": "qwen3:8b", "extraction": EXTRACTION},
        lambda _: jev_response(),
        runner,
    )
    run(*args)
    path = tmp_path / "receipt.json"
    saved = json.loads(path.read_text())
    saved["receipt"]["receiptId"] = str(uuid4())
    path.write_text(json.dumps(saved))

    def forbidden(*args):
        pytest.fail("provider repeated")

    with pytest.raises(CombinedError, match="receipt_submission_mismatch"):
        run(tmp_path, workspace, BINDING, lambda: AIRGAP, forbidden, forbidden, runner)
    assert len(runner.submissions) == 1


@pytest.mark.parametrize(
    "body,error",
    [
        (jev_response(), None),
        ([], "invalid_jev_response_type"),
        ({"large": "x" * 65537}, "jev_response_too_large"),
    ],
)
def test_jev_edge_cookie_accepted_on_only_request_and_response_bounded(
    tmp_path, monkeypatch, body, error
):
    import httpx

    from local_brain.scripts import combined_synthetic_shadow as combined

    tmp_path.chmod(0o700)
    credential = tmp_path / "jev.json"
    credential.write_text(json.dumps({"token": "fake-test-only-jev-token"}))
    credential.chmod(0o600)
    original = httpx.Client
    calls = []
    clients = []

    def handler(request):
        calls.append(request)
        assert "cookie" not in request.headers
        return httpx.Response(200, json=body, headers={"set-cookie": "edge=fake; Secure; HttpOnly"})

    def client(**kwargs):
        result = original(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(result)
        return result

    monkeypatch.setattr(combined.httpx, "Client", client)
    request = bound_request(EXTRACTION, "synthetic", "job", "2026-01-01T00:00:00Z")
    if error:
        with pytest.raises(CombinedError, match=error):
            combined.call_jev(jev_payload(request), credential)
    else:
        result = combined.call_jev(jev_payload(request), credential)
        assert result["answers"] == body["answers"]
        assert result["measured_latency_ms"] >= 0
    assert len(calls) == 1 and clients[0].is_closed
