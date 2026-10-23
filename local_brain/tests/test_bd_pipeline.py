import asyncio
import copy
from functools import wraps

import httpx
import pytest

from local_brain.command_center import bd_pipeline as pipeline


def async_test(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return run


def job(**changes):
    return {"tenant_id": "synthetic-test", "segment": "healthcare", "mode": "replay", **changes}


@async_test
async def test_replay_provenance_events_and_no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted network")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    events = []
    result = await pipeline.run_pipeline(job(), lambda kind, data: events.append((kind, data)))
    packet = result["packets"][0]
    assert packet["mode"] == "replay-only"
    assert packet["outreach_allowed"] is False
    assert packet["contact_policy"] == "hold_pending_tenant_restriction_check"
    assert packet["usage"]["calls"] == 0
    assert packet["contacts"][0]["value"] == "research@example.org"
    for contact in packet["contacts"]:
        assert packet["provenance"]["title"][contact["start"] : contact["end"]] == contact["value"]
    assert "jev_called" not in [kind for kind, _ in events]
    assert events[-1][0] == "packet_assembled"


@async_test
async def test_replay_cannot_target_real_tenant():
    with pytest.raises(ValueError, match="synthetic"):
        await pipeline.run_pipeline(job(tenant_id="customer"))


@async_test
async def test_missing_managed_leads_is_unavailable_not_empty():
    data = await pipeline.ManagedLeadsReadOnly().read("tenant-a")
    assert data["status"] == "data_unavailable"
    assert data["tenant_id"] == "tenant-a"
    assert data["records"] is None


@pytest.mark.parametrize("field,value", [("quote", "invented"), ("organization", "Invented Co")])
def test_qwen_unsupported_extraction_rejected(field, value):
    source, extraction, _ = pipeline.synthetic_fixture("healthcare")
    extraction[field] = value
    with pytest.raises(pipeline.DataUnavailable):
        pipeline.validate_extraction(source, extraction)


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("claim_supported", "noul", float("nan")),
        ("claim_supported", "noul", True),
        ("evidence_strength", "score", 3),
        ("disposition", "choice", "send_email"),
    ],
)
def test_invalid_typed_judgments_fail_closed(key, field, value):
    _, _, data = pipeline.synthetic_fixture("healthcare")
    data["answers"][key][field] = value
    with pytest.raises(pipeline.DataUnavailable):
        pipeline.validate_judgments(data)


@async_test
async def test_live_retrieval_disabled_does_not_call_network(monkeypatch):
    monkeypatch.delenv("AIIA_NEWS_ENABLED", raising=False)
    with pytest.raises(pipeline.DataUnavailable, match="disabled"):
        await pipeline.run_pipeline(job(mode="live"))


@async_test
async def test_denied_egress_never_dials(monkeypatch):
    from local_brain.egress import EgressDecision

    async def denied(*args, **kwargs):
        return EgressDecision(False, "denied")

    monkeypatch.setenv("AIIA_NEWS_ENABLED", "1")
    monkeypatch.setenv("AIIA_SIGNALS_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")
    monkeypatch.setattr(pipeline, "authorize_egress", denied)
    with pytest.raises(pipeline.DataUnavailable, match="egress_denied"):
        await pipeline.retrieve_sources("healthcare")
    source, extraction, _ = pipeline.synthetic_fixture("healthcare")
    with pytest.raises(pipeline.DataUnavailable, match="egress_denied"):
        await pipeline.judge(source, extraction, "healthcare")


@async_test
async def test_local_qwen_real_http_contract(monkeypatch):
    source, extraction, _ = pipeline.synthetic_fixture("healthcare")
    import json

    def handle(request):
        body = json.loads(request.content)
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        assert body["model"] == "qwen3:8b"
        assert body["format"]["additionalProperties"] is False
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(extraction)},
                "prompt_eval_count": 99,
                "eval_count": 30,
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    result, tokens = await pipeline.extract_local(source)
    assert result == extraction
    assert tokens == 129


@async_test
async def test_checkpoint_can_abort_before_provider(monkeypatch):
    async def emit(kind, data):
        raise RuntimeError("kill switch")

    with pytest.raises(RuntimeError, match="kill switch"):
        await pipeline.run_pipeline(job(mode="live"), emit)


def test_contact_fields_never_come_from_model():
    source, extraction, judgments = pipeline.synthetic_fixture("healthcare")
    source["title"] = "Synthetic Orchard opens a clinic."
    extraction["quote"] = source["title"]
    extraction["consumer_insight"] = "Contact invented@example.com"
    packet = pipeline.proposal_packet(job(), source, extraction, judgments, {})
    assert packet["contacts"] == []


@async_test
async def test_client_replay_and_live_missing_tenant_api():
    result = await pipeline.run_pipeline(job(segment="client-bd", payload={"replay_step_delay": 0}))
    assert result["packets"][0]["segment"] == "client-bd"
    with pytest.raises(pipeline.DataUnavailable, match="client_research_segment_unconfigured"):
        await pipeline.run_pipeline(job(segment="client-bd", mode="live"))


@async_test
async def test_no_observations_never_yields_human_packet(monkeypatch):
    async def empty(segment):
        return []

    monkeypatch.setattr(pipeline, "retrieve_sources", empty)
    result = await pipeline.run_pipeline(job(mode="live"))
    assert result["packets"] == []
    assert result["usage"]["calls"] == 1


def test_token_reservation_counts_utf8_bytes():
    ascii_budget = pipeline.token_reservation({"text": "a" * 10}, 512)
    unicode_budget = pipeline.token_reservation({"text": "界" * 10}, 512)
    assert unicode_budget - ascii_budget == 20


@async_test
async def test_oversized_qwen_usage_stops_before_jev(monkeypatch):
    import json

    source, extraction, _ = pipeline.synthetic_fixture("healthcare")

    def handle(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(extraction)},
                "prompt_eval_count": 100000,
                "eval_count": 30,
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    with pytest.raises(pipeline.DataUnavailable, match="usage_exceeded"):
        await pipeline.extract_local(source)


@async_test
async def test_live_failure_emits_unavailable(monkeypatch):
    monkeypatch.delenv("AIIA_NEWS_ENABLED", raising=False)
    events = []
    with pytest.raises(pipeline.DataUnavailable):
        await pipeline.run_pipeline(
            job(mode="live"), lambda kind, data: events.append((kind, data))
        )
    assert events[-1] == (
        "provider_health",
        {
            "provider": "sources",
            "provider_status": "data_unavailable",
            "message": "public_retrieval_disabled",
        },
    )


@async_test
async def test_jev_http_contract_projects_public_only(monkeypatch):
    import json

    from local_brain.egress import EgressDecision

    source, extraction, recording = pipeline.synthetic_fixture("healthcare")
    source["tenant_secret"] = "must-never-leave"

    async def allow(tool, server):
        assert (tool, server) == ("typesafe.signals", "api.typesafe.ai")
        return EgressDecision(True, "test")

    def handle(request):
        payload = json.loads(request.content)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert payload["state"]["proposed_quote"] == extraction["quote"]
        assert "must-never-leave" not in request.content.decode()
        assert {q["type"] for q in payload["questions"].values()} == {"choice", "score", "noul"}
        return httpx.Response(200, json=recording)

    client = httpx.AsyncClient
    monkeypatch.setenv("AIIA_SIGNALS_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(pipeline, "authorize_egress", allow)
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    assert await pipeline.judge(source, extraction, "healthcare") == recording


@async_test
async def test_public_retrieval_uses_segment_query_and_fixed_host(monkeypatch):
    from datetime import datetime, timezone
    from email.utils import format_datetime

    from local_brain.egress import EgressDecision

    date = format_datetime(datetime.now(timezone.utc))
    xml = f"<rss><channel><item><title>Orchard clinic expands</title><link>https://news.google.com/rss/articles/test</link><pubDate>{date}</pubDate><source>Example</source></item></channel></rss>"

    async def allow(tool, server):
        assert (tool, server) == ("news.fetch", "news.google.com")
        return EgressDecision(True, "test")

    def handle(request):
        assert request.url.host == "news.google.com"
        assert "healthcare" in request.url.params["q"]
        return httpx.Response(200, text=xml)

    client = httpx.AsyncClient
    monkeypatch.setenv("AIIA_NEWS_ENABLED", "1")
    monkeypatch.setattr(pipeline, "authorize_egress", allow)
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    sources = await pipeline.retrieve_sources("healthcare")
    assert len(sources) == 1
    assert sources[0]["title"] == "Orchard clinic expands"


def public_excerpt(**changes):
    import hashlib

    text = "Orchard Clinic announces a new outpatient clinic.\nPublic office: research@example.org, 608-555-0100."
    return {
        "url": "https://example.org/company/news",
        "text": text,
        "source_hash": hashlib.sha256(text.encode()).hexdigest(),
        "captured_at": "2026-01-01T12:00:00Z",
        "public": True,
        **changes,
    }


def test_imported_primary_excerpt_full_text_contacts_and_provenance():
    source = pipeline.load_public_excerpt(public_excerpt(), ["example.org"])
    _, extraction, judgments = pipeline.synthetic_fixture("healthcare")
    extraction.update(organization="Orchard Clinic", quote=source["text"].splitlines()[0])
    packet = pipeline.proposal_packet(job(mode="live"), source, extraction, judgments, {})
    assert packet["emails"] == ["research@example.org"]
    assert packet["phones"] == ["608-555-0100"]
    assert packet["provenance"]["source_kind"] == "imported_primary_excerpt"
    assert packet["provenance"]["verification"] == "human_supplied_not_independently_fetched"
    assert packet["provenance"]["captured_at"] == "2026-01-01T12:00:00+00:00"
    for contact in packet["contacts"]:
        assert source["text"][contact["start"] : contact["end"]] == contact["value"]
    assert pipeline.qwen_payload(source)["messages"][1]["content"] == source["text"]
    assert (
        pipeline.jev_payload(source, extraction, "healthcare")["state"]["source_text"]
        == source["text"]
    )


@pytest.mark.parametrize(
    "changes,allowlist",
    [
        ({}, []),
        ({"url": "http://example.org/company"}, ["example.org"]),
        ({"url": "https://user:password@example.org/company"}, ["example.org"]),
        ({"url": "https://127.0.0.1/company"}, ["127.0.0.1"]),
        ({"url": "https://api.internal/company"}, ["api.internal"]),
        ({"url": "https://example.org:8200/company"}, ["example.org"]),
        ({"source_hash": "0" * 64}, ["example.org"]),
        ({"captured_at": "2026-01-01T12:00:00"}, ["example.org"]),
        ({"captured_at": "2099-01-01T12:00:00Z"}, ["example.org"]),
        ({"public": False}, ["example.org"]),
        ({"tenant_id": "other-tenant"}, ["example.org"]),
        ({"text": "界" * 1000}, ["example.org"]),
    ],
)
def test_bad_public_excerpt_rejected(changes, allowlist):
    with pytest.raises(ValueError):
        pipeline.load_public_excerpt(public_excerpt(**changes), allowlist)


@async_test
async def test_imported_live_skips_discovery_and_preserves_guarded_judgment(monkeypatch):
    events = []

    async def forbidden(segment):
        raise AssertionError("No news fetch for imported source")

    async def local(source):
        assert "608-555-0100" in pipeline.source_text(source)
        return {
            "organization": "Orchard Clinic",
            "signal": "new clinic",
            "quote": source["text"].splitlines()[0],
            "consumer_insight": "",
        }, 150

    monkeypatch.setattr(pipeline, "retrieve_sources", forbidden)
    monkeypatch.setattr(pipeline, "extract_local", local)
    monkeypatch.delenv("AIIA_SIGNALS_ENABLED", raising=False)
    with pytest.raises(pipeline.DataUnavailable, match="jev_not_configured"):
        await pipeline.run_pipeline(
            job(
                mode="live",
                payload={"public_source": public_excerpt()},
                source_allowlist=["example.org"],
            ),
            lambda kind, data: events.append((kind, data)),
        )
    assert events[0][1]["reserve_calls"] == 0
    assert any(kind == "source_imported" for kind, _ in events)
    assert events[-1][0] == "jev_error"


@async_test
async def test_replay_cannot_mislabel_imported_real_source():
    with pytest.raises(ValueError, match="imported_sources_require_live"):
        await pipeline.run_pipeline(job(payload={"public_source": public_excerpt()}))


def test_inconsistent_score_and_distribution_rejected():
    _, _, judgments = pipeline.synthetic_fixture("healthcare")
    judgments["answers"]["evidence_strength"]["score"] = 0.1
    with pytest.raises(pipeline.DataUnavailable, match="inconsistent_jev_score"):
        pipeline.validate_judgments(judgments)


def managed_config():
    return {
        "base_url": "https://managed-leads.vercel.app",
        "workspace_id": "tenant-one",
        "client_id_env": "TEST_BD_CLIENT",
        "token_env": "TEST_BD_TOKEN",
    }


def managed_transport(monkeypatch, handler):
    from local_brain.egress import EgressDecision

    async def allow(tool, server):
        assert (tool, server) == ("web.fetch", "managed-leads.vercel.app")
        return EgressDecision(True, "test")

    client = httpx.AsyncClient
    monkeypatch.setenv("TEST_BD_CLIENT", "test-client")
    monkeypatch.setenv("TEST_BD_TOKEN", "synthetic-test-token-" + "x" * 32)
    monkeypatch.setattr(pipeline, "authorize_egress", allow)
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handler), **kw),
    )


@async_test
async def test_managed_list_binds_workspace_and_uses_read_auth(monkeypatch):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/integrations/v1/accounts"
        assert request.headers["X-Integration-Client"] == "test-client"
        assert request.headers["Authorization"].startswith("Bearer synthetic-test-token-")
        assert set(request.url.params) == {"q", "limit", "after"}
        assert "cookie" not in request.headers
        return httpx.Response(
            200,
            json={
                "version": 1,
                "workspaceId": "tenant-one",
                "accounts": [{"id": "one", "organization": "Example"}],
                "nextCursor": None,
            },
        )

    managed_transport(monkeypatch, handler)
    adapter = pipeline.ManagedLeadsReadOnly(managed_config())
    assert adapter.status("tenant-one")["configured"] is True
    result = await adapter.read("tenant-one", query="Example", after="zero")
    assert result["status"] == "available"
    assert result["outreach_allowed"] is False
    assert result["requested_contact_date"] == "data_unavailable"


@async_test
async def test_managed_detail_preserves_suppression_flags(monkeypatch):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/integrations/v1/accounts/one"
        assert not request.url.params
        return httpx.Response(
            200,
            json={
                "version": 1,
                "workspaceId": "tenant-one",
                "account": {"id": "one", "organization": "Example"},
                "people": [
                    {"email": "held@example.org", "hold": True, "optedOut": True, "verified": False}
                ],
                "signals": [],
            },
        )

    managed_transport(monkeypatch, handler)
    result = await pipeline.ManagedLeadsReadOnly(managed_config()).read(
        "tenant-one", account_id="one"
    )
    assert result["status"] == "available"
    assert result["records"]["people"][0]["hold"] is True
    assert result["records"]["people"][0]["optedOut"] is True
    assert result["contact_restrictions"] == "partial_hold_and_opt_out_only"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://evil.example"}),
        httpx.Response(401),
        httpx.Response(503),
        httpx.Response(200, headers={"Set-Cookie": "session=bad"}, json={}),
        httpx.Response(
            200,
            json={"version": 1, "workspaceId": "other-tenant", "accounts": [], "nextCursor": None},
        ),
        httpx.Response(
            200,
            json={"version": True, "workspaceId": "tenant-one", "accounts": [], "nextCursor": None},
        ),
    ],
)
@async_test
async def test_managed_response_fail_closed(monkeypatch, response):
    managed_transport(monkeypatch, lambda request: response)
    result = await pipeline.ManagedLeadsReadOnly(managed_config()).read("tenant-one")
    assert result["status"] == "data_unavailable"
    assert result["records"] is None


@async_test
async def test_managed_config_mismatch_does_not_call_network(monkeypatch):
    def forbidden(request):
        raise AssertionError("invalid configuration must not dial")

    managed_transport(monkeypatch, forbidden)
    result = await pipeline.ManagedLeadsReadOnly(managed_config()).read("tenant-two")
    assert result["reason"] == "managed_leads_workspace_configuration_invalid"
    config = managed_config()
    config["base_url"] = "https://evil.example"
    assert (await pipeline.ManagedLeadsReadOnly(config).read("tenant-one"))["records"] is None


@async_test
async def test_seen_sources_are_skipped_before_qwen(monkeypatch):
    first, extraction, judgments = pipeline.synthetic_fixture("healthcare")
    second = {
        **first,
        "url": "https://example.org/second",
        "title": "Second organization opens a clinic",
    }
    called = []

    async def retrieve(segment):
        return [first, second]

    async def local(source):
        called.append(source)
        return extraction, 10

    async def judge(source, result, segment):
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_local", local)
    monkeypatch.setattr(pipeline, "judge", judge)
    result = await pipeline.run_pipeline(
        job(mode="live", seen_evidence=[pipeline.evidence_key(first)])
    )
    assert called == [second]
    assert result["packets"][0]["evidence_key"] == pipeline.evidence_key(second)
    called.clear()
    result = await pipeline.run_pipeline(
        job(
            mode="live", seen_evidence=[pipeline.evidence_key(first), pipeline.evidence_key(second)]
        )
    )
    assert called == []
    assert result["packets"] == []
    assert result["reason"] == "no_new_evidence"


def test_changed_source_text_has_new_evidence_key():
    source, _, _ = pipeline.synthetic_fixture("healthcare")
    assert pipeline.evidence_key(source) != pipeline.evidence_key(
        {**source, "title": "Updated public announcement"}
    )
