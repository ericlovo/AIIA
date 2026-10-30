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


def selection(extraction, index=0):
    return {key: value for key, value in extraction.items() if key != "quote"} | {
        "quote_index": index
    }


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
        pipeline.validate_judgments(data, pipeline.questions_for("healthcare"))


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
        assert body["keep_alive"] == "30m"
        assert body["format"]["additionalProperties"] is False
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(selection(extraction))},
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
async def test_comcast_replay_extracts_permit_shape_and_only_proposes():
    result = await pipeline.run_pipeline(
        job(
            segment="client-bd",
            research_segment="commercial-fit-out",
            tenant_id="synthetic-comcast",
            payload={"replay_step_delay": 0},
        )
    )
    packet = result["packets"][0]
    assert packet["organization"] == "Synthetic North Loop Tenant"
    assert packet["permit"]["number"] == "BLDC-2026-0100"
    assert packet["permit"]["address"] == "100 Synthetic Avenue"
    assert packet["permit"]["project_title"] == "Ground-floor retail TI"
    assert packet["permit"]["permit_status"] == "Issued"
    assert packet["outreach_allowed"] is False
    assert packet["approved"] is False
    assert packet["contact_policy"] == "hold_pending_tenant_restriction_check"


def test_permit_profile_is_data_unavailable_without_source_spans():
    profile = pipeline.permit_profile(
        {"title": "A city published a generic notice.", "url": "https://example.org/x"}
    )
    assert profile["status"] == "data_unavailable"
    assert profile["number"] is None
    assert profile["address"] is None


def test_operator_query_rejects_urls_and_injection():
    with pytest.raises(pipeline.DataUnavailable, match="invalid_operator_query"):
        pipeline.operator_search_text("https://evil.example/steal")
    with pytest.raises(pipeline.DataUnavailable, match="invalid_operator_query"):
        pipeline.operator_search_text("permits; DROP TABLE")
    assert pipeline.operator_search_text("Minneapolis commercial permits") == (
        "Minneapolis commercial permits"
    )


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
                "message": {"content": json.dumps(selection(extraction))},
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
    import json

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
    assert (
        "".join(
            span["text"]
            for span in json.loads(pipeline.qwen_payload(source)["messages"][1]["content"])[
                "source_spans"
            ]
        )
        == source["text"]
    )
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

    async def local(source, emit=None, segment=None):
        assert "608-555-0100" in pipeline.source_text(source)
        return {
            "organization": "Orchard Clinic",
            "signal": "new clinic",
            "quote": source["text"].splitlines()[0],
            "consumer_insight": "",
        }, 150

    monkeypatch.setattr(pipeline, "retrieve_sources", forbidden)
    monkeypatch.setattr(pipeline, "extract_ready", local)
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
        pipeline.validate_judgments(judgments, pipeline.questions_for("healthcare"))


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
        "title": first["title"] + " New detail.",
    }
    called = []

    async def retrieve(segment):
        return [first, second]

    async def local(source, emit=None, segment=None):
        called.append(source)
        return extraction, 10

    async def judge(source, result, segment):
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
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


@pytest.mark.parametrize(
    "text",
    [
        "Clinic announces a new outpatient service.",
        "  Clínica École\tannounces care.\r\nPublic desk: office@example.org.  ",
        "Clinic\u00a0announces\u2009care — a new service.",
    ],
)
def test_quote_selection_preserves_short_source_verbatim(text):
    import json

    payload = pipeline.qwen_payload({"title": text})
    assert payload["format"]["properties"]["quote_index"]["enum"] == [0]
    assert "quote" not in payload["format"]["properties"]
    assert json.loads(payload["messages"][1]["content"])["source_spans"] == [
        {"index": 0, "text": text}
    ]


@pytest.mark.parametrize(
    "text",
    [
        "First clinic opens. " * 35
        + "\n"
        + "Second clinic expands. " * 40
        + "\r\nPublic office: hello@example.org.",
        "医" * 800,
        "x" * 2000,
    ],
)
def test_long_excerpt_spans_cover_source_exactly_with_bounded_utf8(text):
    import json

    source = {"title": "Announcement", "text": text}
    candidates = pipeline.quote_candidates(source)
    assert 1 < len(candidates) <= 32
    assert all(0 < len(quote.encode()) <= 300 and quote in text for quote in candidates)
    assert "".join(candidates) == text
    payload = pipeline.qwen_payload(source)
    assert [
        span["text"] for span in json.loads(payload["messages"][1]["content"])["source_spans"]
    ] == candidates
    # Quote bytes occur only in the ordered input spans, never repeated in the schema.
    assert payload["format"]["properties"]["quote_index"]["enum"] == list(range(len(candidates)))
    assert all(
        type(value) is int for value in payload["format"]["properties"]["quote_index"]["enum"]
    )


def test_medium_excerpt_combined_reservations_fit_existing_budget():
    source = {"title": "Announcement", "text": "Clinic expands services. " * 40}
    extraction = {
        "organization": "Clinic",
        "signal": "expands services",
        "quote": pipeline.quote_candidates(source)[0],
        "consumer_insight": "",
    }
    qwen = pipeline.qwen_payload(source)
    jev = pipeline.jev_payload(source, extraction, "healthcare")
    total = pipeline.token_reservation(qwen, 512) + pipeline.token_reservation(jev, 1024)
    assert total < 8000
    assert qwen["options"]["num_predict"] == 512


def test_quote_spans_reject_unbounded_or_empty_source():
    with pytest.raises(pipeline.DataUnavailable, match="candidates_exceeded"):
        pipeline.quote_candidates({"title": "x" * 2401})
    with pytest.raises(pipeline.DataUnavailable, match="source_quote_unavailable"):
        pipeline.quote_candidates({"title": " \t\r\n"})


@async_test
async def test_qwen_http_index_resolution_keeps_unicode_and_whitespace(monkeypatch):
    import json

    text = "Clínica\tÉcole announces a new ward.\r\n Public desk."
    source = {"title": text}
    extraction = {
        "organization": "Clínica\tÉcole",
        "signal": "new ward",
        "quote": text,
        "consumer_insight": "",
    }

    def handle(request):
        body = json.loads(request.content)
        assert body["format"]["properties"]["quote_index"]["enum"] == [0]
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(selection(extraction))},
                "prompt_eval_count": 100,
                "eval_count": 40,
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    result, tokens = await pipeline.extract_local(source)
    assert result["quote"] == text
    assert tokens == 140


@pytest.mark.parametrize("bad_index", [-1, 99, True, "0", 0.0, None])
@async_test
async def test_invalid_quote_index_fails_closed_without_retry(monkeypatch, bad_index):
    import json

    source = {"title": "Clinic opens a ward."}
    extraction = {"organization": "Clinic", "signal": "new ward", "consumer_insight": ""}
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(selection(extraction, bad_index))},
                "prompt_eval_count": 100,
                "eval_count": 40,
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    with pytest.raises(ValueError):
        await pipeline.extract_local(source)
    assert len(calls) == 1


@async_test
async def test_forged_quote_field_cannot_override_selected_source(monkeypatch):
    import json

    payload = {
        "organization": "Clinic",
        "signal": "new ward",
        "consumer_insight": "",
        "quote_index": 0,
        "quote": "Fabricated quote",
    }
    client = httpx.AsyncClient

    def handle(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(payload)},
                "prompt_eval_count": 100,
                "eval_count": 40,
            },
        )

    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    with pytest.raises(ValueError):
        await pipeline.extract_local({"title": "Clinic opens a ward."})


@async_test
async def test_blank_span_preserved_in_context_but_not_selectable(monkeypatch):
    import json

    source = {"title": " " * 300 + "Clinic opens a ward."}
    body = pipeline.qwen_payload(source)
    assert body["format"]["properties"]["quote_index"]["enum"] == [1]
    spans = json.loads(body["messages"][1]["content"])["source_spans"]
    assert "".join(span["text"] for span in spans) == source["title"]
    payload = {
        "organization": "Clinic",
        "signal": "new ward",
        "consumer_insight": "",
        "quote_index": 0,
    }
    client = httpx.AsyncClient

    def handle(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(payload)},
                "prompt_eval_count": 100,
                "eval_count": 40,
            },
        )

    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    with pytest.raises(pipeline.DataUnavailable, match="invalid_source_quote_index"):
        await pipeline.extract_local(source)


@pytest.mark.parametrize(
    "bad_quote",
    ["Clinic opens a ward.", "Clinic expands a hospital.", "A fabricated announcement."],
)
def test_strict_resolved_quote_validation_remains_unchanged(bad_quote):
    with pytest.raises(pipeline.DataUnavailable, match="unsupported_source_quote"):
        pipeline.validate_extraction(
            {"title": "Clinic\t opens a ward.\r\n"},
            {
                "organization": "Clinic",
                "signal": "new ward",
                "quote": bad_quote,
                "consumer_insight": "",
            },
        )


@pytest.mark.parametrize(
    "title,publisher,organization,quote,signal,reason",
    [
        (
            "Keep politics out of hospital expansion - The Daily Star",
            "The Daily Star",
            "The Daily Star",
            "Keep politics out of hospital expansion",
            "hospital expansion",
            "publisher_is_not_prospect",
        ),
        (
            "Marshall Islands eyes hiring Filipino doctors, nurses amid healthcare worker shortage - The Filipino Times",
            "The Filipino Times",
            "The Filipino Times",
            "Marshall Islands eyes hiring Filipino doctors, nurses amid healthcare worker shortage",
            "hiring",
            "publisher_is_not_prospect",
        ),
        (
            "Hospital expands services",
            "",
            "Hospital",
            "Hospital expands services",
            "expands",
            "generic_healthcare_reference",
        ),
        (
            "Clinic opening reported",
            "",
            "",
            "Clinic opening reported",
            "opening",
            "no_named_organization",
        ),
        (
            "Orchard Health mentioned. Another clinic expands.",
            "",
            "Orchard Health",
            "Another clinic expands.",
            "expands",
            "organization_not_grounded_in_article_quote",
        ),
        (
            "Orchard Health opens a clinic",
            "",
            "Orchard Health",
            "Orchard Health opens a clinic",
            "  ",
            "no_operational_event",
        ),
    ],
)
@async_test
async def test_healthcare_deterministic_withhold_skips_jev(
    monkeypatch, title, publisher, organization, quote, signal, reason
):
    source, extraction, _ = pipeline.synthetic_fixture("healthcare")
    source.update(title=title, publisher=publisher)
    extraction.update(organization=organization, quote=quote, signal=signal)

    async def retrieve(segment):
        return [source]

    async def local(source, emit=None, segment=None):
        return extraction, 10

    async def forbidden(*args):
        raise AssertionError("unsupported prospect must not call cloud judgment")

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", forbidden)
    events = []
    result = await pipeline.run_pipeline(job(mode="live"), lambda kind, data: events.append(kind))
    assert result["packets"] == []
    assert result["reason"] == reason
    assert result["next_query"]
    assert result["reviewed_evidence"] == [pipeline.evidence_key(source)]
    assert result["usage"]["calls"] == 2
    assert "candidate_proposed" not in events


@pytest.mark.parametrize(
    "failure",
    [
        "country",
        "editorial",
        "publisher",
        "uncertain",
        "low_probability",
        "low_confidence",
        "unsupported_claim",
        None,
        "own_newsroom",
    ],
)
@async_test
async def test_healthcare_semantic_gate_and_positive_provider(monkeypatch, failure):
    source, extraction, judgments = pipeline.synthetic_fixture("healthcare")
    organization = "Marshall Islands" if failure == "country" else "Orchard Health"
    publisher = organization if failure == "own_newsroom" else "The Daily Star"
    article = f"{organization} opens a clinic"
    source.update(title=f"{article} - {publisher}", publisher=publisher)
    extraction.update(organization=organization, quote=article, signal="opens a clinic")
    entity = judgments["answers"]["entity_role"]
    if failure in {"country", "publisher", "uncertain"}:
        role = "other" if failure == "country" else failure
        entity.update(
            choice=role,
            probabilities={key: 0.9 if key == role else 1 / 30 for key in entity["probabilities"]},
        )
    elif failure == "editorial":
        judgments["answers"]["disposition"]["choice"] = "watch"
    elif failure == "low_probability":
        entity["probabilities"] = dict(
            healthcare_operator=0.26, publisher=0.25, other=0.25, uncertain=0.24
        )
    elif failure == "low_confidence":
        entity["confidence"] = 0.79
    elif failure == "unsupported_claim":
        judgments["answers"]["claim_supported"]["noul"] = 0.79

    async def retrieve(segment):
        return [source]

    async def local(source, emit=None, segment=None):
        return extraction, 10

    async def judge(source, extracted, segment):
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", judge)
    result = await pipeline.run_pipeline(job(mode="live"))
    assert result["usage"]["calls"] == 3
    if failure in {None, "own_newsroom"}:
        packet = result["packets"][0]
        assert packet["organization"] == organization
        assert packet["provenance"]["title"] == source["title"]
    else:
        assert result["packets"] == []
        assert result["reason"] == "healthcare_entity_or_event_unconfirmed"
        assert result["reviewed_evidence"] == [pipeline.evidence_key(source)]


def test_publisher_attribution_excluded_from_quote_context_not_provenance():
    source = {"title": "Orchard Health expands — The Daily Star", "publisher": "The Daily Star"}
    assert "".join(pipeline.quote_candidates(source)) == "Orchard Health expands"
    assert pipeline.source_text(source) == source["title"]
    source["source_kind"] = "imported_primary_excerpt"
    assert pipeline.article_text(source) == source["title"]
    assert "entity_role" not in pipeline.questions_for("pe-diligence")


@pytest.mark.parametrize(
    "segment,article,organization",
    [
        (
            "pe-technical-diligence",
            "Orchard Capital, a private equity firm, acquires a software business for technology integration.",
            "Orchard Capital",
        ),
        (
            "alumni-associations",
            "Orchard Alumni Association launches a membership engagement platform.",
            "Orchard Alumni Association",
        ),
    ],
)
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "blank_org",
        "blank_signal",
        "publisher",
        "ungrounded_quote",
        "irrelevant",
        "uncertain",
        "low_probability",
        "watch",
        "unsupported",
    ],
)
@async_test
async def test_segment_prospect_gate(monkeypatch, segment, article, organization, failure):
    source, extraction, judgments = pipeline.synthetic_fixture(segment)
    source.update(title=article + " - Daily Reporter", publisher="Daily Reporter")
    extraction.update(
        organization=organization,
        quote=article,
        signal="launches a platform"
        if segment == "alumni-associations"
        else "acquires a software business",
    )
    role = pipeline.SEGMENT_ENTITIES[segment][0]
    entity = judgments["answers"]["entity_role"]
    if failure == "blank_org":
        extraction["organization"] = "  "
    elif failure == "blank_signal":
        extraction["signal"] = "  "
    elif failure == "publisher":
        extraction["organization"] = "Daily Reporter"
    elif failure == "ungrounded_quote":
        extraction["quote"] = (
            "technology integration" if segment.startswith("pe-") else "membership engagement"
        )
    elif failure in {"irrelevant", "uncertain"}:
        choice = "other" if failure == "irrelevant" else "uncertain"
        entity.update(
            choice=choice,
            probabilities={
                key: 0.9 if key == choice else 1 / 30 for key in entity["probabilities"]
            },
        )
    elif failure == "low_probability":
        entity["probabilities"] = {role: 0.26, "publisher": 0.25, "other": 0.25, "uncertain": 0.24}
    elif failure == "watch":
        judgments["answers"]["disposition"]["choice"] = "watch"
    elif failure == "unsupported":
        judgments["answers"]["claim_supported"]["noul"] = 0.1
    called = []

    async def retrieve(segment):
        return [source]

    async def local(source, emit=None, segment=None):
        return extraction, 10

    async def judge(source, result, segment):
        called.append("jev")
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", judge)
    result = await pipeline.run_pipeline(job(mode="live", segment=segment))
    if failure:
        assert result["packets"] == []
        assert result["status"] == "no_supported_prospect"
        assert result["reviewed_evidence"] == [pipeline.evidence_key(source)]
        assert result["research_outcome"] == (
            "rejected" if failure in {"publisher", "irrelevant"} else "inconclusive"
        )
    else:
        assert result["packets"][0]["organization"] == organization
    assert len(called) == (
        0 if failure in {"blank_org", "blank_signal", "publisher", "ungrounded_quote"} else 1
    )


@pytest.mark.parametrize(
    "segment,title,organization",
    [
        (
            "alumni-associations",
            "Orchard University remembers alumnus in obituary",
            "Orchard University",
        ),
        (
            "alumni-associations",
            "Orchard Alumni Association honors graduate athletic achievement",
            "Orchard Alumni Association",
        ),
        (
            "pe-technical-diligence",
            "Orchard Software acquires another business with no disclosed sponsor",
            "Orchard Software",
        ),
    ],
)
@async_test
async def test_named_but_irrelevant_segment_evidence_is_withheld(
    monkeypatch, segment, title, organization
):
    source, extraction, judgments = pipeline.synthetic_fixture(segment)
    source["title"] = title
    extraction.update(organization=organization, quote=title, signal=title)
    entity = judgments["answers"]["entity_role"]
    entity.update(
        choice="other",
        probabilities={key: 0.9 if key == "other" else 1 / 30 for key in entity["probabilities"]},
    )

    async def retrieve(segment):
        return [source]

    async def local(source, emit=None, segment=None):
        return extraction, 10

    async def judge(source, result, segment):
        payload = pipeline.jev_payload(source, result, segment)
        assert payload["questions"]["entity_role"] == pipeline.questions_for(segment)["entity_role"]
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", judge)
    result = await pipeline.run_pipeline(job(mode="live", segment=segment))
    assert result["packets"] == []
    assert result["reason"] == "segment_entity_or_event_unconfirmed"
    assert result["research_outcome"] == "rejected"


@pytest.mark.parametrize("segment", ["pe-technical-diligence", "alumni-associations"])
@async_test
async def test_segment_typed_fixture_and_medium_budget(segment):
    source, extraction, judgments = pipeline.synthetic_fixture(segment)
    pipeline.validate_judgments(judgments, pipeline.questions_for(segment))
    assert (await pipeline.run_pipeline(job(segment=segment)))["packets"]
    source["text"] = "Orchard announces change. " * 40
    extraction.update(
        organization="Orchard",
        quote=pipeline.quote_candidates(source)[0],
        signal="announces change",
        consumer_insight="",
    )
    total = pipeline.token_reservation(
        pipeline.qwen_payload(source), 512
    ) + pipeline.token_reservation(pipeline.jev_payload(source, extraction, segment), 1024)
    assert total < 8000


@pytest.mark.parametrize("segment", list(pipeline.SEGMENT_ENTITIES))
def test_persisted_answers_validate_without_usage_envelope(segment):
    _, _, judgments = pipeline.synthetic_fixture(segment)
    answers = judgments["answers"]
    assert pipeline.validate_answers(answers, pipeline.questions_for(segment)) is answers
    assert pipeline.validate_judgments(judgments, pipeline.questions_for(segment)) is judgments
    answers["claim_supported"]["noul"] = True
    with pytest.raises(pipeline.DataUnavailable):
        pipeline.validate_answers(answers, pipeline.questions_for(segment))


@pytest.mark.parametrize("answers", [None, [], "", {"unexpected": {}}])
def test_persisted_invalid_answers_fail_closed(answers):
    with pytest.raises(pipeline.DataUnavailable, match="invalid_jev_answers"):
        pipeline.validate_answers(answers)


@async_test
async def test_qwen_warmup_sends_tiny_keep_alive_before_extract(monkeypatch):
    import json

    source, extraction, _ = pipeline.synthetic_fixture("healthcare")
    payloads = []

    def handle(request):
        body = json.loads(request.content)
        payloads.append(body)
        if "format" not in body:
            assert body["model"] == "qwen3:8b"
            assert body["keep_alive"] == "30m"
            assert body["options"]["num_ctx"] == 512
            return httpx.Response(200, json={"message": {"content": "ok"}})
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(selection(extraction))},
                "prompt_eval_count": 40,
                "eval_count": 10,
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        pipeline.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    result, tokens = await pipeline.extract_ready(source)
    assert result == extraction
    assert tokens == 50
    assert payloads[0]["keep_alive"] == "30m"
    assert "format" not in payloads[0]
    assert payloads[1]["keep_alive"] == "30m"
    assert "format" in payloads[1]


@async_test
async def test_qwen_retries_once_after_unavailable_then_succeeds(monkeypatch):
    import json

    source, extraction, _ = pipeline.synthetic_fixture("healthcare")
    extracts = []

    async def fake_ensure():
        return {"ok": True}

    async def fake_extract(current, segment=None):
        extracts.append(current)
        if len(extracts) == 1:
            raise httpx.ConnectError("cold")
        return extraction, 12

    events = []
    monkeypatch.setattr(pipeline, "ensure_qwen_loaded", fake_ensure)
    monkeypatch.setattr(pipeline, "extract_local", fake_extract)
    result, tokens = await pipeline.extract_ready(
        source, emit=lambda kind, **data: events.append(kind)
    )
    assert (result, tokens) == (extraction, 12)
    assert len(extracts) == 2
    assert "qwen_retry" in events


@async_test
async def test_qwen_does_not_retry_invalid_quote_selection(monkeypatch):
    extracts = []

    async def fake_ensure():
        return {}

    async def fake_extract(source, segment=None):
        extracts.append(source)
        raise pipeline.DataUnavailable("invalid_source_quote_index")

    monkeypatch.setattr(pipeline, "ensure_qwen_loaded", fake_ensure)
    monkeypatch.setattr(pipeline, "extract_local", fake_extract)
    with pytest.raises(pipeline.DataUnavailable, match="invalid_source_quote_index"):
        await pipeline.extract_ready({"title": "Clinic opens a ward."})
    assert len(extracts) == 1
