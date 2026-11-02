import httpx
import pytest

from local_brain.command_center import bd_pipeline as pipeline
from local_brain.command_center.bd_queries import (
    QUERY_BANK,
    apply_yield,
    evidence_text,
    named_organization_present,
    select_discovery_query,
)

from .test_bd_pipeline import async_test, job, public_excerpt


def test_query_bank_targets_named_buying_events():
    markers = {
        "healthcare": ("EHR", "CIO", "RFP", "acquisition"),
        "pe-technical-diligence": ("acquire", "platform", "CTO", "ERP"),
        "alumni-associations": ("alumni association", "Blackbaud", "RFP", "Harvard Alumni"),
    }
    for segment, needles in markers.items():
        blob = " ".join(item["q"] for item in QUERY_BANK[segment]).lower()
        for needle in needles:
            assert needle.lower() in blob


def test_select_discovery_query_rotates_unused_then_decays_zero_yield():
    unused = select_discovery_query("healthcare", yield_stats=[], slot=0)
    assert unused["id"] == "hc-ehr-event"
    stats = [
        {
            "segment": "healthcare",
            "query_id": "hc-ehr-event",
            "jobs": 4,
            "supported": 0,
            "failed": 0,
            "inconclusive": {},
        },
        {
            "segment": "healthcare",
            "query_id": "hc-cio-hire",
            "jobs": 2,
            "supported": 1,
            "failed": 0,
            "inconclusive": {},
        },
    ]
    chosen = select_discovery_query("healthcare", yield_stats=stats, slot=0)
    assert chosen["id"] != "hc-ehr-event"
    matured = [
        {
            "segment": "healthcare",
            "query_id": item["id"],
            "jobs": 4,
            "supported": 2 if item["id"] == "hc-cio-hire" else 0,
            "failed": 0,
            "inconclusive": {},
        }
        for item in QUERY_BANK["healthcare"]
    ]
    assert select_discovery_query("healthcare", yield_stats=matured, slot=0)["id"] == "hc-cio-hire"
    forced = select_discovery_query("healthcare", yield_stats=stats, query_id="hc-rfp")
    assert forced["id"] == "hc-rfp"


def test_named_organization_prefilter_rejects_generic_headlines():
    assert named_organization_present(
        {"title": "Atrium Health selects Epic for EHR replacement", "publisher": "News"}
    )
    assert named_organization_present(
        {"title": "Harvard Alumni Association launches membership platform"}
    )
    assert not named_organization_present({"title": "Hospital expands services"})
    assert not named_organization_present({"title": "Clinic opening reported"})
    assert not named_organization_present(
        {
            "title": "Keep politics out of hospital expansion - The Daily Star",
            "publisher": "The Daily Star",
        }
    )
    assert not named_organization_present({"title": "Private equity firms buy software"})


def test_evidence_text_prefers_snippet_and_caps_bytes():
    source = {
        "title": "Orchard Health opens a clinic",
        "text": "<b>Orchard Health</b> will open a clinic next month in Madison.",
    }
    assert "Madison" in evidence_text(source)
    assert len(evidence_text({"title": "x" * 5000}, limit=1800).encode()) <= 1800


def test_apply_yield_counts_inconclusive_reasons():
    row = apply_yield(
        {
            "segment": "healthcare",
            "query_id": "hc-ehr-event",
            "jobs": 0,
            "supported": 0,
            "rejected": 0,
            "failed": 0,
            "inconclusive": {},
        },
        "inconclusive",
        "segment_entity_or_event_unconfirmed",
    )
    assert row["jobs"] == 1
    assert row["inconclusive"]["segment_entity_or_event_unconfirmed"] == 1
    supported = apply_yield(row, "supported")
    assert supported["supported"] == 1 and supported["jobs"] == 2


@async_test
async def test_generic_headlines_skip_qwen(monkeypatch):
    async def retrieve(segment, query=None):
        return [
            {"title": "Hospital expands services", "url": "https://news.google.com/rss/articles/x"}
        ]

    async def forbidden(*args, **kwargs):
        raise AssertionError("generic headlines must not call qwen")

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", forbidden)
    result = await pipeline.run_pipeline(job(mode="live"))
    assert result["reason"] == "headline_lacks_named_organization"
    assert result["research_outcome"] == "inconclusive"
    assert result["usage"]["calls"] == 1
    assert result["discovery_query"]["id"]


@async_test
async def test_qwen_timeout_retries_once_with_bounded_source(monkeypatch):
    calls = []

    async def warm():
        return True

    async def local(source, *, timeout=70, num_ctx=8192):
        text = pipeline.source_text(source)
        calls.append({"timeout": timeout, "num_ctx": num_ctx, "bytes": len(text.encode())})
        if len(calls) == 1:
            raise httpx.TimeoutException("cold start")
        return {
            "organization": "Orchard Health",
            "signal": "ehr replacement",
            "quote": text[:20],
            "consumer_insight": "",
        }, 12

    monkeypatch.setattr(pipeline, "warm_qwen", warm)
    monkeypatch.setattr(pipeline, "extract_local", local)
    source = {"title": "Orchard Health selects Epic", "text": "body " * 400}
    extraction, tokens = await pipeline.extract_ready(source)
    assert tokens == 12
    assert extraction["organization"] == "Orchard Health"
    assert len(calls) == 2
    assert calls[0]["timeout"] == pipeline.QWEN_EXTRACT_TIMEOUT
    assert calls[1]["timeout"] <= pipeline.QWEN_RETRY_TIMEOUT
    assert calls[1]["num_ctx"] == 4096
    assert calls[1]["bytes"] <= pipeline.SOURCE_RETRY_BYTE_LIMIT


@async_test
async def test_qwen_timeout_does_not_retry_when_job_wall_is_nearly_exhausted(monkeypatch):
    clock = {"n": 0}

    def monotonic():
        clock["n"] += 1
        return 0.0 if clock["n"] == 1 else 95.0

    async def warm():
        return True

    async def local(source, *, timeout=70, num_ctx=8192):
        raise httpx.TimeoutException("still cold")

    monkeypatch.setattr(pipeline, "warm_qwen", warm)
    monkeypatch.setattr(pipeline, "extract_local", local)
    monkeypatch.setattr(pipeline.time, "monotonic", monotonic)
    with pytest.raises(httpx.TimeoutException):
        await pipeline.extract_ready({"title": "Orchard Health opens a clinic"})


@async_test
async def test_warmup_failure_does_not_count_as_extract_call(monkeypatch):
    calls = []

    async def warm():
        return False

    async def local(source, *, timeout=70, num_ctx=8192):
        calls.append(source)
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(pipeline, "warm_qwen", warm)
    monkeypatch.setattr(pipeline, "extract_local", local)
    with pytest.raises(httpx.ConnectError):
        await pipeline.extract_ready({"title": "Clinic opens ward"})
    assert len(calls) == 1


@async_test
async def test_imported_excerpt_still_uses_named_org_prefilter(monkeypatch):
    import hashlib

    async def forbidden(*args, **kwargs):
        raise AssertionError("no qwen for generic imported title")

    monkeypatch.setattr(pipeline, "extract_ready", forbidden)
    text = "Hospital expands services in the region."
    excerpt = public_excerpt(text=text, source_hash=hashlib.sha256(text.encode()).hexdigest())
    result = await pipeline.run_pipeline(
        job(mode="live", payload={"public_source": excerpt}, source_allowlist=["example.org"])
    )
    assert result["reason"] == "headline_lacks_named_organization"
    assert result["usage"]["calls"] == 0
