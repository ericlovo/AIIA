"""Offline synthetic replay. No provider, retrieval, approval or staging authority."""

import hashlib
import ipaddress
import json
import re
import time
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ReplayBudgets(StrictModel):
    max_pages: int = Field(default=1, ge=1, le=10)
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    max_elapsed_seconds: float = Field(default=30.0, gt=0, le=300)
    max_spend_usd: float = Field(default=0.0, ge=0, le=10)
    max_attempts: int = Field(default=2, ge=1, le=5)
    max_concurrency: int = Field(default=1, ge=1, le=4)


def public_host(host: str) -> str:
    if host != host.lower() or not re.fullmatch(r"[a-z0-9.-]+", host):
        raise ValueError("exact lowercase public hostname required")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("local hosts excluded")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or any(not part for part in host.split(".")):
            raise ValueError("public hostname required") from None
    else:
        raise ValueError("IP literals excluded")
    return host


class ShadowAssignment(StrictModel):
    tenant_id: str = Field(pattern=r"^synthetic-[a-z0-9-]{1,60}$")
    assignment_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    owner: str = Field(min_length=1, max_length=80)
    config_version: Literal["1"] = "1"
    enabled: bool = False
    schedule: Literal[False] = False
    provider: Literal["replay"] = "replay"
    model_ref: Literal["replay:saved-extraction"] = "replay:saved-extraction"
    policy_version: str = Field(min_length=1, max_length=80)
    source_allowlist: list[str] = Field(min_length=1, max_length=10)
    geography: list[str] = Field(min_length=1, max_length=20)
    signal_taxonomy: list[str] = Field(min_length=1, max_length=20)
    budgets: ReplayBudgets = Field(default_factory=ReplayBudgets)
    retention_days: int = Field(default=7, ge=1, le=30)
    secret_refs: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("source_allowlist")
    @classmethod
    def hosts(cls, values):
        return [public_host(value) for value in values]

    @field_validator("secret_refs")
    @classmethod
    def refs(cls, values):
        if any(not re.fullmatch(r"secret-ref:[a-z0-9_-]{1,64}", v) for v in values):
            raise ValueError("secret reference IDs only")
        return values

    @field_validator("geography", "signal_taxonomy")
    @classmethod
    def bounded_labels(cls, values):
        if any(not v.strip() or len(v) > 100 for v in values):
            raise ValueError("bounded nonempty labels required")
        return values


class EvidenceClaim(StrictModel):
    field: Literal["organization", "location", "signal_type"]
    value: str = Field(min_length=1, max_length=100)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=1000)


class ReplayFixture(StrictModel):
    synthetic: Literal[True] = True
    source_url: str = Field(max_length=2048)
    source_text: str = Field(min_length=1, max_length=100000)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    observed_at: str = Field(
        default="2026-01-01T00:00:00Z", pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    )
    claims: list[EvidenceClaim] = Field(max_length=3)
    # Kept untyped at this boundary intentionally: malformed provider recordings abstain.
    judgments: dict = Field(default_factory=dict)
    judgment_model_ref: Literal["replay:recorded-judgments"] = "replay:recorded-judgments"
    judgment_prompt_version: str = Field(default="synthetic-v1", min_length=1, max_length=80)
    recorded_tokens: int = Field(default=0, ge=0, le=1000000)
    recorded_seconds: float = Field(default=0.0, ge=0, le=100000)
    recorded_spend_usd: float = Field(default=0.0, ge=0, le=1000)
    provider_outage: bool = False


def _validate_source(config, fixture):
    parsed = urlsplit(fixture.source_url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
        or parsed.hostname not in config.source_allowlist
    ):
        raise ValueError("source outside exact HTTPS allowlist")
    public_host(parsed.hostname)
    if hashlib.sha256(fixture.source_text.encode()).hexdigest() != fixture.source_hash:
        raise ValueError("source hash mismatch")
    if len(json.dumps(fixture.judgments)) > 8000:
        raise ValueError("recorded judgments too large")


def enqueue_replay(store, config: ShadowAssignment, fixture: ReplayFixture) -> dict:
    """Save an explicitly enabled synthetic replay; schedules are never installed."""
    _validate_source(config, fixture)
    if not config.enabled:
        raise ValueError("assignment disabled")
    payload = config.model_dump()
    payload["fixture"] = fixture.model_dump()
    return store.enqueue(
        config.tenant_id, config.assignment_id, fixture.source_hash, config.policy_version, payload
    )


def _evidence(config, fixture):
    claims = {claim.field: claim for claim in fixture.claims}
    if len(claims) != 3 or len(fixture.claims) != 3:
        return None, "missing or repeated claims"
    for claim in fixture.claims:
        if (
            not 0 <= claim.start < claim.end <= len(fixture.source_text)
            or fixture.source_text[claim.start : claim.end] != claim.quote
            or claim.value not in claim.quote
        ):
            return None, "unsupported exact span"
    values = {key: claim.value for key, claim in claims.items()}
    if (
        values["location"] not in config.geography
        or values["signal_type"] not in config.signal_taxonomy
    ):
        return None, "outside assignment scope"
    # Deliberately narrow synthetic grammar. A mention/dateline/agency address
    # does not establish an operating location. General extraction is a later adapter.
    sentence = (
        f"{values['organization']} operates in {values['location']} "
        f"and reports {values['signal_type']}."
    )
    if any(claim.quote != sentence for claim in fixture.claims):
        return None, "operating location or event unsupported"
    return values, None


def _judgments(recorded):
    required = {
        "evidence_support",
        "organization_identity",
        "operating_location",
        "actionable_event",
        "relevance",
    }
    if set(recorded) != required:
        return False
    for value in recorded.values():
        if not isinstance(value, dict) or set(value) != {"answer", "probability"}:
            return False
        probability = value["probability"]
        if (
            value["answer"] != "supported"
            or type(probability) not in (float, int)
            or not 0 <= probability <= 1
        ):
            return False
    return True


class MockStagingAdapter:
    """Deterministic local receipt, not a Managed Leads HTTP client."""

    def submit(self, tenant_id, idempotency_key, candidate):
        return {
            "receipt_id": hashlib.sha256(f"{tenant_id}:{idempotency_key}".encode()).hexdigest(),
            "mode": "replay-only",
            "boundary": "mock-no-http",
            "review_status": "correction-needed",
            "approved": False,
            "candidate": candidate,
        }


def public_job(job):
    """Safe configuration/status projection; no source snapshots or secret refs."""
    safe = {
        key: job[key]
        for key in (
            "id",
            "tenant_id",
            "assignment_id",
            "state",
            "attempts",
            "created_at",
            "updated_at",
            "reason",
            "receipt",
        )
        if key in job
    }
    config = dict(job.get("config", {}))
    for key in ("fixture", "secret_refs"):
        config.pop(key, None)
    safe["config"] = config
    safe["mode"] = "replay-only"
    return safe


def run_once(store, tenant_id, job_id, owner):
    """One explicit bounded attempt. No polling, network, model or shell calls."""
    job = store.claim(tenant_id, job_id, owner)
    if job is None:
        return None
    token = job["lease_token"]
    started = time.monotonic()
    try:
        if store.start(tenant_id, job_id, token) is None:
            return None
        if (job.get("checkpoint") or {}).get("receipt"):
            return store.complete(tenant_id, job_id, token, job["checkpoint"]["receipt"])
        payload = dict(job["config"])
        fixture = ReplayFixture.model_validate(payload.pop("fixture"))
        config = ShadowAssignment.model_validate(payload)
        if config.tenant_id != tenant_id or not config.enabled:
            raise ValueError("assignment disabled or tenant mismatch")
        _validate_source(config, fixture)
        budgets = config.budgets
        if (
            fixture.recorded_tokens > budgets.max_tokens
            or fixture.recorded_seconds > budgets.max_elapsed_seconds
            or fixture.recorded_spend_usd > budgets.max_spend_usd
        ):
            return store.fail(tenant_id, job_id, token, "replay_budget_exhausted")
        if fixture.provider_outage:
            return store.fail(tenant_id, job_id, token, "recorded_provider_outage")
        candidate, reason = _evidence(config, fixture)
        if not _judgments(fixture.judgments):
            candidate, reason = None, "judgment abstention"
        if time.monotonic() - started > budgets.max_elapsed_seconds:
            return store.fail(tenant_id, job_id, token, "replay_budget_exhausted")
        receipt = MockStagingAdapter().submit(tenant_id, job_id, candidate)
        receipt["disposition"] = "mock-staged" if candidate else "withheld"
        receipt["reason"] = reason
        receipt["recorded_judgments"] = fixture.judgments if _judgments(fixture.judgments) else {}
        receipt["provenance"] = {
            "source_url": fixture.source_url,
            "source_hash": fixture.source_hash,
            "synthetic_observed_at": fixture.observed_at,
            "claims": [claim.model_dump() for claim in fixture.claims],
            "extraction_model_ref": config.model_ref,
            "config_version": config.config_version,
        }
        receipt["judgment_model_ref"] = fixture.judgment_model_ref
        receipt["judgment_prompt_version"] = fixture.judgment_prompt_version
        receipt["policy_version"] = config.policy_version
        receipt["probability_note"] = (
            "Recorded judgments are not verified facts or calibrated scores."
        )
        receipt["usage"] = {
            "kind": "recorded-replay-not-measured-inference",
            "tokens": fixture.recorded_tokens,
            "seconds": fixture.recorded_seconds,
            "spend_usd": fixture.recorded_spend_usd,
            "network_calls": 0,
            "provider_calls": 0,
        }
        if store.checkpoint(tenant_id, job_id, token, {"receipt": receipt}) is None:
            return None
        return store.complete(tenant_id, job_id, token, receipt)
    except (ValueError, KeyError, TypeError):
        return store.fail(tenant_id, job_id, token, "invalid_replay")


def synthetic_example():
    """Saved synthetic evidence for offline demos; no observation of a real company."""
    text = "Synthetic Orchard operates in Madison and reports expansion."
    config = ShadowAssignment(
        tenant_id="synthetic-demo",
        assignment_id="demo",
        owner="synthetic-operator",
        enabled=True,
        policy_version="synthetic-v1",
        source_allowlist=["example.org"],
        geography=["Madison"],
        signal_taxonomy=["expansion"],
    )
    fixture = ReplayFixture(
        source_url="https://example.org/synthetic",
        source_text=text,
        source_hash=hashlib.sha256(text.encode()).hexdigest(),
        claims=[
            EvidenceClaim(field=key, value=value, start=0, end=len(text), quote=text)
            for key, value in [
                ("organization", "Synthetic Orchard"),
                ("location", "Madison"),
                ("signal_type", "expansion"),
            ]
        ],
        judgments={
            key: {"answer": "supported", "probability": 0.91}
            for key in [
                "evidence_support",
                "organization_identity",
                "operating_location",
                "actionable_event",
                "relevance",
            ]
        },
    )
    return config, fixture
