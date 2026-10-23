"""Pure, replayable JEV-01 shadow judgments. No client, I/O or approval authority.

Span checks establish source integrity, not truth. Thresholds are experimental
shadow defaults, not production calibration or the prior V6 model contract.
The caller must obtain trusted_tenant_id from an authenticated service binding.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Mapping

JUDGMENTS = (
    "evidence_support",
    "organization_identity",
    "operating_location",
    "actionable_event",
    "relevance",
)


class InputError(ValueError):
    """Contains only a safe, fixed error code; never source/provider text."""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash(value) -> str:
    return content_hash(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _number(value) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    source_id: str
    start: int
    end: int
    text: str
    basis: str = "other"


@dataclass(frozen=True)
class ExtractedClaim:
    value: str
    span_ids: tuple[str, ...]


@dataclass(frozen=True)
class JudgmentInput:
    tenant_id: str
    proposal_id: str
    sources: tuple[SourceSnapshot, ...]
    spans: tuple[EvidenceSpan, ...]
    organization: ExtractedClaim
    location: ExtractedClaim
    event: ExtractedClaim
    observed_at: str
    policy_version: str


@dataclass(frozen=True)
class Policy:
    version: str = "shadow-v1"
    min_probability: float = 0.8
    min_link_confidence: float = 0.8


@dataclass(frozen=True)
class JudgmentRequest:
    observation: JudgmentInput
    policy: Policy
    provider: str
    model: str
    prompt_version: str
    binding_hash: str


@dataclass(frozen=True)
class TypedJudgment:
    question: str
    probability: float | None = None
    confidence: float | None = None
    choice: str | None = None


@dataclass(frozen=True)
class JudgmentResult:
    tenant_id: str
    proposal_id: str
    decision: str
    reason: str
    binding_hash: str
    provider: str
    model: str
    prompt_version: str
    policy_version: str
    judgments: tuple[TypedJudgment, ...] = ()
    raw_answers_json: str | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    shadow_only: bool = True


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _contains(text, value):
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I) is not None


def _direct_location(text, organization, location):
    """Recognize a deliberately small subject/operation grammar in one clause.

    No arbitrary words may bridge subject, operation and location. In particular,
    an incidental picnic, press release or a second sentence is not a witness.
    Unsupported wording abstains; this is not a general semantic parser.
    """
    subject = re.escape(organization)
    place = re.escape(location)
    relationship = (
        r"(?:is\s+)?(?:headquartered|based|located)\s+(?:in|at)"
        r"|(?:is\s+)?(?:expanding|building|constructing|opening)\s+"
        r"(?:a|an|the|its)\s+(?:new\s+)?"
        r"(?:factory|plant|facility|project|office|warehouse)\s+(?:in|at)"
    )
    pattern = subject + r"\s+(?:" + relationship + r")\s+" + place
    return any(
        re.fullmatch(pattern, clause.strip(), re.I)
        for clause in re.split(r"[.!?;\n]", text)
        if clause.strip()
    )


def _location_basis(observation, spans):
    """Conservative syntactic binding only; basis annotations are not proof."""
    org, loc = observation.organization.value, observation.location.value
    for sid in observation.location.span_ids:
        s = spans[sid]
        if s.basis == "company_project" and _direct_location(s.text, org, loc):
            return True
    # A bare place/state source span may be joined only to an independent,
    # same-source direct subject/project clause. Explicit datelines are excluded.
    if "," in loc:
        city = loc.split(",", 1)[0].strip()
        for sid in observation.location.span_ids:
            anchor = spans[sid]
            if anchor.basis in ("agency", "dateline") or anchor.text.strip().rstrip(".") != loc:
                continue
            for sid2 in observation.organization.span_ids:
                s = spans[sid2]
                if (
                    s.source_id == anchor.source_id
                    and s.basis == "company_project"
                    and _direct_location(s.text, org, city)
                ):
                    return True
    return False


def validate_input(observation: JudgmentInput, *, trusted_tenant_id: str, policy: Policy) -> None:
    if not isinstance(observation, JudgmentInput) or not isinstance(policy, Policy):
        raise InputError("invalid_input")
    if not _text(trusted_tenant_id) or observation.tenant_id != trusted_tenant_id:
        raise InputError("tenant_mismatch")
    if not _text(observation.proposal_id) or observation.policy_version != policy.version:
        raise InputError("invalid_binding")
    if (
        (policy.version, policy.min_probability, policy.min_link_confidence)
        != ("shadow-v1", 0.8, 0.8)
        or not _number(policy.min_probability)
        or not _number(policy.min_link_confidence)
    ):
        raise InputError("invalid_policy")
    if type(observation.sources) is not tuple or type(observation.spans) is not tuple:
        raise InputError("mutable_input")
    try:
        if datetime.fromisoformat(observation.observed_at.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise InputError("invalid_observation_time") from None
    sources = {}
    for s in observation.sources:
        if (
            not isinstance(s, SourceSnapshot)
            or not _text(s.source_id)
            or s.source_id in sources
            or not _text(s.text)
        ):
            raise InputError("invalid_source")
        if s.content_hash != content_hash(s.text):
            raise InputError("source_hash_mismatch")
        sources[s.source_id] = s
    spans = {}
    for s in observation.spans:
        if (
            not isinstance(s, EvidenceSpan)
            or not _text(s.span_id)
            or s.span_id in spans
            or not _text(s.source_id)
            or s.source_id not in sources
        ):
            raise InputError("invalid_span")
        if s.basis not in ("company_project", "agency", "dateline", "other"):
            raise InputError("invalid_span_basis")
        text = sources[s.source_id].text
        if (
            type(s.start) is not int
            or type(s.end) is not int
            or not 0 <= s.start < s.end <= len(text)
            or text[s.start : s.end] != s.text
        ):
            raise InputError("span_mismatch")
        spans[s.span_id] = s
    for claim in (observation.organization, observation.location, observation.event):
        if (
            not isinstance(claim, ExtractedClaim)
            or not _text(claim.value)
            or type(claim.span_ids) is not tuple
            or not claim.span_ids
        ):
            raise InputError("invalid_claim")
        if any(not _text(sid) or sid not in spans for sid in claim.span_ids) or not any(
            _contains(spans[sid].text, claim.value) for sid in claim.span_ids
        ):
            raise InputError("unsupported_claim")
    if not _location_basis(observation, spans):
        raise InputError("unsupported_location_basis")


def _binding(observation, policy, provider, model, prompt_version):
    return _hash(
        {
            "observation": asdict(observation),
            "policy": asdict(policy),
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
        }
    )


def prepare_request(
    observation: JudgmentInput,
    *,
    trusted_tenant_id: str,
    policy: Policy,
    provider: str,
    model: str,
    prompt_version: str,
) -> JudgmentRequest:
    validate_input(observation, trusted_tenant_id=trusted_tenant_id, policy=policy)
    if not all(_text(x) for x in (provider, model, prompt_version)):
        raise InputError("invalid_model_binding")
    return JudgmentRequest(
        observation,
        policy,
        provider,
        model,
        prompt_version,
        _binding(observation, policy, provider, model, prompt_version),
    )


def evaluate_receipt(
    request: JudgmentRequest, receipt: Mapping, *, trusted_tenant_id: str
) -> JudgmentResult:
    """Consume an externally supplied replay receipt; never invoke a provider.

    Receipt keys: status, binding_hash, tenant_id, proposal_id, policy_version,
    provider, model, prompt_version, answers, optional cost_usd/latency_ms.
    Answers: five {type:noul,noul:number}; evidence_link {type:choice,
    choice:linked|unresolved|unrelated,confidence:number,probabilities:{...}}.
    """
    o = request.observation

    def result(decision, reason, **kwargs):
        return JudgmentResult(
            o.tenant_id,
            o.proposal_id,
            decision,
            reason,
            request.binding_hash,
            request.provider,
            request.model,
            request.prompt_version,
            request.policy.version,
            **kwargs,
        )

    try:
        validate_input(o, trusted_tenant_id=trusted_tenant_id, policy=request.policy)
        if request.binding_hash != _binding(
            o, request.policy, request.provider, request.model, request.prompt_version
        ):
            raise InputError("request_binding_mismatch")
        if not isinstance(receipt, Mapping):
            raise InputError("malformed_receipt")
        expected = dict(
            binding_hash=request.binding_hash,
            tenant_id=o.tenant_id,
            proposal_id=o.proposal_id,
            policy_version=request.policy.version,
            provider=request.provider,
            model=request.model,
            prompt_version=request.prompt_version,
        )
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise InputError("receipt_binding_mismatch")
        if receipt.get("status") != "ok":
            return result("abstain", "provider_unavailable")
        answers = receipt.get("answers")
        if not isinstance(answers, Mapping) or set(answers) != set(JUDGMENTS) | {"evidence_link"}:
            raise InputError("malformed_answers")
        judgments = []
        clean = {}
        for name in JUDGMENTS:
            a = answers[name]
            if not isinstance(a, Mapping) or a.get("type") != "noul" or not _number(a.get("noul")):
                raise InputError("invalid_probability")
            judgments.append(TypedJudgment(name, probability=a["noul"]))
            clean[name] = {"type": "noul", "noul": a["noul"]}
        a = answers["evidence_link"]
        choices = {"linked", "unresolved", "unrelated"}
        if (
            not isinstance(a, Mapping)
            or a.get("type") != "choice"
            or a.get("choice") not in choices
            or not _number(a.get("confidence"))
        ):
            raise InputError("invalid_evidence_link")
        probs = a.get("probabilities")
        if (
            not isinstance(probs, Mapping)
            or set(probs) != choices
            or not all(_number(v) for v in probs.values())
            or not math.isclose(sum(probs.values()), 1, abs_tol=0.01)
            or probs[a["choice"]] < max(probs.values())
        ):
            raise InputError("invalid_distribution")
        judgments.append(
            TypedJudgment("evidence_link", confidence=a["confidence"], choice=a["choice"])
        )
        clean["evidence_link"] = {
            k: a[k] for k in ("type", "choice", "confidence", "probabilities")
        }
        metrics = {}
        for name in ("cost_usd", "latency_ms"):
            value = receipt.get(name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise InputError("invalid_metrics")
            metrics[name] = value
        supported = (
            all(j.probability >= request.policy.min_probability for j in judgments[:-1])
            and a["choice"] == "linked"
            and a["confidence"] >= request.policy.min_link_confidence
        )
        decision = (
            "supported_candidate"
            if supported
            else "withhold"
            if clean["relevance"]["noul"] < 0.5
            else "abstain"
        )
        return result(
            decision,
            "shadow_supported" if supported else "insufficient_support",
            judgments=tuple(judgments),
            raw_answers_json=json.dumps(clean, sort_keys=True),
            **metrics,
        )
    except InputError as error:
        return result("abstain", str(error))
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return result("abstain", "malformed_receipt")
