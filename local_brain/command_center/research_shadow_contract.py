"""Offline subset of Managed Leads runner v1; not a live HTTP client.

Only the minimal submission emitted here is supported. The real server remains
responsible for its full delivery schema, RLS, quota and current scope checks.
Source snapshots and replay judgments remain local, outside this wire envelope.
"""

import hashlib
import hmac
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Brief(Strict):
    version: str = Field(min_length=1, max_length=100)
    icp: str = Field(max_length=2000)
    offer: str = Field(max_length=2000)
    industry: str = Field(default="", max_length=200)
    exclusions: list[str] = Field(default_factory=list, max_length=500)
    evidenceMaxAgeDays: int | None = Field(default=None, ge=1, le=3650)
    contactRequirement: Literal["optional", "required"] = "optional"

    @field_validator("exclusions")
    @classmethod
    def exclusions_supported(cls, values):
        # Conservative subset; never expand the real server's accepted scope.
        for value in values:
            if not (
                re.fullmatch(r"domain:[a-z0-9-]+(?:\.[a-z0-9-]+)+", value)
                or re.fullmatch(r"org:[A-Za-z0-9][A-Za-z0-9 ._-]*", value)
            ):
                raise ValueError("unsupported exclusion rule")
        return values


class AssignmentWire(Strict):
    schemaVersion: Literal[1]
    mode: Literal["shadow"]
    workspaceId: str = Field(pattern=r"^synthetic-[a-z0-9-]{1,60}$")
    assignmentRevision: str = Field(pattern=r"^[a-f0-9]{64}$")
    brief: Brief

    @field_validator("schemaVersion", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("integer schema version required")
        return value


@dataclass(frozen=True, init=False)
class AssignmentSnapshot:
    """Deeply immutable scope: serialized brief, never a mutable dict reference."""

    workspace_id: str
    assignment_revision: str
    _wire_json: str = field(repr=False)

    @classmethod
    def from_wire(cls, wire):
        validated = AssignmentWire.model_validate(wire)
        snapshot = object.__new__(cls)
        object.__setattr__(snapshot, "workspace_id", validated.workspaceId)
        object.__setattr__(snapshot, "assignment_revision", validated.assignmentRevision)
        # Preserve the received brief verbatim, including optional fields' presence.
        object.__setattr__(snapshot, "_wire_json", json.dumps(wire, allow_nan=False))
        return snapshot

    def to_wire(self):
        return json.loads(self._wire_json)


def synthetic_assignment(workspace_id):
    brief = Brief(
        version="synthetic-v1",
        icp="Synthetic organizations only",
        offer="Synthetic research review",
        industry="synthetic",
    ).model_dump()
    revision = hashlib.sha256(json.dumps(brief, sort_keys=True).encode()).hexdigest()
    return AssignmentSnapshot.from_wire(
        {
            "schemaVersion": 1,
            "mode": "shadow",
            "workspaceId": workspace_id,
            "assignmentRevision": revision,
            "brief": brief,
        }
    )


def build_submission(assignment_snapshot, job_id, candidate):
    """Build only unconfirmed pending research; no evidence or probability invention."""
    if not isinstance(assignment_snapshot, AssignmentSnapshot):
        raise ValueError("validated assignment snapshot required")
    if not isinstance(job_id, str) or not 1 <= len(job_id) <= 200:
        raise ValueError("bounded stable job ID required")
    if not isinstance(candidate, dict) or set(candidate) != {
        "organization",
        "location",
        "signal_type",
    }:
        raise ValueError("supported replay candidate required")
    if any(
        not isinstance(value, str) or not value.strip() or len(value) > 100
        for value in candidate.values()
    ):
        raise ValueError("bounded candidate values required")
    key = str(
        uuid5(NAMESPACE_URL, json.dumps([assignment_snapshot.workspace_id, "shadow", job_id]))
    )
    return {
        "schemaVersion": 1,
        "mode": "shadow",
        "idempotencyKey": key,
        "assignmentRevision": assignment_snapshot.assignment_revision,
        "source": {
            "schemaVersion": 1,
            "id": "synthetic-" + key,
            "name": "Synthetic shadow batch",
            "brief": assignment_snapshot.to_wire()["brief"],
            "candidates": [
                {
                    "id": "synthetic-" + key,
                    "org": candidate["organization"],
                    "whyFit": "Proposed relevance; awaiting evidence review.",
                    "identityStatus": "unknown",
                    "decision": "pending",
                    "gaps": ["Identity and evidence require human review"],
                    "claims": [],
                    "contacts": [],
                }
            ],
        },
    }


class ContractError(ValueError):
    def __init__(self, status_code, reason):
        self.status_code = status_code
        super().__init__(reason)


def _normalized(payload):
    """Normalize the emitted minimal subset with real Candidate/Brief defaults."""
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion",
        "mode",
        "idempotencyKey",
        "assignmentRevision",
        "source",
    }:
        raise ValueError("invalid envelope")
    if (
        type(payload["schemaVersion"]) is not int
        or payload["schemaVersion"] != 1
        or payload["mode"] != "shadow"
    ):
        raise ValueError("invalid version")
    key = payload["idempotencyKey"]
    if not isinstance(key, str) or len(key) != 36:
        raise ValueError("invalid idempotency key")
    result = deepcopy(payload)
    result["idempotencyKey"] = str(UUID(key))
    if not re.fullmatch(r"[a-f0-9]{64}", payload["assignmentRevision"]):
        raise ValueError("invalid revision")
    source = result["source"]
    if not isinstance(source, dict) or set(source) != {
        "schemaVersion",
        "id",
        "name",
        "brief",
        "candidates",
    }:
        raise ValueError("invalid source")
    if type(source["schemaVersion"]) is not int or source["schemaVersion"] != 1:
        raise ValueError("invalid source version")
    if any(
        not isinstance(source[k], str) or not 1 <= len(source[k]) <= 200 for k in ("id", "name")
    ):
        raise ValueError("invalid source ID")
    source["brief"] = Brief.model_validate(source["brief"]).model_dump()
    candidates = source["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("mock supports one candidate")
    c = candidates[0]
    if not isinstance(c, dict) or set(c) != {
        "id",
        "org",
        "whyFit",
        "identityStatus",
        "decision",
        "gaps",
        "claims",
        "contacts",
    }:
        raise ValueError("unsupported candidate fields")
    if (
        c["identityStatus"] != "unknown"
        or c["decision"] != "pending"
        or c["claims"] != []
        or c["contacts"] != []
    ):
        raise ValueError("unsupported candidate assertion")
    for key, limit in (("id", 200), ("org", 200), ("whyFit", 8000)):
        if not isinstance(c[key], str) or not c[key].strip() or len(c[key]) > limit:
            raise ValueError("invalid candidate text")
    if (
        not isinstance(c["gaps"], list)
        or len(c["gaps"]) > 100
        or any(not isinstance(gap, str) for gap in c["gaps"])
    ):
        raise ValueError("invalid gaps")
    c.update(website="", segment="", location="", classification="cold", reason="")
    return result


class MockRunnerTransport:
    """In-memory, fake-token-only contract simulation. Never performs HTTP."""

    def __init__(self, assignments, bindings, now=None):
        self._assignments = {a.workspace_id: a for a in assignments}
        self._bindings = {}
        self._receipts = {}
        self._now = now or (lambda: datetime.now(timezone.utc).isoformat())
        for runner, (workspace, fake_token) in bindings.items():
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", runner)
                or workspace not in self._assignments
                or not isinstance(fake_token, str)
                or not fake_token.startswith("fake-")
                or not 32 <= len(fake_token) <= 512
                or not fake_token.isascii()
                or any(not 33 <= ord(c) < 127 for c in fake_token)
            ):
                raise ValueError("synthetic binding and explicit fake token required")
            digest = hashlib.sha256(fake_token.encode()).hexdigest()
            if any(existing[1] == digest for existing in self._bindings.values()):
                raise ValueError("fake token must be unique per runner binding")
            self._bindings[runner] = (workspace, digest)

    def replace_assignment(self, snapshot):
        self._assignments[snapshot.workspace_id] = snapshot

    def _auth(self, headers, query):
        if query:
            raise ContractError(422, "query parameters unsupported")
        if not isinstance(headers, dict):
            raise ContractError(401, "runner credentials required")
        h = {str(key).lower(): value for key, value in headers.items()}
        if len(h) != len(headers) or "cookie" in h:
            raise ContractError(401, "runner credentials required")
        runner = h.get("x-runner-client")
        authorization = h.get("authorization", "")
        if not isinstance(runner, str) or not isinstance(authorization, str):
            raise ContractError(401, "runner credentials required")
        parts = authorization.split(" ")
        binding = self._bindings.get(runner)
        if len(parts) != 2 or parts[0].lower() != "bearer" or binding is None:
            raise ContractError(401, "runner credentials required")
        if not hmac.compare_digest(hashlib.sha256(parts[1].encode()).hexdigest(), binding[1]):
            raise ContractError(401, "runner credentials required")
        return binding[0], runner

    def get_assignment(self, headers, query=None):
        tenant, _ = self._auth(headers, query)
        return 200, self._assignments[tenant].to_wire()

    def submit(self, payload, headers, query=None):
        tenant, runner = self._auth(headers, query)
        try:
            if len(json.dumps(payload, allow_nan=False).encode()) > 1024 * 1024:
                raise ContractError(413, "payload too large")
            normalized = _normalized(payload)
        except (ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(422, "invalid shadow submission") from None
        fingerprint = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        receipt_id = str(
            uuid5(
                NAMESPACE_URL,
                json.dumps([tenant, "runner-shadow", runner, normalized["idempotencyKey"]]),
            )
        )
        key = tenant, runner, receipt_id
        existing = self._receipts.get(key)
        if existing:
            if existing["payloadHash"] != fingerprint:
                raise ContractError(409, "idempotency conflict")
            return 200, {"receipt": deepcopy(existing), "replayed": True}
        assignment = self._assignments[tenant]
        if normalized["assignmentRevision"] != assignment.assignment_revision:
            raise ContractError(409, "assignment changed")
        if (
            normalized["source"]["brief"]
            != Brief.model_validate(assignment.to_wire()["brief"]).model_dump()
        ):
            raise ContractError(422, "brief mismatch")
        receipt = {
            "schemaVersion": 1,
            "receiptId": receipt_id,
            "mode": "shadow",
            "status": "shadow_received",
            "candidateCount": 1,
            "receivedAt": self._now(),
            "assignmentRevision": normalized["assignmentRevision"],
            "payloadHash": fingerprint,
        }
        self._receipts[key] = receipt
        return 201, {"receipt": deepcopy(receipt), "replayed": False}

    def get_receipt(self, receipt_id, headers, query=None):
        tenant, runner = self._auth(headers, query)
        receipt = self._receipts.get((tenant, runner, receipt_id))
        if receipt is None:
            raise ContractError(404, "receipt not found")
        return 200, deepcopy(receipt)
