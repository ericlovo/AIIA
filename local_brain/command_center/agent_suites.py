"""Named Agent Studio suites — catalog + tagging helpers.

Mindmoor is the first client-isolated suite. Membership is by display
name / slug, not live Mini agent IDs (those stay on the Mini).
"""

from __future__ import annotations

import re
from typing import Any

SUITE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

MINDMOOR_SUITE = "mindmoor"
MINDMOOR_MEMORY_NAMESPACE = "mindmoor"
MINDMOOR_MEMORY_SOURCE = "suite:mindmoor"
MINDMOOR_REPO_ID = "mindmoor"
MINDMOOR_MEMORY_COLLECTIONS = ("project", "decisions", "agents", "lessons")

# Canonical members as they appear in Studio. Do not invent agent IDs.
MINDMOOR_MEMBERS: tuple[dict[str, Any], ...] = (
    {
        "slug": "delivery-watch",
        "name": "Delivery Watch",
        "aliases": ("delivery watch",),
        "depth": 0,
        "role": "observe",
        "recommended_max_tokens": 1200,
        "produced_artifacts": ("signal_report",),
    },
    {
        "slug": "mindmoor-scout",
        "name": "Mindmoor Scout",
        "aliases": ("mindmoor scout",),
        "depth": 0,
        "role": "observe",
        "recommended_max_tokens": 1200,
        "produced_artifacts": ("signal_report",),
    },
    {
        "slug": "discovery-bot",
        "name": "Discovery Bot",
        "aliases": ("discovery bot", "mindmoor discovery bot"),
        "depth": 1,
        "role": "interpret",
        "recommended_max_tokens": 1200,
        "produced_artifacts": ("signal_report", "documentation"),
    },
    {
        "slug": "specialty-probe",
        "name": "Specialty Probe",
        "aliases": ("specialty probe",),
        "depth": 1,
        "role": "interpret",
        "recommended_max_tokens": 1200,
        "produced_artifacts": ("signal_report", "failure_analysis"),
    },
    {
        "slug": "cron-test-engineer",
        "name": "Cron Test Engineer",
        "aliases": ("cron test engineer",),
        "depth": 1,
        "role": "interpret",
        "recommended_max_tokens": 1200,
        "produced_artifacts": ("signal_report", "failure_analysis"),
    },
    {
        "slug": "cron-review-gate",
        "name": "Cron Review Gate",
        "aliases": ("cron review gate", "mindmoor cron review gate"),
        "depth": 1,
        "role": "review",
        "recommended_max_tokens": 1600,
        "produced_artifacts": ("risk_review", "audit_record"),
    },
)

KNOWN_SUITES: dict[str, dict[str, Any]] = {
    MINDMOOR_SUITE: {
        "slug": MINDMOOR_SUITE,
        "name": "Mindmoor Agent Suite",
        "memory_namespace": MINDMOOR_MEMORY_NAMESPACE,
        "memory_source": MINDMOOR_MEMORY_SOURCE,
        "repository_id": MINDMOOR_REPO_ID,
        "max_depth": 2,
        "members": MINDMOOR_MEMBERS,
    }
}


def normalize_agent_name(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _member_names(member: dict[str, Any]) -> set[str]:
    names = {normalize_agent_name(member["name"])}
    names.update(normalize_agent_name(alias) for alias in member.get("aliases", ()))
    return {name for name in names if name}


def match_member(name: str, suite: str = MINDMOOR_SUITE) -> dict[str, Any] | None:
    """Return the catalog member for a Studio display name, or None."""
    catalog = KNOWN_SUITES.get(suite)
    if not catalog:
        return None
    needle = normalize_agent_name(name)
    if not needle:
        return None
    for member in catalog["members"]:
        for alias in _member_names(member):
            if needle == alias or needle.endswith(f" {alias}") or needle.startswith(f"{alias} "):
                return member
    return None


def infer_suite(agent: dict[str, Any]) -> str:
    tagged = str(agent.get("suite") or "").strip()
    if tagged:
        return tagged
    if match_member(str(agent.get("name") or "")):
        return MINDMOOR_SUITE
    return ""


def memory_scope_for_suite(suite: str = MINDMOOR_SUITE) -> dict[str, Any]:
    catalog = KNOWN_SUITES.get(suite, {})
    namespace = catalog.get("memory_namespace", suite)
    return {
        "collections": list(MINDMOOR_MEMORY_COLLECTIONS if suite == MINDMOOR_SUITE else ()),
        "write": True,
        "namespace": namespace,
        "source": catalog.get("memory_source", f"suite:{namespace}" if namespace else ""),
    }


def default_memory_namespace(suite: str) -> str:
    catalog = KNOWN_SUITES.get(suite)
    if catalog:
        return str(catalog["memory_namespace"])
    return suite


def normalize_suite_slug(value: Any) -> str:
    slug = str(value or "").strip().lower()
    if not slug:
        return ""
    if not SUITE_SLUG_RE.fullmatch(slug):
        raise ValueError("invalid_suite")
    return slug


def normalize_memory_namespace(value: Any) -> str:
    namespace = str(value or "").strip().lower()
    if not namespace:
        return ""
    if not SUITE_SLUG_RE.fullmatch(namespace):
        raise ValueError("invalid_memory_namespace")
    return namespace


def apply_suite_defaults(suite: str = "", memory_namespace: str = "") -> tuple[str, str]:
    """Fill namespace when a suite is set. Do not infer suite on write."""
    suite = normalize_suite_slug(suite)
    memory_namespace = normalize_memory_namespace(memory_namespace)
    if suite and not memory_namespace:
        memory_namespace = default_memory_namespace(suite)
    return suite, memory_namespace


def suite_prompt_line(agent: dict[str, Any]) -> str:
    """One system-prompt line when the agent is suite-tagged. Empty otherwise."""
    suite = str(agent.get("suite") or "").strip()
    if not suite:
        return ""
    namespace = str(agent.get("memory_namespace") or suite).strip() or suite
    return (
        f"Suite: {suite}. Shared local memory namespace: {namespace} "
        f"(source suite:{namespace}). Stay client-isolated; do not claim "
        "other tenants' memory, email, web fetch, or voice egress.\n"
    )


def describe_suites(agents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Catalog plus any matching in-memory Studio agents (tag or alias)."""
    live = agents or []
    suites = []
    for catalog in KNOWN_SUITES.values():
        members = []
        for member in catalog["members"]:
            matched = [
                {
                    "id": agent.get("id", ""),
                    "name": agent.get("name", ""),
                    "suite": agent.get("suite", ""),
                    "memory_namespace": agent.get("memory_namespace", ""),
                    "max_tokens": agent.get("max_tokens"),
                    "matched_by": "tag"
                    if str(agent.get("suite") or "") == catalog["slug"]
                    else "alias",
                }
                for agent in live
                if str(agent.get("suite") or "") == catalog["slug"]
                or match_member(str(agent.get("name") or ""), catalog["slug"]) is member
            ]
            members.append({**member, "agents": matched})
        suites.append(
            {
                "slug": catalog["slug"],
                "name": catalog["name"],
                "memory_namespace": catalog["memory_namespace"],
                "memory_source": catalog["memory_source"],
                "repository_id": catalog["repository_id"],
                "max_depth": catalog["max_depth"],
                "members": members,
            }
        )
    return suites
