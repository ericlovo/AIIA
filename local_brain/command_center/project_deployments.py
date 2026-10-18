"""Bounded GET-only GitHub deployment reports; these are not live health probes."""

from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from . import repository_tools as repos
from .project_status import _SHA, _now, _text, _timestamp

DEPLOYMENT_LIMIT = 30
ENVIRONMENT_LIMIT = 4
GITHUB_TIMEOUT = 4.0
_STATES = {"error", "failure", "inactive", "pending", "success", "queued", "in_progress"}


def _id(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("invalid deployment id")
    return value


def _url(value: Any) -> str | None:
    if value in (None, ""):
        return None
    value = _text(value, limit=2048)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(c.isspace() for c in value)
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("invalid environment URL")
    return value


def _deployment(raw: Any, slug: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("invalid deployment")
    deployment_id = _id(raw["id"])
    sha = _text(raw["sha"], limit=64)
    if not _SHA.fullmatch(sha):
        raise ValueError("invalid SHA")
    return {
        "environment": _text(raw["environment"], limit=255),
        "sha": sha,
        "deployment_id": deployment_id,
        "state": "unknown",
        "created_at": _timestamp(raw["created_at"]),
        "updated_at": _timestamp(raw["updated_at"]),
        "source_url": f"https://github.com/{slug}/deployments",
        "environment_url": None,
    }


def _error(exc: Exception) -> dict[str, str]:
    code = "github_deployments_invalid_response"
    if isinstance(exc, subprocess.TimeoutExpired):
        code = "github_deployments_timeout"
    elif isinstance(exc, (OSError, RuntimeError)):
        code = "github_deployments_unavailable"
    return {
        "code": code,
        "message": "Deployment evidence could not be fully read. Check GitHub access or inspect the source, then refresh.",
    }


def collect_deployments(slug: str | None) -> dict[str, Any]:
    """Return newest deployment per exact environment in one bounded history page.

    At most five GET calls, each with a four-second process timeout. A newer
    failure or missing status never falls back to an older successful deployment.
    """
    result: dict[str, Any] = {
        "checked_at": _now(),
        "status": "unavailable",
        "provider": "github_deployments",
        "scope": "recent_reported_deployments",
        "coverage": "bounded_recent_history",
        "notice": "Last reported deployments; provider-reported state, not independently probed. Environments absent from this bounded history are unknown.",
        "environments": [],
        "errors": [],
        "has_more": None,
        "truncated": False,
    }
    if (
        not isinstance(slug, str)
        or not repos._GITHUB_SLUG.fullmatch(slug)
        or any(p in {".", ".."} for p in slug.split("/"))
    ):
        result["errors"] = [
            {
                "code": "github_origin_unavailable",
                "message": "No valid GitHub origin is available for deployment evidence.",
            }
        ]
        return result
    try:
        payload = repos._github_api(
            f"repos/{slug}/deployments?per_page={DEPLOYMENT_LIMIT}&page=1", timeout=GITHUB_TIMEOUT
        )
        if not isinstance(payload, list):
            raise ValueError("invalid deployments")
        result["has_more"] = len(payload) >= DEPLOYMENT_LIMIT
        result["truncated"] = result["has_more"]
        # Validate the whole page before selecting: dropping a malformed newest
        # report could falsely present an older success as the newest report.
        rows = [_deployment(raw, slug) for raw in payload[:DEPLOYMENT_LIMIT]]
        if len({r["deployment_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate deployment")
        rows.sort(
            key=lambda r: (
                datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                r["deployment_id"],
            ),
            reverse=True,
        )
    except (
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        result["errors"].append(_error(exc))
        return result
    selected = {}
    for row in rows:
        selected.setdefault(row["environment"], row)
    if len(selected) > ENVIRONMENT_LIMIT:
        result["truncated"] = True
    for row in list(selected.values())[:ENVIRONMENT_LIMIT]:
        try:
            statuses = repos._github_api(
                f"repos/{slug}/deployments/{row['deployment_id']}/statuses?per_page=1&page=1",
                timeout=GITHUB_TIMEOUT,
            )
            if not isinstance(statuses, list):
                raise ValueError("invalid statuses")
            if statuses:
                latest = statuses[0]
                if not isinstance(latest, dict):
                    raise ValueError("invalid status")
                _id(latest["id"])
                _timestamp(latest["created_at"])
                updated = _timestamp(latest["updated_at"])
                state = latest["state"]
                if not isinstance(state, str) or state not in _STATES:
                    raise ValueError("unknown status")
                environment_url = _url(latest.get("environment_url"))
                row.update(state=state, updated_at=updated, environment_url=environment_url)
        except (
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            result["errors"].append(_error(exc))
        result["environments"].append(row)
    result["status"] = "partial" if result["errors"] or result["truncated"] else "available"
    return result
