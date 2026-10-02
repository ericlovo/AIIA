"""Read-only checkout and GitHub Actions evidence for mounted projects."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Response

from . import repository_tools as repos

router = APIRouter(prefix="/api/projects", tags=["projects"])
RUN_LIMIT = 20
GITHUB_TIMEOUT = 8.0
_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ERRORS = {
    "repository_not_mounted": "Restore the configured repository mount, then refresh.",
    "checkout_unavailable": "Check that the mounted checkout has a readable HEAD commit, then refresh.",
    "github_origin_unavailable": "Check the mounted repository's GitHub origin, then refresh.",
    "github_cli_missing": "Install the GitHub CLI on the Command Center host, then refresh.",
    "github_api_unavailable": "Check GitHub CLI authentication, repository Actions read access, rate limits and connectivity, then refresh.",
    "github_api_timeout": "GitHub did not respond in time. Check connectivity, then refresh.",
    "github_api_invalid_response": "GitHub returned incomplete or invalid run evidence. Refresh or inspect the source on GitHub.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(code: str) -> dict[str, str]:
    return {"code": code, "message": _ERRORS[code]}


def _project(repo_id: str) -> dict[str, Any]:
    path = repos.REPO_MOUNTS[repo_id]
    mounted = repos.repo_available(repo_id)
    errors = []
    sha = branch = None
    detached = None
    slug = ""
    if mounted:
        # One process reads both values; never infer a branch from a remote run.
        revision = repos._git_checked(path, "rev-parse", "HEAD", "--abbrev-ref", "HEAD")
        parts = revision.splitlines() if revision else []
        if len(parts) == 2 and _SHA.fullmatch(parts[0]) and parts[1]:
            sha, branch = parts
            detached = branch == "HEAD"
            if detached:
                branch = None
        else:
            errors.append(_error("checkout_unavailable"))
        slug = repos._origin_slug(path)
        if not repos._GITHUB_SLUG.fullmatch(slug) or any(
            part in {".", ".."} for part in slug.split("/")
        ):
            slug = ""
        if not slug:
            errors.append(_error("github_origin_unavailable"))
    else:
        errors.append(_error("repository_not_mounted"))
    return {
        "id": repo_id,
        "name": repos.REPO_NAMES.get(repo_id, path.name),
        "path": str(path),
        "mounted": mounted,
        "checkout_status": "available" if sha else "unavailable",
        "branch": branch,
        "head_sha": sha,
        "detached": detached,
        "github_repo": slug or None,
        "html_url": f"https://github.com/{slug}" if slug else None,
        "checked_at": _now(),
        "errors": errors,
        "deployment": {"status": "unknown", "connection": "not_connected"},
    }


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("invalid timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp missing timezone")
    return value


def _text(value: Any, *, nullable: bool = False, limit: int = 512) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("invalid text")
    return value


def _run(raw: Any, slug: str, fetched_at: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("invalid run")
    run_id = raw["id"]
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("invalid run id")
    url = _text(raw["html_url"], limit=2048)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.path.lower() != f"/{slug}/actions/runs/{run_id}".lower()
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid source URL")
    sha = _text(raw["head_sha"], limit=64)
    if not _SHA.fullmatch(sha):
        raise ValueError("invalid head sha")
    status = _text(raw["status"], limit=80)
    conclusion = _text(raw["conclusion"], nullable=True, limit=80)
    if status != "completed" and conclusion is not None:
        raise ValueError("inconsistent run state")
    return {
        "id": run_id,
        "name": _text(raw.get("name"), nullable=True),
        "html_url": url,
        "head_sha": sha,
        "head_branch": _text(raw["head_branch"], nullable=True, limit=1024),
        "status": status,
        "conclusion": conclusion,
        "created_at": _timestamp(raw["created_at"]),
        "updated_at": _timestamp(raw["updated_at"]),
        "fetched_at": fetched_at,
    }


# FastAPI runs synchronous route functions in its worker thread pool.
@router.get("")
def list_projects(response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return {"projects": [_project(repo_id) for repo_id in repos.REPO_MOUNTS]}


@router.get("/{repo_id}/ci")
def project_ci(repo_id: str, response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    if repo_id not in repos.REPO_MOUNTS:
        raise HTTPException(status_code=404, detail="project_not_found")
    project = _project(repo_id)
    slug = project["github_repo"]
    result = {
        "project": project,
        "status": "unavailable",
        "provider": "github_actions",
        "scope": "repository_recent_runs",
        "source_url": f"https://github.com/{slug}/actions" if slug else None,
        "api_url": (
            f"https://api.github.com/repos/{slug}/actions/runs?per_page={RUN_LIMIT}&page=1"
            if slug
            else None
        ),
        "attempted_at": _now(),
        "fetched_at": None,
        "limit": RUN_LIMIT,
        "total_count": None,
        "has_more": None,
        "runs": [],
        "errors": [],
    }
    if not project["mounted"] or not slug:
        code = "github_origin_unavailable" if project["mounted"] else "repository_not_mounted"
        result["errors"] = [_error(code)]
        return result
    try:
        payload = repos._github_api(
            f"repos/{slug}/actions/runs?per_page={RUN_LIMIT}&page=1",
            timeout=GITHUB_TIMEOUT,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
            raise ValueError("invalid response")
        total = payload.get("total_count")
        raw_runs = payload["workflow_runs"]
        if type(total) is not int or total < len(raw_runs) or (total > 0 and not raw_runs):
            raise ValueError("invalid total")
        fetched_at = _now()
        runs = [_run(raw, slug, fetched_at) for raw in raw_runs[:RUN_LIMIT]]
        if len({run["id"] for run in runs}) != len(runs):
            raise ValueError("duplicate run")
    except subprocess.TimeoutExpired:
        code = "github_api_timeout"
    except OSError:
        code = "github_api_unavailable"
    except RuntimeError as exc:
        code = str(exc) if str(exc) in _ERRORS else "github_api_unavailable"
    except (ValueError, KeyError, TypeError):
        code = "github_api_invalid_response"
    else:
        result.update(
            status="available",
            fetched_at=fetched_at,
            total_count=total,
            has_more=total > len(runs),
            runs=runs,
        )
        return result
    result["errors"] = [_error(code)]
    return result
