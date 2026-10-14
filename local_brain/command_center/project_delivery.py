"""Explicit local operator delivery associations and read-only PR-head evidence."""

from __future__ import annotations

import subprocess
import time

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from . import repository_tools as repos
from .project_deployments import collect_deployments
from .project_status import _SHA, _now, _project, _run

MAX_PRS = 8
REMOTE_BUDGET = 12.0
RUN_LIMIT = 20


class Links(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_updated_at: StrictStr
    assignment_ids: list[StrictStr] = Field(max_length=50)
    pr_numbers: list[StrictInt] = Field(max_length=20)


def _error(code):
    return {"code": code, "message": code.replace("_", " ")}


def _pr(number, slug, deadline):
    result = {
        "number": number,
        "status": "unavailable",
        "errors": [],
        "ci": {
            "status": "unavailable",
            "scope": "pull_request_head",
            "head_sha": None,
            "runs": [],
            "errors": [],
        },
    }

    def fetch(endpoint):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("remote_budget_exhausted")
        return repos._github_api(endpoint, timeout=min(4.0, remaining))

    try:
        raw = fetch(f"repos/{slug}/pulls/{number}")
        if (
            not isinstance(raw, dict)
            or type(raw.get("number")) is not int
            or raw["number"] != number
            or raw.get("state") not in {"open", "closed"}
            or type(raw.get("merged")) is not bool
            or type(raw.get("draft")) is not bool
            or not isinstance(raw.get("title"), str)
            or raw.get("html_url") != f"https://github.com/{slug}/pull/{number}"
        ):
            raise ValueError("invalid PR")
        mergeable = raw.get("mergeable")
        mergeable_state = raw.get("mergeable_state")
        if mergeable is not None and type(mergeable) is not bool:
            raise ValueError("invalid mergeable")
        if mergeable_state is not None and (
            not isinstance(mergeable_state, str)
            or not mergeable_state.strip()
            or len(mergeable_state) > 80
        ):
            raise ValueError("invalid mergeable state")
        sha = raw["head"]["sha"]
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError("invalid head")
        result.update({k: raw[k] for k in ("title", "state", "merged", "draft", "html_url")})
        result.update(
            status="available",
            head_sha=sha,
            fetched_at=_now(),
            mergeable=mergeable,
            mergeable_state=mergeable_state,
        )
        result["ci"]["head_sha"] = sha
    except (
        OSError,
        subprocess.TimeoutExpired,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        code = (
            "remote_budget_exhausted"
            if str(exc) == "remote_budget_exhausted"
            else "github_pr_unavailable"
        )
        result["errors"] = [_error(code)]
        return result
    try:
        raw = fetch(f"repos/{slug}/actions/runs?head_sha={sha}&per_page={RUN_LIMIT}&page=1")
        runs = raw["workflow_runs"]
        total = raw["total_count"]
        if (
            not isinstance(runs, list)
            or type(total) is not int
            or total < len(runs)
            or (total and not runs)
        ):
            raise ValueError("invalid runs")
        now = _now()
        parsed = []
        for run in runs[:RUN_LIMIT]:
            item = _run(run, slug, now)
            workflow_id = run.get("workflow_id")
            if workflow_id is not None:
                if type(workflow_id) is not int or workflow_id <= 0:
                    raise ValueError("invalid workflow id")
                item["workflow_id"] = workflow_id
            parsed.append(item)
        if any(run["head_sha"] != sha for run in parsed) or len({r["id"] for r in parsed}) != len(
            parsed
        ):
            raise ValueError("wrong head or duplicate")
        result["ci"].update(
            status="available",
            runs=parsed,
            fetched_at=now,
            has_more=total > len(parsed),
            total_count=total,
            source_url=f"https://github.com/{slug}/actions",
        )
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError, TypeError):
        result["ci"]["errors"] = [_error("github_head_ci_unavailable")]
    return result


def create_router(roadmap, assignments, agents):
    router = APIRouter(prefix="/api/projects", tags=["projects"])

    def project_exists(repo_id):
        if repo_id not in repos.REPO_MOUNTS:
            raise HTTPException(404, "project_not_found")

    def assignment_view(item):
        agent = agents.get(item.get("agent_id", "")) or {}
        return {
            **{
                k: item.get(k)
                for k in ("id", "title", "status", "agent_id", "review_status", "updated_at")
            },
            "agent_name": agent.get("name"),
            "agent_status": agent.get("status"),
        }

    @router.put("/{repo_id}/features/{story_id}/links")
    def update_links(repo_id: str, story_id: str, body: Links):
        project_exists(repo_id)
        if roadmap.load_error:
            raise HTTPException(503, "roadmap_unavailable")
        if getattr(assignments, "load_error", False):
            raise HTTPException(503, "assignments_unavailable")
        story = roadmap.get(story_id)
        if not story or story.get("product") != repo_id:
            raise HTTPException(404, "feature_not_found")
        if any(not value.strip() or len(value) > 128 for value in body.assignment_ids):
            raise HTTPException(422, "invalid_assignment_id")
        if any(number <= 0 or number > 2147483647 for number in body.pr_numbers):
            raise HTTPException(422, "invalid_pr_number")
        for assignment_id in body.assignment_ids:
            item = assignments.get_assignment(assignment_id)
            if not item or item.get("repo_id") not in (None, "", repo_id):
                raise HTTPException(422, "assignment_missing_or_wrong_project")
        try:
            feature = roadmap.update_delivery_links(
                story_id,
                repo_id,
                list(dict.fromkeys(body.assignment_ids)),
                list(dict.fromkeys(body.pr_numbers)),
                body.expected_updated_at,
            )
        except KeyError as exc:
            raise HTTPException(404, "feature_not_found") from exc
        except ValueError as exc:
            raise HTTPException(409, "delivery_links_changed_refresh_required") from exc
        except OSError as exc:
            raise HTTPException(503, "delivery_links_persistence_failed") from exc
        return {"feature": feature}

    @router.get("/{repo_id}/delivery")
    def delivery(repo_id: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        project_exists(repo_id)
        project = _project(repo_id)
        checked_at = _now()
        deadline = time.monotonic() + REMOTE_BUDGET
        evidence = {}
        features = []
        errors = [_error("roadmap_unavailable")] if roadmap.load_error else []
        if getattr(assignments, "load_error", False):
            errors.append(_error("assignments_unavailable"))
        for story in roadmap.delivery_stories(repo_id):
            linked = []
            ids = story.get("assignment_ids", [])
            numbers = story.get("pr_numbers", [])
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                ids = []
                errors.append(_error("invalid_stored_assignment_links"))
            if not isinstance(numbers, list) or any(type(n) is not int or n <= 0 for n in numbers):
                numbers = []
                errors.append(_error("invalid_stored_pr_links"))
            for assignment_id in ids:
                item = assignments.get_assignment(assignment_id)
                if not item or item.get("repo_id") not in (None, "", repo_id):
                    linked.append({"id": assignment_id, "status": "missing"})
                    errors.append(_error("linked_assignment_missing"))
                else:
                    linked.append(assignment_view(item))
            prs = []
            for number in numbers:
                if number not in evidence:
                    if len(evidence) < MAX_PRS and project["github_repo"]:
                        evidence[number] = _pr(number, project["github_repo"], deadline)
                    else:
                        evidence[number] = {
                            "number": number,
                            "status": "unavailable",
                            "errors": [
                                _error(
                                    "github_origin_unavailable"
                                    if not project["github_repo"]
                                    else "remote_limit_reached"
                                )
                            ],
                        }
                prs.append(evidence[number])
            features.append(
                {
                    **story,
                    "assignment_ids": ids,
                    "pr_numbers": numbers,
                    "assignments": linked,
                    "pull_requests": prs,
                    "deployment": {"status": "unknown", "connection": "not_connected"},
                }
            )
        deployments = collect_deployments(project["github_repo"])
        partial = (
            deployments["status"] != "available"
            or bool(errors)
            or any(
                p["status"] != "available"
                or p.get("ci", {}).get("status") != "available"
                or p.get("ci", {}).get("has_more")
                for p in evidence.values()
            )
        )
        return {
            "repo_id": repo_id,
            "checked_at": checked_at,
            "status": "partial" if partial else "available",
            "features": features,
            "errors": errors,
            "deployments": deployments,
            "assignment_scope": "unscoped_operator_selection",
            "assignment_candidates": [
                assignment_view(a)
                for a in assignments.list_assignments()
                if a.get("repo_id") in (None, "", repo_id)
            ],
            "sources": {
                "roadmap": {
                    "status": "unavailable" if roadmap.load_error else "available",
                    "fetched_at": checked_at,
                },
                "assignments": {
                    "status": "unavailable"
                    if getattr(assignments, "load_error", False)
                    else "available",
                    "fetched_at": checked_at,
                },
            },
        }

    return router
