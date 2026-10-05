"""Deterministic daily digest: triage mounted repos, CI, and behind-main drift.

One line per day. Quiet/green results record without opening a review item;
only material stuck or incomplete evidence goes to the agent's output channel
(Studio inbox, with Slack falling back to inbox until posting exists).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_brain.command_center import repository_tools as repos
from local_brain.command_center.agent_output import (
    DEFAULT_OUTPUT_CHANNEL,
    INBOX_UNAVAILABLE_NOTE,
    inbox_title,
    resolve_delivery,
)
from local_brain.command_center.agent_registry import RunHistoryUnavailable
from local_brain.command_center.assignment_registry import MAX_PENDING_LOOP_REVIEWS

logger = logging.getLogger("aiia.daily_digest")

DIGEST_LINE_MAX = 200
# 07:00 America/Chicago during CDT; 06:00 during CST. The built-in scheduler
# is UTC-only (hour + minute), so the cron is fixed at 12:00 UTC.
DIGEST_CRON_HOUR_UTC = 12
DIGEST_CRON_MINUTE_UTC = 0
DIGEST_TIMEZONE = "America/Chicago"
DIGEST_AGENT_NAME = "Daily Digest"
DIGEST_ONE_LINER = "Triage mounted repos: what moved, what's stuck, CI and drift."
DIGEST_MISSION = (
    "Once a day, read mounted checkouts and GitHub Actions for AIIA, Mindmoor, "
    "and any other mounted repos (Sanction, Morrow, MIA, Proxy AI). Report what "
    "moved, what is stuck (failing CI, merge conflicts), and Mindmoor "
    "production/alumni drift versus main. One line, evidence only."
)
DIGEST_PERSONA = "Evidence first. One line. No speculation."
PREFERRED_REPOS = ("aiia", "mindmoor", "sanction", "proxy-ai", "morrow", "mia")
PR_LIST_LIMIT = 20
PR_DETAIL_CAP = 8
CI_RUN_LIMIT = 10
MAIN_REF_CANDIDATES = ("origin/main", "main", "origin/master", "master")
PRODUCTION_REF_CANDIDATES = ("origin/production", "production")
ALUMNI_EXACT_CANDIDATES = ("origin/alumni", "alumni")
ALUMNI_PREFIXES = ("origin/release/alumni", "release/alumni")
AGENT_MISSING_NOTE = "daily digest agent not enabled; result kept on the task only"
QUIET_CLEAR_NOTE = "quiet clear; recorded without review"

GitRead = Callable[..., str | None]
GitHubApi = Callable[[str], Any]


@dataclass
class DriftSignal:
    repo_id: str
    ref: str
    behind_main: int


@dataclass
class RepoEvidence:
    repo_id: str
    mounted: bool
    complete: bool
    open_prs: int | None = None
    failing_ci: int | None = None
    merge_conflicts: int | None = None
    behind_main: int | None = None
    recent_commits: int | None = None
    drift: list[DriftSignal] = field(default_factory=list)
    failures: tuple[str, ...] = ()
    moved: list[str] = field(default_factory=list)
    stuck: list[str] = field(default_factory=list)


@dataclass
class DigestResult:
    line: str
    severity: str
    fingerprint: str
    evidence: list[RepoEvidence]
    failures: tuple[str, ...] = ()

    @property
    def material(self) -> bool:
        return self.severity in {"stuck", "incomplete"}


def digest_agent_payload(*, output_channel: str = DEFAULT_OUTPUT_CHANNEL) -> dict[str, Any]:
    """Body for creating the standing Daily Digest agent. Never writes runtime JSON."""
    return {
        "name": DIGEST_AGENT_NAME,
        "mission": DIGEST_MISSION,
        "persona": DIGEST_PERSONA,
        "skills": ["Analysis"],
        "tools": ["Repository read", "GitHub read"],
        "one_liner": DIGEST_ONE_LINER,
        "output_channel": output_channel,
        "loop_enabled": False,
        "loop_task": "",
    }


def find_digest_agent(agents: list[dict[str, Any]] | Any) -> dict[str, Any] | None:
    rows = agents.list() if hasattr(agents, "list") else list(agents or [])
    for agent in rows:
        if str(agent.get("name") or "").strip().lower() == DIGEST_AGENT_NAME.lower():
            return agent
    return None


def watched_repo_ids(mounts: dict[str, Path] | None = None) -> list[str]:
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    known = [repo_id for repo_id in PREFERRED_REPOS if repo_id in mounts]
    extras = [repo_id for repo_id in mounts if repo_id not in known]
    return known + extras


def _git_read(path: Path, *args: str) -> str | None:
    return repos._git_checked(path, *args)


def _ref_exists(git_read: GitRead, path: Path, ref: str) -> bool:
    value = git_read(path, "rev-parse", "--verify", "--quiet", ref)
    return bool(value)


def _first_ref(git_read: GitRead, path: Path, candidates: tuple[str, ...]) -> str | None:
    for ref in candidates:
        if _ref_exists(git_read, path, ref):
            return ref
    return None


def _behind_count(git_read: GitRead, path: Path, older: str, newer: str) -> int | None:
    raw = git_read(path, "rev-list", "--count", f"{older}..{newer}")
    if raw is None or not raw.isdigit():
        return None
    return int(raw)


def _alumni_ref(git_read: GitRead, path: Path) -> str | None:
    exact = _first_ref(git_read, path, ALUMNI_EXACT_CANDIDATES)
    if exact:
        return exact
    listed = git_read(
        path, "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes"
    )
    if listed is None:
        return None
    matches = [
        line.strip()
        for line in listed.splitlines()
        if any(line.strip().startswith(prefix) for prefix in ALUMNI_PREFIXES)
    ]
    return sorted(matches)[0] if matches else None


def _mindmoor_drift(git_read: GitRead, path: Path, main_ref: str | None) -> list[DriftSignal]:
    if not main_ref:
        return []
    signals: list[DriftSignal] = []
    production = _first_ref(git_read, path, PRODUCTION_REF_CANDIDATES)
    alumni = _alumni_ref(git_read, path)
    for label, ref in (("production", production), ("alumni", alumni)):
        if not ref:
            continue
        behind = _behind_count(git_read, path, ref, main_ref)
        if behind is None:
            continue
        if behind > 0:
            signals.append(DriftSignal("mindmoor", label, behind))
    return signals


def _open_pr_rows(github_api: GitHubApi, slug: str, failures: list[str]) -> list[dict[str, Any]]:
    try:
        pulls = github_api(f"repos/{slug}/pulls?state=open&per_page={PR_LIST_LIMIT}")
    except (OSError, RuntimeError) as exc:
        failures.append(str(exc) if str(exc).startswith("github_") else "github_api_unavailable")
        return []
    if not isinstance(pulls, list):
        failures.append("github_api_unexpected_shape")
        return []
    return [row for row in pulls if isinstance(row, dict)]


def _failing_ci_count(github_api: GitHubApi, slug: str, failures: list[str]) -> int | None:
    try:
        payload = github_api(f"repos/{slug}/actions/runs?per_page={CI_RUN_LIMIT}&page=1")
    except (OSError, RuntimeError) as exc:
        failures.append(str(exc) if str(exc).startswith("github_") else "github_api_unavailable")
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        failures.append("github_api_unexpected_shape")
        return None
    seen: set[str] = set()
    failing = 0
    for raw in payload["workflow_runs"]:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("head_branch") or "workflow")
        if name in seen:
            continue
        seen.add(name)
        if raw.get("status") == "completed" and raw.get("conclusion") == "failure":
            failing += 1
    return failing


def _conflict_count(
    github_api: GitHubApi, slug: str, pulls: list[dict[str, Any]], failures: list[str]
) -> int | None:
    if not pulls:
        return 0
    conflicts = 0
    checked = 0
    for row in pulls[:PR_DETAIL_CAP]:
        number = row.get("number")
        mergeable = row.get("mergeable")
        state = str(row.get("mergeable_state") or "")
        if mergeable is None and not state and isinstance(number, int):
            try:
                detail = github_api(f"repos/{slug}/pulls/{number}")
            except (OSError, RuntimeError) as exc:
                failures.append(
                    str(exc) if str(exc).startswith("github_") else "github_api_unavailable"
                )
                return None
            if not isinstance(detail, dict):
                failures.append("github_api_unexpected_shape")
                return None
            mergeable = detail.get("mergeable")
            state = str(detail.get("mergeable_state") or "")
        checked += 1
        if mergeable is False or state in {"dirty", "blocked"}:
            conflicts += 1
    if checked == 0:
        return 0
    return conflicts


def collect_repo_evidence(
    repo_id: str,
    *,
    git_read: GitRead | None = None,
    github_api: GitHubApi | None = None,
    mounts: dict[str, Path] | None = None,
) -> RepoEvidence:
    git_read = git_read or _git_read
    github_api = github_api or repos._github_api
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    path = mounts.get(repo_id)
    if not path or not (path / ".git").exists():
        return RepoEvidence(repo_id=repo_id, mounted=False, complete=True)

    failures: list[str] = []
    main_ref = _first_ref(git_read, path, MAIN_REF_CANDIDATES)
    head = git_read(path, "rev-parse", "--verify", "HEAD")
    if head is None:
        failures.append("git_head_failed")

    behind_main = None
    if main_ref and head:
        behind_main = _behind_count(git_read, path, "HEAD", main_ref)
        if behind_main is None:
            failures.append("git_behind_main_failed")

    recent_raw = git_read(path, "rev-list", "--count", "--since=24.hours", "HEAD")
    recent_commits = int(recent_raw) if recent_raw is not None and recent_raw.isdigit() else None
    if recent_raw is None:
        failures.append("git_recent_commits_failed")

    slug = repos._origin_slug(path)
    open_prs = failing_ci = merge_conflicts = None
    if slug:
        pulls = _open_pr_rows(github_api, slug, failures)
        if (
            "github_api_unavailable" not in failures
            and "github_api_unexpected_shape" not in failures
        ):
            open_prs = len(pulls)
            merge_conflicts = _conflict_count(github_api, slug, pulls, failures)
        failing_ci = _failing_ci_count(github_api, slug, failures)
    else:
        failures.append("github_origin_missing")

    drift = _mindmoor_drift(git_read, path, main_ref) if repo_id == "mindmoor" else []

    moved: list[str] = []
    stuck: list[str] = []
    label = repos.REPO_NAMES.get(repo_id, repo_id)
    if open_prs:
        moved.append(f"{label} {open_prs} PR{'s' if open_prs != 1 else ''}")
    elif recent_commits:
        moved.append(f"{label} {recent_commits} commit{'s' if recent_commits != 1 else ''}")
    if failing_ci:
        stuck.append(f"{label} CI")
    if merge_conflicts:
        stuck.append(f"{label} conflict")
    if behind_main and behind_main > 0 and (repo_id != "mindmoor" or not drift):
        stuck.append(f"{label} −{behind_main} behind main")

    unique_failures = tuple(dict.fromkeys(failures))
    return RepoEvidence(
        repo_id=repo_id,
        mounted=True,
        complete=not unique_failures,
        open_prs=open_prs,
        failing_ci=failing_ci,
        merge_conflicts=merge_conflicts,
        behind_main=behind_main,
        recent_commits=recent_commits,
        drift=drift,
        failures=unique_failures,
        moved=moved,
        stuck=stuck,
    )


def format_digest_line(evidence: list[RepoEvidence], *, limit: int = DIGEST_LINE_MAX) -> str:
    mounted = [row for row in evidence if row.mounted]
    if not mounted:
        return "INCOMPLETE: no mounted repos"

    moved = [item for row in mounted for item in row.moved]
    stuck = [item for row in mounted for item in row.stuck]
    drift = [
        f"{signal.repo_id} {signal.ref} −{signal.behind_main} behind main"
        for row in mounted
        for signal in row.drift
    ]
    incomplete = [row.repo_id for row in mounted if not row.complete]
    material = bool(stuck or drift)

    if not material and not incomplete:
        return "CLEAR: no material drift/CI"

    parts: list[str] = []
    if moved:
        parts.append("Moved: " + ", ".join(moved))
    if stuck:
        parts.append("Stuck: " + ", ".join(stuck))
    elif material or incomplete:
        parts.append("Stuck: —")
    if drift:
        parts.append("Drift: " + ", ".join(drift))
    if incomplete:
        parts.append("Incomplete: " + ", ".join(incomplete))
    line = " | ".join(parts) or "CLEAR: no material drift/CI"
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "…"


def classify_digest(evidence: list[RepoEvidence]) -> str:
    mounted = [row for row in evidence if row.mounted]
    if not mounted:
        return "incomplete"
    if any(not row.complete for row in mounted):
        return "incomplete"
    if any(row.stuck or row.drift for row in mounted):
        return "stuck"
    if any((row.behind_main or 0) > 0 for row in mounted):
        return "stuck"
    return "clear"


def _fingerprint(evidence: list[RepoEvidence]) -> str:
    payload = []
    for row in evidence:
        payload.append(
            {
                "id": row.repo_id,
                "mounted": row.mounted,
                "complete": row.complete,
                "open_prs": row.open_prs,
                "failing_ci": row.failing_ci,
                "merge_conflicts": row.merge_conflicts,
                "behind_main": row.behind_main,
                "recent_commits": row.recent_commits,
                "drift": [(s.ref, s.behind_main) for s in row.drift],
                "failures": row.failures,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect_digest(
    *,
    git_read: GitRead | None = None,
    github_api: GitHubApi | None = None,
    mounts: dict[str, Path] | None = None,
) -> DigestResult:
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    evidence = [
        collect_repo_evidence(repo_id, git_read=git_read, github_api=github_api, mounts=mounts)
        for repo_id in watched_repo_ids(mounts)
    ]
    line = format_digest_line(evidence)
    severity = classify_digest(evidence)
    failures = tuple(item for row in evidence for item in row.failures)
    return DigestResult(
        line=line,
        severity=severity,
        fingerprint=_fingerprint(evidence),
        evidence=evidence,
        failures=failures,
    )


def _record_check(
    agents: Any,
    agent: dict[str, Any],
    outcome: str,
    fingerprint: str,
    failures: tuple[str, ...] = (),
) -> None:
    try:
        agents.recover_runs().record_check(
            agent, outcome=outcome, fingerprint=fingerprint, failures=failures
        )
    except (RunHistoryUnavailable, OSError, AttributeError) as exc:
        logger.warning("Daily digest check not recorded for %s: %s", agent.get("id"), exc)


def deliver_digest(
    result: DigestResult,
    *,
    agent: dict[str, Any] | None,
    agents: Any = None,
    assignments: Any = None,
    now: datetime | None = None,
    slack_ready: bool | None = None,
) -> dict[str, Any]:
    """Post today's one-liner through the #101 channel path.

    CLEAR days finish a run and record a quiet check. They open no Work item, so
    they cannot trip awaiting_review and the next day's collection still runs.
    Stuck days open one interval review item. Incomplete days surface as a
    loop_check failure, which does not count toward the review cap.
    """
    now = now or datetime.now(timezone.utc)
    day = now.date().isoformat()
    delivery = (
        resolve_delivery(agent, slack_ready=slack_ready)
        if agent
        else {
            "declared": DEFAULT_OUTPUT_CHANNEL,
            "delivered_channel": "",
            "note": AGENT_MISSING_NOTE,
        }
    )
    report: dict[str, Any] = {
        "line": result.line,
        "severity": result.severity,
        "fingerprint": result.fingerprint,
        "quiet": result.severity == "clear",
        "declared": delivery.get("declared", DEFAULT_OUTPUT_CHANNEL),
        "delivered_channel": delivery.get("delivered_channel", ""),
        "delivery_note": delivery.get("note", ""),
        "assignment_id": "",
        "reason": "",
    }
    if not agent or agents is None:
        report["delivery_note"] = AGENT_MISSING_NOTE
        report["delivered_channel"] = ""
        return report

    if result.severity == "clear":
        _record_check(agents, agent, "quiet_clear", result.fingerprint)
        agents.record_loop_skip(agent["id"], result.fingerprint, reason="quiet_clear")
        note = delivery["note"] or QUIET_CLEAR_NOTE
        agents.finish_run(
            agent["id"],
            result.line,
            result=result.line,
            trigger="interval",
            delivered_channel=delivery["delivered_channel"],
            delivery_note=note,
        )
        report["delivery_note"] = note
        return report

    if result.severity == "incomplete" and assignments is not None:
        item, _created = assignments.record_check_failure(
            agent_id=agent["id"],
            agent_name=agent.get("name") or DIGEST_AGENT_NAME,
            objective=result.line,
            failures=result.failures or ("digest_incomplete",),
        )
        _record_check(agents, agent, "check_incomplete", result.fingerprint, result.failures)
        agents.record_loop_skip(agent["id"], None, reason="check_incomplete")
        agents.finish_run(
            agent["id"],
            result.line,
            result=result.line,
            trigger="interval",
            assignment_id=item["id"] if item else "",
            delivered_channel=delivery["delivered_channel"],
            delivery_note=delivery["note"],
        )
        report["quiet"] = False
        report["reason"] = "check_incomplete"
        report["assignment_id"] = item["id"] if item else ""
        return report

    if assignments is None:
        agents.finish_run(
            agent["id"],
            result.line,
            result=result.line,
            trigger="interval",
            delivered_channel=delivery["delivered_channel"],
            delivery_note=delivery["note"] or INBOX_UNAVAILABLE_NOTE,
        )
        report["delivery_note"] = delivery["note"] or INBOX_UNAVAILABLE_NOTE
        return report

    pending = assignments.pending_loop_reviews(agent["id"])
    if result.fingerprint == agent.get("loop_input_hash") and pending:
        _record_check(agents, agent, "verified_unchanged", result.fingerprint)
        agents.record_loop_skip(
            agent["id"], result.fingerprint, reason="unchanged_repository_input"
        )
        report["quiet"] = True
        report["reason"] = "unchanged_repository_input"
        return report

    if pending >= MAX_PENDING_LOOP_REVIEWS:
        agents.record_loop_skip(agent["id"], None, reason="awaiting_review")
        report["quiet"] = True
        report["reason"] = "awaiting_review"
        report["delivered_channel"] = ""
        return report

    schedule_key = f"daily-digest:{day}"
    item, created = assignments.create_scheduled_assignment(
        agent_id=agent["id"],
        agent_name=agent.get("name") or DIGEST_AGENT_NAME,
        objective=result.line,
        schedule_key=schedule_key,
        interval_minutes=1_440,
        observed_fingerprint=f"{day}:{result.fingerprint}",
    )
    if created and item.get("status") == "queued":
        item["title"] = inbox_title(agent, result.line)
        assignments.set_running(item["id"])
        item = assignments.finish_assignment(item["id"], result=result.line) or item
    agents.record_loop_input(agent["id"], result.fingerprint)
    agents.finish_run(
        agent["id"],
        result.line,
        result=result.line,
        trigger="interval",
        assignment_id=item["id"],
        delivered_channel=delivery["delivered_channel"],
        delivery_note=delivery["note"],
    )
    report["quiet"] = False
    report["reason"] = "stuck"
    report["assignment_id"] = item["id"]
    return report


def collect_and_deliver(
    *,
    agents: Any = None,
    assignments: Any = None,
    git_read: GitRead | None = None,
    github_api: GitHubApi | None = None,
    mounts: dict[str, Path] | None = None,
    slack_ready: bool | None = None,
    now: datetime | None = None,
) -> tuple[DigestResult, dict[str, Any]]:
    result = collect_digest(git_read=git_read, github_api=github_api, mounts=mounts)
    agent = find_digest_agent(agents) if agents is not None else None
    delivery = deliver_digest(
        result,
        agent=agent,
        agents=agents,
        assignments=assignments,
        now=now,
        slack_ready=slack_ready,
    )
    return result, delivery
