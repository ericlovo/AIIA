"""The daily digest: one line per agent and per loop, plus a Repos section.

Spec item 2 of the Studio build-out: "surface one line per day: what moved,
what's stuck." The digest is the only thing the loops should put in front of a
person every day; everything else they file is a proposal that waits for a
decision. Delivery reuses the memory-post outbox, so it reaches the allowlisted
Slack channel when that is configured and is otherwise one inbox row.

Repo, CI, and Mindmoor drift lines are extra evidence in the same body — not a
second scheduled task. Reads reuse repository_tools (no git fetch, no new egress).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from local_brain.command_center import repository_tools as repos

MAX_BODY = 2_800
VERDICT_CHARS = 80
NOTE_CHARS = 90
DIGEST_LINE_MAX = 200
PREFERRED_REPOS = ("aiia", "mindmoor", "sanction", "proxy-ai", "morrow", "mia")
PR_LIST_LIMIT = 20
PR_DETAIL_CAP = 8
CI_RUN_LIMIT = 10
MAIN_REF_CANDIDATES = ("origin/main", "main", "origin/master", "master")
PRODUCTION_REF_CANDIDATES = ("origin/production", "production")
ALUMNI_EXACT_CANDIDATES = ("origin/alumni", "alumni")
ALUMNI_PREFIXES = ("origin/release/alumni", "release/alumni")

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


def loops_registry_path() -> Path:
    override = os.getenv("AIIA_LOOPS_REGISTRY", "")
    return Path(override) if override else Path.home() / ".aiia" / "loops-registry.json"


def load_loops(path: Path | None = None) -> dict:
    target = path or loops_registry_path()
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def digest_key(date: str) -> str:
    return f"digest:{date}"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _first_line(text) -> str:
    for line in str(text or "").splitlines():
        line = line.strip().lstrip("#*- ").strip()
        if line:
            return line[:VERDICT_CHARS]
    return ""


def _agent_line(agent: dict, assignments: list[dict], runs: int, date: str) -> str:
    agent_id = agent.get("id")
    waiting = failed = 0
    for item in assignments:
        if item.get("agent_id") != agent_id:
            continue
        stamp = str(item.get("completed_at") or item.get("updated_at") or "")
        if item.get("status") == "failed" and stamp[:10] == date:
            failed += 1
        elif (
            item.get("status") == "completed"
            and item.get("review_status", "unreviewed") == "unreviewed"
            and not item.get("dismissed_at")
            and str(item.get("result") or "").strip()
        ):
            waiting += 1
    parts = [f"{runs} run{'s' if runs != 1 else ''}"]
    if waiting:
        parts.append(f"{waiting} waiting review")
    if failed:
        parts.append(f"{failed} failed")
    if not agent.get("loop_enabled"):
        parts.append("no schedule")
    line = f"- {agent.get('name') or agent_id}: " + ", ".join(parts)
    if str(agent.get("last_run_at") or "")[:10] == date:
        verdict = _first_line(agent.get("last_error") or agent.get("last_result"))
        if verdict:
            line += f" — {verdict}"
    return line


def watched_repo_ids(mounts: dict[str, Path] | None = None) -> list[str]:
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    known = [repo_id for repo_id in PREFERRED_REPOS if repo_id in mounts]
    extras = [repo_id for repo_id in mounts if repo_id not in known]
    return known + extras


def _git_read(path: Path, *args: str) -> str | None:
    return repos._git_checked(path, *args)


def github_slug_from_remote(remote: str) -> str:
    """Owner/repo from a GitHub remote, including HTTPS URLs with userinfo.

    ``repository_tools._origin_slug`` only accepts bare github.com hosts. Cloud
    checkouts and some Mini remotes rewrite origin to
    ``https://x-access-token:…@github.com/owner/repo.git``. Never return the
    remote itself — only the slug — so tokens cannot leak into the digest line.
    """
    slug = str(remote or "").strip()
    if slug in {"", "unavailable"}:
        return ""
    if slug.startswith("git@github.com:"):
        slug = slug.split(":", 1)[1]
    else:
        parsed = urlsplit(slug)
        host = (parsed.hostname or "").lower()
        if host != "github.com" or not parsed.path:
            return ""
        slug = parsed.path.lstrip("/")
    slug = slug.removesuffix(".git").strip("/")
    return slug if repos._GITHUB_SLUG.fullmatch(slug) else ""


def _github_slug(path: Path) -> str:
    return repos._origin_slug(path) or github_slug_from_remote(
        repos._git(path, "remote", "get-url", "origin")
    )


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

    slug = _github_slug(path)
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
    return DigestResult(
        line=line,
        severity=classify_digest(evidence),
        fingerprint=_fingerprint(evidence),
        evidence=evidence,
        failures=tuple(item for row in evidence for item in row.failures),
    )


def repo_section_lines(evidence: list[RepoEvidence]) -> list[str]:
    """Bullets for the digest's Repos section. Tokens never enter these lines."""
    summary = format_digest_line(evidence)
    parts = [part.strip() for part in summary.split(" | ") if part.strip()]
    return ["", "Repos", *[f"- {part}" for part in parts]]


def build_digest(
    *,
    date: str,
    agents: list[dict],
    assignments: list[dict],
    run_counts: dict[str, int],
    loops: dict,
    tasks: list[dict],
    inbox_counts: dict[str, int],
    repo_evidence: list[RepoEvidence] | None = None,
) -> str:
    lines = [f"AIIA digest {date}", "", "Agents"]
    for agent in sorted(agents, key=lambda a: str(a.get("name") or "")):
        lines.append(_agent_line(agent, assignments, int(run_counts.get(agent.get("id"), 0)), date))
    if not agents:
        lines.append("- no agents")
    lines += ["", "Loops"]
    for name, entry in sorted(loops.items()):
        if not isinstance(entry, dict):
            continue
        last = str(entry.get("last_run") or "")[:16].replace("T", " ")
        status = entry.get("last_status") or "never run"
        note = str(entry.get("last_note") or "").strip()[:NOTE_CHARS]
        lines.append(f"- {name}: {status} {last}".rstrip() + (f" — {note}" if note else ""))
    if not loops:
        lines.append("- no loop registry found")
    failing = [t for t in tasks if t.get("last_status") == "failed"]
    if failing:
        lines += ["", "Built-in tasks failing"]
        for task in failing:
            reason = _first_line(str(task.get("last_result") or "").replace("FAILED: ", ""))
            lines.append(
                f"- {task.get('name') or task.get('task_id')}" + (f": {reason}" if reason else "")
            )
    if repo_evidence is not None:
        lines += repo_section_lines(repo_evidence)
    waiting = sum(inbox_counts.values())
    if waiting:
        detail = ", ".join(f"{count} {source}" for source, count in sorted(inbox_counts.items()))
        lines += ["", f"Inbox waiting review: {waiting} ({detail})"]
    else:
        lines += ["", "Inbox waiting review: 0"]
    body = "\n".join(lines)
    if len(body) > MAX_BODY:
        body = body[: MAX_BODY - 1].rstrip() + "…"
    return body
