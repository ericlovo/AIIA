"""The daily digest: product lines, then customers, then decisions, then a footer.

Spec item 2 of the Studio build-out: "surface one line per day: what moved,
what's stuck." The digest is the only thing the loops should put in front of a
person every day; everything else they file is a proposal that waits for a
decision. Delivery reuses the memory-post outbox, so it reaches the allowlisted
Slack channel when that is configured and is otherwise one inbox row.

Product lines reuse the repo, CI, and Mindmoor drift collectors — not a
second scheduled task. Reads reuse repository_tools (no git fetch, no new egress).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from local_brain.command_center import repository_tools as repos
from local_brain.command_center.deployment_verification import verified_receipt

MAX_BODY = 2_800
SLACK_BODY_MAX = 3_000
VERDICT_CHARS = 80
NOTE_CHARS = 90
DIGEST_LINE_MAX = 200
PRODUCT_LINE_MAX = 220
FOOTER_LINE_MAX = 220
DECISION_LIMIT = 5
DECISION_CHARS = 90
TITLE_CHARS = 36
SHIPPED_WINDOW = timedelta(hours=24)
PREFERRED_REPOS = ("aiia", "mindmoor", "sanction", "proxy-ai", "morrow", "mia")
PR_LIST_LIMIT = 20
PR_DETAIL_CAP = 8
CI_RUN_LIMIT = 10
MAIN_REF_CANDIDATES = ("origin/main", "main", "origin/master", "master")
PRODUCTION_REF_CANDIDATES = ("origin/production", "production")
ALUMNI_EXACT_CANDIDATES = ("origin/alumni", "alumni")
ALUMNI_PREFIXES = ("origin/release/alumni", "release/alumni")
PRODUCTS_ENV = "AIIA_DIGEST_PRODUCTS"
DEPENDABOT_AUTHORS = frozenset({"dependabot[bot]", "dependabot"})
HOLD_LABELS = frozenset({"hold"})
SUPERSEDED_LABELS = frozenset({"superseded"})
CODEX_HEAD_PREFIX = "codex/"
DEPENDABOT_HEAD_PREFIX = "dependabot/"
INBOX_STALE = timedelta(hours=72)
INBOX_STALE_HINTS = ("next best action", "confirm shipped")
STALE_STORY_PREFIX = "stale story:"
_PR_MENTION = re.compile(
    r"(?:(?P<slug>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*)?#(?P<num>\d+)"
    r"|(?P<label>AIIA|MIA|Mindmoor|Sanction|Morrow)\s+#(?P<named>\d+)",
    re.IGNORECASE,
)

# Fallback when the checked-in map is missing. GitHub slugs for Sanction / MIA
# live only in config/digest_products.json so this module stays sanitizer-clean.
FALLBACK_PRODUCTS: tuple[dict[str, Any], ...] = (
    {
        "id": "aiia",
        "name": "AIIA",
        "github": "ericlovo/AIIA",
        "mount": "aiia",
        "holds": [47, 52],
    },
    {
        "id": "mindmoor",
        "name": "Mindmoor",
        "github": "tonybangert/mindmoor",
        "mount": "mindmoor",
        "drift": ["production", "alumni"],
    },
    {"id": "sanction", "name": "Sanction", "github": "", "mount": "sanction", "asks": False},
    {"id": "mia", "name": "MIA", "github": "", "mount": "mia", "holds": [42]},
    {"id": "morrow", "name": "Morrow", "github": "", "mount": "morrow"},
)

FALLBACK_CUSTOMERS: tuple[dict[str, Any], ...] = (
    {
        "id": "trs",
        "code": "TRS",
        "name": "That's Right Sweetie",
        "label": "That's Right Sweetie (TRS)",
        "products": ["mindmoor"],
        "tenant": "trs",
        "branch": "trs",
        "excluded_from_releases": True,
    },
    {
        "id": "alumni-nations",
        "code": "AN",
        "name": "Alumni Nations",
        "products": ["mindmoor"],
        "agents": ["Alumni Nations Research Scout"],
        "drift": ["alumni"],
        "phase": {"name": "Phase 1", "start": "2026-10-15", "end": "2027-01-12"},
    },
    {
        "id": "smart-medical",
        "code": "SM",
        "name": "Smart Medical",
        "kind": "consulting",
        "no_repo": True,
        "status_note": "",
    },
)

GitRead = Callable[..., str | None]
GitHubApi = Callable[[str], Any]


@dataclass(frozen=True)
class DigestProduct:
    id: str
    name: str
    github: str = ""
    mount: str = ""
    drift: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    holds: tuple[int, ...] = ()
    asks: bool = True

    def tokens(self) -> set[str]:
        values = [self.id, self.name, self.mount, self.github, *self.aliases]
        if self.github and "/" in self.github:
            values.append(self.github.rsplit("/", 1)[-1])
        return {value.strip().lower() for value in values if str(value).strip()}


@dataclass(frozen=True)
class DigestPhase:
    name: str = "Phase 1"
    start: str = ""
    end: str = ""


@dataclass(frozen=True)
class DigestCustomer:
    id: str
    name: str
    code: str = ""
    products: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    suites: tuple[str, ...] = ()
    namespaces: tuple[str, ...] = ()
    drift: tuple[str, ...] = ()
    branch: str = ""
    tenant: str = ""
    vercel_project: str = ""
    deploy: str = ""
    excluded_from_releases: bool = False
    phase: DigestPhase | None = None
    label_override: str = ""
    consulting: bool = False
    no_repo: bool = False
    status_note: str = ""

    def tokens(self) -> set[str]:
        values = [self.id, self.name, self.code, *self.agents]
        return {value.strip().lower() for value in values if str(value).strip()}

    def mapped(self) -> bool:
        return bool(
            self.consulting
            or self.no_repo
            or self.products
            or self.agents
            or self.suites
            or self.namespaces
            or self.branch
            or self.tenant
            or self.drift
        )

    def label(self) -> str:
        return self.label_override.strip() or self.name


@dataclass
class CustomerEvidence:
    customer_id: str
    behind_main: int | None = None
    ref_found: bool = False


@dataclass
class DriftSignal:
    repo_id: str
    ref: str
    behind_main: int


@dataclass
class PullSignal:
    number: int
    title: str
    draft: bool = False
    mergeable: bool | None = None
    mergeable_state: str = ""
    checks: str = ""
    sha: str = ""
    merged: bool = False
    author: str = ""
    head_ref: str = ""
    labels: tuple[str, ...] = ()


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
    merged_prs: list[PullSignal] = field(default_factory=list)
    open_pulls: list[PullSignal] = field(default_factory=list)
    remote: bool = False
    verified_deployments: dict[int, str] = field(default_factory=dict)


@dataclass
class DigestResult:
    line: str
    severity: str
    fingerprint: str
    evidence: list[RepoEvidence]
    failures: tuple[str, ...] = ()
    customer_evidence: list[CustomerEvidence] = field(default_factory=list)


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


def products_config_path() -> Path:
    override = os.getenv(PRODUCTS_ENV, "")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "digest_products.json"
        if candidate.is_file():
            return candidate
    return here.parents[2] / "config" / "digest_products.json"


def _product_from_row(row: dict[str, Any]) -> DigestProduct | None:
    product_id = str(row.get("id") or "").strip().lower()
    name = str(row.get("name") or "").strip()
    if not product_id or not name:
        return None
    drift = tuple(str(item).strip() for item in (row.get("drift") or ()) if str(item).strip())
    aliases = tuple(
        str(item).strip().lower() for item in (row.get("aliases") or ()) if str(item).strip()
    )
    asks = True
    if "asks" in row:
        asks = bool(row.get("asks"))
    elif str(row.get("owner") or "").strip().lower() == "codex":
        asks = False
    return DigestProduct(
        id=product_id,
        name=name,
        github=str(row.get("github") or "").strip(),
        mount=str(row.get("mount") or product_id).strip(),
        drift=drift,
        aliases=aliases,
        holds=_ints(row.get("holds")),
        asks=asks,
    )


def load_products(path: Path | None = None) -> list[DigestProduct]:
    target = path or products_config_path()
    rows: list[Any] = []
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("products"), list):
        rows = data["products"]
    elif isinstance(data, list):
        rows = data
    if not rows:
        rows = list(FALLBACK_PRODUCTS)
    products = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        product = _product_from_row(raw)
        if not product or product.id in seen:
            continue
        seen.add(product.id)
        products.append(product)
    return products or [
        _product_from_row(row) for row in FALLBACK_PRODUCTS if _product_from_row(row)
    ]


def _phase_from_row(raw: Any) -> DigestPhase | None:
    if not isinstance(raw, dict):
        return None
    start = str(raw.get("start") or "").strip()
    if not start:
        return None
    return DigestPhase(
        name=str(raw.get("name") or "Phase 1").strip() or "Phase 1",
        start=start,
        end=str(raw.get("end") or "").strip(),
    )


def _strings(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _ints(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list):
        return ()
    numbers: list[int] = []
    for item in raw:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(numbers)


def _customer_from_row(row: dict[str, Any]) -> DigestCustomer | None:
    customer_id = str(row.get("id") or "").strip().lower()
    name = str(row.get("name") or "").strip()
    if not customer_id or not name:
        return None
    kind = str(row.get("kind") or "").strip().lower()
    consulting = bool(row.get("consulting")) or kind == "consulting"
    return DigestCustomer(
        id=customer_id,
        name=name,
        code=str(row.get("code") or "").strip(),
        products=_strings(row.get("products")),
        agents=_strings(row.get("agents")),
        suites=_strings(row.get("suites")),
        namespaces=_strings(row.get("namespaces")),
        drift=_strings(row.get("drift")),
        branch=str(row.get("branch") or "").strip(),
        tenant=str(row.get("tenant") or "").strip(),
        vercel_project=str(row.get("vercel_project") or row.get("vercel") or "").strip(),
        deploy=str(row.get("deploy") or "").strip(),
        excluded_from_releases=bool(row.get("excluded_from_releases")),
        phase=_phase_from_row(row.get("phase")),
        label_override=str(row.get("label") or "").strip(),
        consulting=consulting,
        no_repo=bool(row.get("no_repo")),
        status_note=str(row.get("status_note") or "").strip(),
    )


def _config_payload(path: Path | None = None) -> dict[str, Any] | None:
    target = path or products_config_path()
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_owners(path: Path | None = None) -> frozenset[str]:
    """Author logins that may receive merge asks. Kept in config, not this module."""
    data = _config_payload(path)
    raw = data.get("owners") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(str(item).strip().lower() for item in raw if str(item).strip())


def load_customers(path: Path | None = None) -> list[DigestCustomer]:
    data = _config_payload(path)
    rows: list[Any]
    if data is None or "customers" not in data:
        rows = list(FALLBACK_CUSTOMERS)
    elif isinstance(data.get("customers"), list):
        rows = data["customers"]
    else:
        rows = []
    customers: list[DigestCustomer] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        customer = _customer_from_row(raw)
        if not customer or customer.id in seen:
            continue
        seen.add(customer.id)
        customers.append(customer)
    return customers


def match_product(value: Any, products: Iterable[DigestProduct]) -> DigestProduct | None:
    needle = str(value or "").strip().lower()
    if not needle:
        return None
    compact = needle.replace("_", "-")
    for product in products:
        tokens = product.tokens()
        if needle in tokens or compact in tokens:
            return product
    return None


def map_agent_to_product(
    agent: dict[str, Any], products: Iterable[DigestProduct] | None = None
) -> DigestProduct | None:
    """Map a Studio agent to a digest product. Tolerates missing handles/kind."""
    catalog = list(products) if products is not None else load_products()
    handles = agent.get("handles")
    if isinstance(handles, list):
        for handle in handles:
            matched = match_product(handle, catalog)
            if matched:
                return matched
    elif isinstance(handles, str):
        matched = match_product(handles, catalog)
        if matched:
            return matched
    kind = agent.get("kind")
    if kind:
        matched = match_product(kind, catalog)
        if matched:
            return matched
    for key in ("suite", "memory_namespace", "repo_id", "repo"):
        matched = match_product(agent.get(key), catalog)
        if matched:
            return matched
    try:
        from local_brain.command_center.agent_suites import infer_suite
    except ImportError:
        return None
    return match_product(infer_suite(agent), catalog)


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def map_agent_to_customers(
    agent: dict[str, Any], customers: Iterable[DigestCustomer]
) -> list[DigestCustomer]:
    """Map a Studio agent to customers. Name, handles, namespace; no guessing."""
    catalog = list(customers)
    hits: list[DigestCustomer] = []
    name = _normalize_name(agent.get("name"))
    handles = agent.get("handles")
    handle_tokens = (
        {_normalize_name(item) for item in handles if str(item).strip()}
        if isinstance(handles, list)
        else set()
    )
    namespace = _normalize_name(agent.get("memory_namespace"))
    for customer in catalog:
        named = {_normalize_name(item) for item in customer.agents}
        if name and (
            name in named
            or any(token and (token == name or token in name or name in token) for token in named)
            or customer.name.lower() in name
        ):
            hits.append(customer)
            continue
        if handle_tokens & customer.tokens():
            hits.append(customer)
            continue
        if namespace and namespace in {_normalize_name(item) for item in customer.namespaces}:
            hits.append(customer)
    return hits


def digest_key(date: str) -> str:
    return f"digest:{date}"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _first_line(text, limit: int = VERDICT_CHARS) -> str:
    for line in str(text or "").splitlines():
        line = line.strip().lstrip("#*- ").strip()
        if line:
            return line[:limit]
    return ""


def _short_title(text: Any, limit: int = TITLE_CHARS) -> str:
    title = " ".join(str(text or "").split())
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip() + "…"


def _agent_waiting(item: dict[str, Any]) -> bool:
    return (
        item.get("status") == "completed"
        and item.get("review_status", "unreviewed") == "unreviewed"
        and not item.get("dismissed_at")
        and str(item.get("result") or "").strip()
    )


def _agent_failed_today(item: dict[str, Any], date: str) -> bool:
    stamp = str(item.get("completed_at") or item.get("updated_at") or "")
    return item.get("status") == "failed" and stamp[:10] == date


def _agent_loop_paused(agent: dict[str, Any]) -> bool:
    if agent.get("retired"):
        return False
    if str(agent.get("loop_skip_reason") or "") == "awaiting_review":
        return False
    if agent.get("loop_enabled"):
        return False
    return bool(str(agent.get("loop_task") or "").strip())


def _agent_line(agent: dict, assignments: list[dict], runs: int, date: str) -> str:
    agent_id = agent.get("id")
    waiting = failed = 0
    for item in assignments:
        if item.get("agent_id") != agent_id:
            continue
        if _agent_failed_today(item, date):
            failed += 1
        elif _agent_waiting(item):
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


def watched_repo_ids(
    mounts: dict[str, Path] | None = None,
    products: Iterable[DigestProduct] | None = None,
) -> list[str]:
    catalog = list(products) if products is not None else load_products()
    ids: list[str] = []
    seen: set[str] = set()
    for product in catalog:
        if product.mount and product.mount not in seen:
            ids.append(product.mount)
            seen.add(product.mount)
    if ids:
        return ids
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


def _configured_drift(
    git_read: GitRead,
    path: Path,
    main_ref: str | None,
    repo_id: str,
    labels: tuple[str, ...],
) -> list[DriftSignal]:
    if not main_ref or not labels:
        return []
    signals: list[DriftSignal] = []
    for label in labels:
        if label == "production":
            ref = _first_ref(git_read, path, PRODUCTION_REF_CANDIDATES)
        elif label == "alumni":
            ref = _alumni_ref(git_read, path)
        else:
            ref = _first_ref(git_read, path, (f"origin/{label}", label))
        if not ref:
            continue
        behind = _behind_count(git_read, path, ref, main_ref)
        if behind is None:
            continue
        if behind > 0:
            signals.append(DriftSignal(repo_id, label, behind))
    return signals


def _mindmoor_drift(git_read: GitRead, path: Path, main_ref: str | None) -> list[DriftSignal]:
    return _configured_drift(git_read, path, main_ref, "mindmoor", ("production", "alumni"))


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
        payload = github_api(
            f"repos/{slug}/actions/runs?branch=main&per_page={CI_RUN_LIMIT}&page=1"
        )
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
        head = str(raw.get("head_branch") or "main")
        if head not in {"main", "master"}:
            continue
        name = str(raw.get("name") or head or "workflow")
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
    candidates = [row for row in pulls if not _row_is_inactive(row)]
    for row in candidates[:PR_DETAIL_CAP]:
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
            row["mergeable"] = mergeable
            row["mergeable_state"] = state
            if "draft" in detail:
                row["draft"] = detail.get("draft")
            if detail.get("title"):
                row["title"] = detail.get("title")
        if _row_is_inactive(row):
            continue
        checked += 1
        if mergeable is False or state in {"dirty"}:
            conflicts += 1
    if checked == 0:
        return 0
    return conflicts


def _merged_pr_rows(github_api: GitHubApi, slug: str, since: datetime) -> list[PullSignal]:
    try:
        pulls = github_api(
            f"repos/{slug}/pulls?state=closed&base=main&sort=updated"
            f"&direction=desc&per_page={PR_LIST_LIMIT}"
        )
    except (OSError, RuntimeError):
        return []
    if not isinstance(pulls, list):
        return []
    merged: list[PullSignal] = []
    for row in pulls:
        if not isinstance(row, dict) or not row.get("merged_at"):
            continue
        try:
            when = datetime.fromisoformat(str(row["merged_at"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when < since:
            continue
        pull = _pull_from_row(row)
        if pull:
            pull.merged = True
            pull.sha = str(row.get("merge_commit_sha") or "")
            merged.append(pull)
    return merged


def _label_names(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip().lower()
        else:
            name = str(item or "").strip().lower()
        if name:
            names.append(name)
    return tuple(names)


def _pull_from_row(row: dict[str, Any]) -> PullSignal | None:
    number = row.get("number")
    if not isinstance(number, int):
        return None
    head = row.get("head") if isinstance(row.get("head"), dict) else {}
    user = row.get("user") if isinstance(row.get("user"), dict) else {}
    mergeable = row.get("mergeable")
    author = str(user.get("login") or row.get("author") or "").strip()
    return PullSignal(
        number=number,
        title=str(row.get("title") or "")[:200],
        draft=bool(row.get("draft")),
        mergeable=mergeable if isinstance(mergeable, bool) else None,
        mergeable_state=str(row.get("mergeable_state") or ""),
        sha=str(head.get("sha") or ""),
        merged=bool(row.get("merged_at") or row.get("merged")),
        author=author,
        head_ref=str(head.get("ref") or row.get("head_ref") or "").strip(),
        labels=_label_names(row.get("labels")),
    )


def _row_is_inactive(row: dict[str, Any]) -> bool:
    """Draft or superseded list rows are background, not conflict tallies."""
    if row.get("draft"):
        return True
    return _title_or_labels_superseded(str(row.get("title") or ""), _label_names(row.get("labels")))


def _title_or_labels_superseded(title: str, labels: Iterable[str]) -> bool:
    if SUPERSEDED_LABELS.intersection({str(item).strip().lower() for item in labels}):
        return True
    lowered = title.lower()
    return lowered.startswith("superseded") or "superseded by" in lowered


def _pull_is_dependabot(pull: PullSignal) -> bool:
    login = pull.author.lower()
    ref = pull.head_ref.lower()
    return (
        login in DEPENDABOT_AUTHORS
        or "dependabot" in login
        or ref.startswith(DEPENDABOT_HEAD_PREFIX)
    )


def _pull_is_codex(pull: PullSignal) -> bool:
    return pull.head_ref.lower().startswith(CODEX_HEAD_PREFIX)


def _pull_held(pull: PullSignal, holds: Iterable[int] = ()) -> bool:
    if pull.number in set(holds):
        return True
    return bool(HOLD_LABELS.intersection(pull.labels))


def _pull_superseded(pull: PullSignal) -> bool:
    return _title_or_labels_superseded(pull.title, pull.labels)


def _pull_is_skipped(pull: PullSignal) -> bool:
    """Same set `_skipped_part` reports: Dependabot and Codex drafts."""
    return _pull_is_dependabot(pull) or (pull.draft and _pull_is_codex(pull))


def _pull_counts_as_blocker(pull: PullSignal, holds: Iterable[int] = ()) -> bool:
    """Only non-draft open PRs can turn a product line red."""
    if pull.draft or pull.merged or _pull_superseded(pull) or _pull_held(pull, holds):
        return False
    return not _pull_is_skipped(pull)


def _eric_owned_ready(pull: PullSignal) -> bool:
    if pull.draft or _pull_is_codex(pull) or _pull_is_dependabot(pull):
        return False
    if pull.author.lower() not in load_owners():
        return False
    return _pull_ready(pull)


def _evidence_has_pulls(evidence: RepoEvidence | None) -> bool:
    return bool(evidence and (evidence.mounted or evidence.remote))


def _pr_number_list(pulls: Iterable[PullSignal]) -> str:
    return ", ".join(f"#{pull.number}" for pull in pulls)


def _skipped_part(pulls: Iterable[PullSignal]) -> str:
    rows = list(pulls)
    dependabot = [pull for pull in rows if _pull_is_dependabot(pull)]
    codex_drafts = [pull for pull in rows if pull.draft and _pull_is_codex(pull)]
    bits: list[str] = []
    if dependabot:
        bits.append(f"{len(dependabot)} dependabot ({_pr_number_list(dependabot)})")
    if codex_drafts:
        bits.append(f"{len(codex_drafts)} codex drafts ({_pr_number_list(codex_drafts)})")
    return "skipped: " + ", ".join(bits) if bits else ""


def _enrich_pull_checks(github_api: GitHubApi, slug: str, pulls: list[PullSignal]) -> None:
    for pull in pulls[:PR_DETAIL_CAP]:
        if pull.checks or pull.mergeable_state in {
            "clean",
            "dirty",
            "unstable",
            "blocked",
            "draft",
        }:
            continue
        if not pull.sha:
            continue
        try:
            status = github_api(f"repos/{slug}/commits/{pull.sha}/status")
        except (OSError, RuntimeError):
            continue
        if isinstance(status, dict):
            pull.checks = str(status.get("state") or "")


def _pull_conflicted(pull: PullSignal) -> bool:
    return pull.mergeable is False or pull.mergeable_state in {"dirty"}


def _pull_checks_failing(pull: PullSignal) -> bool:
    if pull.checks in {"failure", "error"}:
        return True
    return pull.mergeable_state in {"unstable"} or (
        pull.mergeable_state == "blocked" and not pull.draft
    )


def _pull_ready(pull: PullSignal) -> bool:
    if _pull_conflicted(pull) or _pull_checks_failing(pull):
        return False
    if pull.checks in {"pending"}:
        return False
    if pull.mergeable_state in {"clean", "has_hooks"}:
        return True
    if pull.mergeable is True and pull.mergeable_state not in {"dirty", "unstable", "blocked"}:
        return True
    return pull.checks == "success" and pull.mergeable is not False


def collect_repo_evidence(
    repo_id: str,
    *,
    git_read: GitRead | None = None,
    github_api: GitHubApi | None = None,
    mounts: dict[str, Path] | None = None,
    drift_refs: tuple[str, ...] | None = None,
    now: datetime | None = None,
    github_slug: str = "",
) -> RepoEvidence:
    git_read = git_read or _git_read
    github_api = github_api or repos._github_api
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    path = mounts.get(repo_id)
    mounted = bool(path and (path / ".git").exists())
    configured = str(github_slug or "").strip()
    slug = _github_slug(path) if mounted else ""
    if not slug and configured and repos._GITHUB_SLUG.fullmatch(configured):
        slug = configured
    if not mounted and not slug:
        return RepoEvidence(repo_id=repo_id, mounted=False, complete=True, remote=False)

    failures: list[str] = []
    behind_main = None
    recent_commits = None
    main_ref = None
    if mounted:
        main_ref = _first_ref(git_read, path, MAIN_REF_CANDIDATES)
        head = git_read(path, "rev-parse", "--verify", "HEAD")
        if head is None:
            failures.append("git_head_failed")
        if main_ref and head:
            behind_main = _behind_count(git_read, path, "HEAD", main_ref)
            if behind_main is None:
                failures.append("git_behind_main_failed")
        recent_raw = git_read(path, "rev-list", "--count", "--since=24.hours", "HEAD")
        recent_commits = (
            int(recent_raw) if recent_raw is not None and recent_raw.isdigit() else None
        )
        if recent_raw is None:
            failures.append("git_recent_commits_failed")

    open_prs = failing_ci = merge_conflicts = None
    merged_prs: list[PullSignal] = []
    open_pulls: list[PullSignal] = []
    if slug:
        pulls = _open_pr_rows(github_api, slug, failures)
        if (
            "github_api_unavailable" not in failures
            and "github_api_unexpected_shape" not in failures
        ):
            open_prs = len(pulls)
            merge_conflicts = _conflict_count(github_api, slug, pulls, failures)
            open_pulls = [pull for row in pulls if (pull := _pull_from_row(row))]
            _enrich_pull_checks(github_api, slug, open_pulls)
        failing_ci = _failing_ci_count(github_api, slug, failures)
        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        merged_prs = _merged_pr_rows(github_api, slug, clock - SHIPPED_WINDOW)
    elif mounted:
        failures.append("github_origin_missing")

    drift: list[DriftSignal] = []
    if mounted:
        if drift_refs is None and repo_id == "mindmoor":
            drift_refs = ("production", "alumni")
        drift = _configured_drift(git_read, path, main_ref, repo_id, drift_refs or ())

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

    verified_deployments: dict[int, str] = {}
    receipt = verified_receipt(slug, now)
    if mounted and receipt:
        for pull in merged_prs:
            if re.fullmatch(r"[0-9a-f]{40}", pull.sha):
                ancestor = git_read(path, "merge-base", pull.sha, receipt["sha"])
                if ancestor == pull.sha:
                    verified_deployments[pull.number] = receipt["environment"]

    unique_failures = tuple(dict.fromkeys(failures))
    return RepoEvidence(
        repo_id=repo_id,
        mounted=mounted,
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
        merged_prs=merged_prs,
        open_pulls=open_pulls,
        remote=bool(slug),
        verified_deployments=verified_deployments,
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
                "merged": [pull.number for pull in row.merged_prs],
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _customer_ref_candidates(customer: DigestCustomer) -> tuple[str, ...]:
    names: list[str] = []
    if customer.branch:
        names.append(customer.branch)
    if customer.tenant and customer.tenant not in names:
        names.extend((customer.tenant, f"tenant/{customer.tenant}", f"release/{customer.tenant}"))
    candidates: list[str] = []
    for name in names:
        candidates.append(f"origin/{name}")
        candidates.append(name)
    return tuple(dict.fromkeys(candidates))


def collect_customer_evidence(
    customers: Iterable[DigestCustomer],
    *,
    products: Iterable[DigestProduct],
    git_read: GitRead | None = None,
    mounts: dict[str, Path] | None = None,
) -> list[CustomerEvidence]:
    """Main-not-on-customer-branch counts. Missing refs stay unset — never invented."""
    git_read = git_read or _git_read
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    catalog = {product.id: product for product in products}
    rows: list[CustomerEvidence] = []
    for customer in customers:
        if not customer.mapped() or not (customer.branch or customer.tenant):
            rows.append(CustomerEvidence(customer_id=customer.id))
            continue
        behind = None
        ref_found = False
        for product_id in customer.products:
            product = catalog.get(product_id)
            if not product or not product.mount:
                continue
            path = mounts.get(product.mount)
            if not path or not (path / ".git").exists():
                continue
            main_ref = _first_ref(git_read, path, MAIN_REF_CANDIDATES)
            customer_ref = _first_ref(git_read, path, _customer_ref_candidates(customer))
            if not main_ref or not customer_ref:
                continue
            ref_found = True
            count = _behind_count(git_read, path, customer_ref, main_ref)
            if count is None:
                continue
            behind = count if behind is None else max(behind, count)
        rows.append(
            CustomerEvidence(customer_id=customer.id, behind_main=behind, ref_found=ref_found)
        )
    return rows


def collect_digest(
    *,
    git_read: GitRead | None = None,
    github_api: GitHubApi | None = None,
    mounts: dict[str, Path] | None = None,
    products: Iterable[DigestProduct] | None = None,
    customers: Iterable[DigestCustomer] | None = None,
    now: datetime | None = None,
) -> DigestResult:
    mounts = mounts if mounts is not None else repos.REPO_MOUNTS
    catalog = list(products) if products is not None else load_products()
    roster = list(customers) if customers is not None else load_customers()
    drift_by_mount = {product.mount: product.drift for product in catalog if product.mount}
    slug_by_mount = {product.mount: product.github for product in catalog if product.mount}
    evidence = [
        collect_repo_evidence(
            repo_id,
            git_read=git_read,
            github_api=github_api,
            mounts=mounts,
            drift_refs=drift_by_mount.get(repo_id) or None,
            now=now,
            github_slug=slug_by_mount.get(repo_id, ""),
        )
        for repo_id in watched_repo_ids(mounts, catalog)
    ]
    line = format_digest_line(evidence)
    return DigestResult(
        line=line,
        severity=classify_digest(evidence),
        fingerprint=_fingerprint(evidence),
        evidence=evidence,
        failures=tuple(item for row in evidence for item in row.failures),
        customer_evidence=collect_customer_evidence(
            roster, products=catalog, git_read=git_read, mounts=mounts
        ),
    )


def repo_section_lines(evidence: list[RepoEvidence]) -> list[str]:
    """Legacy Repos bullets. Tokens never enter these lines."""
    summary = format_digest_line(evidence)
    parts = [part.strip() for part in summary.split(" | ") if part.strip()]
    return ["", "Repos", *[f"- {part}" for part in parts]]


def _pr_bits(pulls: Iterable[PullSignal], suffix: str = "") -> list[str]:
    bits: list[str] = []
    for pull in pulls:
        title = _short_title(pull.title, 28)
        bit = f"#{pull.number}" + (f" {title}" if title else "")
        if suffix:
            bit = f"#{pull.number} {suffix}"
        bits.append(bit)
    return bits


def _join_segments(parts: list[str], empty: str = "none") -> str:
    cleaned = [part for part in parts if part]
    return ", ".join(cleaned) if cleaned else empty


def _shipped_segment(evidence: RepoEvidence | None) -> str:
    if not evidence or not evidence.merged_prs:
        return "none"
    pulls = evidence.merged_prs
    if len(pulls) == 1:
        return _join_segments(_pr_bits(pulls))
    bits = _pr_bits(pulls[:2])
    extra = len(pulls) - len(bits)
    if extra:
        bits.append(f"+{extra}")
    return f"{len(pulls)} PRs (" + ", ".join(bits) + ")"


def _blocked_segment(
    evidence: RepoEvidence | None,
    *,
    paused: list[str],
    failed_agents: int,
    failing_tasks: list[str],
    holds: Iterable[int] = (),
) -> str:
    parts: list[str] = []
    if _evidence_has_pulls(evidence) and evidence is not None:
        if evidence.failing_ci:
            parts.append("CI on main")
        held = set(holds)
        for pull in evidence.open_pulls:
            if not _pull_counts_as_blocker(pull, held):
                continue
            if _pull_conflicted(pull):
                parts.append(f"#{pull.number} conflict")
            elif _pull_checks_failing(pull):
                parts.append(f"#{pull.number} checks")
        for signal in evidence.drift:
            parts.append(f"{signal.ref} −{signal.behind_main}")
    if paused:
        if len(paused) == 1:
            parts.append(f"{paused[0]} paused")
        else:
            parts.append(f"{len(paused)} paused loops")
    if failed_agents:
        parts.append(f"{failed_agents} failed")
    parts.extend(f"{name} failed" for name in failing_tasks)
    return _join_segments(parts)


def _waiting_segment(
    evidence: RepoEvidence | None,
    *,
    review: int,
    holds: Iterable[int] = (),
    asks: bool = True,
) -> str:
    parts: list[str] = []
    if asks and _evidence_has_pulls(evidence) and evidence is not None:
        ready_merge = [
            pull
            for pull in evidence.open_pulls
            if not _pull_held(pull, holds) and _eric_owned_ready(pull)
        ]
        parts.extend(_pr_bits(ready_merge, "merge"))
    if review:
        parts.append(f"{review} review")
    return _join_segments(parts)


def phase_note(phase: DigestPhase | None, date: str) -> str:
    if not phase or not phase.start:
        return ""
    try:
        today = datetime.strptime(date[:10], "%Y-%m-%d").date()
        start = datetime.strptime(phase.start, "%Y-%m-%d").date()
    except ValueError:
        return ""
    if today < start:
        days = (start - today).days
        return f"{days} day{'s' if days != 1 else ''} to kickoff"
    end = None
    if phase.end:
        try:
            end = datetime.strptime(phase.end, "%Y-%m-%d").date()
        except ValueError:
            end = None
    if end and today > end:
        return f"{phase.name} ended"
    elapsed = (today - start).days
    if elapsed == 0:
        return "kickoff today"
    return f"{elapsed} day{'s' if elapsed != 1 else ''} into {phase.name}"


def format_customer_line(
    customer: DigestCustomer,
    *,
    product_evidence: list[RepoEvidence] | None = None,
    customer_evidence: CustomerEvidence | None = None,
    review: int = 0,
    paused: list[str] | None = None,
    failed_agents: int = 0,
    date: str = "",
    limit: int = PRODUCT_LINE_MAX,
) -> str:
    if customer.consulting or customer.no_repo:
        note = customer.status_note.strip() or "no repo, consulting"
        return f"{customer.label()}: {note}"
    if not customer.mapped():
        return f"{customer.label()}: not mapped yet"
    paused = paused or []
    blocked_parts: list[str] = []
    if customer_evidence and customer_evidence.behind_main and customer_evidence.behind_main > 0:
        blocked_parts.append(
            f"main −{customer_evidence.behind_main} not on {customer.code or customer.id}"
        )
    wanted = set(customer.drift)
    for row in product_evidence or []:
        for signal in row.drift:
            if wanted and signal.ref in wanted:
                blocked_parts.append(f"{signal.ref} −{signal.behind_main}")
    if paused:
        if len(paused) == 1:
            blocked_parts.append(f"{paused[0]} paused")
        else:
            blocked_parts.append(f"{len(paused)} paused loops")
    if failed_agents:
        blocked_parts.append(f"{failed_agents} failed")
    line = (
        f"{customer.label()}: shipped none | blocked {_join_segments(blocked_parts)} | "
        f"waiting on you {_waiting_segment(None, review=review)}"
    )
    note = phase_note(customer.phase, date)
    if note:
        line += f" · {note}"
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "…"


def format_product_line(
    product: DigestProduct,
    evidence: RepoEvidence | None = None,
    *,
    paused: list[str] | None = None,
    review: int = 0,
    failed_agents: int = 0,
    failing_tasks: list[str] | None = None,
    limit: int = PRODUCT_LINE_MAX,
) -> str:
    paused = paused or []
    failing_tasks = failing_tasks or []
    readable = bool(evidence and (evidence.mounted or evidence.remote))
    unreadable = bool(readable and not evidence.complete and evidence.open_prs is None)
    block_kwargs = {
        "paused": paused,
        "failed_agents": failed_agents,
        "failing_tasks": failing_tasks,
        "holds": product.holds,
    }
    wait_kwargs = {"review": review, "holds": product.holds, "asks": product.asks}
    if not readable:
        note = "no repo mounted"
        extras: list[str] = []
        blocked = _blocked_segment(None, **block_kwargs)
        waiting = _waiting_segment(None, **wait_kwargs)
        if blocked != "none":
            extras.append(f"blocked {blocked}")
        if waiting != "none":
            extras.append(f"waiting on you {waiting}")
        line = f"{product.name}: " + " | ".join([note, *extras])
    elif unreadable:
        blocked = _blocked_segment(evidence, **block_kwargs)
        waiting = _waiting_segment(evidence, **wait_kwargs)
        extras = ["repo unreadable"]
        if blocked != "none":
            extras.append(f"blocked {blocked}")
        if waiting != "none":
            extras.append(f"waiting on you {waiting}")
        line = f"{product.name}: " + " | ".join(extras)
    else:
        shipped = _shipped_segment(evidence)
        blocked = _blocked_segment(evidence, **block_kwargs)
        waiting = _waiting_segment(evidence, **wait_kwargs)
        line = f"{product.name}: shipped {shipped} | blocked {blocked} | waiting on you {waiting}"
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "…"


def _decision_preferred(item: dict[str, Any], products: list[DigestProduct]) -> bool:
    if str(item.get("source") or "") == "slack":
        return True
    return match_product(item.get("project"), products) is not None


def _decision_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    rank = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
    priority = rank.get(str(item.get("priority") or "normal"), 2)
    return (
        priority,
        str(item.get("created_at") or ""),
    )


def _item_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("subject") or "").strip()


def _parse_stamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def _item_stamp(item: dict[str, Any]) -> datetime | None:
    for key in ("created_at", "updated_at", "ts", "timestamp"):
        when = _parse_stamp(item.get(key))
        if when:
            return when
    return None


def _item_pr_key(item: dict[str, Any]) -> str:
    number = item.get("pr") or item.get("number") or item.get("pr_number")
    repo = str(item.get("repo") or item.get("project") or "").strip().lower()
    if isinstance(number, int):
        return f"{repo}#{number}" if repo else f"#{number}"
    text = _item_text(item)
    match = _PR_MENTION.search(text)
    if not match:
        return ""
    num = match.group("num") or match.group("named")
    slug = (match.group("slug") or match.group("label") or repo or "").lower()
    return f"{slug}#{num}" if slug else f"#{num}"


def _item_subject_key(text: str) -> str:
    cleaned = " ".join(text.lower().split())
    for hint in INBOX_STALE_HINTS:
        prefix = f"{hint}:"
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip(" -")
            break
        if cleaned.startswith(hint):
            cleaned = cleaned[len(hint) :].strip(" :-")
            break
    return cleaned


def _stale_kind(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in INBOX_STALE_HINTS)


def _stale_story(text: str) -> bool:
    return text.lower().startswith(STALE_STORY_PREFIX)


def _pr_keys_from_evidence(
    evidence: Iterable[RepoEvidence] | None, *, merged: bool = False
) -> set[str]:
    keys: set[str] = set()
    for row in evidence or ():
        pulls = row.merged_prs if merged else row.open_pulls
        for pull in pulls:
            keys.add(f"#{pull.number}")
            keys.add(f"{row.repo_id}#{pull.number}")
            if row.repo_id == "aiia":
                keys.add(f"aiia#{pull.number}")
    return keys


def current_inbox_items(
    items: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    resolved_prs: Iterable[str] | None = None,
    open_prs: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Drop digest rows, stale next-best/confirm-shipped, resolved PRs; dedupe."""
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    resolved = {str(key).lower() for key in (resolved_prs or ())}
    opened = {str(key).lower() for key in (open_prs or ())}
    kept: list[dict[str, Any]] = []
    for item in items:
        text = _item_text(item)
        if not text or str(item.get("source") or "") == "digest":
            continue
        key = _item_pr_key(item).lower()
        state = str(item.get("pr_state") or item.get("state") or "").lower()
        if item.get("resolved") or state in {"merged", "closed"}:
            continue
        if key and key in resolved:
            continue
        if _stale_kind(text):
            if key and opened and key not in opened:
                continue
            stamp = _item_stamp(item)
            if stamp and clock - stamp > INBOX_STALE:
                continue
        kept.append(item)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    for item in sorted(kept, key=lambda row: _item_stamp(row) or oldest, reverse=True):
        token = _item_pr_key(item).lower() or _item_subject_key(_item_text(item))
        if not token or token in seen:
            continue
        seen.add(token)
        unique.append(item)
    order = {id(item): index for index, item in enumerate(items)}
    unique.sort(key=lambda item: order.get(id(item), 0))
    return unique


def choose_decisions(
    items: list[dict[str, Any]],
    *,
    products: Iterable[DigestProduct] | None = None,
    inbox_counts: dict[str, int] | None = None,
    limit: int = DECISION_LIMIT,
    now: datetime | None = None,
    resolved_prs: Iterable[str] | None = None,
    open_prs: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Preferred current inbox rows; leftover is unused current items only.

    ``Stale story:`` backlog rows are counted separately so they can be
    summarized instead of listed as decisions.
    """
    catalog = list(products) if products is not None else load_products()
    usable = current_inbox_items(items, now=now, resolved_prs=resolved_prs, open_prs=open_prs)
    stale_count = sum(1 for item in usable if _stale_story(_item_text(item)))
    usable = [item for item in usable if not _stale_story(_item_text(item))]
    preferred = [item for item in usable if _decision_preferred(item, catalog)]
    preferred.sort(key=_decision_sort_key)
    chosen = preferred[:limit]
    leftover: dict[str, int] = {}
    chosen_ids = {id(item) for item in chosen}
    for item in usable:
        if id(item) in chosen_ids:
            continue
        source = str(item.get("source") or "inbox")
        leftover[source] = leftover.get(source, 0) + 1
    _ = inbox_counts
    return chosen, leftover, stale_count


def format_decision_lines(
    items: list[dict[str, Any]],
    *,
    products: Iterable[DigestProduct] | None = None,
    inbox_counts: dict[str, int] | None = None,
    limit: int = DECISION_LIMIT,
    now: datetime | None = None,
    resolved_prs: Iterable[str] | None = None,
    open_prs: Iterable[str] | None = None,
) -> list[str]:
    catalog = list(products) if products is not None else load_products()
    chosen, leftover, stale_count = choose_decisions(
        items,
        products=catalog,
        inbox_counts=inbox_counts,
        limit=limit,
        now=now,
        resolved_prs=resolved_prs,
        open_prs=open_prs,
    )
    lines: list[str] = []
    for item in chosen:
        product = match_product(item.get("project"), catalog)
        tag = product.id if product else str(item.get("source") or "inbox")
        lines.append(f"- {tag} · {_first_line(item.get('text'), DECISION_CHARS)}")
    leftover_total = sum(leftover.values())
    if leftover_total:
        detail = ", ".join(
            f"{count} {source}" for source, count in sorted(leftover.items()) if count
        )
        lines.append(f"- plus {leftover_total} more ({detail})")
    if stale_count:
        noun = "stale story" if stale_count == 1 else "stale stories"
        lines.append(f"- {stale_count} {noun} (backlog)")
    if not lines:
        lines.append("- none")
    return lines


def format_footer(
    *,
    agents: list[dict],
    assignments: list[dict],
    loops: dict,
    tasks: list[dict],
    date: str,
) -> list[str]:
    live = [agent for agent in agents if not agent.get("retired")]
    active = sum(1 for agent in live if agent.get("status") == "running")
    waiting_ids = {
        item.get("agent_id")
        for item in assignments
        if _agent_waiting(item) and item.get("agent_id")
    }
    waiting_ids.update(
        agent.get("id")
        for agent in live
        if str(agent.get("loop_skip_reason") or "") == "awaiting_review"
    )
    waiting = len({agent_id for agent_id in waiting_ids if agent_id})
    loop_entries = [entry for entry in loops.values() if isinstance(entry, dict)]
    loop_ok = sum(1 for entry in loop_entries if (entry.get("last_status") or "") == "ok")
    loop_other = len(loop_entries) - loop_ok
    if not loop_entries:
        loop_bit = "none"
    elif loop_other:
        loop_bit = f"{loop_ok} ok, {loop_other} other"
    else:
        loop_bit = f"{loop_ok} ok"
    failing = [task for task in tasks if task.get("last_status") == "failed"]
    if failing:
        fail_bits = []
        for task in failing[:3]:
            name = str(task.get("name") or task.get("task_id") or "task")
            reason = _first_line(str(task.get("last_result") or "").replace("FAILED: ", ""), 40)
            fail_bits.append(f"{name} ({reason})" if reason else name)
        extra = len(failing) - len(fail_bits)
        fail_bit = ", ".join(fail_bits) + (f" +{extra}" if extra else "")
    else:
        fail_bit = "none"
    combined = (
        f"Agents: {len(live)} ({active} active, {waiting} waiting review) · "
        f"Loops: {loop_bit} · Built-ins failing: {fail_bit}"
    )
    if len(combined) <= FOOTER_LINE_MAX:
        return [combined]
    return [
        f"Agents: {len(live)} ({active} active, {waiting} waiting review)",
        f"Loops: {loop_bit}",
        f"Built-ins failing: {fail_bit}",
    ][:3]


def _signals_for_product(
    product: DigestProduct,
    *,
    agents: list[dict],
    assignments: list[dict],
    date: str,
    products: list[DigestProduct],
) -> tuple[list[str], int, int]:
    mapped = [agent for agent in agents if map_agent_to_product(agent, products) is product]
    ids = {agent.get("id") for agent in mapped}
    paused = [
        str(agent.get("name") or agent.get("id")) for agent in mapped if _agent_loop_paused(agent)
    ]
    review = sum(1 for item in assignments if item.get("agent_id") in ids and _agent_waiting(item))
    review += sum(
        1 for agent in mapped if str(agent.get("loop_skip_reason") or "") == "awaiting_review"
    )
    failed = sum(
        1 for item in assignments if item.get("agent_id") in ids and _agent_failed_today(item, date)
    )
    return paused, review, failed


def _signals_for_customer(
    customer: DigestCustomer,
    *,
    agents: list[dict],
    assignments: list[dict],
    date: str,
    customers: list[DigestCustomer],
) -> tuple[list[str], int, int]:
    mapped = [agent for agent in agents if customer in map_agent_to_customers(agent, customers)]
    ids = {agent.get("id") for agent in mapped}
    paused = [
        str(agent.get("name") or agent.get("id")) for agent in mapped if _agent_loop_paused(agent)
    ]
    review = sum(1 for item in assignments if item.get("agent_id") in ids and _agent_waiting(item))
    review += sum(
        1 for agent in mapped if str(agent.get("loop_skip_reason") or "") == "awaiting_review"
    )
    failed = sum(
        1 for item in assignments if item.get("agent_id") in ids and _agent_failed_today(item, date)
    )
    return paused, review, failed


SLACK_EMOJI = {
    "blocked": "🔴",
    "waiting": "🟡",
    "shipped": "🟢",
    "clean": "🟢",
    "countdown": "🟢",
    "unmapped": "⚪",
    "unmounted": "⚪",
    "consulting": "⚪",
}


@dataclass
class MorningLine:
    """One product or customer row for Slack and the morning-note home."""

    id: str
    name: str
    kind: str
    state: str
    parts: tuple[str, ...] = ()
    waiting: str = ""
    extra: str = ""
    target: str = ""
    note: str = ""


def slack_header_date(date: str) -> str:
    try:
        when = datetime.strptime(date[:10], "%Y-%m-%d")
    except ValueError:
        return date
    return f"{when.strftime('%a')} {when.strftime('%b')} {when.day}"


def _behind_phrase(label: str, count: int, *, suffix: str = "main") -> str:
    return f"{label} {count} behind {suffix}"


def _hard_block_parts(
    evidence: RepoEvidence | None,
    *,
    paused: list[str],
    failed_agents: int,
    failing_tasks: list[str] | None = None,
    holds: Iterable[int] = (),
) -> list[str]:
    parts: list[str] = []
    if _evidence_has_pulls(evidence) and evidence is not None:
        if evidence.failing_ci:
            parts.append("CI red on main")
        held = set(holds)
        for pull in evidence.open_pulls:
            if not _pull_counts_as_blocker(pull, held):
                continue
            if _pull_conflicted(pull):
                parts.append(f"#{pull.number} conflict")
            elif _pull_checks_failing(pull):
                parts.append(f"#{pull.number} checks")
    if paused:
        if len(paused) == 1:
            parts.append(f"{paused[0]} paused")
        else:
            parts.append(f"{len(paused)} paused")
    if failed_agents:
        parts.append(f"{failed_agents} failed")
    parts.extend(f"{name} failed" for name in (failing_tasks or []))
    return parts


def _waiting_parts(
    evidence: RepoEvidence | None,
    *,
    review: int,
    drift: Iterable[DriftSignal] | None = None,
    customer_behind: tuple[str, int] | None = None,
    holds: Iterable[int] = (),
    asks: bool = True,
) -> tuple[list[str], str]:
    """Lead fragments (ready PRs / drift) and the waiting-on-you clause."""
    lead: list[str] = []
    waiting: list[str] = []
    if asks and _evidence_has_pulls(evidence) and evidence is not None:
        ready_merge = [
            pull
            for pull in evidence.open_pulls
            if not _pull_held(pull, holds) and _eric_owned_ready(pull)
        ]
        if len(ready_merge) == 1:
            lead.append(f"#{ready_merge[0].number} ready")
            waiting.append("merge")
        elif ready_merge:
            lead.append(f"{len(ready_merge)} PRs ready")
            waiting.append("merge")
    if review == 1:
        waiting.append("1 review")
    elif review:
        waiting.append(f"{review} reviews")
    for signal in drift or ():
        lead.append(_behind_phrase(signal.ref, signal.behind_main))
    if customer_behind:
        label, count = customer_behind
        lead.append(_behind_phrase("main", count, suffix=label))
    return lead, ", ".join(waiting)


def _shipped_part(evidence: RepoEvidence | None) -> str:
    if not evidence or not evidence.merged_prs:
        return ""
    verified = [p for p in evidence.merged_prs if p.number in evidence.verified_deployments]
    unverified = [p for p in evidence.merged_prs if p.number not in evidence.verified_deployments]
    parts = []
    if verified:
        parts.append("verified deployed " + _pr_number_list(verified))
    if unverified:
        parts.append("merged " + _pr_number_list(unverified) + "; deployment not verified")
    return "; ".join(parts)


def _morning_state(
    *,
    mapped: bool = True,
    mounted: bool = True,
    hard: list[str],
    waiting: str,
    shipped: str,
    pending: bool = False,
) -> str:
    if not mapped:
        return "unmapped"
    if not mounted:
        return "unmounted"
    if hard:
        return "blocked"
    if waiting or pending:
        return "waiting"
    if shipped:
        return "shipped"
    return "clean"


def _compose_morning_note(line: MorningLine) -> str:
    if line.state == "unmapped":
        return "not mapped yet"
    if line.state == "consulting":
        return line.note or "no repo, consulting"
    if line.state == "unmounted":
        return "no repo"
    bits = [part for part in line.parts if part]
    if line.waiting:
        bits.append(f"waiting on you: {line.waiting}")
    if line.extra:
        bits.append(line.extra)
    if not bits or (line.state in {"shipped", "clean"} and "nothing blocked" not in bits):
        bits.append("nothing blocked")
    return " · ".join(bits)


def morning_product_line(
    product: DigestProduct,
    evidence: RepoEvidence | None = None,
    *,
    paused: list[str] | None = None,
    review: int = 0,
    failed_agents: int = 0,
    failing_tasks: list[str] | None = None,
) -> MorningLine:
    paused = paused or []
    hard = _hard_block_parts(
        evidence,
        paused=paused,
        failed_agents=failed_agents,
        failing_tasks=failing_tasks,
        holds=product.holds,
    )
    drift = evidence.drift if evidence and evidence.mounted else ()
    lead, waiting = _waiting_parts(
        evidence, review=review, drift=drift, holds=product.holds, asks=product.asks
    )
    shipped = _shipped_part(evidence)
    mounted = bool(evidence and (evidence.mounted or evidence.remote))
    state = _morning_state(
        mounted=mounted,
        hard=hard,
        waiting=waiting,
        shipped=shipped if evidence and evidence.verified_deployments else "",
        pending=bool(lead),
    )
    parts: list[str] = []
    if shipped:
        parts.append(shipped)
    parts.extend(hard)
    if state != "blocked":
        parts.extend(lead)
    elif lead:
        # Ready PRs stay in the waiting clause; drift stays visible when blocked.
        parts.extend(bit for bit in lead if "behind" in bit)
    extra = _skipped_part(evidence.open_pulls if evidence else [])
    line = MorningLine(
        id=product.id,
        name=product.name,
        kind="product",
        state=state,
        parts=tuple(parts),
        waiting=waiting,
        extra=extra,
    )
    line.note = _compose_morning_note(line)
    return line


def morning_customer_line(
    customer: DigestCustomer,
    *,
    product_evidence: list[RepoEvidence] | None = None,
    customer_evidence: CustomerEvidence | None = None,
    review: int = 0,
    paused: list[str] | None = None,
    failed_agents: int = 0,
    date: str = "",
) -> MorningLine:
    if customer.consulting or customer.no_repo:
        note = customer.status_note.strip() or "no repo, consulting"
        line = MorningLine(
            id=customer.id,
            name=customer.name,
            kind="customer",
            state="consulting",
            note=note,
        )
        return line
    if not customer.mapped():
        line = MorningLine(id=customer.id, name=customer.name, kind="customer", state="unmapped")
        line.note = _compose_morning_note(line)
        return line
    paused = paused or []
    hard = _hard_block_parts(None, paused=paused, failed_agents=failed_agents)
    drift: list[DriftSignal] = []
    wanted = set(customer.drift)
    for row in product_evidence or []:
        for signal in row.drift:
            if wanted and signal.ref in wanted:
                drift.append(signal)
    behind = None
    if customer_evidence and customer_evidence.behind_main and customer_evidence.behind_main > 0:
        behind = (customer.code or customer.id, customer_evidence.behind_main)
    lead, waiting = _waiting_parts(None, review=review, drift=drift, customer_behind=behind)
    extra = phase_note(customer.phase, date)
    target = ""
    if customer.phase and customer.phase.start:
        target = f"{customer.phase.start}T09:00:00"
    state = _morning_state(
        hard=hard,
        waiting=waiting,
        shipped="",
        pending=bool(lead),
    )
    if target and state in {"clean", "shipped"}:
        state = "countdown"
    line = MorningLine(
        id=customer.id,
        name=customer.name,
        kind="customer",
        state=state,
        parts=tuple([*hard, *lead]),
        waiting=waiting,
        extra=extra,
        target=target,
    )
    line.note = _compose_morning_note(line)
    return line


def _slack_line(line: MorningLine) -> str:
    emoji = SLACK_EMOJI[line.state]
    name = escape(line.name)
    if line.state == "unmapped":
        return f"{emoji} *{name}*: not mapped yet"
    if line.state == "consulting":
        return f"{emoji} *{name}*: {escape(line.note or 'no repo, consulting')}"
    if line.state == "unmounted":
        return f"{emoji} *{name}*: no repo"
    bits = [escape(part) for part in line.parts if part]
    if line.waiting:
        bits.append(f"*waiting on you:* {escape(line.waiting)}")
    if line.extra:
        bits.append(escape(line.extra))
    if not bits or (line.state in {"shipped", "clean"} and "nothing blocked" not in bits):
        bits.append("nothing blocked")
    return f"{emoji} *{name}*: " + " · ".join(bits)


def slack_failure_line(
    *,
    tasks: list[dict],
    assignments: list[dict],
    date: str,
) -> str:
    failing = [task for task in tasks if task.get("last_status") == "failed"]
    if failing:
        name = str(failing[0].get("name") or failing[0].get("task_id") or "task")
        extra = f" +{len(failing) - 1}" if len(failing) > 1 else ""
        reason = _first_line(
            str(failing[0].get("last_error") or failing[0].get("last_result") or "").replace(
                "FAILED: ", "", 1
            ),
            80,
        )
        if reason:
            return f"{escape(name)} failed: {escape(reason)}{extra}"
        return f"{escape(name)} failed{extra}"
    failed = sum(1 for item in assignments if _agent_failed_today(item, date))
    if failed == 1:
        return "An agent failed today"
    if failed:
        return f"{failed} agents failed today"
    return ""


def collect_morning_lines(
    *,
    date: str,
    agents: list[dict],
    assignments: list[dict],
    repo_evidence: list[RepoEvidence] | None = None,
    products: Iterable[DigestProduct] | None = None,
    customers: Iterable[DigestCustomer] | None = None,
    customer_evidence: list[CustomerEvidence] | None = None,
) -> tuple[list[MorningLine], list[MorningLine]]:
    catalog = list(products) if products is not None else load_products()
    roster = list(customers) if customers is not None else load_customers()
    evidence_by_mount = {row.repo_id: row for row in (repo_evidence or [])}
    customer_rows = {row.customer_id: row for row in (customer_evidence or [])}
    product_lines = []
    for product in catalog:
        paused, review, failed = _signals_for_product(
            product,
            agents=agents,
            assignments=assignments,
            date=date,
            products=catalog,
        )
        product_lines.append(
            morning_product_line(
                product,
                evidence_by_mount.get(product.mount),
                paused=paused,
                review=review,
                failed_agents=failed,
            )
        )
    customer_lines = []
    for customer in roster:
        paused, review, failed = _signals_for_customer(
            customer,
            agents=agents,
            assignments=assignments,
            date=date,
            customers=roster,
        )
        mounts_for_customer = {
            product.mount
            for product in catalog
            if product.id in customer.products and product.mount
        }
        touched = [
            evidence_by_mount[mount] for mount in mounts_for_customer if mount in evidence_by_mount
        ]
        customer_lines.append(
            morning_customer_line(
                customer,
                product_evidence=touched,
                customer_evidence=customer_rows.get(customer.id),
                review=review,
                paused=paused,
                failed_agents=failed,
                date=date,
            )
        )
    return product_lines, customer_lines


def morning_line_payload(line: MorningLine) -> dict[str, Any]:
    return {
        "id": line.id,
        "name": line.name,
        "kind": line.kind,
        "state": line.state,
        "note": line.note,
        "waiting": line.waiting,
        "extra": line.extra,
        "target": line.target,
    }


def morning_decision_cards(
    items: list[dict[str, Any]],
    *,
    assignments: list[dict] | None = None,
    agents: list[dict] | None = None,
    products: Iterable[DigestProduct] | None = None,
    inbox_counts: dict[str, int] | None = None,
    limit: int = DECISION_LIMIT,
    now: datetime | None = None,
    resolved_prs: Iterable[str] | None = None,
    open_prs: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    catalog = list(products) if products is not None else load_products()
    chosen, leftover, _stale_count = choose_decisions(
        items,
        products=catalog,
        inbox_counts=inbox_counts,
        limit=limit,
        now=now,
        resolved_prs=resolved_prs,
        open_prs=open_prs,
    )
    cards: list[dict[str, Any]] = []
    for item in chosen:
        product = match_product(item.get("project"), catalog)
        title = _first_line(item.get("text"), DECISION_CHARS)
        cards.append(
            {
                "id": str(item.get("id") or ""),
                "kind": "inbox",
                "product": product.name if product else "",
                "title": title,
                "why": _first_line(item.get("review_note") or item.get("text"), NOTE_CHARS),
                "more": str(item.get("text") or ""),
                "expected_version": "",
            }
        )
    reviews = [
        item
        for item in (assignments or [])
        if _agent_waiting(item) and str(item.get("title") or item.get("result") or "").strip()
    ]
    roster = {agent.get("id"): agent for agent in (agents or []) if isinstance(agent, dict)}
    for item in reviews:
        if len(cards) >= limit:
            break
        agent = roster.get(item.get("agent_id")) or {}
        product = map_agent_to_product(agent, catalog) if agent else None
        cards.append(
            {
                "id": str(item.get("id") or ""),
                "kind": "review",
                "product": product.name if product else "",
                "title": str(item.get("title") or "Review this work"),
                "why": _first_line(item.get("result") or item.get("objective"), NOTE_CHARS),
                "more": str(item.get("result") or item.get("objective") or ""),
                "expected_version": str(item.get("review_version") or ""),
            }
        )
    total = len(chosen) + sum(leftover.values()) + len(reviews)
    return cards[:limit], total


def format_slack_digest(
    *,
    date: str,
    agents: list[dict],
    assignments: list[dict],
    loops: dict,
    tasks: list[dict],
    inbox_counts: dict[str, int],
    repo_evidence: list[RepoEvidence] | None = None,
    inbox_items: list[dict] | None = None,
    products: Iterable[DigestProduct] | None = None,
    customers: Iterable[DigestCustomer] | None = None,
    customer_evidence: list[CustomerEvidence] | None = None,
    run_counts: dict[str, int] | None = None,
    now: datetime | None = None,
) -> str:
    """Action-first Slack summary; detailed history remains in Studio."""
    catalog = list(products) if products is not None else load_products()
    roster = list(customers) if customers is not None else load_customers()
    evidence = {row.repo_id: row for row in (repo_evidence or [])}
    attention: list[str] = []
    merged: list[str] = []
    deployed: list[str] = []
    background: list[str] = []
    quiet: list[str] = []
    covered_ids: set[str] = set()

    def label(value: str) -> str:
        return escape(_short_title(value, 80))

    def row(name: str, parts: list[str]) -> str:
        return f"- *{label(name)}*: " + "; ".join(label(part) for part in parts)

    def reviews(count: int) -> str:
        return f"review {count} agent output" + ("s" if count != 1 else "")

    def review_count(ids: set[str]) -> int:
        pending = [
            item for item in assignments if item.get("agent_id") in ids and _agent_waiting(item)
        ]
        represented = {item.get("agent_id") for item in pending}
        # An awaiting-review marker is a fallback, not an extra output to review.
        fallback = {
            agent.get("id")
            for agent in agents
            if agent.get("id") in ids
            and agent.get("id") not in represented
            and agent.get("loop_skip_reason") == "awaiting_review"
        }
        return len(pending) + len(fallback)

    for product in catalog:
        ev = evidence.get(product.mount)
        paused, review, failed = _signals_for_product(
            product, agents=agents, assignments=assignments, date=date, products=catalog
        )
        covered_ids.update(
            agent.get("id") for agent in agents if map_agent_to_product(agent, catalog) is product
        )
        review = review_count(
            {agent.get("id") for agent in agents if map_agent_to_product(agent, catalog) is product}
        )
        parts = []
        if not ev or not _evidence_has_pulls(ev) or not ev.complete:
            parts.append("repository status unavailable or incomplete; check Studio")
        if ev:
            if ev.failing_ci:
                parts.append("investigate failing CI on main")
            pulls = [pull for pull in ev.open_pulls if not _pull_held(pull, product.holds)]
            active = [pull for pull in pulls if _pull_counts_as_blocker(pull, product.holds)]
            conflicts = [pull for pull in active if _pull_conflicted(pull)]
            checks = [pull for pull in active if _pull_checks_failing(pull)]
            if conflicts:
                parts.append("resolve conflicts: " + _pr_number_list(conflicts))
            if checks:
                parts.append("investigate failed checks: " + _pr_number_list(checks))
            ready = [pull for pull in active if _eric_owned_ready(pull)]
            if ready and product.asks and ev.complete:
                parts.append("review for merge: " + _pr_number_list(ready))
            for signal in ev.drift:
                parts.append(_behind_phrase(signal.ref, signal.behind_main))
            if ev.merged_prs:
                verified = [p for p in ev.merged_prs if p.number in ev.verified_deployments]
                unverified = [p for p in ev.merged_prs if p.number not in ev.verified_deployments]
                if verified:
                    environments = sorted({ev.verified_deployments[p.number] for p in verified})
                    deployed.append(
                        row(product.name, [_pr_number_list(verified), ", ".join(environments)])
                    )
                if unverified:
                    merged.append(row(product.name, [_pr_number_list(unverified)]))
            drafts = sum(pull.draft for pull in pulls)
            dependencies = sum(_pull_is_dependabot(pull) and not pull.draft for pull in pulls)
            held = sum(_pull_held(pull, product.holds) for pull in ev.open_pulls)
            notes = []
            if drafts:
                notes.append(f"{drafts} draft PR" + ("s" if drafts != 1 else ""))
            if dependencies:
                notes.append(f"{dependencies} dependency PR" + ("s" if dependencies != 1 else ""))
            if held:
                notes.append(f"{held} PRs on hold")
            if notes:
                background.append(row(product.name, notes))
        if review:
            parts.append(reviews(review))
        if failed:
            parts.append(f"inspect {failed} failed agent runs")
        if paused:
            parts.append(f"check {len(paused)} paused agents")
        if parts:
            attention.append(row(product.name, parts))
        else:
            quiet.append(label(product.name))

    _, customer_lines = collect_morning_lines(
        date=date,
        agents=agents,
        assignments=assignments,
        repo_evidence=repo_evidence,
        products=catalog,
        customers=roster,
        customer_evidence=customer_evidence,
    )
    for line in customer_lines:
        if line.state in {"consulting", "unmapped", "unmounted"}:
            background.append(row(line.name, [line.note]))
            continue
        parts = list(line.parts)
        customer = next(item for item in roster if item.id == line.id)
        covered_ids.update(
            agent.get("id") for agent in agents if customer in map_agent_to_customers(agent, roster)
        )
        count = review_count(
            {
                agent.get("id")
                for agent in agents
                if customer in map_agent_to_customers(agent, roster)
            }
        )
        if count:
            parts.append(reviews(count))
        if line.extra:
            parts.append(line.extra)
        if parts:
            attention.append(row(line.name, parts))
        else:
            quiet.append(label(line.name))

    # Failure detail can contain credentials or private context. Keep it in Studio.
    failing = [task for task in tasks if task.get("last_status") == "failed"]
    for task in failing[:3]:
        attention.insert(
            0,
            row(
                str(task.get("name") or "Scheduled task"),
                ["failed; inspect the run in Studio before retrying"],
            ),
        )
    if len(failing) > 3:
        attention.insert(3, f"- {len(failing) - 3} more scheduled task failures in Studio")

    other_failures = sum(
        _agent_failed_today(item, date)
        for item in assignments
        if item.get("agent_id") not in covered_ids
    )
    if other_failures:
        attention.append(f"- Inspect {other_failures} other failed agent runs in Studio")

    chosen, leftover, stale_count = choose_decisions(
        inbox_items or [],
        products=catalog,
        limit=3,
        now=now,
        resolved_prs=_pr_keys_from_evidence(repo_evidence, merged=True),
        open_prs=_pr_keys_from_evidence(repo_evidence),
    )
    decisions = ["- " + label(_first_line(item.get("text"), DECISION_CHARS)) for item in chosen]
    backlog = stale_count + sum(leftover.values())
    if backlog:
        background.append(f"- {backlog} backlog items in Studio; not counted as today's decisions")

    lines = [f"*AIIA · {slack_header_date(date)}*"]
    for heading, rows in (
        ("Needs attention", attention),
        ("Decisions", decisions),
        ("Verified deployed", deployed),
        ("Merged — deployment not verified", merged),
        ("Background — not merge requests", background),
    ):
        if rows:
            lines.extend(["", f"*{heading}*", *rows])
    if quiet:
        lines.extend(
            [
                "",
                "No active alerts: "
                + ", ".join(quiet)
                + ". Drafts and maintenance are listed separately.",
            ]
        )
    footer = "_Details in Studio_"
    # Keep whole rows and a usable footer; never cut a failure or Slack markup mid-line.
    output: list[str] = []
    budget = SLACK_BODY_MAX - len(footer) - len("\n_More items in Studio_\n")
    for line in lines:
        if len("\n".join([*output, line])) > budget:
            if output and output[-1].startswith("*"):
                output.pop()
            output.append("_More items in Studio_")
            break
        output.append(line)
    return "\n".join([*output, "", footer])


def build_morning_note(
    *,
    date: str,
    agents: list[dict],
    assignments: list[dict],
    tasks: list[dict],
    inbox_counts: dict[str, int],
    repo_evidence: list[RepoEvidence] | None = None,
    inbox_items: list[dict] | None = None,
    products: Iterable[DigestProduct] | None = None,
    customers: Iterable[DigestCustomer] | None = None,
    customer_evidence: list[CustomerEvidence] | None = None,
) -> dict[str, Any]:
    catalog = list(products) if products is not None else load_products()
    product_lines, customer_lines = collect_morning_lines(
        date=date,
        agents=agents,
        assignments=assignments,
        repo_evidence=repo_evidence,
        products=catalog,
        customers=customers,
        customer_evidence=customer_evidence,
    )
    cards, total = morning_decision_cards(
        inbox_items or [],
        assignments=assignments,
        agents=agents,
        products=catalog,
        inbox_counts=inbox_counts,
        resolved_prs=_pr_keys_from_evidence(repo_evidence, merged=True),
        open_prs=_pr_keys_from_evidence(repo_evidence, merged=False),
    )
    return {
        "date": date,
        "products": [morning_line_payload(line) for line in product_lines],
        "customers": [morning_line_payload(line) for line in customer_lines],
        "decisions": cards,
        "needs_you_total": total,
        "failure": slack_failure_line(tasks=tasks, assignments=assignments, date=date) or None,
    }


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
    inbox_items: list[dict] | None = None,
    products: Iterable[DigestProduct] | None = None,
    customers: Iterable[DigestCustomer] | None = None,
    customer_evidence: list[CustomerEvidence] | None = None,
) -> str:
    catalog = list(products) if products is not None else load_products()
    roster = list(customers) if customers is not None else load_customers()
    evidence_by_mount = {row.repo_id: row for row in (repo_evidence or [])}
    customer_rows = {row.customer_id: row for row in (customer_evidence or [])}
    lines = [f"AIIA digest {date}", ""]
    for product in catalog:
        paused, review, failed = _signals_for_product(
            product,
            agents=agents,
            assignments=assignments,
            date=date,
            products=catalog,
        )
        lines.append(
            format_product_line(
                product,
                evidence_by_mount.get(product.mount),
                paused=paused,
                review=review,
                failed_agents=failed,
            )
        )
    if roster:
        lines += ["", "Customers"]
        for customer in roster:
            paused, review, failed = _signals_for_customer(
                customer,
                agents=agents,
                assignments=assignments,
                date=date,
                customers=roster,
            )
            mounts_for_customer = {
                product.mount
                for product in catalog
                if product.id in customer.products and product.mount
            }
            touched = [
                evidence_by_mount[mount]
                for mount in mounts_for_customer
                if mount in evidence_by_mount
            ]
            lines.append(
                format_customer_line(
                    customer,
                    product_evidence=touched,
                    customer_evidence=customer_rows.get(customer.id),
                    review=review,
                    paused=paused,
                    failed_agents=failed,
                    date=date,
                )
            )
    lines += [
        "",
        "Needs a decision",
        *format_decision_lines(
            inbox_items or [],
            products=catalog,
            inbox_counts=inbox_counts,
            resolved_prs=_pr_keys_from_evidence(repo_evidence, merged=True),
            open_prs=_pr_keys_from_evidence(repo_evidence, merged=False),
        ),
    ]
    footer = format_footer(
        agents=agents, assignments=assignments, loops=loops, tasks=tasks, date=date
    )
    lines += ["", *footer]
    # run_counts kept on the signature so callers and older tests still pass it.
    _ = run_counts
    body = "\n".join(lines)
    if len(body) > MAX_BODY:
        body = body[: MAX_BODY - 1].rstrip() + "…"
    return body
