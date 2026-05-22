"""
Obsidian Bridge — Export AIIA memory and roadmap to Obsidian vault.

Reads from AIIA Brain API (:8100), Command Center (:8200), and local
report files, then writes markdown into ~/AIIAVault/:

  10-Daily/YYYY-MM-DD.md           — Shipped code report for that day
  10-Daily/YYYY-MM-DD-activity.md  — Rolling interval activity feed
  30-Decisions/AIIA-Decisions.md   — 100 most recent architectural decisions
  40-Sessions/AIIA-Sessions.md     — Recent session summaries
  50-Stories/AIIA-Backlog.md       — AIIA roadmap stories by status
  80-Resources/AIIA-Patterns.md    — All code patterns and conventions
  80-Resources/AIIA-Lessons.md     — All hard-won lessons

Uses hash-based incremental sync: only rewrites files whose content has
changed since the last run, so Obsidian Sync doesn't get noisy on unchanged
nights.

Usage:
    python -m local_brain.scripts.obsidian_bridge
    python -m local_brain.scripts.obsidian_bridge --dry-run

Shell alias (add to Mini's .zshrc):
    alias vault='python3 -m local_brain.scripts.obsidian_bridge'

Schedule:
    Nightly at 11pm via com.aiia.obsidiansync launchd agent
"""

import hashlib
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# --- Paths & URLs ---
from local_brain.vault_paths import vault_dir as _vault_dir

BRAIN_URL = os.getenv("AIIA_URL", "http://localhost:8100")
CC_URL = os.getenv("AIIA_CC_URL", "http://localhost:8200")
VAULT_DIR = _vault_dir()
REPORTS_DIR = Path.home() / ".aiia" / "eq_data" / "reports"
LOG_DIR = Path.home() / ".aiia" / "logs" / "vault"
STATE_FILE = LOG_DIR / "state.json"

# Files this bridge owns (overwritten every run if content changed)
BRIDGE_FILES = [
    "30-Decisions/AIIA-Decisions.md",
    "40-Sessions/AIIA-Sessions.md",
    "50-Stories/AIIA-Backlog.md",
    "80-Resources/AIIA-Patterns.md",
    "80-Resources/AIIA-Lessons.md",
    "10-Daily/*.md",  # daily code reports + activity feed
]

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "vault.log"),
    ],
)
logger = logging.getLogger("aiia.obsidian_bridge")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get(url: str, params: dict | None = None, timeout: int = 10) -> dict | None:
    """GET request returning parsed JSON, or None on any error."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        logger.warning(f"Unreachable: {url} — {e}")
    except Exception as e:
        logger.warning(f"GET {url} failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def fetch_memories(category: str, limit: int = 200) -> list[dict]:
    result = _get(f"{BRAIN_URL}/v1/aiia/memory", {"category": category, "limit": limit})
    return result.get("memories", []) if result else []


def fetch_workstreams() -> list[dict]:
    result = _get(f"{CC_URL}/api/workstreams")
    return result if isinstance(result, list) else []


def fetch_roadmap() -> list[dict]:
    result = _get(f"{CC_URL}/api/roadmap")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return result.get("stories", [])
    return []


# ---------------------------------------------------------------------------
# Incremental sync state
# ---------------------------------------------------------------------------


class SyncState:
    """Persist SHA256 hashes of written files. Skip unchanged on next run."""

    def __init__(self, path: Path):
        self._path = path
        self._state: dict[str, str] = {}
        if path.exists():
            try:
                self._state = json.loads(path.read_text())
            except Exception:
                pass

    def is_changed(self, key: str, content: str) -> bool:
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        if self._state.get(key) == h:
            return False
        self._state[key] = h
        return True

    def save(self):
        self._path.write_text(json.dumps(self._state, indent=2))


# ---------------------------------------------------------------------------
# Markdown generators
# ---------------------------------------------------------------------------

_AUTOGEN_WARNING = (
    "> **Auto-generated** by Obsidian Bridge — do not edit, changes will be overwritten."
)


def _frontmatter(**kwargs) -> str:
    lines = ["---"]
    for k, v in kwargs.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(i) for i in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def build_decisions_md(memories: list[dict]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fm = _frontmatter(
        type="resource",
        date=today,
        source="aiia-bridge",
        tags=["aiia", "decisions", "architecture"],
    )
    lines = [fm, "", "# AIIA Decisions Digest", "", _AUTOGEN_WARNING, ""]

    # Group by YYYY-MM for readability
    by_month: dict[str, list[dict]] = {}
    for m in memories:
        month = (m.get("created_at") or "")[:7] or "unknown"
        by_month.setdefault(month, []).append(m)

    for month in sorted(by_month, reverse=True):
        lines.append(f"## {month}")
        for m in by_month[month]:
            date = (m.get("created_at") or "")[:10]
            fact = (m.get("fact") or "").strip()
            source = m.get("source", "")
            source_tag = ""
            # Show source only when it's meaningfully traceable
            if source and source not in {"session", "bootstrap", "ask-and-learn", ""}:
                short = source[:60]
                source_tag = f" *({short})*"
            lines.append(f"- **{date}** — {fact}{source_tag}")
        lines.append("")

    lines.append(f"*{len(memories)} decisions exported · last sync {today}*")
    return "\n".join(lines)


def build_patterns_md(memories: list[dict]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fm = _frontmatter(
        type="resource",
        date=today,
        source="aiia-bridge",
        tags=["aiia", "patterns", "conventions"],
    )
    lines = [fm, "", "# AIIA Code Patterns", "", _AUTOGEN_WARNING, ""]
    for m in sorted(memories, key=lambda x: x.get("created_at", ""), reverse=True):
        fact = (m.get("fact") or "").strip()
        date = (m.get("created_at") or "")[:10]
        lines.append(f"- **{date}** — {fact}")
    lines.append("")
    lines.append(f"*{len(memories)} patterns · last sync {today}*")
    return "\n".join(lines)


def build_lessons_md(memories: list[dict]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fm = _frontmatter(
        type="resource",
        date=today,
        source="aiia-bridge",
        tags=["aiia", "lessons", "debugging"],
    )
    lines = [fm, "", "# AIIA Lessons Learned", "", _AUTOGEN_WARNING, ""]
    for m in sorted(memories, key=lambda x: x.get("created_at", ""), reverse=True):
        fact = (m.get("fact") or "").strip()
        date = (m.get("created_at") or "")[:10]
        lines.append(f"- **{date}** — {fact}")
    lines.append("")
    lines.append(f"*{len(memories)} lessons · last sync {today}*")
    return "\n".join(lines)


def build_sessions_md(memories: list[dict]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fm = _frontmatter(
        type="resource",
        date=today,
        source="aiia-bridge",
        tags=["aiia", "sessions"],
    )
    lines = [fm, "", "# AIIA Session Log", "", _AUTOGEN_WARNING, ""]
    for m in sorted(memories, key=lambda x: x.get("created_at", ""), reverse=True):
        fact = (m.get("fact") or "").strip()
        date = (m.get("created_at") or "")[:10]
        source = m.get("source", "")
        lines.append(f"## {date}")
        lines.append(fact)
        if source:
            lines.append(f"*{source}*")
        lines.append("")
    lines.append(f"*{len(memories)} sessions · last sync {today}*")
    return "\n".join(lines)


def build_backlog_md(stories: list[dict]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fm = _frontmatter(
        type="resource",
        date=today,
        source="aiia-bridge",
        tags=["aiia", "stories", "backlog"],
    )
    lines = [fm, "", "# AIIA Roadmap Backlog", "", _AUTOGEN_WARNING, ""]

    status_order = ["active", "in_progress", "backlog", "shipped", "cancelled"]
    by_status: dict[str, list[dict]] = {}
    for s in stories:
        st = s.get("status", "unknown")
        by_status.setdefault(st, []).append(s)

    # Include any statuses not in our defined order
    remaining = [st for st in by_status if st not in status_order]
    ordered = status_order + remaining

    for status in ordered:
        items = by_status.get(status)
        if not items:
            continue
        label = status.replace("_", " ").title()
        lines.append(f"## {label} ({len(items)})")
        for s in sorted(items, key=lambda x: (x.get("priority", "P9"), x.get("title", ""))):
            priority = s.get("priority", "")
            title = (s.get("title") or "").strip()
            product = s.get("product", "")
            desc = (s.get("description") or "").strip()
            lines.append(f"### {title}")
            meta = " | ".join(filter(None, [f"**{priority}**" if priority else "", product]))
            if meta:
                lines.append(meta)
            if desc:
                lines.append(f"> {desc}")
            lines.append("")
        lines.append("")

    lines.append(f"*{len(stories)} stories · last sync {today}*")
    return "\n".join(lines)


def build_daily_report_md(report: dict) -> str:
    """Convert a daily report JSON into an Obsidian-friendly markdown note."""
    date = report.get("date", "unknown")
    summary = report.get("summary", {})
    fm = _frontmatter(
        type="daily-note",
        date=date,
        source="daily-report",
        tags=["daily", "shipped-code", "report"],
    )
    lines = [fm, "", f"# Shipped Code Report -- {date}", "", _AUTOGEN_WARNING, ""]

    # Summary stats
    commits = summary.get("total_commits", 0)
    files = summary.get("total_files_changed", 0)
    adds = summary.get("total_additions", 0)
    dels = summary.get("total_deletions", 0)
    authors = ", ".join(summary.get("authors", [])) or "none"
    types = summary.get("commit_types", {})
    type_str = ", ".join(f"{k}: {v}" for k, v in types.items()) or "none"

    lines.append(f"**Commits:** {commits}  |  **Files:** {files}  |  +{adds} / -{dels}")
    lines.append(f"**Authors:** {authors}")
    lines.append(f"**Types:** {type_str}")
    lines.append("")

    # Per-product breakdown
    products = report.get("products", {})
    if products:
        lines.append("## By Product")
        lines.append("")
        for product, data in sorted(products.items()):
            count = data.get("commit_count", len(data.get("commits", [])))
            lines.append(f"### {product} ({count} commits)")
            for c in data.get("commits", []):
                ctype = c.get("type", "")
                subject = c.get("subject", "")
                sha = c.get("hash", "")[:7]
                lines.append(f"- `{ctype}` {sha} {subject}")
            lines.append("")

    # Category breakdown
    categories = report.get("categories", {})
    if categories:
        lines.append("## By Category")
        lines.append("")
        for cat, data in sorted(categories.items()):
            count = data.get("count", len(data.get("commits", [])))
            lines.append(f"- **{cat}**: {count} commits")
        lines.append("")

    # Syntax check summary
    syntax = report.get("syntax_errors", {})
    if isinstance(syntax, dict) and syntax.get("total_errors", 0) > 0:
        lines.append("## Syntax Errors")
        lines.append("")
        lines.append(
            f"**{syntax['total_errors']} errors** in {syntax.get('total_files', '?')} files"
        )
        for product, errs in syntax.get("by_product", {}).items():
            if isinstance(errs, list):
                for err in errs[:10]:
                    lines.append(f"- [{product}] {err}")
        lines.append("")

    lines.append(f"*Generated {report.get('generated_at', date)}*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cluster generators — monthly .md files with aiia_managed: true
# ---------------------------------------------------------------------------

# Category → vault folder
_CLUSTER_FOLDERS = {
    "decisions": "30-Decisions",
    "patterns": "80-Resources",
    "lessons": "80-Resources",
    "sessions": "40-Sessions",
}

# Category → MOC file that links to clusters
_MOC_FILES = {
    "decisions": "30-Decisions/AIIA-Decisions.md",
    "patterns": "80-Resources/AIIA-Patterns.md",
    "lessons": "80-Resources/AIIA-Lessons.md",
    "sessions": "40-Sessions/AIIA-Sessions.md",
}


def _cluster_frontmatter(category: str, cluster_name: str, today: str) -> str:
    return _frontmatter(
        type=f"aiia-{category}",
        date=today,
        source="aiia-bridge",
        aiia_managed="true",
        aiia_version=2,
        cluster=cluster_name,
        tags=["aiia", category],
    )


def build_cluster_files(category: str, memories: list[dict]) -> dict[str, str]:
    """Build monthly cluster files from memories. Returns {rel_path: content}."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    folder = _CLUSTER_FOLDERS.get(category)
    if not folder:
        return {}

    # Group by YYYY-MM
    by_month: dict[str, list[dict]] = {}
    for m in memories:
        month = (m.get("created_at") or "")[:7] or "unknown"
        by_month.setdefault(month, []).append(m)

    files = {}
    for month, items in by_month.items():
        cluster_name = f"{category}-{month}"
        fm = _cluster_frontmatter(category, cluster_name, today)
        title = category.replace("_", " ").title()

        lines = [
            fm,
            "",
            f"# AIIA {title} — {month}",
            "",
            _AUTOGEN_WARNING,
            "",
        ]

        # Group entries within the month by date
        by_date: dict[str, list[dict]] = {}
        for m in items:
            date = (m.get("created_at") or "")[:10] or "unknown"
            by_date.setdefault(date, []).append(m)

        for date in sorted(by_date.keys(), reverse=True):
            lines.append(f"## {date}")
            for m in by_date[date]:
                mid = m.get("id", "")
                fact = (m.get("fact") or "").strip()
                source = m.get("source", "")
                source_tag = ""
                if source and source not in {"session", "bootstrap", "ask-and-learn", ""}:
                    source_tag = f" *({source[:60]})*"
                if mid:
                    lines.append(f"- **{mid}** — {fact}{source_tag}")
                else:
                    lines.append(f"- **{date}** — {fact}{source_tag}")
            lines.append("")

        lines.append(f"*{len(items)} entries · last sync {today}*")
        rel_path = f"{folder}/{cluster_name}.md"
        files[rel_path] = "\n".join(lines)

    return files


def build_moc_for_category(category: str, cluster_rel_paths: list[str], total: int) -> str:
    """Build a MOC (Map of Content) that links to cluster files."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    title = category.replace("_", " ").title()
    fm = _frontmatter(
        type="moc",
        date=today,
        source="aiia-bridge",
        aiia_managed="true",
        tags=["aiia", category, "moc"],
    )

    lines = [
        fm,
        "",
        f"# AIIA {title}",
        "",
        _AUTOGEN_WARNING,
        "",
        f"Total: **{total}** entries across **{len(cluster_rel_paths)}** months.",
        "",
    ]

    # Sort clusters by month (newest first)
    sorted_paths = sorted(cluster_rel_paths, reverse=True)
    for rel_path in sorted_paths:
        # Extract filename without extension for wikilink
        name = Path(rel_path).stem
        # Extract month from cluster name (e.g., decisions-2026-04 → 2026-04)
        month = name.split("-", 1)[1] if "-" in name else name
        lines.append(f"- [[{name}|{month}]]")

    lines.append("")
    lines.append(f"*Last sync {today}*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bridge engine
# ---------------------------------------------------------------------------


class ObsidianBridge:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.state = SyncState(STATE_FILE)
        self.written = 0
        self.skipped = 0
        self.errors: list[str] = []

    def _write(self, rel_path: str, content: str):
        dest = VAULT_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not self.state.is_changed(rel_path, content):
            logger.info(f"Unchanged: {rel_path}")
            self.skipped += 1
            return

        if self.dry_run:
            logger.info(f"[dry-run] Would write {rel_path} ({len(content):,} chars)")
            self.written += 1
            return

        dest.write_text(content, encoding="utf-8")
        logger.info(f"Written: {rel_path} ({len(content):,} chars)")
        self.written += 1

    def _sync_daily_reports(self, days: int = 7):
        """Sync recent daily report JSONs to 10-Daily/ as markdown notes."""
        if not REPORTS_DIR.exists():
            logger.warning(f"Reports dir missing: {REPORTS_DIR}")
            return
        report_files = sorted(REPORTS_DIR.glob("20??-??-??.json"), reverse=True)
        synced = 0
        for rf in report_files[:days]:
            try:
                report = json.loads(rf.read_text(encoding="utf-8"))
                date = report.get("date", rf.stem)
                md = build_daily_report_md(report)
                self._write(f"10-Daily/{date}.md", md)
                synced += 1
            except Exception as e:
                logger.warning(f"Failed to convert {rf.name}: {e}")
        logger.info(f"Daily reports: synced {synced} of {len(report_files)} available")

    def _sync_category_clusters(self, category: str, limit: int):
        """Fetch memories for a category and write cluster files + MOC."""
        memories = fetch_memories(category, limit=limit)
        logger.info(f"Fetched {len(memories)} {category}")
        if not memories:
            return

        # Write cluster files (one per month)
        cluster_files = build_cluster_files(category, memories)
        cluster_paths = []
        for rel_path, content in cluster_files.items():
            self._write(rel_path, content)
            cluster_paths.append(rel_path)

        # Write MOC that links to clusters (replaces old monolithic file)
        moc_path = _MOC_FILES.get(category)
        if moc_path:
            moc_content = build_moc_for_category(category, cluster_paths, len(memories))
            self._write(moc_path, moc_content)

    def run(self) -> dict:
        logger.info(f"Obsidian Bridge starting — vault={VAULT_DIR} dry_run={self.dry_run}")
        start = datetime.utcnow()

        # 1-4. Memory categories → cluster files + MOC stubs
        for category, limit in [
            ("decisions", 200),
            ("patterns", 200),
            ("lessons", 200),
            ("sessions", 50),
        ]:
            try:
                self._sync_category_clusters(category, limit)
            except Exception as e:
                self.errors.append(f"{category}: {e}")
                logger.error(f"{category.title()} export failed: {e}")

        # 5. Roadmap backlog
        try:
            stories = fetch_roadmap()
            logger.info(f"Fetched {len(stories)} roadmap stories")
            self._write("50-Stories/AIIA-Backlog.md", build_backlog_md(stories))
        except Exception as e:
            self.errors.append(f"roadmap: {e}")
            logger.error(f"Roadmap export failed: {e}")

        # 6. Daily code reports — sync recent JSON reports to 10-Daily/
        try:
            self._sync_daily_reports()
        except Exception as e:
            self.errors.append(f"daily_reports: {e}")
            logger.error(f"Daily reports export failed: {e}")

        # 7. Rolling activity feed (today.md)
        try:
            today_md = REPORTS_DIR / "today.md"
            if today_md.exists():
                content = today_md.read_text(encoding="utf-8")
                today = datetime.now().strftime("%Y-%m-%d")
                self._write(f"10-Daily/{today}-activity.md", content)
        except Exception as e:
            self.errors.append(f"activity_feed: {e}")
            logger.error(f"Activity feed export failed: {e}")

        if not self.dry_run:
            self.state.save()

        elapsed = round((datetime.utcnow() - start).total_seconds(), 2)
        return {
            "timestamp": start.isoformat() + "Z",
            "dry_run": self.dry_run,
            "written": self.written,
            "skipped": self.skipped,
            "errors": self.errors,
            "elapsed_seconds": elapsed,
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(result: dict):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report_dir = LOG_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "vault_report.json").write_text(json.dumps(result, indent=2))

    status = "PASS" if not result.get("errors") else "FAIL"
    lines = [
        f"AIIA Obsidian Bridge — {result.get('timestamp', date)}",
        "=" * 50,
        f"Status: {status}",
        f"Written: {result['written']}  Skipped (unchanged): {result['skipped']}",
        f"Elapsed: {result['elapsed_seconds']}s",
        f"Vault: {VAULT_DIR}",
    ]
    if result.get("errors"):
        lines.append(f"Errors ({len(result['errors'])}):")
        for e in result["errors"][:5]:
            lines.append(f"  - {e}")
    summary = "\n".join(lines)
    (LOG_DIR / "latest.txt").write_text(summary)

    # Symlink latest-report dir
    link = LOG_DIR / "latest-report"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(report_dir)

    print(summary)
    logger.info(f"Report: {report_dir / 'vault_report.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    # Ensure we can find the local_brain package when run standalone
    repo_dir = Path(__file__).parent.parent.parent.parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    bridge = ObsidianBridge(dry_run=dry_run)
    result = bridge.run()
    write_report(result)
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
