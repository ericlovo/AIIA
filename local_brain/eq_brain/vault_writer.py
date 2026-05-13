"""
VaultWriter — Real-time AIIA memory → Obsidian vault sync.

Writes memory entries to monthly cluster .md files in the Obsidian vault
as they happen (not just nightly). Uses an asyncio queue with a background
worker so remember() never blocks.

Cluster granularity: one file per category per month (e.g., decisions-2026-04.md).
All files get aiia_managed: true frontmatter to prevent feedback loops.

iCloud safety: all writes go through tmp+rename to avoid partial-file sync.
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("aiia.vault_writer")

# Category → vault folder mapping
CATEGORY_FOLDERS = {
    "decisions": "30-Decisions",
    "patterns": "80-Resources",
    "lessons": "80-Resources",
    "sessions": "40-Sessions",
    "team": "80-Resources",
    "project": "80-Resources",
    "agents": "80-Resources",
    "meta": "80-Resources",
    # wip is ephemeral — skip vault writes
}

# Categories that skip vault writes (too noisy or ephemeral)
SKIP_CATEGORIES = {"wip"}


def _frontmatter(**kwargs) -> str:
    """Generate YAML frontmatter block."""
    lines = ["---"]
    for k, v in kwargs.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(i) for i in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    slug = slug.strip("-")[:max_len].rstrip("-")
    return slug or "untitled"


def _atomic_write(path: Path, content: str):
    """Write atomically via tmp+rename for iCloud safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.rename(str(tmp), str(path))


def _cluster_key(category: str, created_at: str) -> str:
    """Derive cluster filename from category and timestamp."""
    month = (created_at or "")[:7] or datetime.utcnow().strftime("%Y-%m")
    return f"{category}-{month}"


def _parse_existing_cluster(path: Path) -> tuple:
    """Parse an existing cluster file into (frontmatter_str, header, entries_by_date, footer).

    Returns enough structure to append a new entry without rewriting everything.
    """
    if not path.exists():
        return None, None, {}, None

    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Extract frontmatter
    fm_end = -1
    if lines and lines[0] == "---":
        for i in range(1, len(lines)):
            if lines[i] == "---":
                fm_end = i
                break
    fm_str = "\n".join(lines[: fm_end + 1]) if fm_end >= 0 else ""
    body_start = fm_end + 1 if fm_end >= 0 else 0

    # Parse body into date sections
    header_lines = []
    entries_by_date: dict[str, list[str]] = {}
    current_date = None
    footer_lines = []
    in_footer = False

    for line in lines[body_start:]:
        if line.startswith("## 20") and len(line) >= 13:
            current_date = line[3:].strip()
            entries_by_date.setdefault(current_date, [])
        elif line.startswith("*") and line.endswith("*") and "last sync" in line:
            in_footer = True
            footer_lines.append(line)
        elif in_footer:
            footer_lines.append(line)
        elif current_date is not None:
            entries_by_date[current_date].append(line)
        else:
            header_lines.append(line)

    return fm_str, "\n".join(header_lines), entries_by_date, "\n".join(footer_lines)


class VaultWriter:
    """Async queue-based writer that syncs AIIA memories to Obsidian vault in real time."""

    def __init__(self, vault_dir: str, auto_file_queries: bool = True):
        self._vault_dir = Path(vault_dir)
        self._auto_file_queries = auto_file_queries
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._worker_task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the background worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info(f"VaultWriter started — vault={self._vault_dir}")

    async def stop(self):
        """Stop the background worker, draining remaining items."""
        self._running = False
        if self._worker_task:
            # Signal shutdown
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._worker_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
            self._worker_task = None

    async def _worker(self):
        """Background worker that processes vault write operations."""
        while self._running or not self._queue.empty():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if item is None:  # shutdown signal
                break

            try:
                op = item.get("op")
                if op == "memory":
                    self._do_write_memory(item["entry"], item["category"])
                elif op == "query":
                    self._do_file_query(item["question"], item["answer"], item.get("sources", []))
            except Exception as e:
                logger.warning(f"VaultWriter error: {e}")

            self._queue.task_done()

    async def write_memory(self, entry: dict[str, Any], category: str):
        """Enqueue a memory entry for vault write. Non-blocking."""
        if category in SKIP_CATEGORIES:
            return
        if not entry or not entry.get("fact"):
            return

        try:
            self._queue.put_nowait(
                {
                    "op": "memory",
                    "entry": entry,
                    "category": category,
                }
            )
        except asyncio.QueueFull:
            logger.warning("VaultWriter queue full — dropping vault write")

    async def file_query(self, question: str, answer: str, sources: list[dict] = None):
        """Enqueue a substantive AIIA answer to be filed as a wiki article."""
        if not self._auto_file_queries:
            return
        if len(answer) < 800:
            return

        try:
            self._queue.put_nowait(
                {
                    "op": "query",
                    "question": question,
                    "answer": answer,
                    "sources": sources or [],
                }
            )
        except asyncio.QueueFull:
            logger.warning("VaultWriter queue full — dropping query file")

    def _do_write_memory(self, entry: dict[str, Any], category: str):
        """Synchronous: write/append a memory entry to its cluster file."""
        folder = CATEGORY_FOLDERS.get(category)
        if not folder:
            logger.debug(f"No vault folder for category={category}")
            return

        created_at = entry.get("created_at", datetime.utcnow().isoformat())
        cluster_name = _cluster_key(category, created_at)
        cluster_path = self._vault_dir / folder / f"{cluster_name}.md"

        date = created_at[:10]
        memory_id = entry.get("id", "unknown")
        fact = (entry.get("fact") or "").strip()
        source = entry.get("source", "")

        # Format the entry line
        source_tag = (
            f" *({source})*" if source and source not in {"session", "bootstrap", ""} else ""
        )
        entry_line = f"- **{memory_id}** — {fact}{source_tag}"

        if cluster_path.exists():
            # Append to existing cluster
            fm_str, header, entries_by_date, footer = _parse_existing_cluster(cluster_path)

            entries_by_date.setdefault(date, [])
            entries_by_date[date].append(entry_line)

            # Rebuild file
            content = self._build_cluster_content(category, cluster_name, entries_by_date, fm_str)
        else:
            # Create new cluster
            entries_by_date = {date: [entry_line]}
            content = self._build_cluster_content(category, cluster_name, entries_by_date)

        _atomic_write(cluster_path, content)
        logger.info(f"Vault: wrote {cluster_path.name} ({len(content)} chars)")

    def _build_cluster_content(
        self,
        category: str,
        cluster_name: str,
        entries_by_date: dict[str, list[str]],
        existing_fm: str | None = None,
    ) -> str:
        """Build complete cluster file content."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Count total entries
        total = sum(len([e for e in entries if e.strip()]) for entries in entries_by_date.values())

        if existing_fm:
            fm = existing_fm
        else:
            fm = _frontmatter(
                type=f"aiia-{category}",
                date=today,
                source="aiia-vault-writer",
                aiia_managed=True,
                aiia_version=2,
                cluster=cluster_name,
                tags=["aiia", category],
            )

        title = category.replace("_", " ").title()
        month_label = cluster_name.split("-", 1)[1] if "-" in cluster_name else cluster_name

        lines = [
            fm,
            "",
            f"# AIIA {title} — {month_label}",
            "",
            "> Auto-generated by AIIA VaultWriter — do not edit, changes will be overwritten.",
            "",
        ]

        for date in sorted(entries_by_date.keys(), reverse=True):
            entries = entries_by_date[date]
            non_empty = [e for e in entries if e.strip()]
            if non_empty:
                lines.append(f"## {date}")
                lines.extend(non_empty)
                lines.append("")

        lines.append(f"*{total} entries · last sync {today}*")
        return "\n".join(lines)

    def _do_file_query(self, question: str, answer: str, sources: list[dict]):
        """Synchronous: file a substantive answer as a wiki article."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        slug = _slugify(question)
        filename = f"query-{today}-{slug}.md"
        wiki_dir = self._vault_dir / "85-Wiki"
        dest = wiki_dir / filename

        # Dedup: check if same slug exists for today
        if dest.exists():
            logger.info(f"Query already filed: {filename} — updating")

        # Build source references
        source_links = []
        for s in sources[:10]:
            src = s.get("source", s.get("content", ""))[:80]
            if src:
                source_links.append(f"- {src}")

        fm = _frontmatter(
            type="aiia-query",
            date=today,
            source="aiia-vault-writer",
            aiia_managed=True,
            aiia_version=2,
            tags=["aiia", "query", "wiki"],
        )

        lines = [
            fm,
            "",
            f"# {question[:120]}",
            "",
            "> Auto-generated by AIIA — filed from a substantive answer.",
            "",
            answer,
            "",
        ]
        if source_links:
            lines.append("## Sources")
            lines.extend(source_links)
            lines.append("")

        lines.append(f"*Filed {today}*")
        content = "\n".join(lines)

        _atomic_write(dest, content)
        logger.info(f"Vault: filed query {filename} ({len(content)} chars)")
