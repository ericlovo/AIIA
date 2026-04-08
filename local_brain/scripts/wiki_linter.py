"""
Wiki Linter — Health checks for the AIIA knowledge wiki in Obsidian vault.

Six analysis passes:
1. Stale detection — AIIA-managed files older than 30 days with no updates
2. Contradiction detection — LLM-powered check for conflicting facts
3. Orphan detection — files with zero incoming wikilinks
4. Missing link suggestions — mentions of vault filenames without [[]]
5. Duplicate detection — Jaccard similarity between entries in same category
6. Enhancement suggestions — folders with few articles that could be expanded

Usage:
    python -m local_brain.scripts.wiki_linter
    python -m local_brain.scripts.wiki_linter --fix
    python -m local_brain.scripts.wiki_linter --dry-run
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- Vault path resolution ---

_DEFAULT_VAULT = Path.home() / "Documents" / "Eric's AIIA"
_FALLBACK_VAULT = Path.home() / "AIIAVault"
VAULT_DIR = Path(
    os.getenv(
        "OBSIDIAN_VAULT_DIR",
        str(_DEFAULT_VAULT if _DEFAULT_VAULT.exists() else _FALLBACK_VAULT),
    )
)

SKIP_FOLDERS = {".obsidian", ".trash", "Templates", ".git", "processed"}
INDEX_FILES = {"_Index.md", "_Master-Index.md", "_Wiki-Index.md", "_lint-report.md"}

# Stale threshold in days
STALE_DAYS = 30


def _parse_frontmatter(path: Path) -> Dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end < 0:
        return {}
    fm = {}
    for line in text[3:end].strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
    return fm


def _all_vault_files(vault: Path) -> List[Path]:
    """Get all .md files in the vault (excluding skip folders and index files)."""
    files = []
    for item in vault.iterdir():
        if not item.is_dir() or item.name in SKIP_FOLDERS or item.name.startswith("."):
            # Top-level .md files
            if item.suffix == ".md" and item.name not in INDEX_FILES:
                files.append(item)
            continue
        for f in item.rglob("*.md"):
            if f.name not in INDEX_FILES and "processed" not in f.parts:
                files.append(f)
    return files


def _extract_wikilinks(text: str) -> Set[str]:
    """Extract all [[link]] and [[link|alias]] targets from text."""
    return set(re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]", text))


def _word_set(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase words."""
    return set(re.findall(r"[a-z]{3,}", text.lower()))


# ---------------------------------------------------------------------------
# Pass 1: Stale detection
# ---------------------------------------------------------------------------


def check_stale(files: List[Path]) -> List[Dict]:
    """Find AIIA-managed files with date older than STALE_DAYS and no recent updates."""
    findings = []
    cutoff = (datetime.now() - timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d")

    for f in files:
        fm = _parse_frontmatter(f)
        if fm.get("aiia_managed") not in ("true", "True"):
            continue
        date = fm.get("date", "")
        if date and date < cutoff:
            # Check file mtime as well
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")
            if mtime < cutoff:
                findings.append({
                    "type": "stale",
                    "file": str(f.relative_to(f.parent.parent) if f.parent != VAULT_DIR else f.name),
                    "date": date,
                    "mtime": mtime,
                    "message": f"Last updated {date}, file modified {mtime} (>{STALE_DAYS} days ago)",
                })
    return findings


# ---------------------------------------------------------------------------
# Pass 2: Contradiction detection (LLM-powered, batched)
# ---------------------------------------------------------------------------


def _call_ollama_batch(facts: List[str]) -> Optional[str]:
    """Send facts to LLM to check for contradictions."""
    url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434") + "/api/generate"
    prompt = (
        "Review these facts for contradictions. If any two facts contradict each other, "
        "list each contradiction with the pair of facts and explain why they conflict. "
        "If no contradictions, say 'No contradictions found.'\n\n"
        + "\n".join(f"- {fact}" for fact in facts)
    )
    payload = json.dumps({
        "model": os.getenv("LOCAL_TASK_MODEL", "llama3.1:8b-instruct-q8_0"),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 16384},
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")
    except Exception as e:
        return None


def check_contradictions(files: List[Path], batch_size: int = 10) -> List[Dict]:
    """Check AIIA-managed files for contradicting facts using LLM."""
    findings = []

    # Collect all fact lines from AIIA cluster files
    facts_by_category: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        fm = _parse_frontmatter(f)
        if fm.get("aiia_managed") not in ("true", "True"):
            continue
        cluster = fm.get("cluster", "")
        if not cluster:
            continue
        category = cluster.split("-")[0] if "-" in cluster else "unknown"

        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("- **") and " — " in line:
                # Extract fact content after the ID
                fact_part = line.split(" — ", 1)[1] if " — " in line else ""
                if fact_part:
                    facts_by_category[category].append(fact_part.rstrip())

    # Check each category in batches
    for category, facts in facts_by_category.items():
        if len(facts) < 2:
            continue
        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            if len(batch) < 2:
                continue
            result = _call_ollama_batch(batch)
            if result and "no contradiction" not in result.lower():
                findings.append({
                    "type": "contradiction",
                    "category": category,
                    "batch_start": i,
                    "message": result[:500],
                })

    return findings


# ---------------------------------------------------------------------------
# Pass 3: Orphan detection
# ---------------------------------------------------------------------------


def check_orphans(files: List[Path]) -> List[Dict]:
    """Find files with zero incoming wikilinks from other files."""
    # Build link graph
    all_stems = {f.stem for f in files}
    incoming: Dict[str, int] = {stem: 0 for stem in all_stems}

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        links = _extract_wikilinks(text)
        for link in links:
            # Normalize: strip path prefix if any
            target = link.split("/")[-1]
            if target in incoming:
                incoming[target] += 1

    findings = []
    for stem, count in incoming.items():
        if count == 0:
            # Find the actual file
            matching = [f for f in files if f.stem == stem]
            if matching:
                fm = _parse_frontmatter(matching[0])
                # Skip MOCs and indexes — they don't need incoming links
                if fm.get("type") in ("moc", "daily-note"):
                    continue
                findings.append({
                    "type": "orphan",
                    "file": matching[0].name,
                    "message": f"No incoming wikilinks — consider linking from related pages",
                })
    return findings


# ---------------------------------------------------------------------------
# Pass 4: Missing link suggestions
# ---------------------------------------------------------------------------


def check_missing_links(files: List[Path]) -> List[Dict]:
    """Find mentions of vault filenames in text that aren't wikilinked."""
    all_stems = {f.stem for f in files if not f.name.startswith("_")}
    findings = []

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue

        linked = _extract_wikilinks(text)
        linked_stems = {l.split("/")[-1] for l in linked}

        for stem in all_stems:
            if stem == f.stem:  # Don't suggest self-links
                continue
            if stem in linked_stems:
                continue
            # Check if the stem appears as a word in the text
            if len(stem) > 5 and stem.lower().replace("-", " ") in text.lower():
                findings.append({
                    "type": "missing_link",
                    "file": f.name,
                    "target": stem,
                    "message": f"Mentions '{stem}' but doesn't link to [[{stem}]]",
                })

    return findings


# ---------------------------------------------------------------------------
# Pass 5: Duplicate detection
# ---------------------------------------------------------------------------


def check_duplicates(files: List[Path], threshold: float = 0.7) -> List[Dict]:
    """Find pairs of AIIA entries with high Jaccard similarity."""
    findings = []

    # Collect entries per category
    entries_by_cat: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
    for f in files:
        fm = _parse_frontmatter(f)
        if fm.get("aiia_managed") not in ("true", "True"):
            continue
        cluster = fm.get("cluster", "")
        if not cluster:
            continue
        category = cluster.split("-")[0] if "-" in cluster else "unknown"

        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue

        for line in text.split("\n"):
            if line.strip().startswith("- **") and " — " in line:
                fact = line.split(" — ", 1)[1] if " — " in line else ""
                if fact:
                    words = _word_set(fact)
                    if len(words) >= 3:
                        entries_by_cat[category].append((fact[:100], words))

    for category, entries in entries_by_cat.items():
        for i in range(len(entries)):
            for j in range(i + 1, min(i + 50, len(entries))):  # Cap comparisons
                a_text, a_words = entries[i]
                b_text, b_words = entries[j]
                intersection = len(a_words & b_words)
                union = len(a_words | b_words)
                if union == 0:
                    continue
                similarity = intersection / union
                if similarity >= threshold:
                    findings.append({
                        "type": "duplicate",
                        "category": category,
                        "similarity": round(similarity, 2),
                        "entry_a": a_text,
                        "entry_b": b_text,
                        "message": f"Jaccard {similarity:.0%} similarity — may be duplicates",
                    })
    return findings


# ---------------------------------------------------------------------------
# Pass 6: Enhancement suggestions
# ---------------------------------------------------------------------------


def check_enhancements(files: List[Path]) -> List[Dict]:
    """Suggest folders that could use more wiki articles."""
    findings = []

    folder_counts: Dict[str, int] = defaultdict(int)
    for f in files:
        fm = _parse_frontmatter(f)
        if fm.get("aiia_managed") in ("true", "True"):
            folder = f.parent.name
            folder_counts[folder] += 1

    # Find folders with few AIIA articles
    for folder, count in folder_counts.items():
        if count < 3 and folder not in ("00-Inbox", "Templates"):
            findings.append({
                "type": "enhancement",
                "folder": folder,
                "count": count,
                "message": f"Only {count} AIIA articles in {folder}/ — consider adding more coverage",
            })

    return findings


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(all_findings: List[Dict]) -> str:
    """Build a markdown report of all findings."""
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "---",
        "type: aiia-lint",
        f"date: {today}",
        "source: wiki-linter",
        "aiia_managed: true",
        "tags: [aiia, lint, health]",
        "---",
        "",
        "# Wiki Lint Report",
        "",
        f"> Generated {today} by `brain lint-wiki`.",
        "",
    ]

    # Group by type
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for f in all_findings:
        by_type[f["type"]].append(f)

    type_labels = {
        "stale": "Stale Content",
        "contradiction": "Contradictions",
        "orphan": "Orphaned Pages",
        "missing_link": "Missing Links",
        "duplicate": "Potential Duplicates",
        "enhancement": "Enhancement Opportunities",
    }

    if not all_findings:
        lines.append("All clear — no issues found.")
    else:
        lines.append(f"**{len(all_findings)} findings** across {len(by_type)} categories.")
        lines.append("")

        for type_key in ["contradiction", "duplicate", "stale", "orphan", "missing_link", "enhancement"]:
            items = by_type.get(type_key, [])
            if not items:
                continue
            label = type_labels.get(type_key, type_key)
            lines.append(f"## {label} ({len(items)})")
            lines.append("")
            for item in items[:20]:  # Cap at 20 per category
                msg = item.get("message", "")
                file_ref = item.get("file", "")
                if file_ref:
                    lines.append(f"- **{file_ref}**: {msg}")
                else:
                    lines.append(f"- {msg}")
            if len(items) > 20:
                lines.append(f"- *...and {len(items) - 20} more*")
            lines.append("")

    lines.append(f"*Lint completed {today}*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(dry_run: bool = False, fix: bool = False, skip_llm: bool = False) -> List[Dict]:
    """Run all lint passes and generate report."""
    print(f"Wiki Linter — vault={VAULT_DIR} dry_run={dry_run} fix={fix}")

    files = _all_vault_files(VAULT_DIR)
    print(f"Scanning {len(files)} files...")

    all_findings = []

    # Pass 1: Stale
    print("\nPass 1/6: Stale detection...")
    stale = check_stale(files)
    all_findings.extend(stale)
    print(f"  Found {len(stale)} stale files")

    # Pass 2: Contradictions (LLM-powered, skip if --dry-run or --skip-llm)
    if not dry_run and not skip_llm:
        print("\nPass 2/6: Contradiction detection (LLM)...")
        contradictions = check_contradictions(files)
        all_findings.extend(contradictions)
        print(f"  Found {len(contradictions)} potential contradictions")
    else:
        print("\nPass 2/6: Contradiction detection — skipped (dry-run/no-llm)")

    # Pass 3: Orphans
    print("\nPass 3/6: Orphan detection...")
    orphans = check_orphans(files)
    all_findings.extend(orphans)
    print(f"  Found {len(orphans)} orphaned pages")

    # Pass 4: Missing links
    print("\nPass 4/6: Missing link detection...")
    missing = check_missing_links(files)
    all_findings.extend(missing)
    print(f"  Found {len(missing)} missing link opportunities")

    # Pass 5: Duplicates
    print("\nPass 5/6: Duplicate detection...")
    dupes = check_duplicates(files)
    all_findings.extend(dupes)
    print(f"  Found {len(dupes)} potential duplicates")

    # Pass 6: Enhancements
    print("\nPass 6/6: Enhancement suggestions...")
    enhancements = check_enhancements(files)
    all_findings.extend(enhancements)
    print(f"  Found {len(enhancements)} enhancement opportunities")

    # Write report
    report = build_report(all_findings)
    report_path = VAULT_DIR / "85-Wiki" / "_lint-report.md"

    if dry_run:
        print(f"\n[dry-run] Would write {report_path}")
        print(f"\n{report}")
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(".md.tmp")
        tmp.write_text(report, encoding="utf-8")
        os.rename(str(tmp), str(report_path))
        print(f"\nReport: {report_path}")

    print(f"\nTotal: {len(all_findings)} findings")
    return all_findings


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    fix = "--fix" in sys.argv
    skip_llm = "--skip-llm" in sys.argv or "--no-llm" in sys.argv
    run(dry_run=dry_run, fix=fix, skip_llm=skip_llm)


if __name__ == "__main__":
    main()
