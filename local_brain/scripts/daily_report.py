"""
Daily shipped code report — parses git log, groups by product, runs syntax checks.

Usage:
    python -m local_brain.scripts.daily_report [YYYY-MM-DD]
    # Or from brain CLI: brain report [date]
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from local_brain.scripts.syntax_checker import check_syntax

# Default data directory
DATA_DIR = Path.home() / ".aiia" / "eq_data" / "reports"
REPO_DIR = Path(__file__).parent.parent.parent.parent  # repo root


def _run_git(args: List[str], cwd: str) -> str:
    """Run a git command and return stdout.

    Uses list form (no shell) to prevent shell injection.
    The '--' separator prevents argument injection via crafted inputs.
    """
    # Reject any args that look like they could inject flags
    for arg in args:
        if (
            arg.startswith("-")
            and not arg.startswith("--since=")
            and not arg.startswith("--until=")
            and not arg.startswith("--format=")
            and arg
            not in (
                "--no-commit-id",
                "-r",
                "--name-only",
                "--numstat",
                "--all",
            )
        ):
            raise ValueError(f"Unexpected git argument: {arg}")
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _parse_commit_type(subject: str) -> str:
    """Extract conventional commit type from subject line."""
    match = re.match(r"^(\w+)[\(:]", subject)
    if match:
        ctype = match.group(1).lower()
        if ctype in (
            "feat",
            "fix",
            "chore",
            "refactor",
            "docs",
            "test",
            "style",
            "perf",
            "ci",
            "build",
            "nuke",
        ):
            return ctype
    return "other"


def _classify_file(filepath: str) -> str:
    """Classify a changed file into a product."""
    if filepath.startswith("products/"):
        parts = filepath.split("/")
        if len(parts) >= 2:
            return parts[1]
    elif filepath.startswith("platform/"):
        return "platform"
    elif filepath.startswith("shared/"):
        return "shared"
    elif filepath.startswith("codeword/"):
        return "codeword"
    return "root"


# Business-impact categories for dashboard display
# Customize this map for your product directory names
CATEGORY_MAP = {
    "platform": "Platform",
    "root": "Platform",
    "shared": "Platform",
    # Add your product directories here, e.g.:
    # "my-product": "Primary-Client",
    # "other-product": "Other Tenants",
}

CATEGORY_COLORS = {
    "Primary-Client": "#3b82f6",
    "Platform": "#a855f7",
    "AI/Infra": "#22d3ee",
    "Other Tenants": "#f59e0b",
    "Other": "#666",
}


def _categorize_product(product: str) -> str:
    """Map a product name to a business-impact category."""
    if product in CATEGORY_MAP:
        return CATEGORY_MAP[product]
    # AI/Infra: local_brain, llm_service, mcp, ollama
    # These show up under "platform" product, so we detect via file paths
    return "Other"


def _categorize_files(files: List[str]) -> str:
    """Determine the primary category from a list of changed files."""
    cats: Dict[str, int] = {}
    for f in files:
        product = _classify_file(f)
        # Check for AI/Infra paths within platform
        if product == "platform" and any(
            p in f
            for p in (
                "local_brain/",
                "services/llm",
                "services/google_tts",
                "mcp_server",
                "ollama",
            )
        ):
            cat = "AI/Infra"
        else:
            cat = _categorize_product(product)
        cats[cat] = cats.get(cat, 0) + 1
    if not cats:
        return "Other"
    return max(cats, key=cats.get)


def _get_commits(
    date: str, repo_dir: str, *, since_hours: Optional[int] = None
) -> List[Dict]:
    """Get commits for a given date, or for the last N hours if since_hours is set."""
    if since_hours is not None:
        since_arg = f"--since={since_hours} hours ago"
        until_arg = []
    else:
        since_arg = f"--since={date}T00:00:00"
        until_arg = [f"--until={date}T23:59:59"]

    raw = _run_git(
        [
            "log",
            since_arg,
            *until_arg,
            "--format=%H|%an|%s",
            "--all",
        ],
        cwd=repo_dir,
    )

    if not raw:
        return []

    commits = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        hash_val, author, subject = parts
        commits.append(
            {
                "hash": hash_val[:7],
                "author": author,
                "subject": subject,
                "type": _parse_commit_type(subject),
            }
        )

    return commits


def _get_commit_files(hash_val: str, repo_dir: str) -> Dict:
    """Get files changed and stats for a commit."""
    # Get file list
    files_raw = _run_git(
        ["diff-tree", "--no-commit-id", "-r", "--name-only", hash_val], cwd=repo_dir
    )
    files = [f for f in files_raw.splitlines() if f.strip()]

    # Get numstat (additions/deletions)
    stat_raw = _run_git(
        ["diff-tree", "--no-commit-id", "-r", "--numstat", hash_val], cwd=repo_dir
    )
    additions = 0
    deletions = 0
    for stat_line in stat_raw.splitlines():
        parts = stat_line.split("\t")
        if len(parts) >= 2:
            try:
                additions += int(parts[0]) if parts[0] != "-" else 0
                deletions += int(parts[1]) if parts[1] != "-" else 0
            except ValueError:
                pass

    return {"files": files, "additions": additions, "deletions": deletions}


def generate_report(
    date: Optional[str] = None,
    repo_dir: Optional[str] = None,
    *,
    since_hours: Optional[int] = None,
) -> Dict:
    """Generate a shipped code report.

    If since_hours is set, captures commits from the last N hours
    (ignores date range). Otherwise uses date-based daily window.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    if repo_dir is None:
        repo_dir = str(REPO_DIR)

    commits = _get_commits(date, repo_dir, since_hours=since_hours)

    # Group commits by product
    products: Dict[str, Dict] = {}
    all_authors = set()
    all_types: Dict[str, int] = {}
    total_additions = 0
    total_deletions = 0
    total_files_changed = set()

    for commit in commits:
        details = _get_commit_files(commit["hash"], repo_dir)
        commit["files"] = details["files"]
        commit["category"] = _categorize_files(details["files"])
        all_authors.add(commit["author"])

        ctype = commit["type"]
        all_types[ctype] = all_types.get(ctype, 0) + 1

        # Classify files into products
        file_products = set()
        for f in details["files"]:
            total_files_changed.add(f)
            file_products.add(_classify_file(f))

        # Add commit to each product it touches
        for product in file_products:
            if product not in products:
                products[product] = {
                    "commit_count": 0,
                    "commits": [],
                    "total_additions": 0,
                    "total_deletions": 0,
                }
            products[product]["commit_count"] += 1
            products[product]["commits"].append(
                {
                    "hash": commit["hash"],
                    "subject": commit["subject"],
                    "author": commit["author"],
                    "type": commit["type"],
                    "category": commit["category"],
                    "files": [
                        f for f in details["files"] if _classify_file(f) == product
                    ],
                }
            )
            products[product]["total_additions"] += details["additions"]
            products[product]["total_deletions"] += details["deletions"]

        total_additions += details["additions"]
        total_deletions += details["deletions"]

    # Build category grouping
    categories: Dict[str, Dict] = {}
    for commit in commits:
        cat = commit["category"]
        if cat not in categories:
            categories[cat] = {
                "count": 0,
                "commits": [],
                "color": CATEGORY_COLORS.get(cat, "#666"),
            }
        categories[cat]["count"] += 1
        categories[cat]["commits"].append(
            {
                "hash": commit["hash"],
                "subject": commit["subject"],
                "author": commit["author"],
                "type": commit["type"],
            }
        )

    # Run syntax checks
    syntax = check_syntax(repo_dir)

    report = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "interval" if since_hours else "daily",
        "since_hours": since_hours,
        "summary": {
            "total_commits": len(commits),
            "total_files_changed": len(total_files_changed),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "products_touched": len(products),
            "authors": sorted(all_authors),
            "commit_types": all_types,
        },
        "categories": categories,
        "products": products,
        "syntax_errors": syntax,
    }

    return report


def save_report(report: Dict, data_dir: Optional[str] = None) -> Path:
    """Save report to JSON file."""
    out_dir = Path(data_dir) if data_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['date']}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def load_report(date: str, data_dir: Optional[str] = None) -> Optional[Dict]:
    """Load a saved report by date."""
    out_dir = Path(data_dir) if data_dir else DATA_DIR
    path = out_dir / f"{date}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def list_reports(data_dir: Optional[str] = None) -> List[str]:
    """List all available report dates."""
    out_dir = Path(data_dir) if data_dir else DATA_DIR
    if not out_dir.exists():
        return []
    return sorted([f.stem for f in out_dir.glob("*.json")], reverse=True)


def print_report(report: Dict) -> None:
    """Pretty-print a report to the terminal."""
    s = report["summary"]
    print(f"\n  Shipped Code Report — {report['date']}")
    print(f"  {'=' * 48}")
    print(
        f"  Commits: {s['total_commits']}  |  Files: {s['total_files_changed']}  |  +{s['total_additions']} / -{s['total_deletions']}"
    )
    print(f"  Authors: {', '.join(s['authors']) or 'none'}")
    if s["commit_types"]:
        types_str = ", ".join(f"{k}: {v}" for k, v in sorted(s["commit_types"].items()))
        print(f"  Types:   {types_str}")

    for product, data in report["products"].items():
        print(
            f"\n  [{product}] — {data['commit_count']} commits, +{data['total_additions']}/-{data['total_deletions']}"
        )
        for c in data["commits"]:
            print(f"    {c['type']:8s} {c['hash']} {c['subject'][:70]}")

    sx = report.get("syntax_errors", {})
    if sx.get("total_errors", 0) > 0:
        print(f"\n  SYNTAX ERRORS: {sx['total_errors']}")
        for product, errs in sx.get("by_product", {}).items():
            for e in errs:
                print(f"    [{product}] {e['file']}: {e['error'][:80]}")
    else:
        print(f"\n  Syntax: {sx.get('total_files', 0)} files checked, 0 errors")

    print()


# ── CLI entry point ──


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else None
    # Validate date format to prevent argument injection
    if date is not None:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            print(f"Error: Invalid date format '{date}'. Use YYYY-MM-DD.")
            sys.exit(1)
    report = generate_report(date=date)
    path = save_report(report)
    print_report(report)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
