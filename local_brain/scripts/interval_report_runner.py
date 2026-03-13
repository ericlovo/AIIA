"""
Interval code report — runs every N hours, captures recent git activity,
saves a timestamped report, sends a macOS notification, and pushes to
the Command Center WebSocket.

Usage:
    python -m local_brain.scripts.interval_report_runner [--hours 3]
    # Or from brain CLI: brain report --interval
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_brain.scripts.daily_report import (
    generate_report,
    print_report,
)

REPORTS_DIR = Path.home() / ".aiia" / "eq_data" / "reports" / "intervals"
LATEST_LINK = (
    Path.home() / ".aiia" / "eq_data" / "reports" / "latest-interval.json"
)
DAILY_MD = Path.home() / ".aiia" / "eq_data" / "reports" / "today.md"


def _notify(title: str, message: str) -> None:
    """Send a macOS notification via terminal-notifier (no Script Editor popup)."""
    try:
        subprocess.run(
            [
                "terminal-notifier",
                "-title",
                title,
                "-message",
                message,
                "-group",
                "aiia-report",
                "-sound",
                "default",
            ],
            timeout=5,
            capture_output=True,
        )
    except FileNotFoundError:
        pass


def _post_localhost(port: int, path: str, body: dict) -> None:
    """POST JSON to a localhost service (no external URLs)."""
    import http.client

    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        conn.getresponse()
        conn.close()
    except Exception:
        pass


def _push_to_command_center(report: dict) -> None:
    """POST the report to the Command Center so it broadcasts via WebSocket."""
    _post_localhost(
        8200, "/api/reports/interval", {"event": "interval_report", "report": report}
    )


def _remember_in_aiia(summary: str) -> None:
    """Store report summary in AIIA memory as a project memory."""
    _post_localhost(
        8100,
        "/v1/aiia/remember",
        {
            "fact": summary,
            "category": "project",
            "source": "interval_report",
        },
    )


def save_interval_report(report: dict) -> Path:
    """Save with timestamp-based filename."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M")
    path = REPORTS_DIR / f"{ts}.json"
    path.write_text(json.dumps(report, indent=2))
    # Update latest symlink
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        LATEST_LINK.unlink()
    LATEST_LINK.symlink_to(path)
    return path


def _build_summary_line(report: dict) -> str:
    """One-line summary for notifications."""
    s = report["summary"]
    hours = report.get("since_hours", 3)
    commits = s["total_commits"]
    uncommitted = report.get("uncommitted")
    if commits == 0:
        if uncommitted:
            return f"Last {hours}h: no commits, but {uncommitted}"
        return f"Last {hours}h: no commits"
    files = s["total_files_changed"]
    adds = s["total_additions"]
    dels = s["total_deletions"]
    types = s.get("commit_types", {})
    type_str = ", ".join(
        f"{v} {k}" for k, v in sorted(types.items(), key=lambda x: -x[1])[:3]
    )
    products = list(report.get("products", {}).keys())[:3]
    product_str = ", ".join(products)
    return f"Last {hours}h: {commits} commits ({type_str}) | +{adds}/-{dels} in {files} files | {product_str}"


def _check_uncommitted() -> Optional[str]:
    """Check for uncommitted changes and return a summary if any exist."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "--cached", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )
        staged = result.stdout.strip()

        result2 = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).parent.parent.parent.parent),
        )
        unstaged = result2.stdout.strip()

        lines = []
        if staged:
            count = len([l for l in staged.splitlines() if "|" in l])
            lines.append(f"{count} staged")
        if unstaged:
            count = len([l for l in unstaged.splitlines() if "|" in l])
            lines.append(f"{count} modified")
        if lines:
            return ", ".join(lines) + " (uncommitted)"
        return None
    except Exception:
        return None


def _append_to_daily_md(report: dict, summary: str) -> Path:
    """Append this interval's results to a rolling daily markdown file.

    The file resets each day and accumulates interval snapshots,
    giving Eric a single scannable artifact: ~/.aiia/eq_data/reports/today.md
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    s = report["summary"]

    # Reset the file if the date header doesn't match today
    header = f"# Code Activity — {date_str}\n"
    if DAILY_MD.exists():
        first_line = DAILY_MD.read_text().split("\n", 1)[0]
        if first_line != header.strip():
            DAILY_MD.write_text("")  # new day, fresh file

    lines = []
    if not DAILY_MD.exists() or DAILY_MD.stat().st_size == 0:
        lines.append(header)
        lines.append("")

    lines.append(f"## {time_str}")
    lines.append("")
    lines.append(f"**{summary}**")
    lines.append("")

    commits = s["total_commits"]
    if commits > 0:
        # Per-product breakdown
        for product, data in report.get("products", {}).items():
            product_commits = data.get("commits", [])
            if not product_commits:
                continue
            lines.append(f"### {product}")
            for c in product_commits:
                lines.append(f"- `{c.get('hash', '?')[:7]}` {c.get('subject', '')}")
            lines.append("")

    uncommitted = report.get("uncommitted")
    if uncommitted:
        lines.append(f"> Uncommitted: {uncommitted}")
        lines.append("")

    lines.append("---")
    lines.append("")

    with open(DAILY_MD, "a") as f:
        f.write("\n".join(lines))

    return DAILY_MD


def run(hours: int = 3) -> Optional[dict]:
    """Generate and deliver an interval report."""
    report = generate_report(since_hours=hours)
    commits = report["summary"]["total_commits"]

    # Check for uncommitted work
    uncommitted = _check_uncommitted()
    if uncommitted:
        report["uncommitted"] = uncommitted

    summary = _build_summary_line(report)

    # Always save (even if 0 commits — shows the cadence is working)
    path = save_interval_report(report)

    # Always append to rolling daily markdown
    md_path = _append_to_daily_md(report, summary)

    if commits > 0:
        print_report(report)
        _notify("Code Report", summary)
        _push_to_command_center(report)
        _remember_in_aiia(summary)
    elif uncommitted:
        print(f"\n  {summary}")
        _notify("Code Report", summary)
    else:
        print(f"\n  {summary}")

    print(f"  Saved: {path}")
    print(f"  Daily: {md_path}")
    return report


def main():
    hours = 3
    args = sys.argv[1:]
    if "--hours" in args:
        idx = args.index("--hours")
        if idx + 1 < len(args):
            hours = int(args[idx + 1])

    run(hours=hours)


if __name__ == "__main__":
    main()
