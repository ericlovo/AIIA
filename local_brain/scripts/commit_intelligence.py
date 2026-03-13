"""
Commit Intelligence — extract architectural decisions and patterns from git commits.

Parses recent git commits, sends meaningful ones to local LLM for analysis,
and stores extracted intelligence as AIIA memories.

Usage:
    python -m local_brain.scripts.commit_intelligence [--hours 3]
    # Or from brain CLI: brain commits
"""

import http.client
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_DIR = Path(__file__).parent.parent.parent.parent  # repo root
BRAIN_API = ("localhost", 8100)


def _run_git(args: List[str]) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(REPO_DIR),
        capture_output=True,
        timeout=30,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _get_recent_commits(hours: int = 3) -> List[Dict]:
    """Get commits from the last N hours with file stats.

    Fetches from origin first so we see commits pushed by other agents.
    Then pulls to fast-forward if possible (non-destructive).
    """
    # Fetch + pull so we see all remote commits
    _run_git(["fetch", "--quiet"])
    _run_git(["pull", "--ff-only", "--quiet"])

    log_output = _run_git(
        [
            "log",
            f"--since={hours} hours ago",
            "--format=%H|%s|%an|%ai",
            "--no-merges",
        ]
    )
    if not log_output:
        return []

    commits = []
    for line in log_output.splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        sha, subject, author, date = parts

        # Get files changed
        stat_output = _run_git(
            ["diff-tree", "--no-commit-id", "-r", "--name-only", sha]
        )
        files = (
            [f for f in stat_output.splitlines() if f.strip()] if stat_output else []
        )

        commits.append(
            {
                "sha": sha[:12],
                "subject": subject,
                "author": author,
                "date": date,
                "files": files,
                "file_count": len(files),
            }
        )

    return commits


def _get_diff_summary(sha: str) -> str:
    """Get a condensed diff summary for a commit."""
    diff = _run_git(["diff-tree", "-p", "--stat", sha])
    # Truncate to avoid overwhelming the LLM
    if len(diff) > 4000:
        diff = diff[:4000] + "\n... (truncated)"
    return diff


def _post_aiia(path: str, body: dict) -> Optional[dict]:
    """POST JSON to AIIA API."""
    try:
        conn = http.client.HTTPConnection(*BRAIN_API, timeout=120)
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        return data
    except Exception as e:
        print(f"  AIIA API error: {e}")
        return None


def _extract_intelligence(commits: List[Dict]) -> List[Dict]:
    """Send commits to local LLM and extract architectural intelligence."""
    # Build commit summary for the LLM
    commit_text_parts = []
    for c in commits:
        files_str = ", ".join(c["files"][:10])
        if len(c["files"]) > 10:
            files_str += f" (+{len(c['files']) - 10} more)"
        commit_text_parts.append(
            f"- [{c['sha']}] {c['subject']} ({c['file_count']} files: {files_str})"
        )

    commit_text = "\n".join(commit_text_parts)

    # Get diff details for architecturally significant commits only
    # (feat, refactor, migrate, architect — NOT plain fixes)
    # Cap at 5 to avoid overwhelming the LLM
    diff_details = []
    for c in commits:
        if len(diff_details) >= 5:
            break
        if c["file_count"] > 2 and any(
            kw in c["subject"].lower()
            for kw in ("refactor", "architect", "migrate", "feat")
        ):
            diff = _get_diff_summary(c["sha"])
            if diff:
                diff_details.append(f"### Commit {c['sha']}: {c['subject']}\n{diff}")

    diff_section = ""
    if diff_details:
        combined = "\n\n".join(diff_details)
        # Truncate combined diffs
        if len(combined) > 8000:
            combined = combined[:8000] + "\n... (truncated)"
        diff_section = f"\n\nDIFF DETAILS:\n{combined}"

    question = (
        "Analyze these recent code commits and extract architectural decisions, "
        "patterns, or lessons. Focus on: what was built, why, and what patterns "
        "were established.\n\n"
        f"COMMITS:\n{commit_text}"
        f"{diff_section}\n\n"
        'Return JSON: {"insights": [{"fact": "...", "category": "decisions|patterns|lessons"}]}\n'
        'Return {"insights": []} if nothing significant.'
    )

    response = _post_aiia(
        "/v1/aiia/ask",
        {
            "question": question,
            "depth": "fast",
            "n_results": 0,
            "include_sessions": False,
        },
    )

    if not response:
        return []

    answer = response.get("answer", "")

    # Parse JSON from the answer
    insights = _parse_insights(answer)
    return insights


def _parse_insights(text: str) -> List[Dict]:
    """Extract insights JSON from LLM response."""
    # Try direct parse
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed.get("insights"), list):
            return parsed["insights"]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try code block extraction
    if "```" in text:
        try:
            json_str = text.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            parsed = json.loads(json_str.strip())
            if isinstance(parsed.get("insights"), list):
                return parsed["insights"]
        except (json.JSONDecodeError, IndexError):
            pass

    # Try brace extraction
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed.get("insights"), list):
                return parsed["insights"]
        except json.JSONDecodeError:
            pass

    return []


def _store_memories(insights: List[Dict], commits: List[Dict]) -> int:
    """Store extracted insights as AIIA memories."""
    stored = 0
    shas = [c["sha"] for c in commits[:5]]

    for insight in insights:
        fact = insight.get("fact", "").strip()
        category = insight.get("category", "patterns")
        if len(fact) < 30:
            continue
        if category not in ("decisions", "patterns", "lessons"):
            category = "patterns"

        result = _post_aiia(
            "/v1/aiia/remember",
            {
                "fact": fact,
                "category": category,
                "source": "commit-intelligence",
                "metadata": {"commits": shas},
            },
        )
        if result:
            stored += 1
            print(f"  [{category}] {fact[:80]}...")

    return stored


def run(hours: int = 3) -> Dict:
    """Run commit intelligence extraction."""
    now = datetime.now()
    print(f"\n  Commit Intelligence — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Window: last {hours} hours\n")

    commits = _get_recent_commits(hours)
    if not commits:
        print("  No commits found.")
        return {"commits": 0, "insights": 0, "stored": 0}

    print(f"  Found {len(commits)} commits:")
    for c in commits:
        print(f"    {c['sha']} {c['subject']} ({c['file_count']} files)")
    print()

    # Filter: only analyze if there's something meaningful
    significant = [c for c in commits if c["file_count"] > 0]
    if not significant:
        print("  No significant commits to analyze.")
        return {"commits": len(commits), "insights": 0, "stored": 0}

    print("  Extracting intelligence...")
    insights = _extract_intelligence(significant)

    if not insights:
        print("  No notable architectural insights extracted.")
        return {"commits": len(commits), "insights": 0, "stored": 0}

    print(f"\n  Extracted {len(insights)} insights:")
    stored = _store_memories(insights, significant)
    print(f"\n  Stored {stored} memories from {len(commits)} commits.\n")

    return {"commits": len(commits), "insights": len(insights), "stored": stored}


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
