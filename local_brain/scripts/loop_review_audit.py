"""Read-only audit of scheduled-loop review items on the Mini.

Answers two questions without changing any record:

1. Is the loop-review backpressure guard (#71, MAX_PENDING_LOOP_REVIEWS) in the
   checked-out code, and is there evidence the running process has used it?
2. Where do the unreviewed scheduled items come from: which agents, which days,
   how many failed, and how many are byte-identical repeats?

It reads assignment_data.json and agent_data.json and runs `git` read commands.
It prints counts, dates, lengths and short hashes only, never run output, so the
report is safe to paste. Nothing is reviewed, dismissed, deleted or rewritten.

    python -m local_brain.scripts.loop_review_audit
    python -m local_brain.scripts.loop_review_audit --data-dir /path/to/command_center
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

GUARD_COMMIT = "44ee3a1"  # #71: scheduled-loop review backpressure landed on main
MAX_ASSIGNMENTS = 250  # assignment_registry: beyond this, the oldest finished item is evicted
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "local_brain" / "command_center"


def _git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(  # noqa: S603 - fixed git argv, read-only commands
        ["git", "-C", str(REPO_ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return proc.returncode, proc.stdout.strip() or proc.stderr.strip()


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("assignments", "agents"):
            if isinstance(data.get(key), list):
                return data[key]
        return list(data.values())
    return data


def _short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    print("== Code on disk")
    code, head = _git("rev-parse", "--short", "HEAD")
    print(f"checked-out commit: {head if code == 0 else 'unavailable: ' + head}")
    code, _ = _git("merge-base", "--is-ancestor", GUARD_COMMIT, "HEAD")
    verdict = {0: "yes", 1: "NO"}.get(code, "unknown (commit not found locally)")
    print(f"contains guard commit {GUARD_COMMIT} (#71): {verdict}")
    print("note: this is the code on disk, not proof the running process loaded it;")
    print("      restart time vs. the dates below is the evidence for that.")

    assignments = _load(args.data_dir / "assignment_data.json")
    agents = {a.get("id"): a for a in _load(args.data_dir / "agent_data.json")}

    print("\n== Agents with loops")
    for agent in agents.values():
        if not agent.get("loop_enabled"):
            continue
        print(
            f"- {agent.get('name')}: skip_reason={agent.get('loop_skip_reason') or '-'} "
            f"skipped_at={agent.get('loop_skipped_at') or '-'} "
            f"checked_at={agent.get('loop_checked_at') or '-'} "
            f"last_run_at={agent.get('last_run_at') or '-'} "
            f"tools={sorted(agent.get('tools', []))}"
        )

    scheduled = [a for a in assignments if a.get("trigger") == "interval"]
    print(f"\n== Scheduled assignments: {len(scheduled)} of {len(assignments)} total")
    print(
        f"registry size {len(assignments)} / cap {MAX_ASSIGNMENTS}: "
        + (
            "AT CAP. Each new assignment evicts the oldest finished one, reviewed or not."
            if len(assignments) >= MAX_ASSIGNMENTS
            else "below cap, nothing evicted yet"
        )
    )
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for item in scheduled:
        agent = agents.get(item.get("agent_id"), {})
        by_agent[agent.get("name") or item.get("agent_id", "?")].append(item)

    for name, items in sorted(by_agent.items()):
        statuses = Counter(i.get("status") for i in items)
        pending = [
            i
            for i in items
            if i.get("status") == "completed"
            and i.get("review_status", "unreviewed") == "unreviewed"
            and not i.get("dismissed_at")
        ]
        failed_errors = Counter(i.get("error") or "-" for i in items if i.get("status") == "failed")
        print(f"\n-- {name}: {dict(statuses)}; pending review (guard's count) = {len(pending)}")
        if failed_errors:
            print(f"   failure reasons: {dict(failed_errors)}")
        per_day = Counter((i.get("created_at") or "?")[:10] for i in pending)
        print(
            "   pending by created day: "
            + ", ".join(f"{d}={n}" for d, n in sorted(per_day.items()))
        )
        if pending:
            last = max(i.get("created_at") or "" for i in pending)
            print(f"   newest pending created_at: {last}")
        repeats = Counter(_short(i.get("result", "")) for i in pending)
        dupes = {h: n for h, n in repeats.items() if n > 1}
        print(
            f"   identical-output groups among pending: {len(dupes)} "
            f"(covering {sum(dupes.values())} items); distinct outputs: {len(repeats)}"
        )
        lengths = sorted(len(i.get("result", "")) for i in pending)
        if lengths:
            print(
                f"   output length min/median/max: {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
