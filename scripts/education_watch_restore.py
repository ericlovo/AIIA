"""Restore the immediately preceding cloud run, failing closed on missing state."""

import json
import os
import subprocess
from pathlib import Path


def gh(*args):
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return result.stdout


def main():
    destination = Path("education-watch")
    destination.mkdir(exist_ok=True)
    if int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")) != 1:
        raise RuntimeError(
            "rerun_requires_manual_delivery_audit; dispatch a new run only after reconciliation"
        )
    current = int(os.environ["GITHUB_RUN_ID"])
    repo = os.environ["GITHUB_REPOSITORY"]
    runs = json.loads(
        gh(
            "api",
            f"repos/{repo}/actions/workflows/education-rfp-watch.yml/runs?branch=main&per_page=100",
        )
    )["workflow_runs"]
    previous = next(
        (
            r
            for r in runs
            if r["id"] != current and r["run_number"] < int(os.environ["GITHUB_RUN_NUMBER"])
        ),
        None,
    )
    if previous is None:
        if int(os.environ["GITHUB_RUN_NUMBER"]) != 1:
            raise RuntimeError("previous_run_unavailable; manual delivery audit required")
        print("First run: new delivery ledger")
        return
    if previous["status"] != "completed":
        raise RuntimeError("previous_run_incomplete")
    if previous.get("conclusion") != "success":
        raise RuntimeError(
            "previous_run_failed; inspect its ledger and Slack receipt before recovery"
        )
    gh(
        "run",
        "download",
        str(previous["id"]),
        "--repo",
        repo,
        "--name",
        "education-watch-state",
        "--dir",
        str(destination),
    )
    if not (destination / "state.json").is_file():
        raise RuntimeError("previous_ledger_missing; manual delivery audit required")
    # Summaries/evidence describe only the new execution, not yesterday's result.
    for name in ("summary.md", "result.json"):
        (destination / name).unlink(missing_ok=True)
    print("Previous delivery ledger restored")


if __name__ == "__main__":
    main()
