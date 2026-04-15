#!/usr/bin/env python3
"""
Filter security scan results against .security-baseline.json.

Called by security_scan.sh after all scanners run. Reads each scanner's
raw JSON output, subtracts accepted findings from the baseline, and
rewrites the summary with filtered counts.

Usage:
    python3 scripts/filter_security_baseline.py <report_dir> <summary_file>
"""

import json
import os
import sys
from pathlib import Path


def load_baseline(repo_dir: str) -> list:
    path = os.path.join(repo_dir, ".security-baseline.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("accepted", [])


def is_baselined(scanner: str, rule: str, filepath: str, baseline: list) -> bool:
    for entry in baseline:
        if entry["scanner"] != scanner:
            continue
        if entry["rule"] != rule:
            continue
        pattern = entry.get("file_pattern", "")
        if not pattern or pattern in filepath:
            return True
    return False


def filter_bandit(report_dir: str, baseline: list) -> dict:
    path = os.path.join(report_dir, "bandit.json")
    if not os.path.exists(path):
        return {"total": 0, "new": 0, "baselined": 0}
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    new = [
        r
        for r in results
        if not is_baselined(
            "bandit", r.get("test_id", ""), r.get("filename", ""), baseline
        )
    ]
    return {"total": len(results), "new": len(new), "baselined": len(results) - len(new)}


def filter_semgrep(report_dir: str, baseline: list) -> dict:
    path = os.path.join(report_dir, "semgrep.json")
    if not os.path.exists(path):
        return {"total": 0, "new": 0, "baselined": 0}
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    new = [
        r
        for r in results
        if not is_baselined(
            "semgrep",
            r.get("check_id", "").split(".")[-1],
            r.get("path", ""),
            baseline,
        )
    ]
    return {"total": len(results), "new": len(new), "baselined": len(results) - len(new)}


def filter_shellcheck(report_dir: str, baseline: list) -> dict:
    path = os.path.join(report_dir, "shellcheck.json")
    if not os.path.exists(path):
        return {"total": 0, "new": 0, "baselined": 0}
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else []
    new = [
        r
        for r in results
        if not is_baselined(
            "shellcheck", f"SC{r.get('code', '')}", r.get("file", ""), baseline
        )
    ]
    return {"total": len(results), "new": len(new), "baselined": len(results) - len(new)}


def filter_hadolint(report_dir: str, baseline: list) -> dict:
    path = os.path.join(report_dir, "hadolint.json")
    if not os.path.exists(path):
        return {"total": 0, "new": 0, "baselined": 0}
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else []
    new = [
        r
        for r in results
        if not is_baselined("hadolint", r.get("code", ""), r.get("file", ""), baseline)
    ]
    return {"total": len(results), "new": len(new), "baselined": len(results) - len(new)}


def filter_trufflehog(report_dir: str, baseline: list) -> dict:
    path = os.path.join(report_dir, "trufflehog.json")
    if not os.path.exists(path):
        return {"total": 0, "new": 0, "baselined": 0}
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    new = [
        r
        for r in results
        if not is_baselined(
            "trufflehog",
            r.get("DetectorName", ""),
            (r.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}) or {}).get(
                "file", ""
            ),
            baseline,
        )
    ]
    return {"total": len(results), "new": len(new), "baselined": len(results) - len(new)}


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <report_dir> <summary_file>", file=sys.stderr)
        sys.exit(1)

    report_dir = sys.argv[1]
    summary_file = sys.argv[2]
    repo_dir = os.environ.get("REPO_DIR", str(Path(__file__).parent.parent))

    baseline = load_baseline(repo_dir)
    if not baseline:
        print("No baseline found, skipping filter")
        sys.exit(0)

    results = {
        "bandit": filter_bandit(report_dir, baseline),
        "semgrep": filter_semgrep(report_dir, baseline),
        "shellcheck": filter_shellcheck(report_dir, baseline),
        "hadolint": filter_hadolint(report_dir, baseline),
        "trufflehog": filter_trufflehog(report_dir, baseline),
    }

    filtered_path = os.path.join(report_dir, "baseline_filtered.json")
    with open(filtered_path, "w") as f:
        json.dump(results, f, indent=2)

    total_new = sum(r["new"] for r in results.values())
    total_baselined = sum(r["baselined"] for r in results.values())

    if os.path.exists(summary_file):
        with open(summary_file, "a") as f:
            f.write(f"\nBaseline Filter ({len(baseline)} accepted rules):\n")
            for scanner, counts in results.items():
                if counts["total"] > 0:
                    f.write(
                        f"  {scanner}: {counts['new']} new, "
                        f"{counts['baselined']} baselined "
                        f"(of {counts['total']} total)\n"
                    )
            f.write(f"\nNew findings: {total_new}\n")
            if total_new == 0 and total_baselined > 0:
                f.write("Effective status: PASS (all findings baselined)\n")

    print(f"BASELINE_NEW={total_new}")
    print(f"BASELINE_ACCEPTED={total_baselined}")

    sys.exit(0 if total_new == 0 else 1)


if __name__ == "__main__":
    main()
