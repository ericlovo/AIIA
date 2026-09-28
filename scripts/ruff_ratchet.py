"""Fail when any ignored backlog rule exceeds its reviewed count ceiling."""

import json
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
EXEMPT_RULES = {"E501", "B008"}


def check(root=ROOT, runner=subprocess.run):
    try:
        baseline = json.loads((root / "scripts/ruff-baseline.json").read_text())
        config = tomllib.loads((root / "pyproject.toml").read_text())
        ignored = set(config["tool"]["ruff"]["lint"]["ignore"]) - EXEMPT_RULES
        if not isinstance(baseline, dict) or any(
            type(value) is not int or value < 0 for value in baseline.values()
        ):
            raise ValueError("baseline must map rule codes to nonnegative integer counts")
        if set(baseline) != ignored:
            raise ValueError("baseline rules must match ignored rules except E501 and B008")
        if not baseline:
            print("Ruff ratchet: no ignored backlog rules")
            return 0
        result = runner(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "local_brain/",
                "--select",
                ",".join(sorted(baseline)),
                "--ignore",
                ",".join(sorted(EXEMPT_RULES)),
                "--statistics",
                "--output-format",
                "json",
                "--no-fix",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise ValueError(
                f"ruff failed with exit code {result.returncode}: {result.stderr.strip()}"
            )
        rows = json.loads(result.stdout)
        if not isinstance(rows, list):
            raise ValueError("ruff statistics must be a list")
        counts = dict.fromkeys(baseline, 0)
        seen = set()
        for row in rows:
            code, count = row["code"], row["count"]
            if code not in baseline or code in seen or type(count) is not int or count < 0:
                raise ValueError("unexpected rule or invalid count in ruff statistics")
            seen.add(code)
            counts[code] = count
        if result.returncode == 1 and not any(counts.values()):
            raise ValueError("ruff failed without reporting violations")
        exceeded = False
        for code in sorted(baseline):
            count, ceiling = counts[code], baseline[code]
            print(f"{code}: {count}/{ceiling}")
            exceeded |= count > ceiling
        print("Ruff ratchet: ceiling exceeded" if exceeded else "Ruff ratchet: passed")
        return int(exceeded)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Ruff ratchet error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(check())
