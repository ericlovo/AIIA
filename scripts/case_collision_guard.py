"""Fail when tracked files would collide on a case-insensitive filesystem.

Linux CI is case-sensitive, so `MorningNote.tsx` and `morningNote.ts` can
both exist and `tsc` still succeeds. macOS APFS folds those names together
and TypeScript reports TS1261 / TS2724. This guard walks `git ls-files`
and fails on:

- two paths that differ only by case
- two TS/JS module files in the same directory whose stems differ only by case
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs"})


def tracked_paths(root: Path, runner=subprocess.run) -> list[str]:
    result = runner(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode().strip() or "git ls-files failed")
    return [raw.decode() for raw in result.stdout.split(b"\0") if raw]


def find_collisions(paths: list[str]) -> list[tuple[str, list[str]]]:
    by_path: dict[str, list[str]] = defaultdict(list)
    by_stem: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel in paths:
        posix = rel.replace("\\", "/")
        by_path[posix.lower()].append(posix)
        path = Path(posix)
        if path.suffix.lower() in MODULE_SUFFIXES:
            parent = path.parent.as_posix().lower()
            by_stem[(parent, path.stem.lower())].append(posix)

    found: list[tuple[str, list[str]]] = []
    for group in by_path.values():
        unique = sorted(set(group))
        if len(unique) > 1:
            found.append(("path", unique))
    seen_path_groups = {tuple(unique) for kind, unique in found if kind == "path"}
    for group in by_stem.values():
        unique = sorted(set(group))
        if len(unique) > 1 and tuple(unique) not in seen_path_groups:
            found.append(("module", unique))
    return found


def format_collisions(found: list[tuple[str, list[str]]]) -> str:
    lines = []
    for kind, files in found:
        label = (
            "case-insensitive path collision"
            if kind == "path"
            else "case-insensitive module stem collision"
        )
        lines.append(f"{label}:")
        lines.extend(f"  {name}" for name in files)
    return "\n".join(lines)


def check(root: Path = ROOT, runner=subprocess.run) -> int:
    try:
        found = find_collisions(tracked_paths(root, runner=runner))
    except RuntimeError as exc:
        print(f"Case-collision guard failed: {exc}", file=sys.stderr)
        return 2
    if found:
        print(format_collisions(found), file=sys.stderr)
        print(
            "::error::Case-insensitive path collision guard failed.",
            file=sys.stderr,
        )
        return 1
    print("Case-collision guard: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
