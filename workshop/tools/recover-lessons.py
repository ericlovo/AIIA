#!/usr/bin/env python3
"""Recover knowledge lessons corrupted by the string-evidence bug.

`knowledge_writeback` in scripts/phase-state.py iterates a string `evidence`
value one character at a time (see workshop/findings/04). The damage is
mechanical and therefore mostly reversible: the cited-evidence prose can be
recovered by concatenating the single-character bullets back together.

What is NOT recoverable: the `## TL;DR` body. It was empty at write time,
because the candidate carried no `statement`. Nothing in the file holds it.
This tool reports that gap rather than inventing text for it.

READ-ONLY by default. Pass --write to modify files.

Usage:
    python3 workshop/tools/recover-lessons.py --dir $WRIT_DIR/.writ/knowledge
    python3 workshop/tools/recover-lessons.py --dir ... --write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SINGLE_CHAR_BULLET = re.compile(r"^- (.)$")
YAML_CHAR_ITEM = re.compile(r"^  - (.)$")


def recover_block(lines: list[str], start: int, pattern: re.Pattern) -> tuple[str, int, int]:
    """Concatenate a run of single-character list items starting at `start`.
    Returns (recovered_text, first_index, last_index_exclusive)."""
    chars: list[str] = []
    i = start
    while i < len(lines):
        m = pattern.match(lines[i])
        if not m:
            break
        chars.append(m.group(1))
        i += 1
    return "".join(chars), start, i


def analyze(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # --- corrupted related_artifacts (YAML, two-space indent) ---
    yaml_recovered = None
    yaml_span = None
    for i, ln in enumerate(lines):
        if ln.startswith("related_artifacts:") and i + 1 < len(lines):
            if YAML_CHAR_ITEM.match(lines[i + 1]):
                yaml_recovered, s, e = recover_block(lines, i + 1, YAML_CHAR_ITEM)
                yaml_span = (s, e)
            break

    # --- corrupted cited-evidence bullets ---
    ev_recovered = None
    ev_span = None
    for i, ln in enumerate(lines):
        if ln.strip() == "**Cited evidence:**":
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and SINGLE_CHAR_BULLET.match(lines[j]):
                ev_recovered, s, e = recover_block(lines, j, SINGLE_CHAR_BULLET)
                ev_span = (s, e)
            break

    # --- empty TL;DR ---
    tldr_empty = False
    if "## TL;DR" in lines:
        t = lines.index("## TL;DR")
        body = []
        for ln in lines[t + 1:]:
            if ln.startswith("## "):
                break
            body.append(ln)
        tldr_empty = not "".join(body).strip()

    if not (ev_recovered or yaml_recovered or tldr_empty):
        return None

    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "")
    return {
        "path": path, "lines": lines, "title": title,
        "evidence": ev_recovered, "ev_span": ev_span,
        "yaml": yaml_recovered, "yaml_span": yaml_span,
        "tldr_empty": tldr_empty,
    }


def path_tokens(prose: str) -> list[str]:
    """Extract real repo-path tokens from recovered evidence prose.

    The corrupted `related_artifacts` block is NOT a usable source — it holds
    the path-filtered *characters*, which rejoin into nonsense like `.///../`.
    The recovered evidence prose is the real source, so both the YAML block
    and the `## Related` list are rebuilt from it.
    """
    out: list[str] = []
    for raw in re.split(r"[;,\s]+", prose or ""):
        t = raw.strip().rstrip(".,;:)").lstrip("(")
        if "/" not in t or not any(c.isalnum() for c in t):
            continue
        t = t.split(":")[0]  # drop line-number suffixes (file.py:527)
        if t and t not in out:
            out.append(t)
    return out


def repair(info: dict) -> str:
    lines = list(info["lines"])
    toks = path_tokens(info["evidence"] or "")

    # Bottom-up so earlier spans keep their indices.
    edits: list[tuple[tuple[int, int], list[str]]] = []
    if info["evidence"] and info["ev_span"]:
        edits.append((info["ev_span"], [f"- {info['evidence']}"]))
    if info["yaml_span"]:
        edits.append((info["yaml_span"], [f"  - {t}" for t in toks]))
    for (s, e), repl in sorted(edits, key=lambda x: -x[0][0]):
        lines[s:e] = repl

    # Rebuild the `## Related` list, which carries the same per-character damage.
    if "## Related" in lines:
        r = lines.index("## Related")
        end = r + 1
        while end < len(lines) and (
            not lines[end].strip() or SINGLE_CHAR_BULLET.match(lines[end])
            or lines[end].startswith("- `")
        ):
            end += 1
        lines[r + 1:end] = [""] + ([f"- `{t}`" for t in toks] if toks else [])

    # An empty related_artifacts must be an explicit [], not a dangling key.
    for i, ln in enumerate(lines):
        if ln.rstrip() == "related_artifacts:" and (
            i + 1 >= len(lines) or not lines[i + 1].startswith("  - ")
        ):
            lines[i] = "related_artifacts: []"
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="path to .writ/knowledge")
    ap.add_argument("--write", action="store_true", help="apply repairs (default: dry run)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    damaged = [i for i in (analyze(p) for p in sorted(root.glob("*/*.md"))) if i]
    if not damaged:
        print("No corrupted lessons found.")
        return 0

    print(f"{len(damaged)} corrupted lesson(s) in {root}\n")
    lost_tldr = 0
    for info in damaged:
        print(f"── {info['path'].name}")
        print(f"   title:     {info['title']}")
        if info["evidence"]:
            print(f"   RECOVERED: {info['evidence']}")
        if info["tldr_empty"]:
            lost_tldr += 1
            print("   TL;DR:     EMPTY — unrecoverable, needs a human")
        if args.write:
            info["path"].write_text(repair(info), encoding="utf-8")
            print("   written.")
        print()

    print(f"{'Repaired' if args.write else 'Would repair'}: {len(damaged)} file(s)")
    print(f"Still needing a human-written TL;DR: {lost_tldr}")
    if not args.write:
        print("\nDry run — nothing was modified. Pass --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
