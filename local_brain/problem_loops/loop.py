"""AIIA Problem Loop — pick → attempt → verify → record.

A sibling of story_runner pointed at open mathematics. This module implements
the **computational lane fully** (offline, no LLM, no API key): the prover runs
a deterministic harness, the refuter RE-EXECUTES it in a fresh subprocess and
the result hashes must match, the arbiter promotes the finding to AIIA memory
only if it both verified and reproduced. The LLM lanes (literature / proof_search
/ formalization) are declared but gated — they require claude_agent_sdk + an API
key and are wired in a later pass; see ARCHITECTURE.md.

Run offline now:
    python -m local_brain.problem_loops.loop --offline --budget 2 --memory-dir /tmp/aiia-mem
    python -m local_brain.problem_loops.loop --offline --problem erdos-straus --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from local_brain.problem_loops.harnesses import (
    HARNESSES,
    calibrate_harness,
    run_harness,
)
from local_brain.problem_loops.registry import load_registry, pick_problem

logger = logging.getLogger("aiia.problem_loops")

# Memory categories per verification rung (see ARCHITECTURE.md ladder).
RUNG_CATEGORY = {
    "status": "lessons",
    "computational": "lessons",
    "partial": "project",
    "verified_proof": "decisions",
    "conjecture": "project",
}


def _refuter_subprocess(problem_id: str, params: dict[str, Any]) -> str | None:
    """Re-execute the harness in a FRESH process; return its result_hash.

    This is the real determinism check: a separate interpreter, no shared state.
    Returns None if the subprocess fails (which the arbiter treats as 'not
    reproduced' — refusal to promote).
    """
    code = (
        "import json,sys;"
        "from local_brain.problem_loops.harnesses import run_harness;"
        "p=json.loads(sys.argv[1]);"
        "r=run_harness(sys.argv[2],p);"
        "print(r['result_hash'])"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code, json.dumps(params), problem_id],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).resolve().parents[2]),  # AIIA/
        )
        if out.returncode != 0:
            logger.warning(f"[refuter] subprocess failed: {out.stderr[-300:]}")
            return None
        return out.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("[refuter] subprocess timed out")
        return None


def computational_cycle(
    problem: dict[str, Any],
    budget_seconds: float,
    memory: Any | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One prover → refuter → arbiter cycle on the offline computational lane."""
    pid = problem["id"]
    if pid not in HARNESSES:
        raise KeyError(f"No computational harness for {pid!r}")

    # PROVER — run the deterministic probe.
    params = calibrate_harness(pid, budget_seconds)
    t0 = time.time()
    prover = run_harness(pid, params)
    elapsed = round(time.time() - t0, 2)

    # REFUTER — fresh-process re-execution; hashes must match.
    refuter_hash = _refuter_subprocess(pid, params)
    reproduced = refuter_hash is not None and refuter_hash == prover["result_hash"]

    # ARBITER — promote only if verified AND reproduced.
    promoted = bool(prover["verified"] and reproduced)
    rung = "computational"
    category = RUNG_CATEGORY[rung]

    record = {
        "problem_id": pid,
        "mode": "computational",
        "rung_reached": rung,
        "verified": prover["verified"],
        "reproduced": reproduced,
        "promoted": promoted,
        "result_hash": prover["result_hash"],
        "refuter_hash": refuter_hash,
        "params": params,
        "elapsed_s": elapsed,
        "claim": prover["claim"],
        "evidence": prover["evidence"],
    }

    if promoted and memory is not None and not dry_run:
        memory.remember(
            fact=prover["claim"],
            category=category,
            source=f"erdos-loop:{pid}",
            metadata={
                "problem_id": pid,
                "mode": "computational",
                "rung_reached": rung,
                "result_hash": prover["result_hash"],
                "params": params,
                "evidence": prover["evidence"],
            },
        )
        record["memory_category"] = category
    elif not promoted:
        logger.warning(
            f"[arbiter] NOT promoting {pid}: verified={prover['verified']} "
            f"reproduced={reproduced} — nothing logged (the iron rule)."
        )

    return record


# ── LLM lanes ───────────────────────────────────────────────────


def literature_run(
    problem: dict[str, Any],
    budget_usd: float,
    memory: Any | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the literature lane (async cycle under the hood) and record it."""
    import asyncio

    from local_brain.problem_loops.llm_lane import SDKClient, literature_cycle

    llm = SDKClient()  # raises with a clear message if SDK/key unavailable
    record = asyncio.run(literature_cycle(problem, llm, budget_usd=budget_usd))

    if record["promoted"] and memory is not None and not dry_run:
        category = RUNG_CATEGORY[record["rung_reached"]]
        memory.remember(
            fact=record["claim"],
            category=category,
            source=f"erdos-loop:{record['problem_id']}",
            metadata={
                "problem_id": record["problem_id"],
                "mode": "literature",
                "rung_reached": record["rung_reached"],
                "arbiter": record["arbiter"],
                "citations": record.get("citations_checked"),
                "proposed_registry_update": record.get("proposed_registry_update"),
                "cost_usd": record["cost_usd"],
            },
        )
        record["memory_category"] = category
    return record


def llm_cycle(problem: dict[str, Any], mode: str, **kwargs) -> dict[str, Any]:
    if mode == "literature":
        return literature_run(problem, **kwargs)
    raise NotImplementedError(
        f"The {mode!r} lane is a later pass (proof_search needs the adversarial "
        f"refuter prompts; formalization needs the Lean bridge). literature and "
        f"the offline computational lane are live; see ARCHITECTURE.md."
    )


def _get_memory(memory_dir: str | None):
    if not memory_dir:
        return None
    from local_brain.eq_brain.memory import Memory

    return Memory(memory_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="AIIA Erdős problem loop")
    ap.add_argument(
        "--offline", action="store_true", help="Computational lane only (no LLM, no API key)."
    )
    ap.add_argument(
        "--mode",
        choices=["computational", "literature"],
        help="Lane to run (default: computational when --offline).",
    )
    ap.add_argument("--budget-usd", type=float, default=1.0, help="LLM spend cap for LLM lanes.")
    ap.add_argument("--problem", help="Specific problem id; else picker chooses.")
    ap.add_argument("--budget", type=float, default=2.0, help="Seconds per harness.")
    ap.add_argument("--memory-dir", help="AIIA memory data dir (omit to not persist).")
    ap.add_argument("--dry-run", action="store_true", help="Compute + verify, don't persist.")
    args = ap.parse_args()

    registry = load_registry()
    if args.problem:
        problem = registry["_by_id"].get(args.problem)
        if not problem:
            ap.error(f"Unknown problem id: {args.problem}")
    else:
        problem = pick_problem(registry, offline_harnesses=set(HARNESSES))

    mode = args.mode or ("computational" if args.offline else None)
    if mode is None:
        ap.error("Pass --offline for the computational lane or --mode literature.")

    memory = _get_memory(args.memory_dir) if not args.dry_run else None
    if mode == "computational":
        record = computational_cycle(problem, args.budget, memory, dry_run=args.dry_run)
    else:
        record = llm_cycle(
            problem, mode, budget_usd=args.budget_usd, memory=memory, dry_run=args.dry_run
        )

    print("\n=== problem loop cycle ===")
    print(f"problem   : {record['problem_id']} ({record['mode']})")
    if "verified" in record:
        print(f"verified  : {record['verified']}   reproduced: {record['reproduced']}")
    print(
        f"promoted  : {record['promoted']}"
        + (
            f" → memory[{record.get('memory_category')}]"
            if record["promoted"] and not args.dry_run
            else ""
        )
    )
    if "result_hash" in record:
        print(
            f"hash      : {record['result_hash'][:16]}  (refuter: "
            f"{(record['refuter_hash'] or 'FAILED')[:16]})"
        )
        print(f"elapsed   : {record['elapsed_s']}s")
    if "arbiter" in record:
        print(f"arbiter   : {record['arbiter']}")
    if "cost_usd" in record:
        print(f"llm cost  : ${record['cost_usd']:.4f}")
    print(f"\nFINDING: {record.get('claim', '(not promoted — nothing recorded)')}\n")
    return 0 if record["promoted"] else 1


if __name__ == "__main__":
    sys.exit(main())
