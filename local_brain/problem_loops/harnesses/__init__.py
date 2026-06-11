"""Built-in computational harnesses — the offline lane of the problem loop.

Contract: every harness exposes

    calibrate(budget_seconds: float) -> dict   # pick params that fit the budget
    run(params: dict) -> dict                  # DETERMINISTIC given params

run() returns:
    {
      "claim": str,            # one-sentence finding
      "evidence": dict,        # structured, deterministic (no timings inside)
      "params": dict,          # echo of inputs
      "result_hash": str,      # sha256 over canonical evidence JSON
    }

Determinism is the verification contract: the refuter re-executes run() with
identical params in a fresh process and the result_hash must match exactly.
Time budgets influence ONLY calibrate(), never run().
"""

import hashlib
import json
from typing import Any

from local_brain.problem_loops.harnesses import (
    erdos_straus,
    minimum_overlap,
    multiplication_table,
)

HARNESSES = {
    "erdos-straus": erdos_straus,
    "erdos-multiplication-table": multiplication_table,
    "erdos-minimum-overlap": minimum_overlap,
}


def evidence_hash(evidence: dict[str, Any]) -> str:
    """Canonical hash of the deterministic evidence core."""
    blob = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def run_harness(problem_id: str, params: dict[str, Any]) -> dict[str, Any]:
    mod = HARNESSES[problem_id]
    result = mod.run(params)
    # Hash is computed here, centrally, so no harness can forget it.
    result["result_hash"] = evidence_hash(result["evidence"])
    result["params"] = params
    return result


def calibrate_harness(problem_id: str, budget_seconds: float) -> dict[str, Any]:
    return HARNESSES[problem_id].calibrate(budget_seconds)
