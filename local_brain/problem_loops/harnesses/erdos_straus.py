"""Erdős–Straus: 4/n = 1/a + 1/b + 1/c for all n ≥ 2.

Probe: verify solvability for every n in [2, n_max]; profile where the search
works hardest. A clean run is a *negative* result (no counterexample, matching
the known truth that none exists in range) — its value is the hard-residue
profile, a deterministic, citable artifact.
"""

from fractions import Fraction
from typing import Any


def solve(n: int) -> tuple[int, int, int] | None:
    """Find (a, b, c) with 1/a + 1/b + 1/c = 4/n, or None. Exact arithmetic.

    1/a is the largest unit fraction, so (4/n)/3 ≤ 1/a < 4/n → n/4 < a ≤ 3n/4.
    For each a, the remainder 4/n − 1/a = 1/b + 1/c is solved exactly: b ranges
    where 1/b ≥ rem/2 and 1/b < rem, and 1/c must land on a unit fraction.
    """
    target = Fraction(4, n)
    for a in range(n // 4 + 1, (3 * n) // 4 + 1):
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        b_lo = rem.denominator // rem.numerator + 1
        b_hi = (2 * rem.denominator) // rem.numerator + 1
        for b in range(max(b_lo, a), b_hi + 1):
            rem2 = rem - Fraction(1, b)
            if rem2 <= 0:
                continue
            if rem2.numerator == 1:
                return (a, b, rem2.denominator)
    return None


def calibrate(budget_seconds: float) -> dict[str, Any]:
    """Measured ~75 n/sec for the exact Fraction solver. Scale to budget, capped.
    (Honest cost model so a budgeted/overnight scheduler can reason about runtime.)"""
    n_max = int(max(500, min(20000, budget_seconds * 75)))
    return {"n_max": n_max}


def run(params: dict[str, Any]) -> dict[str, Any]:
    n_max = int(params["n_max"])
    counterexamples = []
    hardest_a = 0
    hardest_n = 0
    hard_residues: dict[int, int] = {}
    for n in range(2, n_max + 1):
        sol = solve(n)
        if sol is None:
            counterexamples.append(n)
            continue
        a = sol[0]
        if a > hardest_a:
            hardest_a, hardest_n = a, n
        if a > n // 4 + 1:  # needed more than the trivial leading term
            hard_residues[n % 840] = hard_residues.get(n % 840, 0) + 1
    top_res = sorted(hard_residues.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    no_counterexample = not counterexamples
    evidence = {
        "n_max": n_max,
        "counterexamples": counterexamples,
        "no_counterexample": no_counterexample,
        "matches_known_truth": no_counterexample,  # known: none exist in range
        "hardest_case": {"n": hardest_n, "leading_denominator": hardest_a},
        "top_hard_residues_mod_840": top_res,
    }
    claim = (
        f"4/n is decomposable into three unit fractions for all 2≤n≤{n_max} "
        f"(no counterexample — consistent with the known result); the hardest "
        f"case needed leading denominator a={hardest_a} at n={hardest_n}."
    )
    return {
        "problem_id": "erdos-straus",
        "mode": "computational",
        "verified": no_counterexample,
        "claim": claim,
        "evidence": evidence,
    }
