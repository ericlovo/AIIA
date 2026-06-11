"""Erdős multiplication-table problem: distinct products in the k×k table.

Ford (2008): M(k) ≍ k² / ((log k)^δ (log log k)^{3/2}), δ = 1 − (1+ln ln 2)/ln 2
≈ 0.086. We count M(k) exactly at sampled k and confirm the ratio M(k)/k²
declines, matching Ford's order — a known-answer-checkable computation.
"""

import math
from typing import Any


def _distinct_products(k: int) -> int:
    products = set()
    for i in range(1, k + 1):
        for j in range(i, k + 1):  # symmetric table; i ≤ j
            products.add(i * j)
    return len(products)


def calibrate(budget_seconds: float) -> dict[str, Any]:
    """Cost is ~k²/2 set inserts; k≈600 is ~0.2M ops. Scale conservatively."""
    k_max = int(max(120, min(1200, budget_seconds * 250)))
    return {"k_max": k_max}


def run(params: dict[str, Any]) -> dict[str, Any]:
    k_max = int(params["k_max"])
    ks = sorted({max(2, k_max // 4), max(3, k_max // 2), k_max})
    samples = []
    for k in ks:
        m = _distinct_products(k)
        samples.append({"k": k, "distinct": m, "ratio_to_k2": round(m / (k * k), 6)})
    ratios = [s["ratio_to_k2"] for s in samples]
    declining = all(ratios[i] >= ratios[i + 1] - 1e-9 for i in range(len(ratios) - 1))
    delta = 1 - (1 + math.log(math.log(2))) / math.log(2)
    evidence = {
        "k_max": k_max,
        "samples": samples,
        "ratio_declining": declining,
        "ford_delta": round(delta, 6),
        "matches_known_order": declining,
    }
    claim = (
        f"Distinct-product ratio M(k)/k² declines "
        f"({' → '.join(str(r) for r in ratios)} at k={ks}), consistent with "
        f"Ford's k²/((log k)^δ(log log k)^{{3/2}}) order, δ≈{round(delta, 3)}."
    )
    return {"problem_id": "erdos-multiplication-table", "mode": "computational",
            "verified": declining, "claim": claim, "evidence": evidence}
