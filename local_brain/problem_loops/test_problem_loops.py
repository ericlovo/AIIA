"""Tests for the Erdős problem loop — registry, harness determinism, arbiter rule.

Run: pytest local_brain/problem_loops/test_problem_loops.py
These are pure/offline (no LLM, no network) — they exercise the computational lane
and the verification contract that gates everything else.
"""

import json
from pathlib import Path

import pytest

from local_brain.problem_loops import harnesses as H
from local_brain.problem_loops.registry import load_registry, pick_problem


def test_registry_valid_and_unique():
    reg = load_registry()
    assert reg["problems"]
    ids = [p["id"] for p in reg["problems"]]
    assert len(ids) == len(set(ids)), "duplicate problem ids"
    for p in reg["problems"]:
        assert p["modes"], f"{p['id']} has no modes"
        assert 0.0 <= p["computational_surface"] <= 1.0
        assert p["status"] in {"open", "partial", "solved", "disproved", "unknown"}


def test_picker_respects_available_harnesses():
    reg = load_registry()
    chosen = {pick_problem(reg, offline_harnesses=set(H.HARNESSES))["id"]
              for _ in range(50)}
    assert chosen, "picker returned nothing"
    assert chosen.issubset(set(H.HARNESSES)), "picker chose a problem with no harness"


@pytest.mark.parametrize("pid", list(H.HARNESSES))
def test_harness_is_deterministic(pid):
    """The verification contract: identical params → identical result_hash."""
    params = H.calibrate_harness(pid, budget_seconds=1.0)
    r1 = H.run_harness(pid, params)
    r2 = H.run_harness(pid, params)
    assert r1["result_hash"] == r2["result_hash"], f"{pid} non-deterministic"
    assert r1["result_hash"] == H.evidence_hash(r1["evidence"])


@pytest.mark.parametrize("pid", list(H.HARNESSES))
def test_harness_shape(pid):
    params = H.calibrate_harness(pid, budget_seconds=1.0)
    r = H.run_harness(pid, params)
    for key in ("problem_id", "mode", "verified", "claim", "evidence", "result_hash"):
        assert key in r, f"{pid} missing {key}"
    assert r["problem_id"] == pid
    assert isinstance(r["verified"], bool)
    assert r["claim"]


def test_minimum_overlap_known_values():
    """Exact small-n optima must match the known minimum-overlap sequence."""
    r = H.run_harness("erdos-minimum-overlap", {"n_max": 9})
    got = {v["n"]: v["M"] for v in r["evidence"]["values"]}
    # M(2..9): the established exact optima.
    assert got == {2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4}


def test_calibrate_budget_is_honest():
    """A smaller budget must not yield a larger workload (monotone)."""
    for pid in H.HARNESSES:
        small = H.calibrate_harness(pid, 0.5)
        big = H.calibrate_harness(pid, 8.0)
        # the single scalar that scales each harness should not shrink with budget
        skey = next(iter(small))
        assert big[skey] >= small[skey], f"{pid} budget scaling inverted"
