"""Tests for the Erdős problem loop — registry, harness determinism, arbiter rule.

Run: pytest local_brain/problem_loops/test_problem_loops.py
These are pure/offline (no LLM, no network) — they exercise the computational lane
and the verification contract that gates everything else.
"""

import json

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
    chosen = {pick_problem(reg, offline_harnesses=set(H.HARNESSES))["id"] for _ in range(50)}
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


# ── literature lane (fake LLM + fake fetcher — no SDK, no network) ──

import asyncio

from local_brain.problem_loops import llm_lane


class FakeLLM:
    """Returns scripted responses in order; records prompts it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def complete(self, prompt, budget_usd):
        self.prompts.append(prompt)
        return {"text": self.responses.pop(0), "cost_usd": 0.01}


def _prover_json(status="open", urls=None):
    urls = urls or ["https://example.org/a", "https://other.net/b"]
    cites = [
        {
            "title": f"src{i}",
            "url": u,
            "quote": f"the problem remains {status} as of today {i}",
            "supports": "status",
        }
        for i, u in enumerate(urls)
    ]
    return (
        "```json\n"
        + json.dumps(
            {
                "status": status,
                "confidence": "high",
                "frontier_summary": "Best known result is X (2024).",
                "citations": cites,
                "proposed_registry_update": None,
            }
        )
        + "\n```"
    )


def _refuter_json(status="open", agrees=True):
    return (
        "```json\n"
        + json.dumps(
            {
                "status": status,
                "agrees_with_prover": agrees,
                "contradicting_evidence": None,
                "citations": [
                    {
                        "title": "r",
                        "url": "https://third.edu/c",
                        "quote": "independent confirmation",
                        "supports": "status",
                    }
                ],
            }
        )
        + "\n```"
    )


def _fake_fetch_factory(pages):
    def fetch(url):
        if url in pages:
            return pages[url]
        raise ConnectionError("no such page")

    return fetch


def _problem():
    reg = load_registry()
    return reg["_by_id"]["erdos-turan-additive-bases"]


def _run_lit(llm, fetch):
    return asyncio.run(llm_lane.literature_cycle(_problem(), llm, 1.0, fetch=fetch))


def test_literature_happy_path_promotes():
    pages = {
        "https://example.org/a": "<html>... the problem remains open as of today 0 ...</html>",
        "https://other.net/b": "<p>and the problem remains open as of today 1</p>",
    }
    llm = FakeLLM([_prover_json("open"), _refuter_json("open")])
    rec = _run_lit(llm, _fake_fetch_factory(pages))
    assert rec["promoted"] is True
    assert rec["arbiter"]["verified_citation_domains"] == 2
    assert "claim" in rec
    # refuter prompt must not leak the prover's sources
    assert "example.org" not in llm.prompts[1]


def test_literature_unverified_quote_blocks_promotion():
    pages = {
        "https://example.org/a": "totally different text",
        "https://other.net/b": "also unrelated",
    }
    llm = FakeLLM([_prover_json("open"), _refuter_json("open")])
    rec = _run_lit(llm, _fake_fetch_factory(pages))
    assert rec["promoted"] is False
    assert rec["arbiter"]["citations_ok"] is False


def test_literature_refuter_disagreement_blocks_promotion():
    pages = {
        "https://example.org/a": "the problem remains open as of today 0",
        "https://other.net/b": "the problem remains open as of today 1",
    }
    llm = FakeLLM([_prover_json("open"), _refuter_json("solved", agrees=False)])
    rec = _run_lit(llm, _fake_fetch_factory(pages))
    assert rec["promoted"] is False
    assert rec["arbiter"]["refuter_agrees"] is False


def test_literature_same_domain_citations_insufficient():
    urls = ["https://example.org/a", "https://example.org/b"]
    pages = {
        "https://example.org/a": "the problem remains open as of today 0",
        "https://example.org/b": "the problem remains open as of today 1",
    }
    llm = FakeLLM([_prover_json("open", urls=urls), _refuter_json("open")])
    rec = _run_lit(llm, _fake_fetch_factory(pages))
    assert rec["promoted"] is False, "two quotes from one domain must not count as independent"


def test_literature_unparseable_prover_fails_closed():
    llm = FakeLLM(["I think it is probably open but here is no JSON.", _refuter_json()])
    rec = _run_lit(llm, _fake_fetch_factory({}))
    assert rec["promoted"] is False
    assert rec["failure"] == "prover output unparseable"


def test_parse_json_block_variants():
    obj = {"status": "open"}
    fenced = "preamble\n```json\n" + json.dumps(obj) + "\n```\ntrailing"
    assert llm_lane.parse_json_block(fenced) == obj
    bare = 'noise before {"status": "open"}'
    assert llm_lane.parse_json_block(bare) == obj
    assert llm_lane.parse_json_block("no json here") is None
