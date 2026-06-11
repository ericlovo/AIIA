"""Problem registry loading and selection."""

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger("aiia.problem_loops")

REGISTRY_PATH = Path(__file__).parent / "registry.json"


def load_registry() -> dict[str, Any]:
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    by_id = {p["id"]: p for p in data["problems"]}
    if len(by_id) != len(data["problems"]):
        raise ValueError("Duplicate problem ids in registry")
    data["_by_id"] = by_id
    return data


def pick_problem(
    registry: dict[str, Any],
    mode: str | None = None,
    offline_harnesses: set[str] | None = None,
    recent_ids: list[str] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Pick a problem, weighted by computational_surface, skipping recents.

    Args:
        mode: restrict to problems supporting this mode.
        offline_harnesses: if given, restrict to problems with a built-in
            harness (offline/computational lane).
        recent_ids: problems attempted recently (cooldown) — deprioritized,
            not excluded, so a small registry never deadlocks.
    """
    rng = rng or random.Random()
    recent = set(recent_ids or [])
    candidates = []
    for p in registry["problems"]:
        if mode and mode not in p["modes"]:
            continue
        if offline_harnesses is not None and p["id"] not in offline_harnesses:
            continue
        weight = 0.05 + p.get("computational_surface", 0.0)
        if p["id"] in recent:
            weight *= 0.15
        candidates.append((weight, p))
    if not candidates:
        raise ValueError(f"No problems match mode={mode!r} with available harnesses")
    total = sum(w for w, _ in candidates)
    roll = rng.uniform(0, total)
    acc = 0.0
    for w, p in candidates:
        acc += w
        if roll <= acc:
            return p
    return candidates[-1][1]
