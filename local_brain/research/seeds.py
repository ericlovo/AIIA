"""
Seed research topics — the curated set the nightly loop keeps alive.

`ensure_seed_topics(store)` is idempotent: it creates any missing seed
topic and leaves existing ones untouched (matching on the same title the
factory uses), so the runner can call it every cycle. Adding a new problem
here is all it takes to put it in the nightly rotation.

The Erdős entries below were each verified against their erdosproblems.com
page before being committed — we don't seed a number we couldn't confirm
maps to a real problem (the erdos profile's whole point is sourced claims).
"""

from dataclasses import dataclass

from local_brain.research.erdos import create_erdos_topic, find_erdos_topic
from local_brain.research.topic import ResearchTopic, TopicStore


@dataclass(frozen=True)
class ErdosSeed:
    number: int
    name: str  # short human label — for logs/reporting, not seeded into the topic
    note: str  # one-line domain/status context


# Verified against erdosproblems.com — a spread across number theory,
# combinatorial geometry, and discrepancy/analysis (one solved, for the
# status-report exercise).
ERDOS_SEEDS: list[ErdosSeed] = [
    ErdosSeed(
        number=28,
        name="Erdős–Turán conjecture on additive bases",
        note="number theory — open; representation function of an order-2 basis is unbounded",
    ),
    ErdosSeed(
        number=107,
        name="Erdős–Szekeres convex polygon (happy ending)",
        note="combinatorial geometry — open; f(n) = 2^(n-2) + 1 conjectured",
    ),
    ErdosSeed(
        number=67,
        name="Erdős discrepancy problem",
        note="discrepancy/analysis — solved (Tao, 2015); good open→solved status exercise",
    ),
]


def ensure_seed_topics(store: TopicStore) -> list[ResearchTopic]:
    """
    Create any missing seed topics; return the ones created this call.

    Idempotent — existing topics (matched by the factory's title) are left
    as-is, so this is safe to call at the start of every research cycle.
    """
    created: list[ResearchTopic] = []
    for seed in ERDOS_SEEDS:
        if find_erdos_topic(store, seed.number) is None:
            created.append(create_erdos_topic(store, seed.number))
    return created
