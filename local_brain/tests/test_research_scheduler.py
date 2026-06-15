"""Offline tests for the autonomous research scheduler + seed topics.

No Ollama, ChromaDB, or network: a FakeEngine stands in for ResearchEngine
(an async generator over event dicts) and a real TopicStore persists to
tmp_path.
"""

from __future__ import annotations

import pytest

from local_brain.autonomy.research_loop import ResearchScheduler
from local_brain.config import AutonomyConfig
from local_brain.research.seeds import ERDOS_SEEDS, ensure_seed_topics
from local_brain.research.topic import TopicStore


class FakeEngine:
    """Records topics it was asked to run; emits a minimal event stream."""

    def __init__(self, *, fail_on: set[str] | None = None, error_event_on: set[str] | None = None):
        self.ran: list[str] = []
        self._fail_on = fail_on or set()
        self._error_event_on = error_event_on or set()

    async def run(self, topic):
        self.ran.append(topic.id)
        topic.run_count += 1  # mirror what the real engine persists
        if topic.id in self._fail_on:
            raise RuntimeError("boom")
        yield {"type": "meta", "session_id": "s1"}
        if topic.id in self._error_event_on:
            yield {"type": "error", "message": "engine error"}
            return
        yield {"type": "done", "answer": "ok", "iterations": 3, "tokens_used": 100}


def _enabled_config(**overrides) -> AutonomyConfig:
    cfg = AutonomyConfig(level="phase2", research_enabled=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def store(tmp_path):
    return TopicStore(data_dir=str(tmp_path))


class TestGate:
    @pytest.mark.asyncio
    async def test_disabled_by_default(self, store):
        sched = ResearchScheduler(AutonomyConfig(), FakeEngine(), store)
        assert sched.enabled is False
        result = await sched.run_cycle()
        assert result == {"skipped": True, "reason": "research_disabled"}

    @pytest.mark.asyncio
    async def test_phase2_without_research_flag_is_disabled(self, store):
        sched = ResearchScheduler(AutonomyConfig(level="phase2"), FakeEngine(), store)
        assert sched.enabled is False

    def test_enabled_when_phase2_and_flagged(self, store):
        sched = ResearchScheduler(_enabled_config(), FakeEngine(), store)
        assert sched.enabled is True


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_runs_sessions_across_active_topics(self, store):
        store.create(title="A", question="qa")
        store.create(title="B", question="qb")
        engine = FakeEngine()
        sched = ResearchScheduler(_enabled_config(), engine, store)

        result = await sched.run_cycle()
        assert result["topics_worked"] == 2
        assert result["sessions_run"] == 2
        assert result["errors"] == 0
        assert len(engine.ran) == 2

    @pytest.mark.asyncio
    async def test_topic_budget_caps_work(self, store):
        for i in range(5):
            store.create(title=f"T{i}", question="q")
        engine = FakeEngine()
        sched = ResearchScheduler(_enabled_config(research_max_topics_per_cycle=2), engine, store)

        result = await sched.run_cycle()
        assert result["topics_worked"] == 2
        assert len(engine.ran) == 2

    @pytest.mark.asyncio
    async def test_sessions_per_topic(self, store):
        store.create(title="A", question="q")
        engine = FakeEngine()
        sched = ResearchScheduler(
            _enabled_config(research_max_topics_per_cycle=1, research_sessions_per_topic=3),
            engine,
            store,
        )

        result = await sched.run_cycle()
        assert result["sessions_run"] == 3
        assert engine.ran == [engine.ran[0]] * 3  # same topic three times

    @pytest.mark.asyncio
    async def test_least_recently_run_first(self, store):
        fresh = store.create(title="Fresh", question="q")
        stale = store.create(title="Stale", question="q")
        # Fresh ran recently; stale never ran (last_run None sorts first).
        fresh.last_run = "2026-06-14T00:00:00Z"
        store.save(fresh)

        engine = FakeEngine()
        sched = ResearchScheduler(_enabled_config(research_max_topics_per_cycle=1), engine, store)
        await sched.run_cycle()
        assert engine.ran == [stale.id]

    @pytest.mark.asyncio
    async def test_skips_non_active_topics(self, store):
        active = store.create(title="Active", question="q")
        done = store.create(title="Done", question="q")
        done.status = "complete"
        store.save(done)

        engine = FakeEngine()
        sched = ResearchScheduler(_enabled_config(), engine, store)
        await sched.run_cycle()
        assert engine.ran == [active.id]

    @pytest.mark.asyncio
    async def test_no_active_topics_is_clean(self, store):
        engine = FakeEngine()
        sched = ResearchScheduler(_enabled_config(), engine, store)
        result = await sched.run_cycle()
        assert result["sessions_run"] == 0
        assert result["topics_worked"] == 0


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_engine_exception_is_counted_not_raised(self, store):
        a = store.create(title="A", question="q")
        b = store.create(title="B", question="q")
        engine = FakeEngine(fail_on={a.id})
        sched = ResearchScheduler(_enabled_config(), engine, store)

        result = await sched.run_cycle()
        # One topic failed, the other still ran — the cycle did not abort.
        assert result["errors"] == 1
        assert result["sessions_run"] == 1
        assert set(engine.ran) == {a.id, b.id}

    @pytest.mark.asyncio
    async def test_error_event_in_stream_is_counted(self, store):
        a = store.create(title="A", question="q")
        engine = FakeEngine(error_event_on={a.id})
        sched = ResearchScheduler(_enabled_config(), engine, store)

        result = await sched.run_cycle()
        assert result["errors"] == 1
        assert result["sessions_run"] == 0


class TestSeeds:
    def test_seeds_are_verified_erdos_numbers(self):
        numbers = {s.number for s in ERDOS_SEEDS}
        assert numbers == {28, 107, 67}

    def test_ensure_creates_then_is_idempotent(self, store):
        created = ensure_seed_topics(store)
        assert len(created) == len(ERDOS_SEEDS)
        # Every seed topic uses the erdos profile and is seeded with its page.
        for t in store.list_all():
            assert t.profile == "erdos"
            assert any("erdosproblems.com" in s for s in t.seeds)

        # Second call creates nothing.
        again = ensure_seed_topics(store)
        assert again == []
        assert len(store.list_all()) == len(ERDOS_SEEDS)
