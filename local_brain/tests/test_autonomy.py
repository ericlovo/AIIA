"""
Autonomy module tests — instantiation and enable/disable gating.

Tests that each autonomy class can be instantiated with mock dependencies
and correctly respects the Phase 2 enable flags.

Run: pytest local_brain/tests/test_autonomy.py -v
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from local_brain.autonomy import (
    GatedDowngradePolicy,
    MemoryQualityLoop,
    ProactiveStoryExecutor,
    SelfHealingMonitor,
)
from local_brain.autonomy import proactive_executor as pe_module
from local_brain.autonomy.gated_downgrade import NEVER_DOWNGRADE_TYPES
from local_brain.autonomy.self_healing import KNOWN_FIX_PATTERNS
from local_brain.config import AutonomyConfig

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def phase1_config():
    """Phase 1 config (all autonomy features disabled)."""
    return AutonomyConfig(level="phase1")


@pytest.fixture
def phase2_config():
    """Phase 2 config (all autonomy features enabled)."""
    return AutonomyConfig(
        level="phase2",
        proactive_story_execution=True,
        gated_downgrade_enabled=True,
        self_healing_enabled=True,
        memory_quality_enabled=True,
    )


@pytest.fixture
def mock_action_queue():
    mock = MagicMock()
    mock.list_actions = MagicMock(return_value=[])
    mock.approve = MagicMock(return_value={"id": "test"})
    mock.create_action = MagicMock()
    return mock


@pytest.fixture
def mock_roadmap_store():
    mock = MagicMock()
    mock.get_stories = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_story_executor():
    mock = MagicMock()
    mock.execute_story = AsyncMock(return_value={"status": "decomposed", "actions_created": 3})
    return mock


# ──────────────────────────────────────────────
# ProactiveStoryExecutor
# ──────────────────────────────────────────────


class TestProactiveStoryExecutor:
    def test_disabled_in_phase1(
        self, phase1_config, mock_action_queue, mock_roadmap_store, mock_story_executor
    ):
        executor = ProactiveStoryExecutor(
            config=phase1_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap_store,
        )
        assert executor.enabled is False

    def test_enabled_in_phase2(
        self, phase2_config, mock_action_queue, mock_roadmap_store, mock_story_executor
    ):
        executor = ProactiveStoryExecutor(
            config=phase2_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap_store,
        )
        assert executor.enabled is True

    @pytest.mark.asyncio
    async def test_disabled_skip_returns_reason(
        self, phase1_config, mock_action_queue, mock_roadmap_store, mock_story_executor
    ):
        executor = ProactiveStoryExecutor(
            config=phase1_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap_store,
        )
        result = await executor.evaluate_pending_stories()
        assert result["skipped"] is True
        assert "disabled" in result["reason"]

    @pytest.mark.parametrize(
        "tz_name,hour,expected",
        [
            ("America/Chicago", 10, True),  # 10am — business hours
            ("America/Chicago", 20, False),  # 8pm — after hours
            ("America/Chicago", 3, False),  # 3am — before hours
            ("Pacific/Auckland", 10, True),
            ("UTC", 12, True),
            ("UTC", 23, False),
        ],
    )
    def test_is_business_hours_uses_configured_tz(
        self,
        phase2_config,
        mock_action_queue,
        mock_roadmap_store,
        mock_story_executor,
        monkeypatch,
        tz_name,
        hour,
        expected,
    ):
        """_is_business_hours must evaluate against
        AutonomyConfig.proactive_business_hours_tz, not the process-local
        timezone. Regression test for a launchd-UTC bug where the check
        silently picked up the wrong zone."""
        phase2_config.proactive_business_hours_tz = tz_name
        executor = ProactiveStoryExecutor(
            config=phase2_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap_store,
        )

        fake_now = datetime(2026, 4, 13, hour, 30, 0, tzinfo=ZoneInfo(tz_name))

        class FakeDatetime:
            @staticmethod
            def now(tz=None):
                return fake_now.astimezone(tz) if tz is not None else fake_now

        monkeypatch.setattr(pe_module, "datetime", FakeDatetime)
        assert executor._is_business_hours() is expected

    def test_is_business_hours_falls_back_to_utc_on_bad_tz(
        self,
        phase2_config,
        mock_action_queue,
        mock_roadmap_store,
        mock_story_executor,
    ):
        phase2_config.proactive_business_hours_tz = "Not/AReal_Zone"
        executor = ProactiveStoryExecutor(
            config=phase2_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap_store,
        )
        assert isinstance(executor._is_business_hours(), bool)

    @pytest.mark.asyncio
    async def test_production_healthy_returns_true_when_no_url_configured(
        self,
        phase2_config,
        mock_action_queue,
        mock_roadmap_store,
        mock_story_executor,
    ):
        """With no health-check URL configured, the gate should pass
        (returns True) so operators who don't want a probe get an
        always-on-when-off-hours executor."""
        phase2_config.proactive_health_check_url = None
        executor = ProactiveStoryExecutor(
            config=phase2_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap_store,
        )
        assert await executor._production_healthy() is True

    def test_eligible_stories_filters_by_priority(
        self, phase2_config, mock_action_queue, mock_story_executor
    ):
        mock_roadmap = MagicMock()
        mock_roadmap.get_stories = MagicMock(
            return_value=[
                {"id": "1", "priority": "P0", "status": "backlog"},
                {"id": "2", "priority": "P1", "status": "active"},
                {"id": "3", "priority": "P2", "status": "backlog"},  # Too low
                {"id": "4", "priority": "P1", "status": "in_progress"},  # Already running
            ]
        )

        executor = ProactiveStoryExecutor(
            config=phase2_config,
            story_executor=mock_story_executor,
            action_queue=mock_action_queue,
            roadmap_store=mock_roadmap,
        )

        eligible = executor._get_eligible_stories()
        eligible_ids = [s["id"] for s in eligible]

        assert "1" in eligible_ids  # P0 backlog
        assert "2" in eligible_ids  # P1 active
        assert "3" not in eligible_ids  # P2 excluded
        assert "4" not in eligible_ids  # Already in_progress


# ──────────────────────────────────────────────
# GatedDowngradePolicy
# ──────────────────────────────────────────────


class TestGatedDowngradePolicy:
    def test_disabled_in_phase1(self, phase1_config, mock_action_queue):
        policy = GatedDowngradePolicy(phase1_config, mock_action_queue)
        assert policy.enabled is False

    def test_enabled_in_phase2(self, phase2_config, mock_action_queue):
        policy = GatedDowngradePolicy(phase2_config, mock_action_queue)
        assert policy.enabled is True

    def test_never_downgrade_types_includes_dangerous_actions(self):
        assert "deploy_production" in NEVER_DOWNGRADE_TYPES
        assert "delete_data" in NEVER_DOWNGRADE_TYPES
        assert "modify_secrets" in NEVER_DOWNGRADE_TYPES
        assert "review" in NEVER_DOWNGRADE_TYPES

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, phase1_config, mock_action_queue):
        policy = GatedDowngradePolicy(phase1_config, mock_action_queue)
        result = await policy.check_stale_gated_actions()
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_promotes_stale_low_severity_action(self, phase2_config):
        """Stale low-severity actions older than 48h should be promoted
        AND have their tier_override set to 'supervised' so the execution
        loop actually picks them up (approve() alone is insufficient)."""
        old_timestamp = (datetime.utcnow() - timedelta(hours=72)).isoformat() + "Z"
        new_timestamp = datetime.utcnow().isoformat() + "Z"

        stale_low = {
            "id": "stale-low",
            "type": "security_fix",
            "severity": "low",
            "created_at": old_timestamp,
            "title": "Stale low-severity action",
        }
        stale_high = {
            "id": "stale-high",
            "type": "security_fix",
            "severity": "critical",
            "created_at": old_timestamp,
            "title": "Stale critical — should NOT promote",
        }
        fresh_low = {
            "id": "fresh-low",
            "type": "security_fix",
            "severity": "low",
            "created_at": new_timestamp,
            "title": "Fresh — should NOT promote",
        }
        never_downgrade = {
            "id": "never-downgrade",
            "type": "deploy_production",
            "severity": "low",
            "created_at": old_timestamp,
            "title": "Never downgrade type",
        }
        actions_by_id = {a["id"]: a for a in [stale_low, stale_high, fresh_low, never_downgrade]}

        mock_queue = MagicMock()
        mock_queue.list_actions = MagicMock(return_value=list(actions_by_id.values()))
        mock_queue.get_action = MagicMock(side_effect=lambda aid: actions_by_id.get(aid))
        mock_queue.approve = MagicMock(return_value={"id": "ok"})
        mock_queue.save = MagicMock()

        policy = GatedDowngradePolicy(phase2_config, mock_queue)
        result = await policy.check_stale_gated_actions()

        # Only "stale-low" should have been approved
        mock_queue.approve.assert_called_once_with("stale-low")
        assert result["promoted"] == 1

        # And its tier_override must have been set to 'supervised'
        assert stale_low.get("tier_override") == "supervised"
        assert "auto_downgrade" in stale_low.get("tier_override_reason", "")

        # Other actions should NOT have tier_override set
        assert "tier_override" not in stale_high
        assert "tier_override" not in fresh_low
        assert "tier_override" not in never_downgrade


# ──────────────────────────────────────────────
# SelfHealingMonitor
# ──────────────────────────────────────────────


class TestSelfHealingMonitor:
    def test_disabled_in_phase1(self, phase1_config, mock_action_queue):
        monitor = SelfHealingMonitor(phase1_config, mock_action_queue)
        assert monitor.enabled is False

    def test_enabled_in_phase2(self, phase2_config, mock_action_queue):
        monitor = SelfHealingMonitor(phase2_config, mock_action_queue)
        assert monitor.enabled is True

    def test_default_monitored_services_is_empty(self, phase1_config):
        """The public default ships with no services — operators must opt in
        via AIIA_SERVICES_CONFIG. An empty default means a misconfigured
        monitor does nothing rather than probing accidentally."""
        assert phase1_config.monitored_services == []

    def test_known_fix_patterns_are_defined(self):
        assert "health_timeout" in KNOWN_FIX_PATTERNS
        assert "high_error_rate" in KNOWN_FIX_PATTERNS
        assert "deploy_stale" in KNOWN_FIX_PATTERNS

        for name, pattern in KNOWN_FIX_PATTERNS.items():
            assert callable(pattern["detection"]), f"{name} missing detection"
            assert "action_type" in pattern
            assert "severity" in pattern
            assert "title_template" in pattern

    @pytest.mark.asyncio
    async def test_disabled_skip(self, phase1_config, mock_action_queue):
        monitor = SelfHealingMonitor(phase1_config, mock_action_queue)
        result = await monitor.check_and_heal()
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_empty_services_list_returns_zero_counts(self, phase2_config):
        """Phase 2 enabled but no services configured → zero-count no-op."""
        phase2_config.monitored_services = []
        mock_queue = MagicMock()
        monitor = SelfHealingMonitor(phase2_config, mock_queue)
        result = await monitor.check_and_heal()
        assert result["services_checked"] == 0
        assert result["actions_created"] == 0

    @pytest.mark.asyncio
    async def test_handle_unhealthy_creates_action_with_correct_kwargs(self, phase2_config):
        """Regression test: self-healing must use ActionQueue's real kwarg
        name (`action_type=`) not `type=`. Using a MagicMock that records
        calls lets us assert the exact call signature used."""
        mock_queue = MagicMock()
        mock_queue.list_actions = MagicMock(return_value=[])  # No existing dupes

        monitor = SelfHealingMonitor(phase2_config, mock_queue)
        monitor._consecutive_failures["test-service"] = 2

        service = {
            "name": "test-service",
            "url": "http://localhost:9999/health",
        }
        status = {
            "healthy": False,
            "status_code": 503,
            "latency_ms": 5000,
        }

        actions_created = await monitor._handle_unhealthy(service, status)

        assert actions_created == 1
        mock_queue.create_action.assert_called_once()
        # The call must use `action_type=`, not `type=`
        _, kwargs = mock_queue.create_action.call_args
        assert kwargs["action_type"] == "tech_debt"
        assert kwargs["severity"] == "error"
        assert "test-service" in kwargs["title"]


# ──────────────────────────────────────────────
# MemoryQualityLoop
# ──────────────────────────────────────────────


class TestMemoryQualityLoop:
    def _make_memory_mock(self, tmp_path=None):
        """Build a Memory mock that only exposes recall() — the actual public API.
        Deliberately does NOT expose get_all() to catch regressions where the
        quality loop calls a method the real Memory class doesn't have.
        """
        mock = MagicMock(spec=["recall", "_data_dir"])
        mock.recall = MagicMock(return_value=[])
        mock._data_dir = str(tmp_path) if tmp_path else "/tmp/mem"
        return mock

    def test_disabled_in_phase1(self, phase1_config, tmp_path):
        loop = MemoryQualityLoop(
            config=phase1_config,
            memory=self._make_memory_mock(tmp_path),
            knowledge_store=MagicMock(),
            ollama=MagicMock(),
            state_dir=str(tmp_path),
        )
        assert loop.enabled is False

    def test_enabled_in_phase2(self, phase2_config, tmp_path):
        loop = MemoryQualityLoop(
            config=phase2_config,
            memory=self._make_memory_mock(tmp_path),
            knowledge_store=MagicMock(),
            ollama=MagicMock(),
            state_dir=str(tmp_path),
        )
        assert loop.enabled is True

    @pytest.mark.asyncio
    async def test_disabled_skip(self, phase1_config, tmp_path):
        loop = MemoryQualityLoop(
            config=phase1_config,
            memory=self._make_memory_mock(tmp_path),
            knowledge_store=MagicMock(),
            ollama=MagicMock(),
            state_dir=str(tmp_path),
        )
        result = await loop.run_quality_cycle()
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_uses_recall_not_get_all(self, phase2_config, tmp_path):
        """Regression test: quality loop must call Memory.recall(), not the
        non-existent get_all() method. The memory mock uses spec=['recall']
        so calling get_all would raise AttributeError."""
        memory = self._make_memory_mock(tmp_path)
        memory.recall = MagicMock(return_value=[])

        loop = MemoryQualityLoop(
            config=phase2_config,
            memory=memory,
            knowledge_store=MagicMock(),
            ollama=MagicMock(),
            state_dir=str(tmp_path),
        )
        result = await loop.run_quality_cycle()

        assert memory.recall.called
        # Three promotable categories: decisions, patterns, lessons
        assert memory.recall.call_count == 3
        assert "scored" in result

    @pytest.mark.asyncio
    async def test_state_file_tracks_promoted_ids(self, phase2_config, tmp_path):
        """Promoted IDs should be persisted to a local state file, not
        mutated into Memory's internal storage."""
        entry_high = {
            "id": "high-1",
            "fact": "Always use asyncio for I/O bound tasks — avoid blocking calls in async contexts",
            "category": "patterns",
        }

        memory = self._make_memory_mock(tmp_path)
        memory.recall = MagicMock(
            side_effect=lambda category, limit: [entry_high] if category == "patterns" else []
        )

        mock_ollama = MagicMock()
        mock_ollama.chat = AsyncMock(
            return_value={"message": {"content": "0.95"}, "_latency_ms": 50}
        )

        mock_ks = MagicMock()
        mock_ks.add_document = AsyncMock(return_value=None)

        loop = MemoryQualityLoop(
            config=phase2_config,
            memory=memory,
            knowledge_store=mock_ks,
            ollama=mock_ollama,
            state_dir=str(tmp_path),
        )
        result = await loop.run_quality_cycle()

        assert result["promoted"] == 1
        assert "high-1" in loop._promoted_ids

        state_file = tmp_path / "memory_quality_promoted.json"
        assert state_file.exists()
        data = __import__("json").loads(state_file.read_text())
        assert "high-1" in data["promoted_ids"]

    @pytest.mark.asyncio
    async def test_score_memory_parses_float(self, phase2_config):
        """Quality scoring should handle LLM float responses."""
        mock_ollama = MagicMock()
        mock_ollama.chat = AsyncMock(
            return_value={"message": {"content": "0.85"}, "_latency_ms": 50}
        )

        loop = MemoryQualityLoop(
            config=phase2_config,
            memory=MagicMock(),
            knowledge_store=MagicMock(),
            ollama=mock_ollama,
        )

        entry = {
            "fact": "Always use asyncio for I/O-bound tasks",
            "category": "patterns",
        }
        score = await loop._score_memory(entry)
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_memory_handles_invalid_response(self, phase2_config):
        """Non-float LLM responses should return 0.0."""
        mock_ollama = MagicMock()
        mock_ollama.chat = AsyncMock(
            return_value={"message": {"content": "not a number"}, "_latency_ms": 50}
        )

        loop = MemoryQualityLoop(
            config=phase2_config,
            memory=MagicMock(),
            knowledge_store=MagicMock(),
            ollama=mock_ollama,
        )

        entry = {"fact": "x" * 100, "category": "decisions"}
        score = await loop._score_memory(entry)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_memory_rejects_short_content(self, phase2_config):
        """Very short content should score 0 without an LLM call."""
        mock_ollama = MagicMock()
        mock_ollama.chat = AsyncMock()  # Should NOT be called

        loop = MemoryQualityLoop(
            config=phase2_config,
            memory=MagicMock(),
            knowledge_store=MagicMock(),
            ollama=mock_ollama,
        )

        entry = {"fact": "short", "category": "patterns"}  # < 30 chars
        score = await loop._score_memory(entry)
        assert score == 0.0
        mock_ollama.chat.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
