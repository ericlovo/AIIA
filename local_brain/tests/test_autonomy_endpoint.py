"""
Tests for GET /v1/autonomy/status.

Calls the endpoint function directly with module-level state patched in,
avoiding FastAPI/startup side effects (Ollama + ChromaDB init).

Run: pytest local_brain/tests/test_autonomy_endpoint.py -v
"""

import json

import pytest

from local_brain import local_api
from local_brain.config import AutonomyConfig, LocalBrainConfig


def _make_config(level: str = "phase2") -> LocalBrainConfig:
    """Build a LocalBrainConfig without triggering __post_init__ env reads."""
    config = LocalBrainConfig.__new__(LocalBrainConfig)
    config.autonomy = AutonomyConfig(
        level=level,
        proactive_story_execution=True,
        gated_downgrade_enabled=True,
        self_healing_enabled=True,
        memory_quality_enabled=True,
    )
    return config


@pytest.mark.asyncio
async def test_autonomy_status_phase2_structure(monkeypatch, tmp_path):
    monkeypatch.setattr(local_api, "_config", _make_config("phase2"))
    monkeypatch.setattr(local_api, "_TASK_DATA_FILE", tmp_path / "missing.json")

    response = await local_api.autonomy_status()

    assert response["level"] == "phase2"
    assert response["phase2_active"] is True
    assert set(response["loops"].keys()) == {
        "proactive_story_eval",
        "gated_downgrade_check",
        "self_healing_monitor",
        "memory_quality_loop",
    }
    for loop in response["loops"].values():
        assert loop["enabled"] is True
        assert loop["run_count"] == 0
        assert loop["last_run"] is None
    # Default business-hours TZ is UTC in the public default config
    assert response["policy"]["proactive"]["business_hours_tz"] == "UTC"
    assert response["policy"]["proactive"]["business_hours"] == [9, 18]
    assert response["policy"]["proactive"]["health_check_url"] is None
    assert response["policy"]["self_healing"]["monitored_service_count"] == 0
    assert "forbidden_files" in response["safety"]
    assert "forbidden_actions" in response["safety"]
    assert response["runtime_state_available"] is False


@pytest.mark.asyncio
async def test_autonomy_status_enabled_requires_phase2_level(monkeypatch, tmp_path):
    """Loop.enabled must be False when level=phase1, even if individual flags are True.

    Regression guard: an earlier implementation reported the raw flag, which
    would lie if the master switch was downgraded at runtime.
    """
    config = LocalBrainConfig.__new__(LocalBrainConfig)
    config.autonomy = AutonomyConfig(
        level="phase1",
        proactive_story_execution=True,
        gated_downgrade_enabled=True,
        self_healing_enabled=True,
        memory_quality_enabled=True,
    )
    monkeypatch.setattr(local_api, "_config", config)
    monkeypatch.setattr(local_api, "_TASK_DATA_FILE", tmp_path / "missing.json")

    response = await local_api.autonomy_status()

    assert response["level"] == "phase1"
    assert response["phase2_active"] is False
    for loop_id, loop in response["loops"].items():
        assert loop["enabled"] is False, f"{loop_id} should be disabled in phase1"


@pytest.mark.asyncio
async def test_autonomy_status_reads_runtime_state(monkeypatch, tmp_path):
    task_file = tmp_path / "task_data.json"
    task_file.write_text(
        json.dumps(
            {
                "tasks": {
                    "self_healing_monitor": {
                        "last_run": "2026-04-13T14:00:00+00:00",
                        "last_result": "5 healthy, 0 unhealthy",
                        "last_duration_ms": 2000,
                        "run_count": 5,
                        "fail_count": 0,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(local_api, "_config", _make_config("phase2"))
    monkeypatch.setattr(local_api, "_TASK_DATA_FILE", task_file)

    response = await local_api.autonomy_status()

    assert response["runtime_state_available"] is True
    sh = response["loops"]["self_healing_monitor"]
    assert sh["last_run"] == "2026-04-13T14:00:00+00:00"
    assert sh["last_result"] == "5 healthy, 0 unhealthy"
    assert sh["run_count"] == 5


@pytest.mark.asyncio
async def test_autonomy_status_handles_malformed_task_data(monkeypatch, tmp_path):
    """A corrupt or partially-written task_data.json must degrade gracefully,
    not 500. The task scheduler's save is non-atomic, so mid-write reads are possible."""
    task_file = tmp_path / "task_data.json"
    task_file.write_text("{ not valid json")
    monkeypatch.setattr(local_api, "_config", _make_config("phase2"))
    monkeypatch.setattr(local_api, "_TASK_DATA_FILE", task_file)

    response = await local_api.autonomy_status()

    assert response["runtime_state_available"] is False
    for loop in response["loops"].values():
        assert loop["run_count"] == 0


@pytest.mark.asyncio
async def test_autonomy_status_503_when_config_uninitialized(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(local_api, "_config", None)
    with pytest.raises(HTTPException) as exc_info:
        await local_api.autonomy_status()
    assert exc_info.value.status_code == 503
