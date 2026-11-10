from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from local_brain.command_center import server
from local_brain.command_center.aiia_tasks import (
    TaskRunner,
    classify_monitor_services,
    is_configured_monitor_service,
)


@pytest.mark.asyncio
async def test_unconfigured_probe_skipped_and_history_preserved(monkeypatch):
    monkeypatch.setitem(server.MONITORED_SERVICES["default"], "configured", False)
    client = AsyncMock()
    assert await server.check_service(client, "default") is None
    client.get.assert_not_called()
    monitor = server.MonitorState()
    monitor.load_persisted(
        {
            "statuses": {"default": "offline"},
            "total_checks": {"default": 12},
            "error_counts": {"default": 12},
        }
    )
    snapshot = monitor.get_service_snapshot("default")
    assert snapshot["status"] == "not_configured"
    assert snapshot["total_checks"] == 12
    assert monitor.to_persist()["statuses"]["default"] == "offline"
    monkeypatch.setitem(server.MONITORED_SERVICES["default"], "configured", True)
    client.get.return_value.status_code = 503
    result = await server.check_service(client, "default")
    assert result.status == "offline"
    assert result.error == "HTTP 503"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected", [("online", "success"), ("offline", "warn"), ("unknown", None)]
)
async def test_journal_only_reports_configured_failures(status, expected):
    runner = SimpleNamespace(
        monitor_state=SimpleNamespace(
            get_full_snapshot=lambda: {
                "services": {
                    "optional": {"name": "Unused", "configured": False, "status": "offline"},
                    "aiia": {"name": "Brain", "status": status},
                }
            }
        ),
        _progress=AsyncMock(),
        _aiia_request=AsyncMock(),
        _emit_insight=AsyncMock(),
    )
    await TaskRunner._task_health_journal(runner)
    fact = runner._aiia_request.call_args.args[2]["fact"]
    assert "Unused" not in fact
    if expected:
        assert runner._emit_insight.call_args.args[1] == expected
    else:
        runner._emit_insight.assert_not_called()


def test_not_configured_status_is_not_a_configured_monitor():
    assert is_configured_monitor_service({"configured": False, "status": "offline"}) is False
    assert is_configured_monitor_service({"status": "not_configured"}) is False
    assert is_configured_monitor_service({"configured": True, "status": "online"}) is True
    probed, skipped, failures = classify_monitor_services(
        {
            "default": {"name": "App Analyst", "configured": False, "status": "not_configured"},
            "aiia": {"name": "Brain", "status": "degraded"},
        }
    )
    assert "default" in skipped
    assert "aiia" in probed
    assert failures == ["Brain"]
