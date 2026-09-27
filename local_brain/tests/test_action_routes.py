from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from local_brain.command_center import action_queue as queue_module


@pytest.fixture
def actions(tmp_path, monkeypatch):
    from local_brain.command_center import server

    monkeypatch.setattr(queue_module, "ACTION_DATA_FILE", tmp_path / "actions.json")
    queue = queue_module.ActionQueue()
    monkeypatch.setattr(server, "action_queue", queue)
    monkeypatch.setattr(server, "_execution_engine", None)
    broadcast = AsyncMock()
    monkeypatch.setattr(server.manager, "broadcast", broadcast)
    # No lifespan: never start live schedulers or inference services.
    client = TestClient(server.app)
    yield client, queue, broadcast
    client.close()


def create(client):
    response = client.post(
        "/api/actions",
        json={"action_type": "review", "severity": "warn", "title": "Review fixture"},
    )
    assert response.status_code == 200
    return response.json()["action"]


def test_create_list_summary_and_filters(actions):
    client, _, _ = actions
    action = create(client)
    assert action["status"] == "pending"
    assert action["source_task"] == "api"
    assert action["files_affected"] == []
    summary = {"total": 1, "by_status": {"pending": 1}, "pending_by_severity": {"warn": 1}}
    assert client.get("/api/actions").json() == {"actions": [action], "summary": summary}
    assert client.get("/api/actions/summary").json() == summary
    assert client.get("/api/actions?status=approved").json()["actions"] == []
    assert client.get("/api/actions?action_type=lint_fix").json()["actions"] == []
    assert create(client)["id"] == action["id"]


@pytest.mark.parametrize("operation", ["approve", "reject", "complete"])
def test_unknown_id_is_404(actions, operation):
    client, _, broadcast = actions
    assert client.post(f"/api/actions/missing/{operation}", json={}).status_code == 404
    broadcast.assert_not_awaited()


@pytest.mark.parametrize(
    "operation,allowed",
    [
        ("approve", {"pending"}),
        ("reject", {"pending", "approved"}),
        ("complete", {"approved", "executing"}),
    ],
)
@pytest.mark.parametrize("status", sorted(queue_module.ActionQueue.VALID_STATUSES))
def test_transition_matrix(actions, operation, allowed, status):
    client, queue, broadcast = actions
    action = queue.get_action(create(client)["id"])
    action["status"] = status
    queue.save()
    before = dict(action)
    response = client.post(
        f"/api/actions/{action['id']}/{operation}", json={"reason": "skip", "result": "done"}
    )
    if status in allowed:
        assert response.status_code == 200
        assert (
            response.json()["status"]
            == {"approve": "approved", "reject": "rejected", "complete": "completed"}[operation]
        )
        broadcast.assert_awaited_once()
    else:
        assert response.status_code == 409
        assert action == before
        broadcast.assert_not_awaited()


@pytest.mark.parametrize("path", ["/api/actions", "/api/actions/summary"])
def test_expiry_on_read_is_persisted(actions, path):
    client, queue, _ = actions
    action = queue.get_action(create(client)["id"])
    action["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=73)).isoformat()
    queue.save()
    assert client.get(path).status_code == 200
    assert action["status"] == "expired"
    assert queue_module.ActionQueue().get_action(action["id"])["status"] == "expired"


def test_expired_pending_action_cannot_be_approved(actions):
    client, queue, broadcast = actions
    action = queue.get_action(create(client)["id"])
    action["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=73)).isoformat()
    assert client.post(f"/api/actions/{action['id']}/approve").status_code == 409
    assert action["status"] == "expired"
    broadcast.assert_not_awaited()


def test_auto_execution_and_repeat_approval(actions, monkeypatch):
    from local_brain.command_center import server

    client, queue, _ = actions
    action = queue.create_action("lint_fix", "info", "Lint fixture")

    async def execute(action_id):
        queue.set_executing(action_id, "fixture-log")
        queue.complete(action_id, "fixed")
        return {"success": True}

    engine = AsyncMock()
    engine.execute_now.side_effect = execute
    monkeypatch.setattr(server, "_execution_engine", engine)
    response = client.post(f"/api/actions/{action['id']}/approve")
    assert response.status_code == 200
    assert response.json()["auto_executed"] is True
    assert response.json()["status"] == "completed"
    assert client.post(f"/api/actions/{action['id']}/approve").status_code == 409
    engine.execute_now.assert_awaited_once_with(action["id"])


def test_approved_actions_do_not_expire(actions):
    client, queue, _ = actions
    action = queue.get_action(create(client)["id"])
    queue.approve(action["id"])
    action["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    assert client.get("/api/actions").json()["actions"][0]["status"] == "approved"
