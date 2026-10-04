"""One-liner, output channel, Slack fallback, and inbox delivery."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from local_brain.command_center.agent_output import (
    DEFAULT_OUTPUT_CHANNEL,
    RUN_FAILED_NOTE,
    SLACK_NOT_CONFIGURED,
    SLACK_POSTING_PENDING,
    derive_one_liner,
    inbox_title,
    present_agent,
    present_agents,
    resolve_delivery,
)
from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.assignment_registry import AssignmentRegistry
from local_brain.scripts.propose_agent_one_liners import main as propose_main
from local_brain.scripts.propose_agent_one_liners import proposals

REAL_ASYNC_CLIENT = httpx.AsyncClient


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_chat = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/chat":
            if self.fail_chat:
                return httpx.Response(500, json={"detail": "boom"})
            return httpx.Response(
                200, json={"content": "GREEN", "model": "qwen3:8b", "latency_ms": 5}
            )
        raise AssertionError(f"unexpected upstream call {request.url}")


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    agents = AgentRegistry(tmp_path / "agents.json")
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    events: list[tuple[str, str, dict[str, Any]]] = []
    upstream = Upstream()

    async def capture(entity: str, event: str, item: dict[str, Any]) -> None:
        events.append((entity, event, dict(item)))

    async def capture_assignment(event: str, item: dict[str, Any]) -> None:
        events.append(("assignment", event, dict(item)))

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(upstream.handler))
        return REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(server, "agent_registry", agents)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    monkeypatch.setattr(server, "broadcast_studio_event", capture)
    monkeypatch.setattr(server, "broadcast_assignment_event", capture_assignment)
    monkeypatch.setattr(server.httpx, "AsyncClient", client_factory)
    return server, agents, assignments, events, upstream


def _call(server, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        async with REAL_ASYNC_CLIENT(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(go())


def _agent(registry: AgentRegistry, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Signal Officer",
        "mission": "Report current CI state. File only evidence-backed findings.",
        "persona": "Evidence first.",
        "skills": ["Analysis"],
    }
    payload.update(overrides)
    return registry.create(**payload)


def test_derive_one_liner_uses_first_sentence_and_truncates():
    assert derive_one_liner("Watch CI. Then write a brief.") == "Watch CI."
    assert derive_one_liner("No terminator here") == "No terminator here"
    long_mission = "A" * 200
    assert derive_one_liner(long_mission) == "A" * 119 + "…"
    assert len(derive_one_liner(long_mission)) == 120
    assert derive_one_liner("   ") == ""


def test_legacy_agent_json_derives_one_liner_without_rewriting(tmp_path):
    data_file = tmp_path / "agents.json"
    data_file.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "legacy1",
                        "name": "Legacy",
                        "mission": "Keep the build green. Ignore noise.",
                        "persona": "Terse.",
                        "skills": [],
                        "tools": [],
                        "updated_at": "2026-09-01T00:00:00+00:00",
                        "created_at": "2026-09-01T00:00:00+00:00",
                        "runs": [],
                    }
                ],
                "pending_runs": [],
            }
        )
    )
    before = data_file.read_text()
    registry = AgentRegistry(data_file)
    raw = registry.get("legacy1")
    assert "one_liner" not in raw
    assert "output_channel" not in raw
    shown = present_agent(raw, slack_ready=False)
    assert shown["one_liner"] == "Keep the build green."
    assert shown["one_liner_derived"] is True
    assert shown["output_channel"] == DEFAULT_OUTPUT_CHANNEL
    assert shown["output_channel_note"] == ""
    assert data_file.read_text() == before
    registry.update("legacy1", temperature=0.4)
    stored = json.loads(data_file.read_text())["agents"][0]
    assert "one_liner" not in stored
    assert "output_channel" not in stored


def test_create_and_patch_persist_one_liner_and_channel(studio):
    server, registry, _assignments, _events, _upstream = studio
    created = _call(
        server,
        "POST",
        "/api/agents",
        json={
            "name": "Digest",
            "mission": "Post the daily client brief.",
            "one_liner": "Write the client daily.",
            "output_channel": "slack",
        },
    )
    assert created.status_code == 200
    agent = created.json()["agent"]
    assert agent["one_liner"] == "Write the client daily."
    assert agent["one_liner_derived"] is False
    assert agent["output_channel"] == "slack"
    assert agent["output_channel_note"] == SLACK_NOT_CONFIGURED
    restored = AgentRegistry(registry.data_file).get(agent["id"])
    assert restored["one_liner"] == "Write the client daily."
    assert restored["output_channel"] == "slack"

    patched = _call(
        server,
        "PATCH",
        f"/api/agents/{agent['id']}",
        json={"one_liner": "Shorter job.", "output_channel": "studio_inbox"},
    )
    assert patched.status_code == 200
    assert patched.json()["agent"]["one_liner"] == "Shorter job."
    assert patched.json()["agent"]["output_channel"] == "studio_inbox"
    assert patched.json()["agent"]["output_channel_note"] == ""


@pytest.mark.parametrize("channel", ["email", "both", ["studio_inbox", "slack"], ""])
def test_unknown_output_channel_is_422(studio, channel):
    server, registry, _assignments, events, _upstream = studio
    agent = _agent(registry)
    disk = registry.data_file.read_bytes()
    for method, path, body in (
        ("POST", "/api/agents", {"name": "N", "mission": "M", "output_channel": channel}),
        (
            "PUT",
            f"/api/agents/{agent['id']}",
            {"name": "N", "mission": "M", "output_channel": channel},
        ),
        ("PATCH", f"/api/agents/{agent['id']}", {"output_channel": channel}),
        ("PATCH", "/api/agent-suites/ops/agents", {"output_channel": channel}),
    ):
        response = _call(server, method, path, json=body)
        assert response.status_code == 422, (method, path, response.text)
    assert registry.data_file.read_bytes() == disk
    assert events == []


def test_registry_rejects_unknown_channel_without_api(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    with pytest.raises(ValueError, match="unknown_output_channel"):
        registry.create("N", "M", "P", [], output_channel="telegram")
    agent = registry.create("N", "M", "P", [])
    with pytest.raises(ValueError, match="unknown_output_channel"):
        registry.update(agent["id"], output_channel="sms")


def test_slack_not_configured_falls_back_to_studio_inbox(studio, monkeypatch):
    server, registry, assignments, _events, _upstream = studio
    monkeypatch.setattr(
        server,
        "resolve_delivery",
        lambda agent, **kwargs: resolve_delivery(agent, slack_ready=False),
    )
    from local_brain.command_center import agent_output

    monkeypatch.setattr(agent_output, "slack_outbound_configured", lambda: False)
    agent = _agent(registry, output_channel="slack")
    listed = _call(server, "GET", "/api/agents").json()["agents"][0]
    assert listed["output_channel"] == "slack"
    assert listed["output_channel_note"] == SLACK_NOT_CONFIGURED

    result = asyncio.run(server._execute_agent(agent["id"], "Inspect current checks."))
    run = result["agent"]["runs"][0]
    assert run["delivered_channel"] == "studio_inbox"
    assert run["delivery_note"] == SLACK_NOT_CONFIGURED
    assert registry.ledger.get(run["id"])["delivered_channel"] == "studio_inbox"
    work = assignments.list_assignments()
    assert len(work) == 1
    assert work[0]["status"] == "completed"
    assert work[0]["result"] == "GREEN"
    assert work[0]["review_status"] == "unreviewed"
    assert work[0]["trigger"] == "manual"


def test_configured_slack_still_delivers_to_studio_inbox_until_posting_exists(studio, monkeypatch):
    """Nothing posts agent output to Slack yet, so a configured Slack must not swallow a run."""
    server, registry, assignments, _events, upstream = studio
    from local_brain.command_center import agent_output, slack_memory_posts

    monkeypatch.setattr(slack_memory_posts, "configured", lambda: True)
    monkeypatch.setattr(agent_output, "slack_outbound_configured", lambda: True)
    agent = _agent(registry, output_channel="slack")
    shown = _call(server, "GET", "/api/agents").json()["agents"][0]
    assert shown["output_channel"] == "slack"
    assert shown["output_channel_note"] == SLACK_POSTING_PENDING

    result = asyncio.run(server._execute_agent(agent["id"], "Inspect current checks."))
    run = result["agent"]["runs"][0]
    assert run["delivered_channel"] == "studio_inbox"
    assert run["delivery_note"] == SLACK_POSTING_PENDING
    work = assignments.list_assignments()
    assert len(work) == 1
    assert work[0]["id"] == run["assignment_id"]
    assert work[0]["result"] == "GREEN"
    assert not any(request.url.host == "slack.com" for request in upstream.requests)


def test_receipt_token_alone_does_not_count_as_slack_outbound(monkeypatch):
    from local_brain.command_center import agent_output, slack_memory_posts, slack_receipts

    monkeypatch.setattr(slack_memory_posts, "configured", lambda: False)
    monkeypatch.setattr(slack_receipts, "configured", lambda: True)
    assert agent_output.slack_outbound_configured() is False


def test_failed_manual_run_opens_no_work_item_and_is_not_delivered(studio):
    server, registry, assignments, _events, upstream = studio
    agent = _agent(registry, output_channel="studio_inbox")
    upstream.fail_chat = True

    with pytest.raises(Exception) as excinfo:
        asyncio.run(server._execute_agent(agent["id"], "Inspect current checks."))
    assert getattr(excinfo.value, "status_code", None) == 503

    saved = registry.get(agent["id"])
    run = saved["runs"][0]
    assert run["error"] == "local_model_error_500"
    assert run["delivered_channel"] == ""
    assert run["delivery_note"] == RUN_FAILED_NOTE
    assert assignments.list_assignments() == []
    assert registry.ledger.get(run["id"])["delivered_channel"] == ""


def test_inbox_title_leads_with_the_task():
    agent = {"name": "Signal Officer", "mission": "Report current CI state. More."}
    assert (
        inbox_title(agent, "Inspect current checks.") == "Inspect current checks. — Signal Officer"
    )
    assert inbox_title(agent, "") == "Signal Officer"
    long_task = "x" * 300
    title = inbox_title(agent, long_task)
    assert len(title) <= 120
    assert title.endswith("— Signal Officer")


def test_manual_studio_inbox_run_surfaces_in_existing_review_queue(studio):
    server, registry, assignments, events, _upstream = studio
    agent = _agent(registry, output_channel="studio_inbox")
    result = asyncio.run(server._execute_agent(agent["id"], "Inspect current checks."))
    run = result["agent"]["runs"][0]
    assert run["delivered_channel"] == "studio_inbox"
    work = assignments.list_assignments()[0]
    assert work["id"] == run["assignment_id"]
    assert work["status"] == "completed"
    assert work["review_status"] == "unreviewed"
    assert work["source_ref"] == run["id"]
    # The ledger row links to the Work item too, not only the in-memory run.
    assert registry.ledger.get(run["id"])["assignment_id"] == work["id"]
    assert (
        any(entity == "assignment" for entity, _event, _item in events) or work["result"] == "GREEN"
    )


def test_assignment_run_does_not_create_a_second_inbox_item(studio):
    server, registry, assignments, _events, _upstream = studio
    agent = _agent(registry)
    work = assignments.create_assignment(
        title="Manual", objective="Inspect current checks.", agent_id=agent["id"]
    )
    result = asyncio.run(server._run_assignment(work["id"]))
    assert result["assignment"]["id"] == work["id"]
    assert len(assignments.list_assignments()) == 1
    run = result["agent"]["runs"][0]
    assert run["assignment_id"] == work["id"]
    assert run["delivered_channel"] == "studio_inbox"


def test_think_false_and_review_backpressure_are_unchanged(studio, monkeypatch):
    server, registry, assignments, _events, upstream = studio
    agent = _agent(registry, loop_enabled=True, loop_task="Inspect.")
    for index in range(3):
        monkeypatch.setattr(server, "_loop_schedule_key", lambda _agent, i=index: str(i))
        asyncio.run(server._run_scheduled_agent(agent))
    assert assignments.pending_loop_reviews(agent["id"]) == 3
    monkeypatch.setattr(
        server,
        "_observe_scheduled_inputs",
        lambda _agent, memory=None: server.LoopObservation("new-input", True),
    )
    monkeypatch.setattr(server, "_loop_schedule_key", lambda _agent: "next")
    blocked = asyncio.run(server._run_scheduled_agent(agent))
    assert blocked["reason"] == "awaiting_review"
    payload = json.loads(upstream.requests[0].content)
    assert payload["think"] is False


def test_value_glance_counts_reviewed_and_unreviewed():
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    agent = {"id": "a1", "last_run_at": "2026-10-03T12:00:00+00:00"}
    assignments = [
        {
            "agent_id": "a1",
            "status": "completed",
            "result": "Done",
            "review_status": "accepted",
            "reviewed_at": "2026-10-01T00:00:00+00:00",
            "dismissed_at": None,
        },
        {
            "agent_id": "a1",
            "status": "completed",
            "result": "Waiting",
            "review_status": "unreviewed",
            "completed_at": "2026-10-02T00:00:00+00:00",
            "dismissed_at": None,
        },
        {
            "agent_id": "a1",
            "status": "completed",
            "result": "Old",
            "review_status": "accepted",
            "reviewed_at": "2026-08-01T00:00:00+00:00",
            "dismissed_at": None,
        },
    ]
    shown = present_agents(
        [agent],
        assignments,
        {"agent_days": [{"agent_id": "a1", "day": "2026-10-03", "total": 4}]},
        slack_ready=False,
        now=now,
    )[0]
    assert shown["value"] == {
        "window_days": 14,
        "runs": 4,
        "last_run_at": "2026-10-03T12:00:00+00:00",
        "reviewed": 1,
        "unreviewed": 1,
    }


def test_backfill_script_prints_a_diff_and_does_not_write(tmp_path, capsys):
    path = tmp_path / "agents.json"
    path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "one",
                        "name": "CI Officer",
                        "mission": "Track current CI state. Ignore flakes.",
                    },
                    {
                        "id": "two",
                        "name": "Named",
                        "mission": "Unused because stored.",
                        "one_liner": "Already set.",
                    },
                ]
            }
        )
    )
    before = path.read_bytes()
    assert propose_main(["--file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "-one CI Officer: (missing)" in out
    assert "+one CI Officer: Track current CI state." in out
    assert " two Named: Already set." in out
    assert "No files were written" in out
    assert path.read_bytes() == before
    rows = proposals(json.loads(path.read_text())["agents"])
    assert rows[0]["changed"] is True
    assert rows[1]["changed"] is False
