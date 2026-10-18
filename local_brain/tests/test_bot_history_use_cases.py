import asyncio
from copy import deepcopy

import httpx
import pytest

from local_brain.command_center.agent_output import present_agent
from local_brain.command_center.agent_registry import AgentRegistry, RunHistoryUnavailable
from local_brain.command_center.bot_use_cases import DEFAULTS, resolve_use_cases


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from local_brain.command_center import server

    registry = AgentRegistry(tmp_path / "agents.json")
    monkeypatch.setattr(server, "agent_registry", registry)

    async def broadcast(*args):
        pass

    monkeypatch.setattr(server, "broadcast_studio_event", broadcast)
    return server, registry


def call(server, method, path, **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


@pytest.mark.parametrize("name", list(DEFAULTS))
def test_existing_roster_defaults_are_three_bounded_tasks(name):
    cases = resolve_use_cases({"name": name})
    assert len(cases) == 3
    assert len({item["label"] for item in cases}) == 3
    assert all(0 < len(item["task"]) <= 6000 for item in cases)


def test_legacy_config_gets_defaults_without_rewriting_settings_or_disk(tmp_path):
    data_file = tmp_path / "agents.json"
    registry = AgentRegistry(data_file)
    agent = registry.create("Custom scout", "Inspect supplied customer feedback.", "Careful.", [])
    agent.pop("use_cases")
    registry.save()
    before = data_file.read_bytes()
    original = deepcopy(agent)
    shown = present_agent(agent, slack_ready=False)
    assert len(shown["use_cases"]) == 3
    assert all(agent["mission"] in item["task"] for item in shown["use_cases"])
    assert agent == original
    assert data_file.read_bytes() == before


def test_config_api_defaults_custom_cases_and_partial_edits(studio):
    server, registry = studio
    response = call(
        server,
        "POST",
        "/api/agents",
        json={"name": "Code Reviewer", "mission": "Review supplied code."},
    )
    assert response.status_code == 200
    agent = response.json()["agent"]
    assert len(agent["use_cases"]) == 3
    before = deepcopy(registry.get(agent["id"]))
    cases = [{"label": "Review tests", "task": "Check coverage for the supplied diff."}]
    response = call(server, "PATCH", f"/api/agents/{agent['id']}", json={"use_cases": cases})
    assert response.status_code == 200
    restored = AgentRegistry(registry.data_file).get(agent["id"])
    assert restored["use_cases"] == cases
    assert {k: v for k, v in restored.items() if k not in {"use_cases", "updated_at"}} == {
        k: v for k, v in before.items() if k not in {"use_cases", "updated_at"}
    }
    response = call(
        server,
        "PUT",
        f"/api/agents/{agent['id']}",
        json={"name": agent["name"], "mission": agent["mission"]},
    )
    assert response.json()["agent"]["use_cases"] == cases
    assert (
        call(server, "PATCH", f"/api/agents/{agent['id']}", json={"use_cases": []}).json()["agent"][
            "use_cases"
        ]
        == []
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "bad",
        [{}],
        [{"label": " ", "task": "x"}],
        [{"label": "x", "task": " "}],
        [{"label": "x", "task": "x"}] * 4,
        [{"label": "x", "task": "x" * 6001}],
    ],
)
def test_invalid_use_cases_do_not_change_config(studio, value):
    server, registry = studio
    agent = registry.create("Scout", "Research", "Careful", [])
    before = registry.data_file.read_bytes()
    response = call(server, "PATCH", f"/api/agents/{agent['id']}", json={"use_cases": value})
    assert response.status_code == 422
    assert registry.data_file.read_bytes() == before


def test_history_survives_recent_cache_and_restart_with_pagination(studio, monkeypatch):
    server, registry = studio
    agent = registry.create("Scout", "Research", "Careful", [])
    other = registry.create("Other", "Other work", "Careful", [])
    registry.finish_run(other["id"], "Other task", result="Do not leak into Scout history")
    for index in range(25):
        registry.finish_run(
            agent["id"],
            f"Task {index}",
            result=f"Result {index}" if index != 24 else "",
            error="Fixture failure" if index == 24 else "",
        )
    assert len(agent["runs"]) == 12
    monkeypatch.setattr(server, "agent_registry", AgentRegistry(registry.data_file))
    page = call(server, "GET", f"/api/agents/{agent['id']}/history").json()
    assert page["total"] == 25
    assert len(page["runs"]) == 20
    assert [run["task"] for run in page["runs"]] == [f"Task {index}" for index in range(24, 4, -1)]
    assert page["runs"][0]["status"] == "failed"
    assert page["runs"][0]["error"] == "Fixture failure"
    assert page["runs"][1]["result"] == "Result 23"
    older = call(server, "GET", f"/api/agents/{agent['id']}/history?offset=20").json()
    assert [run["task"] for run in older["runs"]] == [f"Task {index}" for index in range(4, -1, -1)]
    assert all(run["agent_id"] == agent["id"] for run in page["runs"] + older["runs"])
    assert call(server, "GET", "/api/agents/missing/history").status_code == 404
    assert call(server, "GET", f"/api/agents/{other['id']}/history?offset=-1").status_code == 422
    assert call(server, "GET", f"/api/agents/{other['id']}/history?limit=51").status_code == 422


def test_history_empty_and_unavailable_are_distinct(studio, monkeypatch):
    server, registry = studio
    agent = registry.create("Scout", "Research", "Careful", [])
    assert call(server, "GET", f"/api/agents/{agent['id']}/history").json()["runs"] == []

    def unavailable():
        raise RunHistoryUnavailable("fixture")

    monkeypatch.setattr(registry, "recover_runs", unavailable)
    assert call(server, "GET", f"/api/agents/{agent['id']}/history").status_code == 503
