import json

import httpx
import pytest

from local_brain.command_center import typesafe_advisor as module

REAL_CLIENT = httpx.AsyncClient
CANDIDATES = [
    {
        "id": "a",
        "name": "CI reviewer",
        "skills": ["Analysis"],
        "mission": "PRIVATE",
        "runs": ["PRIVATE"],
    }
]


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv("AIIA_TYPESAFE_ENABLED", "true")
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key")
    state = {
        "requests": [],
        "status": 200,
        "payload": {
            "model": "jev-test",
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "answers": {
                "specialist": {
                    "type": "choice",
                    "choice": "candidate_0",
                    "confidence": 0.8,
                    "probabilities": {"candidate_0": 0.9, "no_match": 0.1},
                }
            },
        },
    }

    def handler(request):
        state["requests"].append(request)
        if state.get("timeout"):
            raise httpx.ReadTimeout("secret provider detail", request=request)
        return httpx.Response(state["status"], json=state["payload"])

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kwargs: REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs),
    )
    return state


def request(**overrides):
    return module.RoutingRequest(
        **{
            "brief": "Review CI results",
            "candidate_agent_ids": ["a"],
            "allow_external": True,
            **overrides,
        }
    )


async def test_explicit_minimal_payload_and_cooldown(upstream):
    advisor = module.RoutingAdvisor()
    result = await advisor.suggest(request(), CANDIDATES)
    assert result["agent_id"] == "a"
    assert result["requires_confirmation"] is True
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 10}
    sent = json.loads(upstream["requests"][0].content)
    assert sent["state"] == {"brief": "Review CI results"}
    assert "PRIVATE" not in json.dumps(sent)
    assert "synthetic-key" not in json.dumps(result)
    assert str(upstream["requests"][0].url) == "https://api.typesafe.ai/v1/systemone"
    with pytest.raises(ValueError, match="routing_advisor_busy"):
        await advisor.suggest(request(), CANDIDATES)
    assert len(upstream["requests"]) == 1


@pytest.mark.parametrize("gate", ["consent", "disabled", "key", "blank"])
async def test_opt_in_gates_make_no_external_call(upstream, monkeypatch, gate):
    req = request()
    if gate == "consent":
        req.allow_external = False
    elif gate == "disabled":
        monkeypatch.delenv("AIIA_TYPESAFE_ENABLED")
    elif gate == "key":
        monkeypatch.delenv("TYPESAFE_API_KEY")
    else:
        req.brief = "   "
    with pytest.raises(ValueError):
        await module.RoutingAdvisor().suggest(req, CANDIDATES)
    assert not upstream["requests"]


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "401",
        "429",
        "500",
        "redirect",
        "unknown",
        "probabilities",
        "confidence",
        "usage",
        "shape",
    ],
)
async def test_fail_closed_without_provider_details(upstream, failure):
    answer = upstream["payload"]["answers"]["specialist"]
    if failure == "timeout":
        upstream["timeout"] = True
    elif failure.isdigit():
        upstream["status"] = int(failure)
    elif failure == "redirect":
        upstream["status"] = 302
    elif failure == "unknown":
        answer["choice"] = "not-an-agent"
    elif failure == "probabilities":
        answer["probabilities"]["candidate_0"] = 2
    elif failure == "confidence":
        answer["confidence"] = "high"
    elif failure == "usage":
        upstream["payload"]["usage"]["input_tokens"] = -1
    else:
        upstream["payload"] = []
    with pytest.raises(ValueError, match="^typesafe_unavailable$"):
        await module.RoutingAdvisor().suggest(request(), CANDIDATES)
    assert len(upstream["requests"]) == 1


async def test_no_match_is_not_an_agent(upstream):
    answer = upstream["payload"]["answers"]["specialist"]
    answer.update(choice="no_match", probabilities={"candidate_0": 0.1, "no_match": 0.9})
    result = await module.RoutingAdvisor().suggest(request(), CANDIDATES)
    assert result["status"] == "no_match"
    assert result["agent_id"] is None


async def test_endpoint_validates_candidates_and_never_creates_work(
    tmp_path, monkeypatch, upstream
):
    from local_brain.command_center import server
    from local_brain.command_center.agent_registry import AgentRegistry
    from local_brain.command_center.assignment_registry import AssignmentRegistry

    registry = AgentRegistry(tmp_path / "agents.json")
    agent = registry.create("CI reviewer", "PRIVATE", "PRIVATE", ["Analysis"])
    assignments = AssignmentRegistry(tmp_path / "assignments.json")
    monkeypatch.setattr(server, "agent_registry", registry)
    monkeypatch.setattr(server, "assignment_registry", assignments)
    monkeypatch.setattr(server, "typesafe_advisor", module.RoutingAdvisor())
    async with REAL_CLIENT(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        for ids in [["missing"], [agent["id"], agent["id"]], [], ["a"] * 33]:
            response = await client.post(
                "/api/assignments/suggest-agent",
                json=request(candidate_agent_ids=["a"]).model_dump() | {"candidate_agent_ids": ids},
            )
            assert response.status_code == 422
        assert not upstream["requests"]
        response = await client.post(
            "/api/assignments/suggest-agent",
            json=request(candidate_agent_ids=[agent["id"]]).model_dump(),
        )
        assert response.status_code == 200
        assert response.json()["agent_id"] == agent["id"]
    assert not assignments.list_assignments()
    assert not agent["runs"]


@pytest.mark.asyncio
async def test_a_governance_denial_stops_the_call(upstream, monkeypatch):
    """Consent from the caller is not authorisation to leave the Mini.

    The advisor asks the egress governance like every other cloud call site, so
    Sanction can refuse it centrally and the denial lands in the audit trail.
    """
    from local_brain.egress import EgressDecision

    async def deny(tool, server=None):
        assert tool == "typesafe.routing"
        return EgressDecision(False, "denied: air-gapped")

    monkeypatch.setattr(module, "authorize_egress", deny)

    with pytest.raises(ValueError, match="egress_denied"):
        await module.RoutingAdvisor().suggest(
            module.RoutingRequest(
                brief="route this", candidate_agent_ids=["a"], allow_external=True
            ),
            CANDIDATES,
        )

    assert upstream["requests"] == []


@pytest.mark.asyncio
async def test_nothing_dials_when_the_feature_is_switched_off(upstream, monkeypatch):
    monkeypatch.delenv("AIIA_TYPESAFE_ENABLED", raising=False)

    with pytest.raises(ValueError, match="typesafe_not_configured"):
        await module.RoutingAdvisor().suggest(
            module.RoutingRequest(
                brief="route this", candidate_agent_ids=["a"], allow_external=True
            ),
            CANDIDATES,
        )

    assert upstream["requests"] == []


@pytest.mark.asyncio
async def test_an_air_gapped_mini_allows_it_only_once_switched_on(upstream, monkeypatch):
    monkeypatch.setenv("AIIA_AIRGAP", "1")
    monkeypatch.setenv("AIIA_TYPESAFE_ENABLED", "1")

    result = await module.RoutingAdvisor().suggest(
        module.RoutingRequest(brief="route this", candidate_agent_ids=["a"], allow_external=True),
        CANDIDATES,
    )

    assert result["status"] == "suggested"
    assert len(upstream["requests"]) == 1


def test_the_advisor_is_a_registered_egress_point():
    """A call site /health cannot see is a hole in the egress inventory."""
    from local_brain import egress

    assert "typesafe.routing" in egress.EGRESS_POINTS
    # Never a static exception: it has to be switched on deliberately.
    assert "typesafe.routing" not in egress.AIRGAP_ALLOWED_EGRESS
    assert egress.airgap_allows_tool("typesafe.routing") is False
