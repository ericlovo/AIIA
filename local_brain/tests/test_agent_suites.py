from local_brain.command_center.agent_registry import AgentRegistry
from local_brain.command_center.agent_suites import (
    MINDMOOR_MEMORY_SOURCE,
    MINDMOOR_SUITE,
    apply_suite_defaults,
    describe_suites,
    infer_suite,
    match_member,
    memory_scope_for_suite,
    normalize_suite_slug,
)


def test_system_prompt_states_suite_namespace():
    from local_brain.command_center.agent_suites import suite_prompt_line

    prompt = suite_prompt_line(
        {
            "name": "Cron Review Gate",
            "suite": "mindmoor",
            "memory_namespace": "mindmoor",
        }
    )
    assert "Suite: mindmoor" in prompt
    assert "Shared local memory namespace: mindmoor" in prompt
    assert "source suite:mindmoor" in prompt
    assert suite_prompt_line({"name": "CI Signal Officer"}) == ""


def test_match_member_accepts_studio_aliases():
    assert match_member("Mindmoor Discovery Bot")["slug"] == "discovery-bot"
    assert match_member("Cron Review Gate")["slug"] == "cron-review-gate"
    assert match_member("Mindmoor Cron Review Gate")["recommended_max_tokens"] == 1600
    assert match_member("CI Signal Officer") is None


def test_infer_suite_uses_tag_then_alias():
    assert infer_suite({"name": "Weekend Repo Brief"}) == ""
    assert infer_suite({"name": "Mindmoor Scout"}) == MINDMOOR_SUITE
    assert infer_suite({"name": "Other", "suite": "mindmoor"}) == MINDMOOR_SUITE


def test_memory_scope_is_local_mindmoor_namespace():
    scope = memory_scope_for_suite()
    assert scope["namespace"] == "mindmoor"
    assert scope["source"] == MINDMOOR_MEMORY_SOURCE
    assert scope["write"] is True
    assert "decisions" in scope["collections"]


def test_apply_suite_defaults_fills_namespace_without_inferring():
    suite, namespace = apply_suite_defaults(suite="mindmoor")
    assert suite == "mindmoor"
    assert namespace == "mindmoor"
    suite, namespace = apply_suite_defaults()
    assert suite == ""
    assert namespace == ""


def test_invalid_suite_slug_is_rejected():
    try:
        normalize_suite_slug("Mindmoor Suite")
    except ValueError as exc:
        assert str(exc) == "invalid_suite"
    else:
        raise AssertionError("spaces should not be a suite slug")


def test_registry_persists_suite_and_memory_namespace(tmp_path):
    data_file = tmp_path / "agents.json"
    registry = AgentRegistry(data_file)
    agent = registry.create(
        name="Cron Review Gate",
        mission="Review cron evidence.",
        persona="Evidence first.",
        skills=["Review"],
        suite="mindmoor",
    )
    assert agent["suite"] == "mindmoor"
    assert agent["memory_namespace"] == "mindmoor"

    restored = AgentRegistry(data_file).get(agent["id"])
    assert restored["suite"] == "mindmoor"
    assert restored["memory_namespace"] == "mindmoor"

    updated = registry.update(agent["id"], memory_namespace="mindmoor")
    assert updated["suite"] == "mindmoor"
    unchanged = registry.update(agent["id"], persona="Still terse.")
    assert unchanged["suite"] == "mindmoor"
    assert unchanged["memory_namespace"] == "mindmoor"


def test_describe_suites_matches_tag_or_alias(tmp_path):
    registry = AgentRegistry(tmp_path / "agents.json")
    tagged = registry.create(
        name="Delivery Watch",
        mission="Watch delivery signals.",
        persona="Terse.",
        skills=["Observe"],
        suite="mindmoor",
    )
    alias_only = registry.create(
        name="Mindmoor Discovery Bot",
        mission="Discover local context.",
        persona="Structured.",
        skills=["Analysis"],
    )
    registry.create(
        name="CI Signal Officer",
        mission="Track CI.",
        persona="Evidence first.",
        skills=["Analysis"],
    )

    suites = describe_suites(registry.list())
    mindmoor = next(suite for suite in suites if suite["slug"] == "mindmoor")
    members = {member["slug"]: member for member in mindmoor["members"]}

    watch_ids = {row["id"] for row in members["delivery-watch"]["agents"]}
    discovery_ids = {row["id"] for row in members["discovery-bot"]["agents"]}
    assert tagged["id"] in watch_ids
    assert alias_only["id"] in discovery_ids
    assert all(
        row["id"] not in watch_ids | discovery_ids
        for member in members.values()
        for row in member["agents"]
        if row["name"] == "CI Signal Officer"
    )
    assert members["discovery-bot"]["agents"][0]["matched_by"] == "alias"
    assert members["delivery-watch"]["agents"][0]["matched_by"] == "tag"
