from local_brain.command_center.agent_registry import AgentRegistry


def test_run_provenance_persists(tmp_path):
    data_file = tmp_path / "agents.json"
    registry = AgentRegistry(data_file)
    agent = registry.create(
        name="CI Signal Officer",
        mission="Track current CI state.",
        persona="Evidence first.",
        skills=["Analysis"],
    )

    registry.finish_run(
        agent["id"],
        "Inspect current checks.",
        result="GREEN",
        trigger="assignment",
        assignment_id="asg_123",
        model="qwen3:8b",
        latency_ms=1234.56,
    )

    run = AgentRegistry(data_file).get(agent["id"])["runs"][0]
    assert run["trigger"] == "assignment"
    assert run["assignment_id"] == "asg_123"
    assert run["model"] == "qwen3:8b"
    assert run["latency_ms"] == 1234.6
