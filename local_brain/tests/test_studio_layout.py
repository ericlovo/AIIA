import json

from local_brain.command_center.studio_layout import StudioLayoutRegistry


def test_layout_updates_are_merged_and_persisted(tmp_path):
    data_file = tmp_path / "agent_world_layout.json"
    registry = StudioLayoutRegistry(data_file)

    first = registry.update({"agent:architect": {"x": 20, "y": 25}})
    second = registry.update({"assignment:brief": {"x": 36.1254, "y": 48.5}})

    assert first["revision"] == 1
    assert second["positions"] == {
        "agent:architect": {"x": 20.0, "y": 25.0},
        "assignment:brief": {"x": 36.125, "y": 48.5},
    }
    restored = StudioLayoutRegistry(data_file)
    assert restored.snapshot() == second
    second["positions"]["agent:architect"]["x"] = 99
    assert restored.snapshot()["positions"]["agent:architect"]["x"] == 20
    assert not data_file.with_suffix(".json.tmp").exists()


def test_layout_filters_unknown_nodes_without_persisting_them(tmp_path):
    registry = StudioLayoutRegistry(tmp_path / "agent_world_layout.json")
    allowed = {"agent:known"}

    snapshot = registry.update(
        {
            "agent:known": {"x": 25, "y": 30},
            "agent:deleted": {"x": 50, "y": 50},
        },
        allowed,
    )

    assert snapshot["positions"] == {"agent:known": {"x": 25.0, "y": 30.0}}
    assert "agent:deleted" not in registry.positions


def test_layout_rejects_invalid_nodes_and_coordinates(tmp_path):
    registry = StudioLayoutRegistry(tmp_path / "agent_world_layout.json")

    for positions, expected in [
        ({"secret:prompt": {"x": 20, "y": 20}}, "invalid_layout_node"),
        ({"agent:a": {"x": -1, "y": 20}}, "layout_position_out_of_bounds"),
        ({"agent:a": {"x": float("nan"), "y": 20}}, "invalid_layout_position"),
    ]:
        try:
            registry.update(positions)
        except ValueError as exc:
            assert str(exc) == expected
        else:
            raise AssertionError(f"expected {expected}")


def test_layout_clear_is_durable(tmp_path):
    data_file = tmp_path / "agent_world_layout.json"
    registry = StudioLayoutRegistry(data_file)
    registry.update({"agent:a": {"x": 10, "y": 20}})

    cleared = registry.clear()

    assert cleared["positions"] == {}
    assert json.loads(data_file.read_text())["positions"] == {}
