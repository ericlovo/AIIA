import json

from local_brain.command_center.studio_layout import (
    StudioLayoutRegistry,
    boxes_overlap,
    count_overlaps,
    default_lane_layout,
    hit_test,
    pixel_box,
    reconcile_positions,
)


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


def _representative_fleet():
    agents = [f"agent:{index:02d}" for index in range(6)]
    assignments = [f"assignment:{index:02d}" for index in range(12)]
    owners = {assignments[index]: agents[index % 6] for index in range(12)}
    return agents + assignments, owners


def test_reconcile_unstacks_persisted_collisions(tmp_path):
    registry = StudioLayoutRegistry(tmp_path / "agent_world_layout.json")
    stacked = {f"agent:{index:02d}": {"x": 50.0, "y": 50.0} for index in range(6)}
    stacked.update({f"assignment:{index:02d}": {"x": 50.0, "y": 50.0} for index in range(12)})

    snapshot = registry.update(stacked)

    assert count_overlaps(stacked) >= 18
    assert count_overlaps(snapshot["positions"]) == 0
    restored = StudioLayoutRegistry(tmp_path / "agent_world_layout.json")
    assert count_overlaps(restored.snapshot()["positions"]) == 0


def _assert_unique_click_targets(layout, viewports=((1280, 800), (1440, 900))):
    assert count_overlaps(layout) == 0
    for width, height in viewports:
        boxes = {
            node_id: pixel_box(node_id, point, width, height) for node_id, point in layout.items()
        }
        ids = list(boxes)
        for index, left_id in enumerate(ids):
            for right_id in ids[index + 1 :]:
                assert not boxes_overlap(boxes[left_id], boxes[right_id]), (
                    f"{left_id} overlaps {right_id} at {width}x{height}"
                )
        for node_id, point in layout.items():
            center_x = point["x"] / 100 * width
            center_y = point["y"] / 100 * height
            assert hit_test(boxes, center_x, center_y) == [node_id]


def test_representative_fleet_has_no_overlap_or_buried_click_targets():
    node_ids, owners = _representative_fleet()
    stacked = {node_id: {"x": 48.0, "y": 44.0} for node_id in node_ids}
    _assert_unique_click_targets(
        reconcile_positions({**default_lane_layout(node_ids, owners), **stacked})
    )
    _assert_unique_click_targets(reconcile_positions(default_lane_layout(node_ids, owners)))
