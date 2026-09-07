"""Durable node geometry for the Agent Studio World canvas."""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AGENT_WORLD_LAYOUT_FILE = Path(__file__).parent / "agent_world_layout.json"
MAX_POSITIONS = 500
VALID_NODE_PREFIXES = ("agent:", "assignment:")
MIN_SEPARATION_X = 16.0
MIN_SEPARATION_Y = 14.0
LAYOUT_X_BOUNDS = (8.0, 92.0)
LAYOUT_Y_BOUNDS = (12.0, 88.0)
AGENT_BOX_PX = (192, 104)
ASSIGNMENT_BOX_PX = (160, 58)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def node_kind(node_id: str) -> str:
    return "assignment" if node_id.startswith("assignment:") else "agent"


def positions_overlap(
    left: dict[str, float],
    right: dict[str, float],
    min_dx: float = MIN_SEPARATION_X,
    min_dy: float = MIN_SEPARATION_Y,
) -> bool:
    return abs(left["x"] - right["x"]) < min_dx and abs(left["y"] - right["y"]) < min_dy


def count_overlaps(
    positions: dict[str, dict[str, float]],
    min_dx: float = MIN_SEPARATION_X,
    min_dy: float = MIN_SEPARATION_Y,
) -> int:
    node_ids = list(positions)
    overlaps = 0
    for index, left_id in enumerate(node_ids):
        for right_id in node_ids[index + 1 :]:
            if positions_overlap(positions[left_id], positions[right_id], min_dx, min_dy):
                overlaps += 1
    return overlaps


def pixel_box(
    node_id: str,
    point: dict[str, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    box_w, box_h = ASSIGNMENT_BOX_PX if node_kind(node_id) == "assignment" else AGENT_BOX_PX
    center_x = point["x"] / 100 * width
    center_y = point["y"] / 100 * height
    return (center_x - box_w / 2, center_y - box_h / 2, center_x + box_w / 2, center_y + box_h / 2)


def boxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1]
    )


def hit_test(
    boxes: dict[str, tuple[float, float, float, float]],
    x: float,
    y: float,
) -> list[str]:
    return [
        node_id
        for node_id, (left, top, right, bottom) in boxes.items()
        if left <= x <= right and top <= y <= bottom
    ]


def default_lane_layout(
    node_ids: list[str],
    owners: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Place agents on a grid and stack each agent's assignments in a lane below."""
    owners = owners or {}
    agents = [node_id for node_id in node_ids if node_kind(node_id) == "agent"]
    assignments = [node_id for node_id in node_ids if node_kind(node_id) == "assignment"]
    columns = min(4, max(1, math.ceil(math.sqrt(len(agents) or 1))))
    rows = math.ceil((len(agents) or 1) / columns)
    layout: dict[str, dict[str, float]] = {}

    for index, agent_id in enumerate(agents):
        column = index % columns
        row = index // columns
        x = 50.0 if columns == 1 else 16 + column * 68 / (columns - 1)
        y = 22.0 if rows == 1 else 16 + row * 36 / max(rows - 1, 1)
        layout[agent_id] = {"x": round(x, 3), "y": round(y, 3)}

    work_index: dict[str, int] = {}
    for assignment_id in assignments:
        owner_id = owners.get(assignment_id, "")
        owner = layout.get(owner_id) or {"x": 50.0, "y": 22.0}
        slot = work_index.get(owner_id, 0)
        work_index[owner_id] = slot + 1
        layout[assignment_id] = {
            "x": round(_clamp(owner["x"], LAYOUT_X_BOUNDS), 3),
            "y": round(_clamp(owner["y"] + 14 + slot * 14, LAYOUT_Y_BOUNDS), 3),
        }
    return layout


def reconcile_positions(
    positions: dict[str, dict[str, float]],
    min_dx: float = MIN_SEPARATION_X,
    min_dy: float = MIN_SEPARATION_Y,
) -> dict[str, dict[str, float]]:
    """Nudge colliding nodes into free lanes without changing already-clear ones."""
    placed: dict[str, dict[str, float]] = {}
    for node_id in sorted(positions):
        origin = positions[node_id]
        placed[node_id] = _first_free_point(origin, placed, min_dx, min_dy)
    return placed


def _lane_grid(min_dx: float, min_dy: float) -> list[dict[str, float]]:
    slots: list[dict[str, float]] = []
    y = LAYOUT_Y_BOUNDS[0]
    while y <= LAYOUT_Y_BOUNDS[1] + 1e-6:
        x = LAYOUT_X_BOUNDS[0]
        while x <= LAYOUT_X_BOUNDS[1] + 1e-6:
            slots.append({"x": round(x, 3), "y": round(y, 3)})
            x += min_dx
        y += min_dy
    return slots


def _first_free_point(
    origin: dict[str, float],
    placed: dict[str, dict[str, float]],
    min_dx: float,
    min_dy: float,
) -> dict[str, float]:
    candidate = {
        "x": round(_clamp(origin["x"], LAYOUT_X_BOUNDS), 3),
        "y": round(_clamp(origin["y"], LAYOUT_Y_BOUNDS), 3),
    }
    if not any(positions_overlap(candidate, other, min_dx, min_dy) for other in placed.values()):
        return candidate
    slots = sorted(
        _lane_grid(min_dx, min_dy),
        key=lambda point: (point["x"] - candidate["x"]) ** 2 + (point["y"] - candidate["y"]) ** 2,
    )
    for slot in slots:
        if not any(positions_overlap(slot, other, min_dx, min_dy) for other in placed.values()):
            return slot
    return candidate


class StudioLayoutRegistry:
    def __init__(self, data_file: Path | None = None):
        self.data_file = data_file or AGENT_WORLD_LAYOUT_FILE
        self.positions: dict[str, dict[str, float]] = {}
        self.revision = 0
        self.updated_at: str | None = None
        self.load()

    def snapshot(self, allowed_node_ids: set[str] | None = None) -> dict[str, Any]:
        positions = {
            node_id: dict(point)
            for node_id, point in self.positions.items()
            if allowed_node_ids is None or node_id in allowed_node_ids
        }
        return {
            "version": 1,
            "revision": self.revision,
            "positions": reconcile_positions(positions),
            "updated_at": self.updated_at,
        }

    def update(
        self,
        positions: dict[str, dict[str, float]],
        allowed_node_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize(positions, allowed_node_ids)
        if normalized:
            self.positions.update(normalized)
            self.positions = reconcile_positions(
                dict(list(self.positions.items())[-MAX_POSITIONS:])
            )
            self.revision += 1
            self.updated_at = _now()
            self.save()
        return self.snapshot(allowed_node_ids)

    def _normalize(
        self,
        positions: dict[str, dict[str, float]],
        allowed_node_ids: set[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        if len(positions) > MAX_POSITIONS:
            raise ValueError("too_many_layout_positions")

        normalized: dict[str, dict[str, float]] = {}
        for node_id, point in positions.items():
            if not isinstance(node_id, str) or len(node_id) > 160:
                raise ValueError("invalid_layout_node")
            if not node_id.startswith(VALID_NODE_PREFIXES):
                raise ValueError("invalid_layout_node")
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            try:
                x = float(point["x"])
                y = float(point["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid_layout_position") from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("invalid_layout_position")
            if not 0 <= x <= 100 or not 0 <= y <= 100:
                raise ValueError("layout_position_out_of_bounds")
            normalized[node_id] = {"x": round(x, 3), "y": round(y, 3)}
        return normalized

    def clear(self) -> dict[str, Any]:
        self.positions = {}
        self.revision += 1
        self.updated_at = _now()
        self.save()
        return self.snapshot()

    def save(self) -> None:
        try:
            payload = {
                "version": 1,
                "revision": self.revision,
                "positions": self.positions,
                "updated_at": self.updated_at,
            }
            temporary_file = self.data_file.with_suffix(f"{self.data_file.suffix}.tmp")
            temporary_file.write_text(json.dumps(payload, indent=2))
            temporary_file.replace(self.data_file)
        except OSError as exc:
            logger.error("Could not save Agent World layout: %s", exc)

    def load(self) -> None:
        if not self.data_file.exists():
            return
        try:
            payload = json.loads(self.data_file.read_text())
            raw_positions = payload.get("positions", {})
            if not isinstance(raw_positions, dict):
                raise ValueError("invalid_layout_positions")
            self.revision = max(0, int(payload.get("revision", 0)))
            self.updated_at = payload.get("updated_at")
            self.positions = reconcile_positions(self._normalize(raw_positions))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.positions = {}
            self.revision = 0
            self.updated_at = None
            logger.warning("Could not load Agent World layout: %s", exc)
