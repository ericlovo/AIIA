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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            "positions": positions,
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
            self.positions = dict(list(self.positions.items())[-MAX_POSITIONS:])
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
            self.positions = self._normalize(raw_positions)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.positions = {}
            self.revision = 0
            self.updated_at = None
            logger.warning("Could not load Agent World layout: %s", exc)
