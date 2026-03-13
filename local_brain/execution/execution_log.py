from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("aiia.execution.log")


@dataclass
class ExecutionRecord:
    id: str
    action_id: str
    action_type: str
    strategy: str  # "direct" or "claude_code"
    safety_tier: str
    started_at: str
    completed_at: str | None = None
    duration_ms: int | None = None
    status: str = "running"  # running, success, failed, timeout, killed
    input_summary: str = ""
    output_summary: str = ""
    error: str | None = None
    files_changed: list[str] = field(default_factory=list)
    verified: bool | None = None
    retry_count: int = 0


class ExecutionLog:
    def __init__(self, data_dir: str = ""):
        if not data_dir:
            data_dir = os.path.join(
                os.path.expanduser("~"),
                ".aiia",
                "eq_data",
                "execution",
            )
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, record_id: str) -> Path:
        return self._dir / f"{record_id}.json"

    def _write(self, record: ExecutionRecord) -> None:
        try:
            self._path(record.id).write_text(
                json.dumps(asdict(record), indent=2, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to write execution record {record.id}: {e}")

    def _read(self, path: Path) -> ExecutionRecord | None:
        try:
            data = json.loads(path.read_text())
            return ExecutionRecord(**data)
        except Exception:
            return None

    def start(
        self,
        action_id: str,
        action_type: str,
        strategy: str,
        safety_tier: str,
        input_summary: str,
    ) -> ExecutionRecord:
        record = ExecutionRecord(
            id=uuid.uuid4().hex[:12],
            action_id=action_id,
            action_type=action_type,
            strategy=strategy,
            safety_tier=safety_tier,
            started_at=datetime.now(timezone.utc).isoformat(),
            input_summary=input_summary,
        )
        self._write(record)
        logger.info(f"Execution started: {record.id} for action {action_id}")
        return record

    def complete(
        self,
        record_id: str,
        status: str,
        output_summary: str = "",
        files_changed: list[str] | None = None,
        verified: bool | None = None,
        error: str | None = None,
    ) -> ExecutionRecord | None:
        record = self.get(record_id)
        if not record:
            return None

        record.status = status
        record.output_summary = output_summary[:2000]
        record.files_changed = files_changed or []
        record.verified = verified
        record.error = error
        record.completed_at = datetime.now(timezone.utc).isoformat()

        # Calculate duration
        try:
            started = datetime.fromisoformat(record.started_at)
            completed = datetime.fromisoformat(record.completed_at)
            record.duration_ms = int((completed - started).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

        self._write(record)
        logger.info(f"Execution {status}: {record_id} ({record.duration_ms or 0}ms)")
        return record

    def get(self, record_id: str) -> ExecutionRecord | None:
        path = self._path(record_id)
        if not path.exists():
            return None
        return self._read(path)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        records = []
        paths = sorted(
            self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for path in paths[:limit]:
            rec = self._read(path)
            if rec:
                records.append(asdict(rec))
        return records

    def get_by_action(self, action_id: str) -> list[dict[str, Any]]:
        results = []
        for path in self._dir.glob("*.json"):
            rec = self._read(path)
            if rec and rec.action_id == action_id:
                results.append(asdict(rec))
        return results

    def get_stats(self) -> dict[str, Any]:
        total = 0
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        durations: list[int] = []
        successes = 0

        for path in self._dir.glob("*.json"):
            rec = self._read(path)
            if not rec:
                continue
            total += 1
            by_status[rec.status] = by_status.get(rec.status, 0) + 1
            by_type[rec.action_type] = by_type.get(rec.action_type, 0) + 1
            if rec.duration_ms is not None:
                durations.append(rec.duration_ms)
            if rec.status == "success":
                successes += 1

        return {
            "total": total,
            "by_status": by_status,
            "by_type": by_type,
            "avg_duration_ms": (
                int(sum(durations) / len(durations)) if durations else 0
            ),
            "success_rate": (round(successes / total, 2) if total else 0.0),
        }
