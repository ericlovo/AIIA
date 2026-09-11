"""Local durable history, independent of the agent's bounded recent-run cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class RunLedger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, agent_name TEXT NOT NULL,
                repo_id TEXT NOT NULL, at TEXT NOT NULL, status TEXT NOT NULL,
                trigger TEXT NOT NULL, assignment_id TEXT NOT NULL, model TEXT NOT NULL,
                latency_ms REAL NOT NULL, legacy INTEGER NOT NULL, payload TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS runs_at ON runs(at)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, agent: dict, run: dict, *, legacy: bool = False) -> str:
        identity = json.dumps([agent["id"], run], sort_keys=True, ensure_ascii=True)
        run_id = run.get("id") or hashlib.sha256(identity.encode()).hexdigest()[:32]
        at = datetime.fromisoformat(run["at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        at = at.astimezone(timezone.utc).isoformat()
        status = (
            "failed"
            if run.get("error") or not str(run.get("result") or "").strip()
            else "completed"
        )
        trigger = run.get("trigger") or "unknown"
        payload = {
            **run,
            "temperature": None if legacy else agent.get("temperature"),
            "max_tokens": None if legacy else agent.get("max_tokens"),
        }
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    agent["id"],
                    agent["name"],
                    agent.get("repo_id", ""),
                    at,
                    status,
                    trigger,
                    run.get("assignment_id", ""),
                    run.get("model", ""),
                    float(run.get("latency_ms", 0) or 0),
                    int(legacy),
                    json.dumps(payload),
                ),
            )
        return run_id

    def backfill(self, agents: list[dict]) -> None:
        for agent in agents:
            for run in agent.get("runs", []):
                try:
                    self.record(agent, run, legacy=True)
                except (KeyError, TypeError, ValueError, AttributeError):
                    logger.warning("Skipped invalid legacy run for agent %s", agent.get("id"))

    def activity(
        self,
        *,
        days: int = 91,
        agent_id: str = "",
        day: str = "",
        status: str = "",
        now: datetime | None = None,
    ) -> dict:
        now = now or datetime.now(timezone.utc)
        start = (now - timedelta(days=days - 1)).date().isoformat()
        clauses, args = ["at >= ?", "at < ?"], [start, (now + timedelta(days=1)).date().isoformat()]
        if agent_id:
            clauses.append("agent_id = ?")
            args.append(agent_id)
        where = " AND ".join(clauses)
        with self.connect() as db:
            agent_days = [
                dict(row)
                for row in db.execute(
                    """SELECT agent_id,
                substr(at,1,10) AS day, count(*) AS total, sum(status='failed') AS failed
                FROM runs WHERE at >= ? AND at < ? GROUP BY agent_id,day""",
                    args[:2],
                )
            ]
            daily = [
                dict(row)
                for row in db.execute(
                    f"""SELECT substr(at,1,10) AS day,
                count(*) AS total, sum(status='completed') AS completed,
                sum(status='failed') AS failed, sum(latency_ms) AS latency_ms
                FROM runs WHERE {where} GROUP BY day ORDER BY day""",
                    args,
                )
            ]
            total = db.execute("SELECT count(*) FROM runs").fetchone()[0]
            earliest = db.execute("SELECT min(at) FROM runs").fetchone()[0]
            imported = db.execute("SELECT count(*) FROM runs WHERE legacy=1").fetchone()[0]
            if day:
                clauses.append("substr(at,1,10) = ?")
                args.append(day)
            if status:
                clauses.append("status = ?")
                args.append(status)
            where = " AND ".join(clauses)
            matching = db.execute(f"SELECT count(*) FROM runs WHERE {where}", args).fetchone()[0]
            runs = [
                dict(row)
                for row in db.execute(
                    f"""SELECT id,agent_id,agent_name,repo_id,
                at,status,trigger,assignment_id,model,latency_ms,legacy FROM runs
                WHERE {where} ORDER BY at DESC,id DESC LIMIT 200""",
                    args,
                )
            ]
        return {
            "days": daily,
            "agent_days": agent_days,
            "runs": runs,
            "matching": matching,
            "total": total,
            "earliest": earliest,
            "imported": imported,
            "start": start,
            "today": now.date().isoformat(),
            "timezone": "UTC",
        }

    def get(self, run_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        payload = json.loads(result.pop("payload"))
        return {**payload, **result}
