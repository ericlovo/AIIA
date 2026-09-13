"""Durable, unreviewed idea captures; separate from confirmed Brain facts."""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class MemoryInbox:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a"):
            pass
        self.path.chmod(0o600)
        connection = sqlite3.connect(self.path, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS ideas (
                    id TEXT PRIMARY KEY, source_key TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL, source TEXT NOT NULL, project TEXT NOT NULL,
                    workspace_id TEXT NOT NULL, channel_id TEXT NOT NULL,
                    author_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'unreviewed'
                )""")
                connection.execute("""CREATE TABLE IF NOT EXISTS capture_receipts (
                    idea_id TEXT PRIMARY KEY, thread_ts TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt REAL NOT NULL DEFAULT 0, lease TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '', slack_ts TEXT NOT NULL DEFAULT ''
                )""")
                yield connection
        finally:
            connection.close()

    def capture(
        self,
        *,
        text: str,
        source_key: str,
        source: str,
        project: str,
        workspace_id: str = "",
        channel_id: str = "",
        author_id: str = "",
        receipt_thread_ts: str = "",
    ) -> dict:
        if not text.strip() or len(text) > 8_000:
            raise ValueError("idea_requires_1_to_8000_characters")
        with self.connect() as db:
            db.execute(
                """INSERT INTO ideas (id,source_key,text,source,project,workspace_id,
                channel_id,author_id,created_at) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_key) DO NOTHING""",
                (
                    uuid.uuid4().hex,
                    source_key,
                    text,
                    source,
                    project,
                    workspace_id,
                    channel_id,
                    author_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            idea = dict(
                db.execute("SELECT * FROM ideas WHERE source_key=?", (source_key,)).fetchone()
            )
            if receipt_thread_ts:
                db.execute(
                    "INSERT INTO capture_receipts (idea_id,thread_ts) VALUES (?,?) "
                    "ON CONFLICT(idea_id) DO NOTHING",
                    (idea["id"], receipt_thread_ts),
                )
            return idea

    def claim_receipt(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT r.*,i.workspace_id,i.channel_id FROM capture_receipts r "
                "JOIN ideas i ON i.id=r.idea_id "
                "WHERE r.status IN ('pending','sending') AND r.next_attempt<=? "
                "ORDER BY r.next_attempt,r.idea_id LIMIT 1",
                (time.time(),),
            ).fetchone()
            if row is None:
                return None
            receipt = dict(row)
            receipt["lease"] = uuid.uuid4().hex
            receipt["attempts"] += 1
            db.execute(
                "UPDATE capture_receipts SET status='sending',attempts=?,lease=?,"
                "next_attempt=? WHERE idea_id=?",
                (receipt["attempts"], receipt["lease"], time.time() + 120, receipt["idea_id"]),
            )
            return receipt

    def finish_receipt(self, receipt, *, status, error="", slack_ts="", delay=0):
        with self.connect() as db:
            db.execute(
                "UPDATE capture_receipts SET status=?,error=?,slack_ts=?,next_attempt=? "
                "WHERE idea_id=? AND lease=? AND status='sending'",
                (
                    status,
                    error,
                    slack_ts,
                    time.time() + delay,
                    receipt["idea_id"],
                    receipt["lease"],
                ),
            )

    def receipt_status(self):
        with self.connect() as db:
            return dict(
                db.execute(
                    "SELECT status,count(*) FROM capture_receipts GROUP BY status"
                ).fetchall()
            )

    def retry_receipt(self, idea_id):
        with self.connect() as db:
            result = db.execute(
                "UPDATE capture_receipts SET status='pending',attempts=0,next_attempt=0,"
                "error='' WHERE idea_id=? AND status='failed'",
                (idea_id,),
            )
            return result.rowcount == 1

    def list(self, *, project: str = "", query: str = "", offset: int = 0) -> dict:
        clauses, args = [], []
        if project:
            clauses.append("project=?")
            args.append(project)
        if query:
            clauses.append("instr(lower(text), lower(?)) > 0")
            args.append(query)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            total = db.execute("SELECT count(*) FROM ideas" + where, args).fetchone()[0]
            rows = db.execute(
                "SELECT ideas.*,r.status AS acknowledgement_status,"
                "r.error AS acknowledgement_error,r.slack_ts AS acknowledgement_ts "
                "FROM ideas LEFT JOIN capture_receipts r ON r.idea_id=ideas.id"
                + where
                + " ORDER BY ideas.created_at DESC,ideas.id DESC LIMIT 50 OFFSET ?",
                [*args, offset],
            )
            return {"ideas": [dict(row) for row in rows], "total": total, "offset": offset}
