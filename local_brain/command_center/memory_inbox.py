"""Durable, unreviewed idea captures; separate from confirmed Brain facts."""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

IDEA_STATUSES = ("unreviewed", "promoted", "dismissed")
# Highest first; list sorting and memory post delivery both use this order.
PRIORITIES = ("urgent", "high", "normal", "low")
PRIORITY_RANK = (
    "CASE ideas.priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END"
)
IDEA_SORTS = ("newest", "priority")
# Receipt kinds map to fixed tables; never interpolate caller strings into SQL.
# Queries below carry `# nosec B608` for that reason: the only interpolated
# identifiers are these literal table names, and every value is bound through `?`.
RECEIPT_TABLES = {"capture": "capture_receipts", "promotion": "promotion_receipts"}
RECEIPT_COLUMNS = """(
    idea_id TEXT PRIMARY KEY, thread_ts TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt REAL NOT NULL DEFAULT 0, lease TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '', slack_ts TEXT NOT NULL DEFAULT ''
)"""
IDEA_REVIEW_COLUMNS = ("memory_id", "memory_category", "review_note", "reviewed_at")
IDEA_POST_COLUMNS = {
    "priority": "TEXT NOT NULL DEFAULT 'normal'",
    "post_requested": "INTEGER NOT NULL DEFAULT 0",
}
IDEA_SELECT = (
    "SELECT ideas.*,c.status AS acknowledgement_status,"
    "c.error AS acknowledgement_error,c.slack_ts AS acknowledgement_ts,"
    "p.status AS promotion_status,p.error AS promotion_error,p.slack_ts AS promotion_ts "
    "FROM ideas LEFT JOIN capture_receipts c ON c.idea_id=ideas.id "
    "LEFT JOIN promotion_receipts p ON p.idea_id=ideas.id"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                for table in RECEIPT_TABLES.values():
                    connection.execute(f"CREATE TABLE IF NOT EXISTS {table} {RECEIPT_COLUMNS}")
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(ideas)")}
                for column in IDEA_REVIEW_COLUMNS:
                    if column not in columns:
                        connection.execute(
                            f"ALTER TABLE ideas ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                        )
                for column, definition in IDEA_POST_COLUMNS.items():
                    if column not in columns:
                        connection.execute(f"ALTER TABLE ideas ADD COLUMN {column} {definition}")
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
                    _now(),
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

    def get(self, idea_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(IDEA_SELECT + " WHERE ideas.id=?", (idea_id,)).fetchone()
            return dict(row) if row else None

    def promote(
        self,
        idea_id: str,
        *,
        memory_id: str,
        category: str,
        note: str = "",
        priority: str = "normal",
    ) -> dict:
        """Mark a capture as logged to Brain memory and queue one fixed Slack receipt.

        The receipt is queued at most once per idea and only when the capture
        arrived through a Slack thread; slash-command captures have no thread.
        """
        if priority not in PRIORITIES:
            raise ValueError("invalid_priority")
        if not memory_id or len(note) > 2_000:
            raise ValueError("invalid_promotion")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT i.status,c.thread_ts FROM ideas i "
                "LEFT JOIN capture_receipts c ON c.idea_id=i.id WHERE i.id=?",
                (idea_id,),
            ).fetchone()
            if row is None:
                raise ValueError("idea_not_found")
            if row["status"] == "promoted":
                raise ValueError("idea_already_promoted")
            db.execute(
                "UPDATE ideas SET status='promoted',memory_id=?,memory_category=?,"
                "review_note=?,reviewed_at=?,priority=? WHERE id=?",
                (memory_id, category, note.strip(), _now(), priority, idea_id),
            )
            if row["thread_ts"]:
                db.execute(
                    "INSERT INTO promotion_receipts (idea_id,thread_ts) VALUES (?,?) "
                    "ON CONFLICT(idea_id) DO NOTHING",
                    (idea_id, row["thread_ts"]),
                )
            return dict(db.execute(IDEA_SELECT + " WHERE ideas.id=?", (idea_id,)).fetchone())

    def dismiss(self, idea_id: str, *, note: str = "") -> dict:
        if len(note) > 2_000:
            raise ValueError("invalid_review_note")
        return self._transition(
            idea_id,
            allowed=("unreviewed",),
            status="dismissed",
            note=note,
            error="idea_not_dismissable",
        )

    def restore(self, idea_id: str) -> dict:
        return self._transition(
            idea_id,
            allowed=("dismissed",),
            status="unreviewed",
            note="",
            error="idea_not_restorable",
        )

    def _transition(self, idea_id, *, allowed, status, note, error) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM ideas WHERE id=?", (idea_id,)).fetchone()
            if row is None:
                raise ValueError("idea_not_found")
            if row["status"] not in allowed:
                raise ValueError(error)
            db.execute(
                "UPDATE ideas SET status=?,review_note=?,reviewed_at=? WHERE id=?",
                (status, note.strip(), _now() if status != "unreviewed" else "", idea_id),
            )
            return dict(db.execute(IDEA_SELECT + " WHERE ideas.id=?", (idea_id,)).fetchone())

    def claim_receipt(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT 'capture' AS kind,r.*,i.workspace_id,i.channel_id,i.memory_id "
                "FROM capture_receipts r JOIN ideas i ON i.id=r.idea_id "
                "WHERE r.status IN ('pending','sending') AND r.next_attempt<=? "
                "UNION ALL "
                "SELECT 'promotion' AS kind,r.*,i.workspace_id,i.channel_id,i.memory_id "
                "FROM promotion_receipts r JOIN ideas i ON i.id=r.idea_id "
                "WHERE r.status IN ('pending','sending') AND r.next_attempt<=? "
                "ORDER BY next_attempt,idea_id,kind LIMIT 1",
                (time.time(), time.time()),
            ).fetchone()
            if row is None:
                return None
            receipt = dict(row)
            receipt["lease"] = uuid.uuid4().hex
            receipt["attempts"] += 1
            db.execute(
                f"UPDATE {RECEIPT_TABLES[receipt['kind']]} SET status='sending',attempts=?,"  # nosec B608
                "lease=?,next_attempt=? WHERE idea_id=?",
                (receipt["attempts"], receipt["lease"], time.time() + 120, receipt["idea_id"]),
            )
            return receipt

    def finish_receipt(self, receipt, *, status, error="", slack_ts="", delay=0):
        table = RECEIPT_TABLES[receipt.get("kind", "capture")]
        with self.connect() as db:
            db.execute(
                f"UPDATE {table} SET status=?,error=?,slack_ts=?,next_attempt=? "  # nosec B608
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

    def receipt_status(self, kind: str = "capture"):
        table = RECEIPT_TABLES[kind]
        with self.connect() as db:
            return dict(
                db.execute(
                    f"SELECT status,count(*) FROM {table} GROUP BY status"  # nosec B608
                ).fetchall()
            )

    def retry_receipt(self, idea_id, kind: str = "capture"):
        table = RECEIPT_TABLES[kind]
        with self.connect() as db:
            result = db.execute(
                f"UPDATE {table} SET status='pending',attempts=0,next_attempt=0,"  # nosec B608
                "error='' WHERE idea_id=? AND status='failed'",
                (idea_id,),
            )
            return result.rowcount == 1

    def list(
        self,
        *,
        project: str = "",
        query: str = "",
        offset: int = 0,
        status: str = "",
        priority: str = "",
        sort: str = "newest",
    ) -> dict:
        if status and status not in IDEA_STATUSES:
            raise ValueError("invalid_idea_status")
        if (priority and priority not in PRIORITIES) or sort not in IDEA_SORTS:
            raise ValueError("invalid_idea_query")
        clauses, args = [], []
        if project:
            clauses.append("ideas.project=?")
            args.append(project)
        if query:
            clauses.append("instr(lower(ideas.text), lower(?)) > 0")
            args.append(query)
        if priority:
            clauses.append("ideas.priority=?")
            args.append(priority)
        scope = " WHERE " + " AND ".join(clauses) if clauses else ""
        if status:
            clauses.append("ideas.status=?")
            args.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            total = db.execute("SELECT count(*) FROM ideas" + where, args).fetchone()[0]  # nosec B608
            counts = {name: 0 for name in IDEA_STATUSES}
            counts.update(
                db.execute(
                    "SELECT ideas.status,count(*) FROM ideas"  # nosec B608
                    + scope
                    + " GROUP BY ideas.status",
                    args[: len(args) - (1 if status else 0)],
                ).fetchall()
            )
            order = PRIORITY_RANK + "," if sort == "priority" else ""
            rows = db.execute(
                IDEA_SELECT
                + where
                + " ORDER BY "
                + order
                + "ideas.created_at DESC,ideas.id DESC LIMIT 50 OFFSET ?",
                [*args, offset],
            )
            return {
                "ideas": [dict(row) for row in rows],
                "total": total,
                "offset": offset,
                "counts": counts,
            }
