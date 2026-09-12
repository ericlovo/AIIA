"""Durable, unreviewed idea captures; separate from confirmed Brain facts."""

import sqlite3
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
            return dict(
                db.execute("SELECT * FROM ideas WHERE source_key=?", (source_key,)).fetchone()
            )

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
                "SELECT * FROM ideas"
                + where
                + " ORDER BY created_at DESC,id DESC LIMIT 50 OFFSET ?",
                [*args, offset],
            )
            return {"ideas": [dict(row) for row in rows], "total": total, "offset": offset}
