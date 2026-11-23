"""Shared client calls and their attributed source notes on the Mini."""

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import AwareDatetime, BaseModel, Field, StringConstraints

from .daily_digest import load_customers

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
Author = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
NoteText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100000)]


class NewMeeting(BaseModel):
    title: Title
    starts_at: AwareDatetime


class NewNote(BaseModel):
    author: Author
    source: Literal["zoom", "morrow", "manual"]
    text: NoteText
    source_ref: str = Field(default="", max_length=500)


class MeetingStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY, client_id TEXT NOT NULL, title TEXT NOT NULL,
                    title_key TEXT NOT NULL, starts_at TEXT NOT NULL,
                    UNIQUE (client_id, starts_at, title_key)
                );
                CREATE TABLE IF NOT EXISTS meeting_notes (
                    id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL REFERENCES meetings(id),
                    author TEXT NOT NULL, source TEXT NOT NULL, source_ref TEXT NOT NULL,
                    text TEXT NOT NULL, created_at TEXT NOT NULL, import_key TEXT NOT NULL,
                    UNIQUE (meeting_id, import_key)
                );
            """)
            yield db

    def create(self, client: str, draft: NewMeeting):
        starts = draft.starts_at.astimezone(UTC).isoformat()
        key = " ".join(draft.title.casefold().split())
        with self.connection() as db:
            db.execute(
                "INSERT INTO meetings VALUES (?,?,?,?,?) "
                "ON CONFLICT(client_id, starts_at, title_key) DO NOTHING",
                (uuid4().hex, client, draft.title, key, starts),
            )
            row = db.execute(
                "SELECT id FROM meetings WHERE client_id=? AND starts_at=? AND title_key=?",
                (client, starts, key),
            ).fetchone()
            return self._call(db, client, row["id"])

    def meetings(self, client: str):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT m.id, m.client_id, m.title, m.starts_at, count(n.id) AS note_count
                   FROM meetings m LEFT JOIN meeting_notes n ON n.meeting_id=m.id
                   WHERE m.client_id=? GROUP BY m.id ORDER BY m.starts_at DESC, m.id""",
                    (client,),
                )
            ]

    @staticmethod
    def _call(db, client, meeting_id):
        row = db.execute(
            "SELECT id, client_id, title, starts_at FROM meetings WHERE client_id=? AND id=?",
            (client, meeting_id),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Meeting not found for this client")
        return dict(row)

    def require_call(self, client, meeting_id):
        with self.connection() as db:
            return self._call(db, client, meeting_id)

    def get(self, client, meeting_id):
        with self.connection() as db:
            call = self._call(db, client, meeting_id)
            call["notes"] = [
                dict(row)
                for row in db.execute(
                    "SELECT id, author, source, source_ref, created_at, text FROM meeting_notes "
                    "WHERE meeting_id=? ORDER BY created_at, id",
                    (meeting_id,),
                )
            ]
            return call

    def append(self, client, meeting_id, draft: NewNote):
        ref = draft.source_ref.strip()
        identity = [
            draft.author.casefold(),
            draft.source,
            "ref" if ref else "text",
            ref or draft.text,
        ]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._call(db, client, meeting_id)
            db.execute(
                """INSERT INTO meeting_notes
                   (id, meeting_id, author, source, source_ref, text, created_at, import_key)
                   VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(meeting_id, import_key) DO NOTHING""",
                (
                    uuid4().hex,
                    meeting_id,
                    draft.author,
                    draft.source,
                    ref,
                    draft.text,
                    datetime.now(UTC).isoformat(),
                    key,
                ),
            )
            row = db.execute(
                "SELECT id, text FROM meeting_notes WHERE meeting_id=? AND import_key=?",
                (meeting_id, key),
            ).fetchone()
            if row["text"] != draft.text:
                raise HTTPException(409, "This source reference already has a different note")
            return {"id": row["id"], "meeting_id": meeting_id}

    def agent_context(self, client, meeting_id=""):
        """Bound excerpts at read time. Full originals remain on the shared call."""
        with self.connection() as db:
            if meeting_id:
                calls = [self._call(db, client, meeting_id)]
            else:
                calls = [
                    dict(row)
                    for row in db.execute(
                        """SELECT m.id, m.title, m.starts_at FROM meetings m WHERE m.client_id=?
                       AND EXISTS (SELECT 1 FROM meeting_notes n WHERE n.meeting_id=m.id)
                       ORDER BY m.starts_at DESC, m.id LIMIT 3""",
                        (client,),
                    )
                ]
            sections = []
            for call in calls:
                # Take one from each contributor/source before taking their older notes.
                rows = db.execute(
                    """SELECT id, author, source, source_ref, created_at, total,
                              substr(text,1,?) AS excerpt, length(text) AS size
                       FROM (SELECT *, count(*) OVER () AS total, row_number() OVER (
                           PARTITION BY lower(author), source ORDER BY created_at DESC, id
                       ) AS position FROM meeting_notes WHERE meeting_id=?)
                       ORDER BY position, created_at DESC, id LIMIT 8""",
                    (8000, call["id"]),
                ).fetchall()
                allowance = 8000 // (max(1, len(calls)) * max(1, len(rows)))
                notes = [
                    {
                        "note_id": row["id"],
                        "author": row["author"],
                        "source": row["source"],
                        "source_ref": row["source_ref"],
                        "added_at": row["created_at"],
                        "text": row["excerpt"][:allowance],
                        "excerpted": row["size"] > allowance,
                    }
                    for row in rows
                ]
                sections.append(
                    {
                        "meeting_id": call["id"],
                        "title": call["title"],
                        "starts_at": call["starts_at"],
                        "total_notes": rows[0]["total"] if rows else 0,
                        "notes": notes,
                    }
                )
            return {"client_id": client, "meetings": sections}


store = MeetingStore(Path.home() / ".aiia" / "studio" / "meetings.sqlite3")
router = APIRouter(prefix="/api/clients")


def check_scope(client: str, meeting_id: str = ""):
    if client not in {customer.id for customer in load_customers()}:
        raise HTTPException(404, "Unknown client")
    if meeting_id:
        store.require_call(client, meeting_id)


@router.get("")
def clients():
    return {"clients": [{"id": item.id, "name": item.name} for item in load_customers()]}


@router.get("/{client}/meetings")
def list_meetings(client: str):
    check_scope(client)
    return {"meetings": store.meetings(client)}


@router.post("/{client}/meetings")
def create_meeting(client: str, body: NewMeeting):
    check_scope(client)
    return store.create(client, body)


@router.get("/{client}/meetings/{meeting_id}")
def meeting(client: str, meeting_id: str):
    check_scope(client)
    return store.get(client, meeting_id)


@router.post("/{client}/meetings/{meeting_id}/notes")
def append_note(client: str, meeting_id: str, body: NewNote):
    check_scope(client)
    return store.append(client, meeting_id, body)


@router.get("/{client}/context")
def client_context(client: str, meeting_id: str = ""):
    check_scope(client)
    return store.agent_context(client, meeting_id)
