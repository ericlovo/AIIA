"""Human qualification of public signals, independent of execution approval."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from local_brain.command_center.slack_capture import inbox

router = APIRouter()


@router.get("/api/public-signals/leads")
def list_leads(
    status: Literal["all", "unreviewed", "research", "watch", "qualified", "rejected"] = "all",
    company: str = Query(default="", max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
):
    try:
        with inbox().connect() as db:
            db.execute("BEGIN")
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lead_reviews'"
            ).fetchone()
            reviews = (
                "lead_reviews"
                if exists
                else "(SELECT NULL AS idea_id, NULL AS payload, NULL AS version, NULL AS updated_at)"
            )
            base = f"""WITH signals AS (
                SELECT i.id, i.text, i.created_at, i.status AS inbox_status, i.assignment_id,
                    r.payload, r.version, r.updated_at,
                    CASE WHEN r.idea_id IS NULL THEN 'unreviewed'
                         ELSE json_extract(r.payload, '$.status') END AS decision,
                    COALESCE(json_extract(r.payload, '$.company'), '') AS company
                FROM ideas i LEFT JOIN {reviews} r ON r.idea_id=i.id
                WHERE i.source='public_signals')
            """
            where = "WHERE (?='all' OR decision=?) AND instr(lower(company),lower(?))>0"
            args = (status, status, company.strip())
            # SQL fragments above are fixed; all request values are bound in args.
            total = db.execute(base + "SELECT COUNT(*) FROM signals " + where, args).fetchone()[0]  # nosec B608
            rows = db.execute(
                base
                + "SELECT * FROM signals "  # nosec B608
                + where
                + " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            for row in rows:
                if row["payload"] is not None:
                    Qualification.model_validate(
                        {**json.loads(row["payload"]), "expected_version": row["version"]}
                    )
            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "leads": [
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "text",
                            "created_at",
                            "inbox_status",
                            "assignment_id",
                            "decision",
                            "company",
                        )
                    }
                    | {"review": record(row) if row["payload"] is not None else None}
                    for row in rows
                ],
            }
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="lead_queue_unavailable") from exc


class Qualification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: StrictInt = Field(ge=0)
    status: Literal["research", "watch", "qualified", "rejected"]
    company: str = Field(default="", max_length=200)
    evidence_url: str = Field(default="", max_length=2000)
    account_fit: str = Field(default="", max_length=2000)
    observed_change: str = Field(default="", max_length=2000)
    note: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_evidence(self):
        for field in ("company", "evidence_url", "account_fit", "observed_change", "note"):
            setattr(self, field, getattr(self, field).strip())
        if not self.note:
            raise ValueError("review_note_required")
        if self.evidence_url:
            parsed = urlsplit(self.evidence_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or any(char.isspace() for char in self.evidence_url)
            ):
                raise ValueError("https_evidence_url_required")
        if self.status == "qualified" and not all(
            (self.company, self.evidence_url, self.account_fit, self.observed_change)
        ):
            raise ValueError("qualification_requires_company_primary_source_fit_and_change")
        return self


def prepare(db, idea_id):
    row = db.execute("SELECT source FROM ideas WHERE id=?", (idea_id,)).fetchone()
    if not row or row["source"] != "public_signals":
        raise HTTPException(status_code=404, detail="public_signal_not_found")
    db.execute("""CREATE TABLE IF NOT EXISTS lead_reviews (
        idea_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL,
        updated_at TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS lead_review_history (
        idea_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
        updated_at TEXT NOT NULL, PRIMARY KEY(idea_id,version))""")


def record(row):
    return {
        **json.loads(row["payload"]),
        "version": row["version"],
        "updated_at": row["updated_at"],
    }


def review_snapshot(storage, idea_id):
    """Read an immutable handoff snapshot without initializing review storage."""
    with storage.connect() as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lead_reviews'"
        ).fetchone()
        if not exists:
            return None
        row = db.execute("SELECT * FROM lead_reviews WHERE idea_id=?", (idea_id,)).fetchone()
        if not row:
            return None
        saved = record(row)
        # A corrupt review must not silently become an unqualified assignment.
        Qualification.model_validate(
            {
                **{
                    key: value
                    for key, value in saved.items()
                    if key not in {"version", "updated_at"}
                },
                "expected_version": saved["version"],
            }
        )
        return saved


@router.get("/api/public-signals/{idea_id}/qualification")
def get_qualification(idea_id: str):
    try:
        with inbox().connect() as db:
            prepare(db, idea_id)
            row = db.execute("SELECT * FROM lead_reviews WHERE idea_id=?", (idea_id,)).fetchone()
            history = db.execute(
                "SELECT * FROM lead_review_history WHERE idea_id=? ORDER BY version DESC LIMIT 20",
                (idea_id,),
            ).fetchall()
            return {
                "review": record(row) if row else None,
                "history": [record(item) for item in history],
            }
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="qualification_storage_unavailable") from exc


@router.put("/api/public-signals/{idea_id}/qualification")
def save_qualification(idea_id: str, body: Qualification):
    try:
        with inbox().connect() as db:
            prepare(db, idea_id)
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT version FROM lead_reviews WHERE idea_id=?", (idea_id,)
            ).fetchone()
            version = row["version"] if row else 0
            if body.expected_version != version:
                raise HTTPException(
                    status_code=409, detail="qualification_changed_reload_before_saving"
                )
            payload = json.dumps(body.model_dump(exclude={"expected_version"}))
            now = datetime.now(timezone.utc).isoformat()
            args = (idea_id, version + 1, payload, now)
            db.execute(
                "INSERT INTO lead_reviews VALUES (?,?,?,?) ON CONFLICT(idea_id) DO UPDATE SET version=excluded.version,payload=excluded.payload,updated_at=excluded.updated_at",
                args,
            )
            db.execute("INSERT INTO lead_review_history VALUES (?,?,?,?)", args)
            return {"review": {**json.loads(payload), "version": version + 1, "updated_at": now}}
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(status_code=503, detail="qualification_storage_unavailable") from exc
