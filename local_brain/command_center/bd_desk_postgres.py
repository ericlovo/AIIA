"""Desk and Record for a trusted tenant login in the hosted Studio replay pilot."""

from contextlib import contextmanager
from uuid import UUID, uuid5

import psycopg
from psycopg.rows import dict_row

from .bd_desk import DESK_LIMIT, RECORD_LIMIT

# A configured tenant ID alone is insufficient. The database must attest that
# this exact login is mapped to it and cannot maintain or bypass the schema.
LOGIN_CHECK = """
    SELECT studio.current_tenant_id() = %s
       AND current_user = session_user
       AND pg_has_role('aiia_studio_app', 'USAGE')
       AND NOT (login.rolsuper OR login.rolbypassrls OR login.rolcreaterole)
       AND NOT has_table_privilege('studio.tenant_principals', 'INSERT,UPDATE,DELETE,TRUNCATE')
       AND NOT pg_has_role(schema.nspowner, 'MEMBER')
       AND NOT EXISTS (
           SELECT 1 FROM pg_class rel
            WHERE rel.relnamespace = schema.oid AND rel.relkind = 'r'
              AND pg_has_role(rel.relowner, 'MEMBER')
       ) AS allowed
      FROM pg_roles login CROSS JOIN pg_namespace schema
     WHERE login.rolname = session_user AND schema.nspname = 'studio'
"""

PACKET_FIELDS = """
    p.*, r.agent_id, d.id AS receipt_id, d.recorded_at, d.decision
"""
RECEIPT_FIELDS = """
    p.tenant_id, p.run_id, p.packet_index, p.organization, p.mode, r.agent_id,
    d.id AS receipt_id, d.recorded_at, d.decision
"""
ROWS = """
    FROM studio.bd_packets p
    JOIN studio.runs r ON (r.tenant_id, r.work_item_id, r.id)
                        = (p.tenant_id, p.work_item_id, p.run_id)
    LEFT JOIN studio.decision_receipts d ON (d.tenant_id, d.run_id, d.packet_index)
                                         = (p.tenant_id, p.run_id, p.packet_index)
    WHERE p.tenant_id = %s AND p.mode = 'replay' AND p.source_kind = 'synthetic_fixture'
"""


def _packet(row):
    return {
        "id": f"{row['run_id']}:{row['packet_index']}",
        "job_id": str(row["run_id"]),
        "bot_id": str(row["agent_id"]),
        "tenant_id": str(row["tenant_id"]),
        "index": row["packet_index"],
        "created_at": row["completed_at"].isoformat(),
        "review_status": row["decision"] or "human_review_required",
        "contact_policy": "hold_pending_tenant_restriction_check",
        "insight_status": "unverified_hypothesis",
        "next_query": None,
        **{
            key: row[key]
            for key in (
                "candidate_status",
                "candidate_reason",
                "mode",
                "segment",
                "disposition",
                "organization",
                "source_url",
                "source_quote",
                "emails",
                "phones",
                "signals",
                "insights",
                "summary",
            )
        },
    }


def _receipt(row):
    return {
        "id": str(row["receipt_id"]),
        "created_at": row["recorded_at"].isoformat(),
        "bot_id": str(row["agent_id"]),
        "job_id": str(row["run_id"]),
        "tenant_id": str(row["tenant_id"]),
        "mode": row["mode"],
        "packet_index": row["packet_index"],
        "status": row["decision"],
        "organization": row["organization"],
    }


def _object_uuid(value):
    try:
        return UUID(str(value))
    except ValueError as exc:
        raise KeyError(value) from exc


class PostgresDesk:
    def __init__(self, dsn: str, tenant_id: str):
        if not dsn or not dsn.strip():
            raise ValueError("A tenant database connection is required")
        self._dsn = dsn
        self.tenant_id = UUID(str(tenant_id))

    @contextmanager
    def _connection(self):
        with psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=10) as db:
            # Each statement sees decisions committed by concurrent requests while
            # waiting on the unique index in decide().
            db.isolation_level = psycopg.IsolationLevel.READ_COMMITTED
            db.execute("SET LOCAL search_path = pg_catalog, pg_temp")
            login = db.execute(LOGIN_CHECK, (self.tenant_id,)).fetchone()
            if not login or not login["allowed"]:
                raise PermissionError("Studio tenant login required")
            yield db

    def desk(self):
        with self._connection() as db:
            rows = db.execute(
                f"SELECT {PACKET_FIELDS} {ROWS}"
                """ AND p.candidate_status = 'supported' AND r.status = 'completed'
                    AND d.id IS NULL
                    ORDER BY p.completed_at DESC, p.run_id, p.packet_index LIMIT %s""",
                (self.tenant_id, DESK_LIMIT + 1),
            ).fetchall()
        return {
            "packets": [_packet(r) for r in rows[:DESK_LIMIT]],
            "has_more": len(rows) > DESK_LIMIT,
        }

    def record(self):
        with self._connection() as db:
            rows = db.execute(
                f"SELECT {RECEIPT_FIELDS} {ROWS}"
                " AND d.id IS NOT NULL ORDER BY d.recorded_at DESC, d.id DESC LIMIT %s",
                (self.tenant_id, RECORD_LIMIT + 1),
            ).fetchall()
        return {
            "receipts": [_receipt(r) for r in rows[:RECORD_LIMIT]],
            "has_more": len(rows) > RECORD_LIMIT,
        }

    def detail(self, receipt_id):
        receipt_uuid = _object_uuid(receipt_id)
        with self._connection() as db:
            row = db.execute(
                f"SELECT {PACKET_FIELDS} {ROWS} AND d.id = %s", (self.tenant_id, receipt_uuid)
            ).fetchone()
        if not row:
            raise KeyError(receipt_id)
        packet = _packet(row)
        receipt = _receipt(row)
        scope = {key: receipt[key] for key in ("bot_id", "job_id", "mode")}
        stages = [
            {
                **scope,
                "id": packet["job_id"],
                "kind": "job_completed",
                "created_at": packet["created_at"],
                "message": "Synthetic replay completed",
            },
            {
                **scope,
                "id": receipt["id"],
                "kind": "packet_reviewed",
                "created_at": receipt["created_at"],
                "message": "Packet reviewed",
            },
        ]
        return {"packet": packet, "receipt": receipt, "stages": stages}

    def decide(self, bot_id, job_id, index, expected_tenant_id, status):
        if str(self.tenant_id) != expected_tenant_id:
            raise KeyError(job_id)
        if status not in ("research_next", "watch", "dismissed"):
            raise ValueError("packet_changed")
        scope = (self.tenant_id, _object_uuid(job_id), _object_uuid(bot_id), index)
        lookup = (
            f"SELECT {RECEIPT_FIELDS}, p.work_item_id, p.candidate_status, r.status {ROWS}"
            " AND p.run_id = %s AND r.agent_id = %s AND p.packet_index = %s"
        )
        with self._connection() as db:
            # Hold the run's completion status stable until the decision commits.
            row = db.execute(lookup + " FOR SHARE OF r", scope).fetchone()
            if not row:
                raise KeyError(job_id)
            if row["candidate_status"] != "supported":
                raise ValueError("packet_changed")
            if row["receipt_id"] is None:
                if row["status"] != "completed":
                    raise ValueError("packet_changed")
                db.execute(
                    """INSERT INTO studio.decision_receipts
                           (tenant_id, work_item_id, run_id, packet_index, decision, rationale)
                       VALUES (%s, %s, %s, %s, %s, 'Operator review of synthetic replay packet')
                       ON CONFLICT (tenant_id, run_id, packet_index)
                           WHERE packet_index IS NOT NULL DO NOTHING""",
                    (self.tenant_id, row["work_item_id"], row["run_id"], index, status),
                )
                row = db.execute(lookup, scope).fetchone()
            if row["decision"] != status:
                raise ValueError("packet_changed")
            return {"receipt": _receipt(row)}

    def seed(self) -> str:
        """Insert one tenant-specific replay fixture. Re-running preserves its receipt."""
        from .bd_pipeline import proposal_packet, synthetic_fixture
        from .bd_service import candidate_assessment

        segment = "healthcare"
        agent_id, work_id, run_id = (
            uuid5(self.tenant_id, f"studio:bd-replay:v1:{kind}")
            for kind in ("agent", "work", "run")
        )
        job = {"tenant_id": str(self.tenant_id), "segment": segment, "mode": "replay"}
        source, extraction, judgments = synthetic_fixture(segment)
        packet = proposal_packet(
            job, source, extraction, judgments, {"calls": 0, "network_calls": 0}
        )
        candidate, reason = candidate_assessment(packet, segment)
        if candidate != "supported":
            raise ValueError("Synthetic replay fixture no longer qualifies for Desk")
        with self._connection() as db:
            db.execute(
                """INSERT INTO studio.agents (tenant_id, id, name)
                   VALUES (%s, %s, 'Synthetic healthcare research') ON CONFLICT (tenant_id, id) DO NOTHING""",
                (self.tenant_id, agent_id),
            )
            db.execute(
                """INSERT INTO studio.work_items (tenant_id, id, kind, title, status, agent_id)
                   VALUES (%s, %s, 'assignment', 'Review synthetic healthcare packet', 'completed', %s)
                   ON CONFLICT (tenant_id, id) DO NOTHING""",
                (self.tenant_id, work_id, agent_id),
            )
            db.execute(
                """INSERT INTO studio.runs (tenant_id, id, work_item_id, agent_id, status)
                   VALUES (%s, %s, %s, %s, 'completed') ON CONFLICT (tenant_id, id) DO NOTHING""",
                (self.tenant_id, run_id, work_id, agent_id),
            )
            db.execute(
                """INSERT INTO studio.bd_packets
                       (tenant_id, work_item_id, run_id, packet_index, mode, source_kind, segment,
                        organization, source_url, source_quote, signals, insights, emails, phones,
                        summary, disposition, candidate_status, candidate_reason)
                   VALUES (%s, %s, %s, 0, 'replay', 'synthetic_fixture', %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (tenant_id, run_id, packet_index) DO NOTHING""",
                (
                    self.tenant_id,
                    work_id,
                    run_id,
                    segment,
                    packet["organization"],
                    packet["source_url"],
                    packet["market_signals"][0]["quote"],
                    packet["signals"],
                    packet["insights"],
                    packet["emails"],
                    packet["phones"],
                    packet["summary"],
                    packet["disposition"],
                    candidate,
                    reason,
                ),
            )
        return str(run_id)
