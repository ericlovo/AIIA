"""Operator-local Desk and Record projections over configured bot/tenant pairs."""

import json
import sqlite3
from contextlib import closing

from .bd_service import iso

DESK_LIMIT = 50
RECORD_LIMIT = 100

# Match both durable event scope and current configuration before exposing a receipt.
RECEIPTS = """
    SELECT e.id, e.timestamp, e.data, j.data
      FROM bd_events e JOIN bd_jobs j ON j.id=e.job_id
       AND j.tenant_id=json_extract(e.data,'$.tenant_id')
       AND json_extract(j.data,'$.bot_id')=json_extract(e.data,'$.bot_id')
      JOIN json_each(?) scope ON scope.key=json_extract(e.data,'$.bot_id')
       AND scope.value=j.tenant_id AND j.tenant_id=json_extract(j.data,'$.tenant_id')
     WHERE e.kind='packet_reviewed'
"""


def desk_packets(service):
    store = service.store
    # Read connection only; eligible older packets remain discoverable as decisions are made.
    with closing(sqlite3.connect(store.path, timeout=15)) as db:
        jobs = db.execute(
            """SELECT j.data FROM bd_jobs j JOIN json_each(?) scope
                 ON scope.key=json_extract(j.data,'$.bot_id') AND j.tenant_id=scope.value
               WHERE j.state='completed' AND j.tenant_id=json_extract(j.data,'$.tenant_id')
               ORDER BY json_extract(j.data,'$.finished_at') DESC, j.rowid DESC""",
            (store.configured_scope(),),
        )
        packets = []
        for (raw,) in jobs:
            job = json.loads(raw)
            result = job.get("result") or {}
            for index, item in enumerate(
                result.get("packets", [result] if result.get("organization") else [])
            ):
                if not isinstance(item, dict):
                    continue
                packet = service._packet(job, item, index)
                if (
                    packet["review_status"] == "human_review_required"
                    and packet["candidate_status"] == "supported"
                ):
                    if len(packets) == DESK_LIMIT:
                        return {"packets": packets, "has_more": True}
                    packets.append(packet)
    return {"packets": packets, "has_more": False}


def unpack(store, row):
    event = json.loads(row[2])
    job = json.loads(row[3])
    index = event["packet_index"]
    packet = store.packet_at(job, index)
    receipt = {
        "id": row[0],
        "created_at": iso(row[1]),
        "bot_id": job["bot_id"],
        "job_id": job["id"],
        "tenant_id": job["tenant_id"],
        "mode": event["mode"],
        "packet_index": index,
        "status": event["review_status"],
        "organization": packet["organization"],
    }
    return receipt, job, packet


def record_list(store):
    with closing(sqlite3.connect(store.path, timeout=15)) as db:
        rows = db.execute(
            RECEIPTS + " ORDER BY e.id DESC LIMIT ?", (store.configured_scope(), RECORD_LIMIT + 1)
        ).fetchall()
    return {
        "receipts": [unpack(store, row)[0] for row in rows[:RECORD_LIMIT]],
        "has_more": len(rows) > RECORD_LIMIT,
    }


def record_detail(service, receipt_id):
    store = service.store
    with closing(sqlite3.connect(store.path, timeout=15)) as db:
        row = db.execute(
            RECEIPTS + " AND e.id=?", (store.configured_scope(), receipt_id)
        ).fetchone()
        if not row:
            raise KeyError(receipt_id)
        receipt, job, packet = unpack(store, row)
        rows = db.execute(
            """SELECT id,timestamp,kind,data FROM bd_events
               WHERE job_id=? AND json_extract(data,'$.tenant_id')=?
                 AND json_extract(data,'$.bot_id')=? AND id<=?
                 AND kind IN ('qwen_called','qwen_completed','qwen_replayed','qwen_error',
                              'jev_called','jev_completed','jev_replayed','jev_error',
                              'job_completed','packet_reviewed') ORDER BY id""",
            (job["id"], job["tenant_id"], job["bot_id"], receipt_id),
        )
        stages = []
        for event_id, stamp, kind, data in rows:
            detail = json.loads(data)
            if kind == "packet_reviewed" and detail.get("packet_index") != receipt["packet_index"]:
                continue
            stages.append(
                {
                    "id": event_id,
                    "kind": kind,
                    "created_at": iso(stamp),
                    "message": detail.get("message") or kind.replace("_", " "),
                    "bot_id": job["bot_id"],
                    "job_id": job["id"],
                    "mode": detail["mode"],
                    **{
                        key: detail[key]
                        for key in ("provider", "model", "latency_ms", "primitive")
                        if key in detail
                    },
                }
            )
    projected = service._packet(job, packet, receipt["packet_index"])
    projected["review_status"] = receipt["status"]
    return {"receipt": receipt, "packet": projected, "stages": stages}


def decide(service, bot_id, job_id, index, expected_tenant_id, status):
    store = service.store
    bot = store.bots.get(bot_id)
    if not bot or bot["tenant_id"] != expected_tenant_id:
        raise KeyError(job_id)
    with store._transaction() as db:
        row = db.execute(
            """SELECT data FROM bd_jobs WHERE tenant_id=? AND tenant_id=json_extract(data,'$.tenant_id') AND id=?
                 AND json_extract(data,'$.bot_id')=?""",
            (bot["tenant_id"], job_id, bot_id),
        ).fetchone()
        if not row:
            raise KeyError(job_id)
        job = json.loads(row[0])
        packet = service._packet(job, store.packet_at(job, index), index)
        if packet["candidate_status"] != "supported":
            raise ValueError("packet_changed")
        if packet["review_status"] == status:
            row = db.execute(
                RECEIPTS
                + """ AND j.id=? AND json_extract(e.data,'$.packet_index')=?
                               ORDER BY e.id DESC LIMIT 1""",
                (store.configured_scope(), job_id, index),
            ).fetchone()
            if not row or json.loads(row[2])["review_status"] != status:
                raise ValueError("packet_changed")
        elif packet["review_status"] == "human_review_required":
            receipt_id = store.write_review(db, job, index, status)
            row = db.execute(
                RECEIPTS + " AND e.id=?", (store.configured_scope(), receipt_id)
            ).fetchone()
        else:
            raise ValueError("packet_changed")
        return {"receipt": unpack(store, row)[0]}
