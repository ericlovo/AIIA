"""Read-only research aggregates over the configured bot and tenant scopes."""

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .bd_service import iso

SEGMENT_LABELS = {
    "healthcare": "Healthcare",
    "pe-technical-diligence": "PE technical diligence",
    "alumni-associations": "Alumni and associations",
    "client-bd": "Client research",
}
PROVENANCE_FIELDS = (
    "published_at",
    "captured_at",
    "processed_at",
    "publisher",
    "source_kind",
    "verification",
    "source_hash",
)
SCOPED_JOBS = """
    SELECT j.data FROM bd_jobs j JOIN json_each(?) scope
      ON scope.key=json_extract(j.data,'$.bot_id') AND j.tenant_id=scope.value
    WHERE j.tenant_id=json_extract(j.data,'$.tenant_id')
      AND j.state=json_extract(j.data,'$.state')
    ORDER BY json_extract(j.data,'$.finished_at') DESC, j.rowid DESC
"""


def _source_url(value):
    try:
        parsed = urlsplit(value or "")
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
        ):
            return value.strip()
    except (ValueError, TypeError):
        pass
    return ""


def _synthetic(packet):
    provenance = packet.get("provenance") or {}
    return (
        packet.get("mode") in {"replay", "replay-only"}
        or provenance.get("source_kind") == "synthetic_fixture"
        or str(packet.get("judgment_model", "")).startswith("replay:")
        or bool(
            re.match(r"^(?:SYNTHETIC QA|QA[- ]ONLY)\b", str(packet.get("organization", "")), re.I)
        )
    )


def insights(service, segment="all"):
    store = service.store
    segment_ids = sorted({bot["segment"] for bot in store.bots.values()})
    if segment != "all" and segment not in segment_ids:
        raise ValueError("unknown_insights_segment")
    packets = []
    job_counts = Counter()
    job_outcomes = Counter()
    excluded_replay = excluded_synthetic = excluded_unknown = 0
    database = Path(store.path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database, uri=True, timeout=15)) as db:
        for (encoded,) in db.execute(SCOPED_JOBS, (store.configured_scope(),)):
            job = json.loads(encoded)
            bot = store.bots[job["bot_id"]]
            if segment != "all" and bot["segment"] != segment:
                continue
            if job.get("mode") == "replay":
                excluded_replay += 1
                continue
            if job.get("mode") != "live":
                excluded_unknown += 1
                continue
            job_counts[job["state"]] += 1
            if job["state"] != "completed":
                continue
            result = job.get("result") or {}
            assessments = []
            for index, raw in enumerate(
                result.get("packets", [result] if result.get("organization") else [])
            ):
                if not isinstance(raw, dict):
                    continue
                if _synthetic(raw):
                    excluded_synthetic += 1
                    continue
                packet = service._packet(job, raw, index)
                # The exact run and index stay the source identity even if a
                # pipeline supplies its own display ID.
                packet["id"] = f"{job['id']}:{index}"
                packet["source_url"] = _source_url(packet["source_url"])
                provenance = raw.get("provenance") or {}
                packet["provenance"] = {
                    key: provenance[key]
                    for key in PROVENANCE_FIELDS
                    if isinstance(provenance.get(key), str)
                }
                packets.append(packet)
                assessments.append(packet["candidate_status"])
            outcome = result.get("research_outcome", "unknown")
            if assessments:
                outcome = "supported" if "supported" in assessments else assessments[0]
            job_outcomes[
                outcome if outcome in {"supported", "inconclusive", "rejected"} else "unknown"
            ] += 1

    prospects = {}
    evidence = Counter()
    review = Counter()
    segment_packets = Counter()
    trend_packets = Counter()
    trend_prospects = {}
    source_urls = set()
    with_quote = with_url = 0
    for packet in packets:
        status = packet.get("candidate_status")
        evidence[status if status in {"supported", "inconclusive", "rejected"} else "unknown"] += 1
        segment_packets[packet["segment"]] += 1
        day = (packet.get("created_at") or "")[:10]
        if day:
            trend_packets[day] += 1
        with_quote += bool(str(packet.get("source_quote") or "").strip())
        with_url += bool(packet["source_url"])
        if packet["source_url"]:
            source_urls.add(packet["source_url"])
        organization = str(packet.get("organization") or "").strip()
        if status != "supported" or not organization:
            continue
        identity = (
            packet["tenant_id"],
            packet["segment"],
            " ".join(organization.casefold().split()),
        )
        prospect_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        if day:
            trend_prospects.setdefault(day, set()).add(prospect_id)
        review[packet["review_status"]] += 1
        if prospect_id not in prospects:
            prospects[prospect_id] = {
                "id": prospect_id,
                "organization": organization,
                "segment": packet["segment"],
                "tenant_id": packet["tenant_id"],
                "packet_count": 0,
                "pending_packet_count": 0,
                "latest_at": packet["created_at"],
                "review_status": packet["review_status"],
                "source_url": packet["source_url"],
                "source_quote": packet["source_quote"],
                "signals": [],
                "insights": [],
                "insight_status": "unverified_hypothesis",
                "job_id": packet["job_id"],
                "index": packet["index"],
                "bot_id": packet["bot_id"],
                "href": "#/insights?"
                + urlencode({"segment": packet["segment"], "prospect": prospect_id}),
                "sources": [],
            }
        prospect = prospects[prospect_id]
        prospect["packet_count"] += 1
        prospect["pending_packet_count"] += packet["review_status"] == "human_review_required"
        prospect["signals"].extend(packet["signals"])
        prospect["insights"].extend(packet["insights"])
        prospect["sources"].append(
            {
                key: packet[key]
                for key in (
                    "id",
                    "job_id",
                    "index",
                    "bot_id",
                    "created_at",
                    "source_url",
                    "source_quote",
                    "signals",
                    "insights",
                    "review_status",
                    "candidate_status",
                    "provenance",
                )
            }
        )

    rows = list(prospects.values())
    for prospect in rows:
        prospect["signals"] = list(dict.fromkeys(prospect["signals"]))
        prospect["insights"] = list(dict.fromkeys(prospect["insights"]))
    rows.sort(key=lambda item: (item["latest_at"] or "", item["id"]), reverse=True)
    segment_rows = []
    for value in segment_ids:
        if segment != "all" and value != segment:
            continue
        group = [row for row in rows if row["segment"] == value]
        segment_rows.append(
            {
                "id": value,
                "label": SEGMENT_LABELS.get(value, value),
                "packets": segment_packets[value],
                "prospects": len(group),
                "insights": sum(len(row["insights"]) for row in group),
            }
        )
    return {
        "as_of": iso(store.clock()),
        "segment": segment,
        "scope": "all_retained_live_jobs",
        "trend_timezone": "UTC",
        "available_segments": [
            {"id": value, "label": SEGMENT_LABELS.get(value, value)} for value in segment_ids
        ],
        "totals": {
            "research_jobs": sum(job_counts.values()),
            "completed_jobs": job_counts["completed"],
            "failed_jobs": job_counts["failed"],
            "packets": len(packets),
            "prospects": len(rows),
            "insights": sum(len(row["insights"]) for row in rows),
            "sources": len(source_urls),
            "awaiting_review": review["human_review_required"],
        },
        "segments": segment_rows,
        "review": [{"status": key, "count": count} for key, count in sorted(review.items())],
        "evidence": {
            key: evidence[key] for key in ("supported", "inconclusive", "rejected", "unknown")
        },
        "source_coverage": {
            "with_quote": with_quote,
            "with_url": with_url,
            "domains": len({urlsplit(url).hostname for url in source_urls}),
        },
        "trend": [
            {"date": day, "packets": count, "prospects": len(trend_prospects.get(day, set()))}
            for day, count in sorted(trend_packets.items())
        ],
        "job_outcomes": {
            key: job_outcomes[key] for key in ("supported", "inconclusive", "rejected", "unknown")
        },
        "excluded_replay_jobs": excluded_replay,
        "excluded_unknown_jobs": excluded_unknown,
        "excluded_synthetic_packets": excluded_synthetic,
        "prospects": rows,
    }
