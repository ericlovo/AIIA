"""Standing BD research cadence. Importing this module starts no work.

Working hours and weekday/weekend intervals live in versioned roster config.
The worker only enqueues during open local hours; the kill switch and
schedule_enabled flag still gate execution.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

RESEARCH_SEGMENTS = {"healthcare", "pe-technical-diligence", "alumni-associations"}
WEEKDAY_KEYS = {"start_hour", "end_hour", "interval_seconds"}
NEXT_QUERY = {
    "no_new_evidence": (
        "Retry with a named-operator query or a narrower event window for this segment"
    ),
    "no_named_organization": "Search for a named organization plus a concrete operational event",
    "publisher_is_not_prospect": (
        "Skip publisher-only headlines; require the operator in the article body"
    ),
    "generic_healthcare_reference": "Require a named hospital, clinic, or health system",
    "generic_segment_reference": "Require a named firm, fund, or association, not a generic theme",
    "no_operational_event": "Look for expansion, acquisition, hiring, or platform-change language",
    "organization_not_grounded_in_article_quote": (
        "Use a source where the organization name appears in the supporting excerpt"
    ),
    "healthcare_entity_or_event_unconfirmed": (
        "Find a named healthcare operator and a confirmed operational event"
    ),
    "segment_entity_or_event_unconfirmed": (
        "Find a named in-segment operator and a confirmed relevant event"
    ),
    "qwen_unavailable": "Warm qwen3:8b and retry this source after the model is resident",
    "source_evidence_missing": "Retain the source quote and URL before treating this as a candidate",
    "source_evidence_unverified": "Re-check grounding: named organization, quote, and supported claim",
    "segment_rejected": "Move to a different source; this segment judgment rejected the evidence",
}


def validate_cadence(cadence):
    if not isinstance(cadence, dict):
        raise ValueError("cadence must be an object")
    if set(cadence) != {"timezone", "weekday", "weekend"}:
        raise ValueError("cadence requires timezone, weekday and weekend")
    try:
        ZoneInfo(cadence["timezone"])
    except (TypeError, ZoneInfoNotFoundError) as exc:
        raise ValueError("cadence.timezone must be a valid IANA zone") from exc
    if not isinstance(cadence["timezone"], str) or not cadence["timezone"].strip():
        raise ValueError("cadence.timezone must be a nonempty string")
    for name in ("weekday", "weekend"):
        part = cadence[name]
        if not isinstance(part, dict) or set(part) != WEEKDAY_KEYS:
            raise ValueError(f"cadence.{name} requires start_hour, end_hour and interval_seconds")
        for key in ("start_hour", "end_hour"):
            value = part[key]
            if type(value) is not int or not 0 <= value <= 24:
                raise ValueError(f"cadence.{name}.{key} must be an integer hour 0-24")
        if part["end_hour"] <= part["start_hour"]:
            raise ValueError(f"cadence.{name} end_hour must be after start_hour")
        interval = part["interval_seconds"]
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or interval < 60
            or interval != int(interval)
        ):
            raise ValueError(f"cadence.{name}.interval_seconds must be an integer of at least 60")


def local_now(timestamp, zone):
    return datetime.fromtimestamp(timestamp, timezone.utc).astimezone(ZoneInfo(zone))


def daypart(cadence, timestamp):
    local = local_now(timestamp, cadence["timezone"])
    return cadence["weekend"] if local.weekday() >= 5 else cadence["weekday"]


def cadence_open(cadence, timestamp):
    local = local_now(timestamp, cadence["timezone"])
    part = daypart(cadence, timestamp)
    return part["start_hour"] <= local.hour < part["end_hour"]


def cadence_interval(cadence, timestamp):
    return int(daypart(cadence, timestamp)["interval_seconds"])


def next_open_timestamp(cadence, timestamp):
    if cadence_open(cadence, timestamp):
        return timestamp
    tz = ZoneInfo(cadence["timezone"])
    local = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(tz)
    for offset in range(8):
        day = local + timedelta(days=offset)
        part = cadence["weekend"] if day.weekday() >= 5 else cadence["weekday"]
        start = day.replace(hour=part["start_hour"], minute=0, second=0, microsecond=0)
        if start.timestamp() > timestamp:
            return start.timestamp()
    return None


def slot_index(cadence, timestamp):
    return int(timestamp // cadence_interval(cadence, timestamp))


def has_research_contract(bot):
    segment = bot.get("segment")
    if segment == "client-bd" or bot.get("kind") == "client":
        managed = bot.get("managed_leads")
        return (
            bot.get("research_segment") in RESEARCH_SEGMENTS
            and isinstance(managed, dict)
            and managed
        )
    return segment in RESEARCH_SEGMENTS


def eligible_bots(bots, *, shadow=False, standing=False):
    chosen = []
    for bot in bots:
        if not bot.get("enabled"):
            continue
        if shadow:
            chosen.append(bot)
            continue
        if not has_research_contract(bot):
            continue
        if standing or bot.get("schedule", {}).get("enabled"):
            chosen.append(bot)
    return chosen


def rotate_bot(eligible, slot):
    if not eligible:
        return None
    return eligible[slot % len(eligible)]


def next_cadence_timestamp(cadence, timestamp, existing, eligible):
    if not eligible:
        return None
    cursor = timestamp
    deadline = timestamp + 8 * 86400
    while cursor < deadline:
        if not cadence_open(cadence, cursor):
            opened = next_open_timestamp(cadence, cursor)
            if opened is None:
                return None
            cursor = opened
            continue
        interval = cadence_interval(cadence, cursor)
        slot = int(cursor // interval)
        bot = rotate_bot(eligible, slot)
        key = (bot["tenant_id"], f"{bot['id']}:{slot}", "public-research")
        if key not in existing:
            return cursor
        cursor = (slot + 1) * interval
    return None


def next_query_for(reason):
    if not isinstance(reason, str) or not reason:
        return NEXT_QUERY["source_evidence_unverified"]
    return NEXT_QUERY.get(reason, reason.replace("_", " "))


def status_banner(kill_switch, schedule_enabled, running=False):
    if kill_switch:
        return {"state": "paused_kill_switch", "label": "Paused: kill switch on"}
    if not schedule_enabled:
        return {"state": "paused_schedule_off", "label": "Paused: schedule off"}
    if running:
        return {"state": "running", "label": "Running"}
    return {"state": "waiting", "label": "Waiting for next run"}
