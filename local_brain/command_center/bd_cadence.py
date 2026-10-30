"""Standing BD research cadence. Importing this module starts no work.

Working hours, weekend intervals, and an opt-in overnight window live in
versioned roster config. Overnight is off by default; only bots marked
around_the_clock run in that window, at a lower rate. The worker only
enqueues during an open window; the kill switch and schedule_enabled flag
still gate execution.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

RESEARCH_SEGMENTS = {
    "healthcare",
    "pe-technical-diligence",
    "alumni-associations",
    "commercial-fit-out",
}
WEEKDAY_KEYS = {"start_hour", "end_hour", "interval_seconds"}
CADENCE_REQUIRED = {"timezone", "weekday", "weekend"}
OVERNIGHT_KEYS = {"enabled", "interval_seconds"}
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
    "generic_permit_reference": (
        "Require a named business or property on a Minneapolis commercial permit, not a generic theme"
    ),
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
    "permit_fields_not_in_source": (
        "Find a Minneapolis building permit with number, address, project title, and status"
    ),
    "qwen_unavailable": "Warm qwen3:8b and retry this source after the model is resident",
    "source_evidence_missing": "Retain the source quote and URL before treating this as a candidate",
    "source_evidence_unverified": "Re-check grounding: named organization, quote, and supported claim",
    "segment_rejected": "Move to a different source; this segment judgment rejected the evidence",
}


def _interval(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 60
        or value != int(value)
    ):
        raise ValueError(f"{name} must be an integer of at least 60")
    return int(value)


def validate_cadence(cadence):
    if not isinstance(cadence, dict):
        raise ValueError("cadence must be an object")
    extra = set(cadence) - CADENCE_REQUIRED - {"overnight"}
    if extra or not set(cadence) >= CADENCE_REQUIRED:
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
        _interval(part["interval_seconds"], f"cadence.{name}.interval_seconds")
    overnight = cadence.get("overnight")
    if overnight is not None:
        if not isinstance(overnight, dict) or set(overnight) != OVERNIGHT_KEYS:
            raise ValueError("cadence.overnight requires enabled and interval_seconds")
        if not isinstance(overnight["enabled"], bool):
            raise ValueError("cadence.overnight.enabled must be boolean")
        _interval(overnight["interval_seconds"], "cadence.overnight.interval_seconds")


def local_now(timestamp, zone):
    return datetime.fromtimestamp(timestamp, timezone.utc).astimezone(ZoneInfo(zone))


def daypart(cadence, timestamp):
    local = local_now(timestamp, cadence["timezone"])
    return cadence["weekend"] if local.weekday() >= 5 else cadence["weekday"]


def working_hours(cadence, timestamp):
    local = local_now(timestamp, cadence["timezone"])
    part = daypart(cadence, timestamp)
    return part["start_hour"] <= local.hour < part["end_hour"]


def overnight_enabled(cadence):
    overnight = cadence.get("overnight") if isinstance(cadence, dict) else None
    return bool(isinstance(overnight, dict) and overnight.get("enabled") is True)


def cadence_window(cadence, timestamp):
    if working_hours(cadence, timestamp):
        local = local_now(timestamp, cadence["timezone"])
        return "weekend" if local.weekday() >= 5 else "weekday"
    if overnight_enabled(cadence):
        return "overnight"
    return None


def cadence_open(cadence, timestamp):
    return cadence_window(cadence, timestamp) is not None


def cadence_interval(cadence, timestamp):
    window = cadence_window(cadence, timestamp)
    if window == "overnight":
        return int(cadence["overnight"]["interval_seconds"])
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
        return bot.get("research_segment") in RESEARCH_SEGMENTS
    return segment in RESEARCH_SEGMENTS


def around_the_clock(bot):
    schedule = bot.get("schedule") if isinstance(bot, dict) else None
    return bool(isinstance(schedule, dict) and schedule.get("around_the_clock") is True)


def eligible_bots(
    bots,
    *,
    shadow=False,
    standing=False,
    paused=(),
    overnight=False,
    focus=None,
):
    paused = set(paused or ())
    chosen = []
    for bot in bots:
        if not bot.get("enabled"):
            continue
        if bot.get("id") in paused:
            continue
        if shadow:
            chosen.append(bot)
            continue
        if not has_research_contract(bot):
            continue
        if standing or bot.get("schedule", {}).get("enabled"):
            if overnight and not around_the_clock(bot):
                continue
            chosen.append(bot)
    if focus and chosen:
        kind = focus.get("kind")
        ident = focus.get("id")
        if kind == "bot":
            chosen = [bot for bot in chosen if bot.get("id") == ident]
        elif kind == "segment":
            chosen = [
                bot
                for bot in chosen
                if bot.get("segment") == ident or bot.get("research_segment") == ident
            ]
    return chosen


def rotate_bot(eligible, slot):
    if not eligible:
        return None
    return eligible[slot % len(eligible)]


def next_cadence_timestamp(cadence, timestamp, existing, eligible, *, overnight_eligible=None):
    if overnight_eligible is None:
        overnight_eligible = [bot for bot in eligible if around_the_clock(bot)]
    cursor = timestamp
    deadline = timestamp + 8 * 86400
    while cursor < deadline:
        window = cadence_window(cadence, cursor)
        if window is None:
            opened = next_open_timestamp(cadence, cursor)
            if opened is None:
                return None
            cursor = opened
            continue
        pool = overnight_eligible if window == "overnight" else eligible
        if not pool:
            if window == "overnight":
                opened = next_open_timestamp(
                    {
                        **cadence,
                        "overnight": {**(cadence.get("overnight") or {}), "enabled": False},
                    },
                    cursor,
                )
                if opened is None or opened <= cursor:
                    return None
                cursor = opened
                continue
            if not eligible:
                return cursor
            cursor = (int(cursor // cadence_interval(cadence, cursor)) + 1) * cadence_interval(
                cadence, cursor
            )
            continue
        interval = cadence_interval(cadence, cursor)
        slot = int(cursor // interval)
        bot = rotate_bot(pool, slot)
        key = (bot["tenant_id"], f"{bot['id']}:{slot}", "public-research")
        if key not in existing:
            return cursor
        cursor = (slot + 1) * interval
    return None


def next_query_for(reason):
    if not isinstance(reason, str) or not reason:
        return NEXT_QUERY["source_evidence_unverified"]
    return NEXT_QUERY.get(reason, reason.replace("_", " "))


def status_banner(kill_switch, schedule_enabled):
    if kill_switch:
        return {"state": "paused_kill_switch", "label": "Paused: kill switch on"}
    if not schedule_enabled:
        return {"state": "paused_schedule_off", "label": "Paused: schedule off"}
    return {"state": "running", "label": "Running"}
