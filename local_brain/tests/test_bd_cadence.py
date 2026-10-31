from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from local_brain.command_center.bd_cadence import (
    cadence_interval,
    cadence_open,
    eligible_bots,
    next_cadence_timestamp,
    next_open_timestamp,
    next_query_for,
    rotate_bot,
    slot_index,
    status_banner,
    validate_cadence,
)

CHICAGO = ZoneInfo("America/Chicago")
CADENCE = {
    "timezone": "America/Chicago",
    "weekday": {"start_hour": 8, "end_hour": 18, "interval_seconds": 1800},
    "weekend": {"start_hour": 9, "end_hour": 17, "interval_seconds": 3600},
}


def at(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=CHICAGO).timestamp()


def test_default_roster_cadence_is_valid():
    import json

    from local_brain.command_center.bd_conductor import CONFIG_PATH

    config = json.loads(CONFIG_PATH.read_text())
    validate_cadence(config["cadence"])
    assert config["schedule_enabled"] is False


@pytest.mark.parametrize(
    "cadence",
    [
        None,
        {},
        {"timezone": "America/Chicago"},
        {
            "timezone": "Not/AZone",
            "weekday": {"start_hour": 8, "end_hour": 18, "interval_seconds": 1800},
            "weekend": {"start_hour": 9, "end_hour": 17, "interval_seconds": 3600},
        },
        {
            "timezone": "America/Chicago",
            "weekday": {"start_hour": 18, "end_hour": 8, "interval_seconds": 1800},
            "weekend": {"start_hour": 9, "end_hour": 17, "interval_seconds": 3600},
        },
        {
            "timezone": "America/Chicago",
            "weekday": {"start_hour": 8, "end_hour": 18, "interval_seconds": 30},
            "weekend": {"start_hour": 9, "end_hour": 17, "interval_seconds": 3600},
        },
    ],
)
def test_invalid_cadence_fails_closed(cadence):
    with pytest.raises(ValueError):
        validate_cadence(cadence)


def test_weekday_and_weekend_hours_and_intervals():
    weekday = at(2026, 10, 27, 10)  # Tuesday
    early = at(2026, 10, 27, 7, 59)
    late = at(2026, 10, 27, 18)
    sunday = at(2026, 10, 25, 11)
    sunday_early = at(2026, 10, 25, 8, 30)
    assert cadence_open(CADENCE, weekday) is True
    assert cadence_open(CADENCE, early) is False
    assert cadence_open(CADENCE, late) is False
    assert cadence_open(CADENCE, sunday) is True
    assert cadence_open(CADENCE, sunday_early) is False
    assert cadence_interval(CADENCE, weekday) == 1800
    assert cadence_interval(CADENCE, sunday) == 3600
    assert next_open_timestamp(CADENCE, early) == at(2026, 10, 27, 8)
    assert next_open_timestamp(CADENCE, late) == at(2026, 10, 28, 8)
    assert next_open_timestamp(CADENCE, at(2026, 10, 24, 18)) == at(2026, 10, 25, 9)


def test_rotation_spreads_one_bot_per_slot():
    bots = [
        {
            "id": "healthcare",
            "tenant_id": "synthetic-healthcare",
            "enabled": True,
            "segment": "healthcare",
        },
        {
            "id": "pe-diligence",
            "tenant_id": "synthetic-pe",
            "enabled": True,
            "segment": "pe-technical-diligence",
        },
        {
            "id": "alumni-associations",
            "tenant_id": "synthetic-alumni",
            "enabled": True,
            "segment": "alumni-associations",
        },
    ]
    first = slot_index(CADENCE, at(2026, 10, 27, 10))
    assert {rotate_bot(bots, first + offset)["id"] for offset in range(3)} == {
        "healthcare",
        "pe-diligence",
        "alumni-associations",
    }
    existing = {
        (
            bots[first % 3]["tenant_id"],
            f"{bots[first % 3]['id']}:{first}",
            "public-research",
        )
    }
    nxt = next_cadence_timestamp(CADENCE, at(2026, 10, 27, 10), existing, bots)
    assert nxt is not None
    later_slot = slot_index(CADENCE, nxt)
    assert later_slot != first
    assert rotate_bot(bots, later_slot)["id"] != rotate_bot(bots, first)["id"]


def test_client_bots_need_a_bd_contract():
    srg = {
        "id": "srg",
        "enabled": True,
        "kind": "client",
        "segment": "client-bd",
        "tenant_id": "synthetic-srg",
        "schedule": {"enabled": True},
    }
    contracted = {
        **srg,
        "research_segment": "healthcare",
        "managed_leads": {
            "base_url": "https://managed-leads.vercel.app",
            "workspace_id": "synthetic-srg",
        },
    }
    healthcare = {
        "id": "healthcare",
        "enabled": True,
        "segment": "healthcare",
        "tenant_id": "synthetic-healthcare",
        "schedule": {"enabled": False},
    }
    assert [bot["id"] for bot in eligible_bots([healthcare, srg], standing=True)] == ["healthcare"]
    assert [bot["id"] for bot in eligible_bots([healthcare, contracted], standing=True)] == [
        "healthcare",
        "srg",
    ]
    assert [bot["id"] for bot in eligible_bots([healthcare, srg], shadow=True)] == [
        "healthcare",
        "srg",
    ]


def test_banner_states_are_explicit():
    assert status_banner(True, False) == {
        "state": "paused_kill_switch",
        "label": "Paused: kill switch on",
    }
    assert status_banner(True, True) == {
        "state": "paused_kill_switch",
        "label": "Paused: kill switch on",
    }
    assert status_banner(False, False) == {
        "state": "paused_schedule_off",
        "label": "Paused: schedule off",
    }
    assert status_banner(False, True) == {"state": "waiting", "label": "Waiting for next run"}
    assert status_banner(False, True, True) == {"state": "running", "label": "Running"}


def test_next_query_explains_inconclusive_reasons():
    assert "named" in next_query_for("healthcare_entity_or_event_unconfirmed").lower()
    assert "qwen3:8b" in next_query_for("qwen_unavailable")
    assert next_query_for("") != ""
