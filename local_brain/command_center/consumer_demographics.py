"""Public ACS estimates, delivered by Census Reporter with bounded caching."""

import asyncio
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from time import monotonic
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException

YEAR = 2024
PERIOD = "2020–2024"
RELEASE = "acs2024_5yr"
TABLES = ("B01001", "B01002", "B11001", "B19013", "B19001")
API_URL = "https://api.censusreporter.org/1.0/data/show/" + RELEASE
PARAMS = {"table_ids": ",".join(TABLES), "geo_ids": "01000US,040|01000US"}
SOURCE = {
    "name": "U.S. Census Bureau · ACS via Census Reporter",
    "provider": "Census Reporter",
    "url": "https://www.census.gov/programs-surveys/acs",
    "api_url": API_URL + "?" + urlencode(PARAMS),
    "tables": list(TABLES),
}
CACHE_SECONDS = 24 * 60 * 60
RETRY_SECONDS = 5 * 60
MAX_RESPONSE_BYTES = 1_000_000
GEOGRAPHIES = {
    "us": "United States",
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
}

# B01001 supplies matching male and female age columns, offset by 24.
AGE_BANDS = (
    ("Under 5", (3,)),
    ("5–9", (4,)),
    ("10–14", (5,)),
    ("15–19", (6, 7)),
    ("20–24", (8, 9, 10)),
    ("25–34", (11, 12)),
    ("35–44", (13, 14)),
    ("45–54", (15, 16)),
    ("55–59", (17,)),
    ("60–64", (18, 19)),
    ("65–74", (20, 21, 22)),
    ("75–84", (23, 24)),
    ("85+", (25,)),
)
INCOME_BANDS = (
    ("Under $10k", (2,)),
    ("$10k–15k", (3,)),
    ("$15k–25k", (4, 5)),
    ("$25k–35k", (6, 7)),
    ("$35k–50k", (8, 9, 10)),
    ("$50k–75k", (11, 12)),
    ("$75k–100k", (13,)),
    ("$100k–150k", (14, 15)),
    ("$150k–200k", (16,)),
    ("$200k+", (17,)),
)
METRICS = {
    "population": "B01001001",
    "households": "B11001001",
    "median_household_income": "B19013001",
    "median_age": "B01002001",
}


def _estimate(raw):
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        return None
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return None
    # Census uses negative sentinels for suppressed or unavailable estimates.
    if not math.isfinite(value) or value < 0:
        return None
    return int(value) if value.is_integer() else value


def _bands(estimates, definitions, table, denominator, offsets=(0,)):
    result = []
    for label, columns in definitions:
        values = [
            estimates[f"{table}{column + offset:03d}"] for column in columns for offset in offsets
        ]
        value = None if None in values else sum(values)
        share = None
        if denominator and value is not None and value <= denominator:
            share = round(100 * value / denominator, 2)
        result.append({"label": label, "value": value, "share": share})
    return result


def _parse_snapshot(payload):
    release = payload["release"]
    if release["id"] != RELEASE or release["years"] != "2020-2024":
        raise ValueError("unexpected_acs_release")
    data = payload["data"]
    if not isinstance(data, dict) or len(data) > 60:
        raise ValueError("invalid_geographies")
    rows = {}
    for state in GEOGRAPHIES:
        geoid = "01000US" if state == "us" else "04000US" + state
        estimates = {}
        for table in TABLES:
            table_estimates = data[geoid][table]["estimate"]
            if not isinstance(table_estimates, dict):
                raise ValueError("invalid_estimates")
            estimates.update({key: _estimate(raw) for key, raw in table_estimates.items()})
        metrics = {name: estimates[column] for name, column in METRICS.items()}
        rows[state] = {
            "metrics": metrics,
            "age_bands": _bands(estimates, AGE_BANDS, "B01001", metrics["population"], (0, 24)),
            "income_bands": _bands(estimates, INCOME_BANDS, "B19001", estimates["B19001001"]),
        }
    return rows


async def _fetch_snapshot():
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5), follow_redirects=False) as client,
        client.stream("GET", API_URL, params=PARAMS) as response,
    ):
        response.raise_for_status()
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("census_response_too_large")
    return _parse_snapshot(json.loads(body))


class DemographicsCache:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._rows = {}
        self._fetched_at = None
        self._refresh_after = 0.0
        self._error = False

    async def get(self, state):
        if monotonic() >= self._refresh_after:
            async with self._lock:
                if monotonic() >= self._refresh_after:
                    try:
                        rows = await asyncio.wait_for(_fetch_snapshot(), timeout=20)
                    except (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError, TypeError):
                        self._error = True
                        self._refresh_after = monotonic() + RETRY_SECONDS
                    else:
                        self._rows = rows
                        self._fetched_at = datetime.now(timezone.utc).isoformat()
                        self._error = False
                        self._refresh_after = monotonic() + CACHE_SECONDS
        selected = self._rows.get(state)
        status = "stale" if self._error and selected else "ok"
        error = None
        if self._error:
            error = "Census refresh failed. Showing the last retrieved estimates."
        if not selected:
            status = "unavailable"
            error = "Census estimates are temporarily unavailable."
            selected = {
                "metrics": dict.fromkeys(METRICS),
                "age_bands": [
                    {"label": label, "value": None, "share": None} for label, _ in AGE_BANDS
                ],
                "income_bands": [
                    {"label": label, "value": None, "share": None} for label, _ in INCOME_BANDS
                ],
            }
        return {
            "status": status,
            "geography": {"id": state, "name": GEOGRAPHIES[state]},
            "geographies": [{"id": key, "name": name} for key, name in GEOGRAPHIES.items()],
            "year": YEAR,
            "period": PERIOD,
            "source": deepcopy(SOURCE),
            "fetched_at": self._fetched_at,
            "error": error,
            **deepcopy(selected),
        }


cache = DemographicsCache()
router = APIRouter(prefix="/api/consumer-insights")


@router.get("/demographics")
async def demographics(state: str = "us"):
    if state not in GEOGRAPHIES:
        raise HTTPException(422, "Choose us or a supported two-digit state/DC FIPS code.")
    return await cache.get(state)
