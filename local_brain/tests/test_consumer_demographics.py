import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from local_brain.command_center import consumer_demographics as module

REAL_CLIENT = httpx.AsyncClient


@pytest.fixture
def upstream(monkeypatch):
    data = {}
    for state in module.GEOGRAPHIES:
        scale = 2 if state == "06" else 1
        tables = {
            "B01001": {f"B01001{i:03d}": 10 * scale for i in range(1, 50)},
            "B01002": {"B01002001": 38.5},
            "B11001": {"B11001001": 16 * scale},
            "B19013": {"B19013001": 80000 * scale},
            "B19001": {f"B19001{i:03d}": scale for i in range(1, 18)},
        }
        tables["B01001"]["B01001001"] = 460 * scale
        tables["B19001"]["B19001001"] = 16 * scale
        geoid = "01000US" if state == "us" else "04000US" + state
        data[geoid] = {table: {"estimate": cells} for table, cells in tables.items()}
    payload = {
        "release": {"id": "acs2024_5yr", "years": "2020-2024"},
        "data": data,
    }
    state = {"payload": payload, "calls": [], "failure": None, "delay": False}

    async def handler(request):
        state["calls"].append(request)
        if state["delay"]:
            await asyncio.sleep(0.01)
        if state["failure"] == "timeout":
            raise httpx.ReadTimeout("private transport detail", request=request)
        if state["failure"] == "redirect":
            return httpx.Response(302, headers={"Location": "https://example.invalid"})
        if state["failure"] == "oversized":
            return httpx.Response(200, content=b" " * (module.MAX_RESPONSE_BYTES + 1))
        if state["failure"] == "html":
            return httpx.Response(200, text="<html>Upstream unavailable</html>")
        if state["failure"] == "http":
            return httpx.Response(503, text="private upstream detail")
        return httpx.Response(200, json=state["payload"])

    def client(**kwargs):
        return REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(
        module,
        "httpx",
        SimpleNamespace(AsyncClient=client, Timeout=httpx.Timeout, HTTPError=httpx.HTTPError),
    )
    monkeypatch.setattr(module, "cache", module.DemographicsCache())
    return state


async def test_us_state_contract_and_real_column_aggregation(upstream):
    us = await module.demographics()
    california = await module.demographics("06")
    assert len(us["geographies"]) == 52
    assert us["metrics"] == {
        "population": 460,
        "households": 16,
        "median_household_income": 80000,
        "median_age": 38.5,
    }
    assert california["metrics"]["population"] == 920
    assert california["geography"] == {"id": "06", "name": "California"}
    assert us["age_bands"][3] == {"label": "15–19", "value": 40, "share": 8.7}
    assert us["income_bands"][4] == {"label": "$35k–50k", "value": 3, "share": 18.75}
    assert sum(band["value"] for band in us["age_bands"]) == 460
    assert sum(band["value"] for band in us["income_bands"]) == 16
    assert us["status"] == "ok" and us["error"] is None
    assert us["year"] == 2024 and us["period"] == "2020–2024"
    assert us["fetched_at"].endswith("+00:00")
    assert us["source"]["provider"] == "Census Reporter"
    request = upstream["calls"][0]
    assert str(request.url).startswith(module.API_URL)
    assert request.url.params["geo_ids"] == "01000US,040|01000US"
    assert len(upstream["calls"]) == 1
    us["metrics"]["population"] = 0
    us["source"]["tables"].clear()
    again = await module.demographics()
    assert again["metrics"]["population"] == 460 and again["source"]["tables"]


async def test_sentinels_and_partial_bands_remain_null(upstream):
    row = upstream["payload"]["data"]["01000US"]
    row["B19013"]["estimate"]["B19013001"] = -666666666
    row["B01002"]["estimate"]["B01002001"] = None
    row["B01001"]["estimate"]["B01001003"] = "-888888888"
    row["B19001"]["estimate"]["B19001009"] = "not available"
    row["B19001"]["estimate"]["B19001002"] = 0
    result = await module.demographics()
    assert result["metrics"]["median_household_income"] is None
    assert result["metrics"]["median_age"] is None
    assert result["age_bands"][0]["value"] is None
    assert result["age_bands"][0]["share"] is None
    assert result["income_bands"][4]["value"] is None
    assert result["income_bands"][0]["value"] == 0
    assert result["income_bands"][0]["share"] == 0


async def test_zero_or_missing_denominator_never_becomes_zero_share(upstream):
    row = upstream["payload"]["data"]["01000US"]
    row["B01001"]["estimate"]["B01001001"] = 0
    row["B19001"]["estimate"]["B19001001"] = None
    result = await module.demographics()
    assert all(band["share"] is None for band in result["age_bands"])
    assert all(band["share"] is None for band in result["income_bands"])


async def test_single_flight_and_24_hour_cache(upstream, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    upstream["delay"] = True
    results = await asyncio.gather(*(module.demographics() for _ in range(10)))
    assert all(row["status"] == "ok" for row in results)
    assert len(upstream["calls"]) == 1
    clock[0] += module.CACHE_SECONDS - 1
    await module.demographics("11")
    assert len(upstream["calls"]) == 1
    clock[0] += 1
    await module.demographics("11")
    assert len(upstream["calls"]) == 2


async def test_failed_refresh_retains_snapshot_and_retries_after_cooldown(upstream, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    first = await module.demographics()
    clock[0] += module.CACHE_SECONDS
    upstream["failure"] = "timeout"
    failed = await module.demographics()
    assert failed["status"] == "stale"
    assert failed["metrics"] == first["metrics"]
    assert failed["fetched_at"] == first["fetched_at"]
    assert "last retrieved" in failed["error"]
    await module.demographics("06")
    assert len(upstream["calls"]) == 2
    clock[0] += module.RETRY_SECONDS
    upstream["failure"] = None
    recovered = await module.demographics()
    assert recovered["status"] == "ok" and recovered["error"] is None
    assert len(upstream["calls"]) == 3


@pytest.mark.parametrize("failure", ["timeout", "redirect", "oversized", "html", "http"])
async def test_upstream_failure_is_unavailable_and_never_exposes_details(upstream, failure):
    upstream["failure"] = failure
    result = await module.demographics()
    assert result["status"] == "unavailable" and result["fetched_at"] is None
    assert all(value is None for value in result["metrics"].values())
    assert all(row["value"] is None for row in result["age_bands"])
    assert "private" not in str(result)
    await module.demographics("06")
    assert len(upstream["calls"]) == 1


@pytest.mark.parametrize("malformed", ["release", "period", "region", "table", "column", "shape"])
async def test_incomplete_or_wrong_release_does_not_replace_valid_snapshot(upstream, malformed):
    first = await module.demographics()
    payload = deepcopy(upstream["payload"])
    if malformed == "release":
        payload["release"]["id"] = "acs2023_5yr"
    elif malformed == "period":
        payload["release"]["years"] = "2019-2023"
    elif malformed == "region":
        del payload["data"]["04000US11"]
    elif malformed == "table":
        del payload["data"]["01000US"]["B19013"]
    elif malformed == "column":
        del payload["data"]["01000US"]["B01001"]["estimate"]["B01001049"]
    else:
        payload = ["invalid"]
    upstream["payload"] = payload
    module.cache._refresh_after = 0
    result = await module.demographics()
    assert result["status"] == "stale" and result["metrics"] == first["metrics"]


async def test_endpoint_validates_geography_before_fetch(upstream):
    app = FastAPI()
    app.include_router(module.router)
    async with REAL_CLIENT(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for invalid in ("00", "03", "72", "6", "CA", "state:*", "https://example.com"):
            response = await client.get(
                "/api/consumer-insights/demographics", params={"state": invalid}
            )
            assert response.status_code == 422
        assert not upstream["calls"]
        response = await client.get("/api/consumer-insights/demographics")
        assert response.status_code == 200 and response.json()["geography"]["id"] == "us"
