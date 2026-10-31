"""Bounded Delaware public feed adapter; unknown solicitation types are withheld."""

import hashlib
import json
import re
from datetime import date
from html.parser import HTMLParser

BASE = "https://mmp.delaware.gov"
SOURCE = BASE + "/Bids/"
MAX_BYTES = 1_000_000


class Agencies(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cells, self.text = {}, [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells = []
        if tag == "td":
            self.text = []

    def handle_data(self, text):
        if self.text is not None:
            self.text.append(text)

    def handle_endtag(self, tag):
        if tag == "td" and self.text is not None:
            self.cells.append(" ".join("".join(self.text).split()))
            self.text = None
        if tag == "tr" and len(self.cells) == 2:
            self.rows[self.cells[0]] = self.cells[1]


def _read(client, method, path, kind, **kwargs):
    with client.stream(method, BASE + path, follow_redirects=False, **kwargs) as response:
        if (
            response.status_code != 200
            or kind not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("delaware_source_unavailable")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise ValueError("delaware_source_too_large")
    return body.decode("utf-8", errors="strict")


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def fetch(client, source, today):
    """One page, at most 100 rows. Fail on truncation rather than claim coverage.

    Same-day deadlines are withheld because the listing omits cutoff time.
    Agency identity comes from the official table, never acronym guesses.
    """
    if source.get("state") not in ("DE", "Delaware") or source.get("url") != SOURCE:
        raise ValueError("invalid_delaware_source")
    current = date.fromisoformat(today)
    raw = json.loads(
        _read(
            client,
            "POST",
            "/Bids/GetBids?status=Open",
            "json",
            json={
                "page": 1,
                "rows": 100,
                "sidx": "OpenDate",
                "sord": "desc",
                "_search": False,
            },
        )
    )
    if (
        not isinstance(raw, dict)
        or type(raw.get("records")) is not int
        or type(raw.get("page")) is not int
        or raw["page"] != 1
        or type(raw.get("total")) is not int
        or raw["total"] not in (0, 1)
        or not isinstance(raw.get("rows"), list)
        or not 0 <= raw["records"] <= 100
        or len(raw["rows"]) != raw["records"]
    ):
        raise ValueError("delaware_feed_incomplete")
    agencies = Agencies()
    agencies.feed(_read(client, "GET", "/Agency/Index", "html"))
    if not agencies.rows:
        raise ValueError("delaware_agency_mapping_unavailable")
    found = {}
    for row in raw["rows"]:
        if not isinstance(row, dict):
            raise ValueError("delaware_invalid_row")
        bid_id, title = row.get("Id"), row.get("Title")
        if (
            type(bid_id) is not int
            or bid_id <= 0
            or not isinstance(title, str)
            or len(title) > 1000
        ):
            raise ValueError("delaware_invalid_row")
        try:
            deadline = date.fromisoformat(row.get("DeadlineDate", ""))
            opened = date.fromisoformat(row.get("OpenDate", ""))
        except (ValueError, TypeError):
            continue
        if deadline <= current or opened > current:
            continue
        agency = agencies.rows.get(row.get("AgencyCode"), "")
        if not re.search(r"\bschool\b|\beducation\b", agency, re.I):
            continue
        kind = row.get("Type")
        if kind is not None and not isinstance(kind, str):
            continue
        if kind:
            if not re.fullmatch(
                r"RFP|RFI|Request for Proposals?|Request for Information", kind.strip(), re.I
            ):
                continue
        else:
            match = re.search(
                r"\b(RFP|RFI)\b|\bRequest for (?:Proposals?|Information)\b", title, re.I
            )
            if not match:
                continue
            kind = match.group()
        url = f"{BASE}/Bids/Details/{bid_id}"
        quote = json.dumps(
            {
                "title": title,
                "agency": agency,
                "agency_code": row.get("AgencyCode"),
                "type": kind,
                "open_date": opened.isoformat(),
                "deadline_date": deadline.isoformat(),
                "deadline_time": "unknown; check official notice",
                "listing_status": "Open",
                "as_of": today,
            },
            sort_keys=True,
        )
        item = {"state": "DE", "url": url, "source_url": SOURCE, "title": title, "quote": quote}
        item["identity"] = _hash({"state": "DE", "url": url})
        item["hash"] = _hash(item)
        found[item["identity"]] = item
    return list(found.values())
