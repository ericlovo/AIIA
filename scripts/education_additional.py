"""Verified Iowa DOE JSON and Indiana DOE table subsets; other states documented."""

import json
import re
import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo

from scripts.education_agency_pages import Node, Tree, _hash

SOURCES = {
    "IA": "https://bidopportunities.iowa.gov/",
    "IN": "https://www.in.gov/idoa/procurement/current-business-opportunities/",
}
IOWA_AGENCY = "548843e7-193a-e611-80f0-0050569c6707"
IOWA_API = (
    SOURCES["IA"] + "Home/DT_HostedBidsSearch?agencyId=" + IOWA_AGENCY + "&enteredSearchText="
)
MAX_BYTES = 2_000_000
MAX_RECORDS = 500


def _record(state, identity, title, deadline, kind, url):
    item = {
        "state": state,
        "url": url,
        "source_url": SOURCES[state],
        "title": title,
        "quote": f"{title}. Solicitation type: {kind}. Buyer: Department of Education. Response deadline: {deadline.isoformat()}.",
        "notice_number": identity,
        "deadline_date": deadline.isoformat(),
        "buyer": ("Iowa" if state == "IA" else "Indiana") + " Department of Education",
        "coverage_scope": "State Department of Education only; not all school districts",
    }
    item["notice_id"] = _hash({"state": state, "url": url, "notice_number": identity})
    item["hash"] = _hash(item)
    return item


def _kind(text):
    if re.search(r"\b(cancelled|canceled|withdrawn|awarded|closed|expired)\b", text, re.I):
        return None
    if re.search(
        r"\bgrant applications?\b|\bgrant awards?\b|\buniversity\b|\bcollege\b|\bhigher education\b",
        text,
        re.I,
    ):
        return None
    if not re.search(
        r"\bk[- ]?12\b|\bschools?\b|\bstudents?\b|\belementary\b|\bsecondary\b|\bteachers?\b",
        text,
        re.I,
    ):
        return None
    if re.search(r"\bRFP\b|Request for Proposals?", text, re.I):
        return "RFP"
    if re.search(r"\bRFI\b|Request for Information", text, re.I):
        return "RFI"
    return None


def _iowa_date(value):
    match = re.fullmatch(r"/Date\((-?\d{1,15})\)/", value) if isinstance(value, str) else None
    if not match:
        raise ValueError("iowa_date_invalid")
    try:
        return datetime.fromtimestamp(
            int(match.group(1)) / 1000, ZoneInfo("America/Chicago")
        ).date()
    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError("iowa_date_invalid") from exc


def parse_iowa(data, today):
    if (
        not isinstance(data, dict)
        or type(data.get("iTotalRecords")) is not int
        or not isinstance(data.get("aaData"), list)
    ):
        raise ValueError("iowa_schema_changed")
    rows = data["aaData"]
    # Live endpoint returns the complete filtered collection, not requested page sizes.
    if not 0 <= data["iTotalRecords"] <= MAX_RECORDS or len(rows) != data["iTotalRecords"]:
        raise ValueError("iowa_incomplete_results")
    records = []
    seen = set()
    for row in rows:
        required = [
            "ID",
            "BidNumber",
            "AgencyID",
            "AgencyName",
            "Solicitation",
            "Description",
            "EffectiveDate",
            "ExpirationDate",
            "IsActive",
            "Status",
        ]
        if not isinstance(row, dict) or any(k not in row for k in required):
            raise ValueError("iowa_record_schema_changed")
        if (
            not isinstance(row["AgencyID"], str)
            or row["AgencyID"].lower() != IOWA_AGENCY
            or row["AgencyName"] != "Education, Dept Of"
        ):
            raise ValueError("iowa_agency_mismatch")
        try:
            identity = str(uuid.UUID(row["ID"]))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("iowa_record_id_invalid") from exc
        if identity in seen:
            raise ValueError("iowa_duplicate_notice")
        seen.add(identity)
        if row["IsActive"] is not True or row["Status"] != "Open":
            continue
        deadline = _iowa_date(row["ExpirationDate"])
        if not _iowa_date(row["EffectiveDate"]) <= date.fromisoformat(today) < deadline:
            continue
        text = " ".join(str(row[k]) for k in ["BidNumber", "Solicitation", "Description"])
        kind = _kind(text)
        if not kind:
            continue
        # Contact fields and free-text descriptions are inspected but never persisted.
        if (
            not isinstance(row["Solicitation"], str)
            or len(row["Solicitation"]) > 1000
            or not isinstance(row["BidNumber"], str)
            or len(row["BidNumber"]) > 150
        ):
            raise ValueError("iowa_title_invalid")
        records.append(
            _record(
                "IA",
                row["BidNumber"],
                row["Solicitation"],
                deadline,
                kind,
                SOURCES["IA"] + "Home/BidInfo?bidId=" + identity,
            )
        )
    return records


def parse_indiana(body, today):
    tree = Tree(body)
    if re.search(
        r"access denied|request rejected|verify you are human|captcha challenge",
        tree.root.text(),
        re.I,
    ):
        raise ValueError("indiana_source_challenge")
    tables = [
        n for n in tree.root.walk() if n.tag == "table" and n.attrs.get("id") == "events-table"
    ]
    if len(tables) != 1:
        raise ValueError("indiana_events_missing")
    table = tables[0]
    if any(
        n.tag == "a" and ("Page$" in n.attrs.get("href", "") or n.text().lower() == "next")
        for n in table.walk()
    ):
        raise ValueError("indiana_unhandled_pagination")
    headers = None
    records = []
    count = 0
    seen = set()
    for tr in table.walk():
        if tr.tag != "tr":
            continue
        cells = [n for n in tr.children if isinstance(n, Node) and n.tag in ("th", "td")]
        texts = [n.text() for n in cells]
        if texts == [
            "Event Name",
            "Agency",
            "Event ID",
            "Event Description",
            "Response Due By",
            "Contact",
        ]:
            headers = texts
            continue
        if not headers:
            continue
        if len(cells) != 6:
            raise ValueError("indiana_row_shape_changed")
        count += 1
        if count > MAX_RECORDS:
            raise ValueError("indiana_record_budget")
        row = dict(zip(headers, texts, strict=True))
        if row["Agency"] not in (
            "Education",
            "Department of Education",
            "Indiana Department of Education",
        ):
            continue
        identity = row["Event ID"]
        if not re.fullmatch(r"\d{5,25}", identity) or identity in seen:
            raise ValueError("indiana_notice_id_invalid")
        seen.add(identity)
        kind = _kind(row["Event Name"] + " " + row["Event Description"])
        if not kind:
            continue
        try:
            deadline = datetime.strptime(row["Response Due By"].split()[0], "%m/%d/%Y").date()
        except (ValueError, IndexError) as exc:
            raise ValueError("indiana_deadline_invalid") from exc
        if deadline <= date.fromisoformat(today):
            continue
        title = row["Event Name"].removesuffix(" Bid Documents")
        if len(title) > 1000:
            raise ValueError("indiana_title_invalid")
        # Portal table is stable URL; event identifier disambiguates notice identity.
        records.append(_record("IN", identity, title, deadline, kind, SOURCES["IN"]))
    if headers is None or not count:
        raise ValueError("indiana_empty_structure_unverified")
    return records


def fetch(client, source, today):
    state = {"Iowa": "IA", "Indiana": "IN"}.get(source.get("state"), source.get("state"))
    if state not in SOURCES or source.get("url") != SOURCES[state]:
        raise ValueError("invalid_additional_source")
    url = IOWA_API if state == "IA" else SOURCES[state]
    with client.stream("GET", url, follow_redirects=False, timeout=20) as response:
        expected = "json" if state == "IA" else "html"
        if (
            response.status_code != 200
            or expected not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("additional_source_unavailable")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError("additional_source_too_large")
    if state == "IA":
        return parse_iowa(json.loads(content), today)
    return parse_indiana(content.decode("utf-8", errors="strict"), today)
