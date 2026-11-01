"""Public Jaggaer MT/NM listings. No login, signed URLs, or attachments retained."""

import re
from datetime import date, datetime
from urllib.parse import urlencode

from scripts.education_agency_pages import Tree, _hash

BASE = "https://bids.sciquest.com/apps/Router/PublicEvent"
ORGS = {"MT": "StateOfMontana", "NM": "StateOfNewMexico"}
SOURCES = {
    state: BASE + "?" + urlencode({"CustomerOrg": org, "tab": "PHX_NAV_SourcingOpenForBid"})
    for state, org in ORGS.items()
}
MAX_BYTES = 3_000_000
PAGE_SIZE = 200
MAX_PAGES = 7


def parse_page(body, state, today):
    if state not in ORGS:
        raise ValueError("invalid_jaggaer_state")
    tree = Tree(body)
    visible = tree.root.text()
    if re.search(
        r"access denied|request rejected|verify you are human|captcha challenge", visible, re.I
    ):
        raise ValueError("jaggaer_source_challenge")
    if ORGS[state] + " Public Bid Opportunities" not in visible:
        raise ValueError("jaggaer_workspace_mismatch")
    ranges = set(re.findall(r"\b(\d+)-(\d+) of (\d+) Results\b", visible))
    if len(ranges) != 1:
        if re.search(r"\b0 Results\b", visible):
            return [], {"start": 0, "end": 0, "total": 0, "records_checked": 0}
        raise ValueError("jaggaer_pagination_missing")
    start, end, total = map(int, next(iter(ranges)))
    if not 1 <= start <= end <= total <= PAGE_SIZE * MAX_PAGES or end - start + 1 > PAGE_SIZE:
        raise ValueError("jaggaer_pagination_invalid")
    rows = []
    scanned = 0
    numbers = set()
    for node in tree.root.walk():
        if node.tag != "tr":
            continue
        text = node.text()
        # Actual record rows include all four field labels. Header/nested tables excluded.
        match = re.fullmatch(
            r"Open (.*?) Open (\d{1,2}/\d{1,2}/\d{4}), .*? Close (\d{1,2}/\d{1,2}/\d{4}), .*? Type (.*?) Number (.*?) Contact .*",
            text,
        )
        if not match:
            continue
        description, opening, closing, kind, number = match.groups()
        scanned += 1
        if number in numbers:
            raise ValueError("jaggaer_duplicate_notice")
        numbers.add(number)
        try:
            opened = datetime.strptime(opening, "%m/%d/%Y").date()
            deadline = datetime.strptime(closing, "%m/%d/%Y").date()
        except ValueError as exc:
            raise ValueError("jaggaer_invalid_date") from exc
        if re.search(r"\b(cancelled|canceled|withdrawn|awarded|closed)\b", description, re.I):
            continue
        if not opened <= date.fromisoformat(today) < deadline:
            continue
        if kind not in ("RFP", "RFI", "RFP/RFQ"):
            continue
        if not re.search(
            r"\bK[- ]?12\b|\bschools?\b|\belementary\b|\bsecondary\b|\bpublic instruction\b|\bpublic education\b",
            description,
            re.I,
        ):
            continue
        if re.search(
            r"\buniversity\b|\bcollege\b|\bhigher education\b", description, re.I
        ) and not re.search(r"\bK[- ]?12\b|\bschool districts?\b", description, re.I):
            continue
        title = next(
            (
                n.text()
                for n in node.walk()
                if n.tag == "a" and "ViewSourcingEvent" in n.attrs.get("href", "")
            ),
            None,
        )
        if not title or len(description) > 10000:
            raise ValueError("jaggaer_notice_structure_missing")
        # Never retain public ephemeral AuthToken / signed document links or contact names.
        quote = f"{description} Open {opening}; Close {closing}; Type {kind}; Number {number}"
        item = {
            "state": state,
            "url": SOURCES[state],
            "source_url": SOURCES[state],
            "title": title,
            "quote": quote,
            "notice_number": number,
            "deadline_date": deadline.isoformat(),
            "buyer": ORGS[state],
            "coverage_scope": "State central procurement portal only; not all school districts",
        }
        item["notice_id"] = _hash({"state": state, "url": SOURCES[state], "notice_number": number})
        item["hash"] = _hash(item)
        rows.append(item)
    if scanned != end - start + 1:
        raise ValueError("jaggaer_record_count_mismatch")
    return rows, {"start": start, "end": end, "total": total, "records_checked": scanned}


def fetch(client, source, today):
    state = {"Montana": "MT", "New Mexico": "NM"}.get(source.get("state"), source.get("state"))
    if state not in SOURCES or source.get("url") not in (
        SOURCES[state],
        BASE + "?CustomerOrg=" + ORGS[state],
    ):
        raise ValueError("invalid_jaggaer_source")
    records = []
    expected_total = None
    seen = set()
    for page in range(1, MAX_PAGES + 1):
        url = SOURCES[state] + "&" + urlencode({"PageSize": PAGE_SIZE, "PageNum": page})
        with client.stream("GET", url, follow_redirects=False, timeout=20) as response:
            if (
                response.status_code != 200
                or "html" not in response.headers.get("content-type", "").lower()
            ):
                raise ValueError("jaggaer_source_unavailable")
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError("jaggaer_source_too_large")
        rows, info = parse_page(data.decode("utf-8", errors="strict"), state, today)
        if expected_total is None:
            expected_total = info["total"]
        if info["total"] != expected_total or (
            expected_total and info["start"] != (page - 1) * PAGE_SIZE + 1
        ):
            raise ValueError("jaggaer_pagination_changed")
        for row in rows:
            if row["notice_id"] in seen:
                raise ValueError("jaggaer_repeated_page")
            seen.add(row["notice_id"])
        records.extend(rows)
        if info["end"] == expected_total:
            return records
        if info["end"] != page * PAGE_SIZE:
            raise ValueError("jaggaer_short_page")
    raise ValueError("jaggaer_page_budget_exceeded")
