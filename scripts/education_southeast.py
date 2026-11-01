"""Louisiana official notification-board subset; MS/TN/GA gaps documented separately."""

import re
from datetime import date, datetime

from scripts.education_agency_pages import Tree, _hash

SOURCE = "https://co.apps.laregents.edu/"
DETAIL = SOURCE + "Detail.aspx?id="
MAX_BYTES = 1_000_000
MAX_PAGES = 7
PREFIX = "MainContent_DV1_"


def _tree(body):
    tree = Tree(body)
    if re.search(
        r"access denied|request rejected|verify you are human|captcha challenge",
        tree.root.text(),
        re.I,
    ):
        raise ValueError("louisiana_source_challenge")
    return tree


def listing(body):
    tree = _tree(body)
    grids = [
        n
        for n in tree.root.walk()
        if n.tag == "table" and n.attrs.get("id") == "MainContent_GV_oppo"
    ]
    if len(grids) != 1 or "CURRENT OPPORTUNITIES" not in grids[0].text():
        raise ValueError("louisiana_current_listing_missing")
    grid = grids[0]
    candidates = {}
    for node in grid.walk():
        if node.tag != "a":
            continue
        href = node.attrs.get("href", "")
        if "Page$" in href or node.text().lower() in ("next", "next page"):
            raise ValueError("louisiana_unhandled_pagination")
        if "Detail.aspx" not in href:
            continue
        match = re.fullmatch(r"Detail.aspx\?id=(\d+)", href)
        if not match:
            raise ValueError("louisiana_link_changed")
        text = node.text()
        if "Funding Source(s): Department of Education" not in text:
            continue
        identity = match.group(1)
        if identity in candidates:
            raise ValueError("louisiana_duplicate_notice")
        # Dollar amounts displayed on the listing are not necessary evidence.
        title = re.split(r"\s+\$", text, maxsplit=1)[0].strip()
        if not title:
            raise ValueError("louisiana_title_missing")
        candidates[identity] = title
    if len(candidates) + 1 > MAX_PAGES:
        raise ValueError("louisiana_page_budget")
    return candidates


def detail(body, identity, title, today):
    if not re.fullmatch(r"\d+", identity):
        raise ValueError("louisiana_invalid_id")
    tree = _tree(body)
    tables = [n for n in tree.root.walk() if n.attrs.get("id") == "MainContent_DV1"]
    if len(tables) != 1:
        raise ValueError("louisiana_detail_missing")
    content = tables[0]
    if re.search(r"\b(cancelled|canceled|withdrawn|closed|awarded)\b", content.text(), re.I):
        return None
    values = {
        n.attrs["id"].removeprefix(PREFIX): n.text()
        for n in content.walk()
        if n.attrs.get("id", "").startswith(PREFIX)
    }
    required = ["opportunity_name", "fund", "month", "Label14", "agency", "Label15"]
    if any(k not in values for k in required):
        raise ValueError("louisiana_detail_fields_missing")
    if values["opportunity_name"] != title or values["fund"] != "Department of Education":
        raise ValueError("louisiana_detail_correlation_mismatch")
    if values["Label14"] not in ("RFP", "RFI"):
        return None
    try:
        closing = datetime.strptime(values["month"], "%b %d, %Y").date()
    except ValueError as exc:
        raise ValueError("louisiana_deadline_invalid") from exc
    if closing <= date.fromisoformat(today):
        return None
    evidence = title + " " + values["Label15"]
    if re.search(
        r"\bgrant applications?\b|\bgrant awards?\b|\bcollege\b|\buniversity\b", evidence, re.I
    ):
        return None
    if not re.search(
        r"\bstudent assessments?\b|\bk[- ]?12\b|\bschools?\b|\belementary\b|\bsecondary\b",
        evidence,
        re.I,
    ):
        return None
    # Whitelist fields, not whole-page text: omit contacts, addresses and amounts.
    quote = f"{title}. Funding Source(s): Department of Education. Respond By: {values['month']}. Award Type: {values['Label14']}. Agency: {values['agency']}. Scope of Service: {values['Label15']}"
    if len(quote) > 12000:
        return None
    item = {
        "state": "LA",
        "url": DETAIL + identity,
        "source_url": SOURCE,
        "title": title,
        "quote": quote,
        "notice_number": None,
        "source_record_id": identity,
        "deadline_date": closing.isoformat(),
        "buyer": values["agency"],
        "coverage_scope": "Louisiana Regents notification-board DOE-funded subset; not all education procurement",
        "review_note": "Official notification record; verify linked procurement portal and amendments before bidding.",
    }
    item["notice_id"] = _hash({"state": "LA", "url": DETAIL + identity})
    item["hash"] = _hash(item)
    return item


def _get(client, url):
    if url != SOURCE and not re.fullmatch(re.escape(DETAIL) + r"\d+", url):
        raise ValueError("louisiana_url_invalid")
    with client.stream("GET", url, follow_redirects=False, timeout=20) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("louisiana_source_unavailable")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError("louisiana_source_too_large")
    return content.decode("utf-8", errors="strict")


def fetch(client, source, today):
    if source.get("state") not in ("LA", "Louisiana") or source.get("url") != SOURCE:
        raise ValueError("invalid_southeast_source")
    candidates = listing(_get(client, SOURCE))
    records = []
    for identity, title in candidates.items():
        record = detail(_get(client, DETAIL + identity), identity, title, today)
        if record:
            records.append(record)
    return records
