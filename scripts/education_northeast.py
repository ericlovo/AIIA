"""Verified Vermont education notices; other Northeast access gaps documented separately."""

import re
from datetime import date, datetime

from scripts.education_agency_pages import Tree, _hash

SOURCE = "https://www.vermontbusinessregistry.com/bidsearch.aspx?type=5"
DETAIL = "https://www.vermontbusinessregistry.com/BidPreview.aspx?BidID="
MAX_BYTES = 2_000_000
MAX_PAGES = 7


def _tree(body):
    tree = Tree(body)
    if re.search(
        r"access denied|request rejected|verify you are human|captcha challenge",
        tree.root.text(),
        re.I,
    ):
        raise ValueError("vermont_source_challenge")
    return tree


def listing(body):
    tree = _tree(body)
    if "Search Results - Open State Bids" not in tree.root.text():
        raise ValueError("vermont_listing_structure_missing")
    grids = [n for n in tree.root.walk() if n.tag == "table" and n.attrs.get("id") == "gvResults"]
    if len(grids) != 1:
        raise ValueError("vermont_grid_missing")
    all_ids = set()
    education = {}
    for row in grids[0].walk():
        if row.tag != "tr":
            continue
        links = [
            n for n in row.walk() if n.tag == "a" and "openPrintView" in n.attrs.get("href", "")
        ]
        if len(links) != 1:
            continue
        link = links[0]
        match = re.fullmatch(
            r"javascript:openPrintView\('BidPreview.aspx\?BidID=(\d+)', 'Window\d+'\);",
            link.attrs["href"],
        )
        if not match:
            raise ValueError("vermont_link_changed")
        identity = match.group(1)
        all_ids.add(identity)
        text = row.text()
        if "Agency of Education" in text:
            closing = re.findall(r"Close Date:\s*(\d{1,2}/\d{1,2}/\d{4})", text)
            if len(closing) != 1:
                raise ValueError("vermont_list_deadline_missing")
            education[identity] = {"title": link.text(), "closing": closing[0]}
    if not all_ids:
        raise ValueError("vermont_empty_listing_unverified")
    pages = set()
    for node in grids[0].walk():
        if node.tag == "a" and "Page$" in node.attrs.get("href", ""):
            m = re.fullmatch(
                r"javascript:__doPostBack\('gvResults','Page\$(\d+)'\)", node.attrs["href"]
            )
            if not m:
                raise ValueError("vermont_pager_changed")
            pages.add(int(m.group(1)))
    fields = {
        n.attrs["name"]: n.attrs.get("value", "")
        for n in tree.root.walk()
        if n.tag == "input" and n.attrs.get("type") == "hidden" and n.attrs.get("name")
    }
    if not {"__VIEWSTATE", "__EVENTVALIDATION"}.issubset(fields):
        raise ValueError("vermont_pagination_state_missing")
    return all_ids, education, pages, fields


def detail(body, identity, listed, today):
    if not re.fullmatch(r"\d+", identity):
        raise ValueError("vermont_invalid_id")
    tree = _tree(body)
    # Check complete visible context for cancellation before removing contact fields.
    if re.search(r"\b(cancelled|canceled|withdrawn|closed|awarded)\b", tree.root.text(), re.I):
        return None
    values = {}
    for node in tree.root.walk():
        if node.attrs.get("id", "").startswith("lbl"):
            values[node.attrs["id"]] = node.text()
    required = [
        "lblBidTitle",
        "lblAuthorName",
        "lblBidCloseDate",
        "lblBidDescription",
        "lblSpecialInstructions",
        "lblBidType",
    ]
    if any(k not in values for k in required):
        raise ValueError("vermont_detail_fields_missing")
    if values["lblAuthorName"] != "Agency of Education" or values["lblBidTitle"] != listed["title"]:
        raise ValueError("vermont_detail_correlation_mismatch")
    try:
        closing = datetime.strptime(values["lblBidCloseDate"].split()[0], "%m/%d/%Y").date()
        listclosing = datetime.strptime(listed["closing"], "%m/%d/%Y").date()
    except (ValueError, IndexError) as exc:
        raise ValueError("vermont_deadline_invalid") from exc
    if closing != listclosing:
        raise ValueError("vermont_deadline_mismatch")
    if closing <= date.fromisoformat(today):
        return None
    title = values["lblBidTitle"]
    description = values["lblBidDescription"]
    instructions = values["lblSpecialInstructions"]
    text = title + " " + description + " " + instructions
    if not re.search(r"\bRFP\b|\bRFI\b|Request for Proposals?|Request for Information", text, re.I):
        return None
    if re.search(
        r"\bgrant applications?\b|\bgrant awards?\b|\bhigher education\b|\buniversity\b", text, re.I
    ):
        return None
    if values["lblBidType"] != "Job Contract" or not re.search(
        r"\bpreK[- ]?12\b|\bk[- ]?12\b|\bschools?\b|\bstudents?\b|\bgraduation\b|\battendance\b",
        text,
        re.I,
    ):
        return None
    # Contacts/address fields and attachments never enter evidence or output.
    quote = f"{title}. {description}. {instructions} Closing Date: {values['lblBidCloseDate']}. Agency of Education."
    if len(quote) > 12000:
        return None
    item = {
        "state": "VT",
        "url": DETAIL + identity,
        "source_url": SOURCE,
        "title": title,
        "quote": quote,
        "notice_number": values.get("lblRFQNumber") or None,
        "source_record_id": identity,
        "deadline_date": closing.isoformat(),
        "buyer": "Vermont Agency of Education",
        "coverage_scope": "Agency of Education state bids only; not all school districts",
    }
    item["notice_id"] = _hash({"state": "VT", "url": DETAIL + identity})
    item["hash"] = _hash(item)
    return item


def _request(client, method, url, data=None):
    if url != SOURCE and not re.fullmatch(re.escape(DETAIL) + r"\d+", url):
        raise ValueError("vermont_url_invalid")
    with client.stream(method, url, data=data, follow_redirects=False, timeout=20) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("vermont_source_unavailable")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError("vermont_source_too_large")
    return content.decode("utf-8", errors="strict")


def fetch(client, source, today):
    if source.get("state") not in ("VT", "Vermont") or source.get("url") != SOURCE:
        raise ValueError("invalid_northeast_source")
    body = _request(client, "GET", SOURCE)
    visited = set()
    seen_ids = set()
    candidates = {}
    calls = 1
    page = 1
    while True:
        identities, selected, pages, fields = listing(body)
        if seen_ids.intersection(identities):
            raise ValueError("vermont_repeated_page")
        seen_ids.update(identities)
        candidates.update(selected)
        visited.add(page)
        pending = sorted(pages - visited)
        if not pending:
            break
        if calls >= MAX_PAGES or max(pending) > MAX_PAGES:
            raise ValueError("vermont_page_budget")
        page = pending[0]
        fields.update(
            {
                "__EVENTTARGET": "gvResults",
                "__EVENTARGUMENT": f"Page${page}",
                "ddlSortBy": "BidAddDate",
            }
        )
        body = _request(client, "POST", SOURCE, fields)
        calls += 1
    if calls + len(candidates) > MAX_PAGES:
        raise ValueError("vermont_detail_budget")
    records = []
    for identity, selected in candidates.items():
        record = detail(_request(client, "GET", DETAIL + identity), identity, selected, today)
        if record:
            records.append(record)
    return records
