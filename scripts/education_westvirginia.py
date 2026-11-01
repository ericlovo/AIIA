"""WVDE's public bid cards, with explicit status and complete-result validation."""

import re
from datetime import date, datetime
from urllib.parse import urljoin, urlsplit

from scripts.education_agency_pages import Tree

SOURCE = "https://wvde.us/about-us/purchasing/bids"
MAX_BYTES = 2_000_000


def field(card, name):
    found = [n for n in card.walk() if "c-field--name-" + name in n.attrs.get("class", "").split()]
    if len(found) != 1:
        raise ValueError("westvirginia_field_changed")
    return found[0]


def parse(body, today):
    tree = Tree(body)
    counts = re.findall(r"Displaying (\d+)\s*-\s*(\d+) of (\d+)", tree.root.text())
    cards = [
        n
        for n in tree.root.walk()
        if n.tag == "article" and "c-search-result-card" in n.attrs.get("class", "").split()
    ]
    if len(counts) != 1:
        raise ValueError("westvirginia_listing_unverified")
    start, end, total = map(int, counts[0])
    if total > 200 or end != total or len(cards) != total or start != (1 if total else 0):
        raise ValueError("westvirginia_listing_incomplete")
    results, seen = [], set()
    for card in cards:
        links = [n for n in card.walk() if n.tag == "a" and n.text()]
        if len(links) != 1:
            raise ValueError("westvirginia_notice_link_changed")
        link = links[0]
        url = urljoin(SOURCE, link.attrs.get("href", ""))
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "wvde.us"
            or not parsed.path.startswith("/about-us/purchasing/bids/")
            or parsed.query
            or parsed.fragment
            or url in seen
        ):
            raise ValueError("westvirginia_notice_link_invalid")
        seen.add(url)
        status = field(card, "field-solicitation-status").text()
        kind = field(card, "field-solicitation-type").text()
        closing = field(card, "field-close-datetime")
        times = [n.attrs.get("datetime", "") for n in closing.walk() if n.tag == "time"]
        if len(times) != 1:
            raise ValueError("westvirginia_deadline_missing")
        try:
            deadline = datetime.fromisoformat(times[0].replace("Z", "+00:00")).date()
        except ValueError as exc:
            raise ValueError("westvirginia_deadline_invalid") from exc
        if (
            status != "Open"
            or deadline <= date.fromisoformat(today)
            or not re.search(r"\b(RFP|RFI)\b", kind)
        ):
            continue
        evidence = card.text()
        if len(evidence) > 12000 or re.search(
            r"\b(cancelled|canceled|withdrawn|closed|awarded)\b", evidence, re.I
        ):
            continue
        results.append(
            {
                "state": "WV",
                "url": url,
                "source_url": SOURCE,
                "title": link.text(),
                "quote": "West Virginia Department of Education. " + evidence,
                "deadline_date": deadline.isoformat(),
                "coverage_scope": "WVDE public bid cards only; not all state or district procurements",
            }
        )
    return results


def fetch(client, source, today):
    if source.get("state") != "WV" or source.get("url") != SOURCE:
        raise ValueError("invalid_westvirginia_source")
    with client.stream("GET", SOURCE, follow_redirects=False, timeout=20) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("westvirginia_source_unavailable")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("westvirginia_source_too_large")
    return parse(data.decode("utf-8", errors="strict"), today)
