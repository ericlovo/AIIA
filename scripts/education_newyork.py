"""NYC DOE public RFP fallback: district coverage, not statewide NYSED coverage.

TLS verification remains the caller's default. No alternate CA or proxy is used.
"""

import hashlib
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser

SOURCE = "https://infohub.nyced.org/in-our-schools/working-with-the-doe/contracts-and-purchasing/open-doe-solicitations/request-for-proposals"
MAX_BYTES = 1_000_000


class Sections(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections = []
        self.current = None
        self.in_heading = False
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if self.hidden:
            return
        if tag in ("h1", "h2"):
            self.current = {"heading": [], "parts": []}
            self.sections.append(self.current)
            self.in_heading = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag in ("h1", "h2"):
            self.in_heading = False

    def handle_data(self, text):
        if self.hidden or self.current is None:
            return
        clean = " ".join(text.split())
        if clean:
            self.current["parts"].append(clean)
            if self.in_heading:
                self.current["heading"].append(clean)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def parse(body, today):
    """Extract section-bound notices; unknown, indefinite and same-day dates withheld."""
    current = date.fromisoformat(today)
    parser = Sections()
    parser.feed(body)
    records = []
    seen = set()
    recognized = 0
    for section in parser.sections:
        title = " ".join(section["heading"])
        number = re.match(r"^(R\d{3,6})(?:\b|\s*[-:])", title)
        if not number:
            continue
        recognized += 1
        key = number.group(1)
        if key in seen:
            raise ValueError("newyork_duplicate_notice")
        seen.add(key)
        text = " ".join(section["parts"])
        if re.search(r"\b(withdrawn|cancelled|canceled|closed|awarded)\b", text, re.I):
            continue
        # Explicit proposal deadline only: conference and question deadlines never qualify.
        dates = re.findall(
            r"Proposal Due Date(?: and Time)?\s*:\s*(?:\w+,?\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            text,
        )
        try:
            deadlines = {
                datetime.strptime(re.sub(r",", "", value), "%B %d %Y").date() for value in dates
            }
        except ValueError:
            continue
        if len(deadlines) != 1 or next(iter(deadlines)) <= current:
            continue
        if not re.search(r"Request for Proposals?|\bRFP\b", text, re.I):
            continue
        deadline = next(iter(deadlines)).isoformat()
        # Keep exact source text from this section, with the qualifying deadline.
        # Oversized sections are withheld rather than dropping contradictory context.
        if len(text) > 12000:
            continue
        item = {
            "state": "NY",
            "url": SOURCE,
            "source_url": SOURCE,
            "title": title,
            "quote": text,
            "notice_number": key,
            "deadline_date": deadline,
            "coverage_scope": "New York City Department of Education only",
        }
        item["notice_id"] = _hash({"state": "NY", "url": SOURCE, "notice_number": key})
        item["hash"] = _hash(item)
        records.append(item)
    if not recognized:
        raise ValueError("newyork_notice_structure_missing")
    if recognized > 100:
        raise ValueError("newyork_notice_limit")
    return records


def fetch(client, source, today):
    if source.get("state") not in ("NY", "New York") or source.get("url") != SOURCE:
        raise ValueError("invalid_newyork_source")
    with client.stream("GET", SOURCE, follow_redirects=False) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("newyork_source_unavailable")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("newyork_source_too_large")
    body = data.decode("utf-8", errors="strict")
    if re.search(
        r"captcha|request rejected|access denied|verify you are human", body[:20000], re.I
    ):
        raise ValueError("newyork_source_challenge")
    return parse(body, today)
