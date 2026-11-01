"""Verified Midwest collection paths; currently Nebraska DAS current bid table.

Agency-level notices without explicit submission deadlines remain pending and
cannot alert. Other researched states deliberately have no guessed adapters.
"""

import hashlib
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlsplit

SOURCE = "https://das.nebraska.gov/materiel/bid-opportunities.html"
MAX_BYTES = 2_000_000


class CurrentBids(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.seen = False
        self.tables = 0
        self.rows = []
        self.cells = []
        self.cell = None
        self.link = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and attrs.get("id") == "current-bid-opportunities":
            self.active = self.seen = True
        if not self.active:
            return
        if tag == "table":
            self.tables += 1
        if tag == "tr":
            self.cells, self.link = [], ""
        if tag in ("td", "th"):
            self.cell = []
        if tag == "a" and self.cell is not None:
            self.link = attrs.get("href", "")

    def handle_data(self, text):
        if self.active and self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if not self.active:
            return
        if tag in ("td", "th") and self.cell is not None:
            self.cells.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        if tag == "tr":
            self.rows.append((self.link, self.cells))
        if tag == "section":
            self.active = False


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parse_nebraska(body, today):
    page = CurrentBids()
    page.feed(body)
    if not page.seen or page.tables != 1 or not page.rows or len(page.rows) > 201:
        raise ValueError("nebraska_current_table_unavailable")
    header = page.rows[0][1]
    expected = [
        "Posted",
        "Description",
        "Category - NIGP Code",
        "Opening",
        "Type",
        "PCO/Buyer",
        "Solicitation Number",
        "Agency",
        "Updated",
    ]
    if header != expected:
        raise ValueError("nebraska_current_table_schema_changed")
    current = date.fromisoformat(today)
    found = []
    seen = set()
    for href, cells in page.rows[1:]:
        if len(cells) != 9:
            raise ValueError("nebraska_current_row_invalid")
        posted, title, _, opening, kind, _, number, agency, _ = cells
        if not any(cells):
            continue
        if not number or number in seen:
            raise ValueError("nebraska_current_table_incomplete")
        seen.add(number)
        if not re.fullmatch(
            r"Request for (?:Proposals?|Information)", kind, re.I
        ) or agency not in ("Nebraska Department of Education", "Department of Education"):
            continue
        try:
            opened = datetime.strptime(opening, "%m/%d/%y").date()
            posted_date = datetime.strptime(posted, "%m/%d/%y").date()
        except ValueError:
            continue
        if opened <= current or posted_date > current:
            continue
        url = urljoin(SOURCE, href)
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "das.nebraska.gov"
            or not parsed.path.startswith("/materiel/purchasing/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("nebraska_invalid_detail_url")
        url = quote(url, safe=":/%")
        evidence = {
            "title": title,
            "agency": agency,
            "type": kind,
            "notice_number": number,
            "listing_section": "Current Bid Opportunities",
            "posted_date": posted_date.isoformat(),
            "bid_opening_date": opened.isoformat(),
            "submission_deadline": "not verified; opening date is not a submission deadline",
            "k12_applicability": "requires detail qualification; agency also procures non-K12 services",
            "as_of": today,
        }
        item = {
            "state": "NE",
            "source_url": SOURCE,
            "url": url,
            "title": title,
            "notice_number": number,
            "quote": json.dumps(evidence, sort_keys=True),
            "alert_eligible": False,
            "qualification_status": "pending_submission_deadline_and_k12_verification",
            "coverage_scope": "Nebraska DAS current NDE agency rows only; not all school districts",
        }
        item["identity"] = _digest({"state": "NE", "url": url})
        item["hash"] = _digest(item)
        found.append(item)
    return found


def fetch(client, source, today):
    if source.get("state") != "NE" or source.get("url") != SOURCE:
        raise ValueError("unsupported_midwest_source")
    with client.stream("GET", SOURCE, follow_redirects=False, timeout=20) as response:
        if response.status_code != 200:
            raise ValueError("nebraska_source_unavailable")
        raw = bytearray()
        for chunk in response.iter_bytes():
            raw.extend(chunk)
            if len(raw) > MAX_BYTES:
                raise ValueError("nebraska_source_too_large")
    return parse_nebraska(raw.decode("utf-8", errors="strict"), today)
