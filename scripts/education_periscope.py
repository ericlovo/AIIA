"""Bounded public Periscope listings. Fail closed on incomplete JSF pagination."""

import hashlib
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlsplit
from xml.etree import ElementTree

BASES = {
    "NJ": "https://www.njstart.gov",
    "HI": "https://alohaebuys.hawaii.gov",
    "IL": "https://www.bidbuy.illinois.gov",
    "NV": "https://nevadaepro.com",
    "OR": "https://oregonbuys.gov",
    "MA": "https://www.commbuys.com",
}
LIMITS = {"MA": (1000, 40)}
PATH = "/bso/view/search/external/advancedSearchBid.xhtml?openBids=true"
TABLE = "bidSearchResultsForm:bidResultId"
MAX_BYTES = 2_000_000
TYPE = re.compile(r"\b(?:RFP|RFI)\b|\bRequest for (?:Proposals?|Information)\b", re.I)
BUYER = re.compile(
    r"\bschools?\b|\b(?:department|board) of (?:elementary and secondary )?education\b", re.I
)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.fields, self.text = [], {}, []
        self.cells, self.cell, self.link, self.form = [], None, "", False
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "form":
            self.form = a.get("id") == "bidSearchResultsForm"
        if tag == "input" and self.form and a.get("type") == "hidden":
            self.fields[a.get("name", "")] = a.get("value", "")
        if tag == "tr":
            self.cells, self.link = [], ""
        if tag == "td":
            self.cell = []
        if tag == "a" and "bidDetail.sda?" in a.get("href", ""):
            self.link = a["href"]

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)
            if self.cell is not None:
                self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if tag == "form":
            self.form = False
        if tag == "td" and self.cell is not None:
            self.cells.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.link:
            self.rows.append((self.link, self.cells))

    @property
    def plain(self):
        return " ".join(" ".join(self.text).split())


def _read(client, method, url, **kwargs):
    with client.stream(method, url, follow_redirects=False, timeout=25, **kwargs) as response:
        kind = response.headers.get("content-type", "").lower()
        if response.status_code != 200 or not any(t in kind for t in ("html", "xml")):
            raise ValueError("periscope_source_unavailable")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise ValueError("periscope_source_too_large")
    return body.decode("utf-8", errors="strict")


def _parse(body):
    page = Page()
    page.feed(body)
    return page


def _hash(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def fetch(client, source, today):
    """Retrieve complete bounded listings; preserve pending RFP/RFI K12 notices.

    Massachusetts permits 1000 rows/40 pages; others retain 200 rows/10 pages.
    MA uses explicit title types and no detail requests; its retained records
    remain pending document review and are never alert-eligible here.

    Bid opening is preserved as a labeled date, not invented as a submission cutoff.
    Same-day notices are withheld. State/local coverage is limited to portal users.
    """
    state = source.get("state")
    base = BASES.get(state)
    if not base or source.get("url") != base + PATH:
        raise ValueError("invalid_periscope_source")
    max_records, max_pages = LIMITS.get(state, (200, 10))
    current = date.fromisoformat(today)
    page = _parse(_read(client, "GET", source["url"]))
    totals = re.findall(r"\b(\d+)-(\d+) of (\d+)\b", page.plain)
    if not totals:
        raise ValueError("periscope_count_unavailable")
    start, end, total = map(int, totals[0])
    if (
        total > max_records
        or (total + 24) // 25 > max_pages
        or start != 1
        or end != min(25, total)
        or len(page.rows) != end
    ):
        raise ValueError("periscope_feed_incomplete")
    rows = list(page.rows)
    fields = page.fields.copy()
    if total > 25 and not fields.get("javax.faces.ViewState"):
        raise ValueError("periscope_pagination_unavailable")
    for first in range(25, total, 25):
        data = dict(fields)
        data.update(
            {
                "javax.faces.partial.ajax": "true",
                "javax.faces.source": TABLE,
                "javax.faces.partial.execute": TABLE,
                "javax.faces.partial.render": TABLE,
                TABLE + "_pagination": "true",
                TABLE + "_first": str(first),
                TABLE + "_rows": "25",
                TABLE + "_skipChildren": "true",
                TABLE + "_encodeFeature": "true",
            }
        )
        raw = _read(
            client, "POST", source["url"], data=data, headers={"Faces-Request": "partial/ajax"}
        )
        try:
            root = ElementTree.fromstring(raw)
            updates = {x.attrib.get("id"): x.text or "" for x in root.iter("update")}
            counts = [
                json.loads(x.text or "{}").get("totalRecords")
                for x in root.iter("extension")
                if x.attrib.get("type") == "args"
            ]
            fragment = _parse(updates[TABLE])
        except (ElementTree.ParseError, KeyError, ValueError) as exc:
            raise ValueError("periscope_pagination_invalid") from exc
        if counts != [total] or len(fragment.rows) != min(25, total - first):
            raise ValueError("periscope_feed_incomplete")
        for key, value in updates.items():
            if key and "javax.faces.ViewState" in key:
                fields["javax.faces.ViewState"] = value
        rows.extend(fragment.rows)
    links = [urljoin(base, row[0]) for row in rows]
    if len(set(links)) != total:
        raise ValueError("periscope_feed_incomplete")
    found = []
    for link, cells in rows:
        if len(cells) != 11:
            raise ValueError("periscope_row_schema_changed")
        number, agency, title, opening, status = cells[0], cells[2], cells[6], cells[7], cells[9]
        if (
            status != "Sent"
            or not BUYER.search(agency)
            or re.search(r"university|college", agency, re.I)
        ):
            continue
        if source.get("buyer_scope") == "state_agencies" and re.search(
            r"\bdistrict\b|\bpublic schools\b", agency, re.I
        ):
            continue
        try:
            opening_date = datetime.strptime(opening, "%m/%d/%Y %I:%M %p").date()
        except ValueError:
            continue
        if opening_date <= current:
            continue
        url = urljoin(base, link)
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or parsed.netloc != urlsplit(base).netloc
            or parsed.path != "/bso/external/bidDetail.sda"
            or parse_qs(parsed.query).get("docId") != [number]
        ):
            raise ValueError("periscope_invalid_detail_link")
        # MA is a larger statewide portal. Complete the listing, but leave
        # document/deadline review pending rather than fan out to 1000 details.
        bulletin = None
        if state != "MA":
            detail = _parse(_read(client, "GET", url)).plain
            if f"Bid Number: {number}" not in detail:
                raise ValueError("periscope_detail_unavailable")
            bulletin = re.search(r"Bulletin Desc:\s*(.*?)\s*Ship-to Address:", detail)
        type_evidence = title + " " + (bulletin.group(1) if bulletin else "")
        if not TYPE.search(type_evidence) or re.search(
            r"\b(?:cancelled|canceled|withdrawn)\b", type_evidence, re.I
        ):
            continue
        quote = json.dumps(
            {
                "title": title,
                "agency": agency,
                "notice_number": number,
                "bid_opening_date": opening_date.isoformat(),
                "bid_opening_time": opening,
                "submission_deadline": "not separately verified; check official notice",
                "listing_status": status,
                "bulletin": bulletin.group(1) if bulletin else "",
                "as_of": today,
                **(
                    {"evidence_scope": "public listing only; detail not fetched"}
                    if state == "MA"
                    else {}
                ),
            },
            sort_keys=True,
        )
        item = {
            "state": state,
            "url": url,
            "source_url": source["url"],
            "title": title,
            "quote": quote,
            "notice_number": number,
            "alert_eligible": False,
            "qualification_status": (
                "pending_document_review"
                if state == "MA"
                else "pending_submission_deadline_verification"
            ),
            "coverage_scope": "Participating portal buyers only; not exhaustive district coverage",
        }
        item["identity"] = _hash({"state": state, "url": url})
        item["hash"] = _hash(item)
        found.append(item)
    return found
