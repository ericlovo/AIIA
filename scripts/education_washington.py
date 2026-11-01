"""Public WEBS OSPI filter; no login, statewide/district coverage inference, or TLS bypass."""

import hashlib
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin

SOURCE = "https://pr-webs-vendor.des.wa.gov/BidCalendar.aspx"
AGENCY = "Superintendent of Public Instruction, Office of"
MAX_BYTES = 2_000_000


class Node:
    def __init__(self, tag="", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        return " ".join(
            " ".join(c.text() if isinstance(c, Node) else c for c in self.children).split()
        )

    def find(self, tag):
        return [
            n
            for c in self.children
            if isinstance(c, Node)
            for n in ([c] if c.tag == tag else []) + c.find(tag)
        ]


class Tree(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.root = Node()
        self.stack = [self.root]
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        self.stack[-1].children.append(n)
        if tag not in (
            "input",
            "br",
            "hr",
            "img",
            "meta",
            "link",
            "area",
            "base",
            "embed",
            "source",
            "wbr",
        ):
            self.stack.append(n)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _read(client, method, **kwargs):
    with client.stream(method, SOURCE, follow_redirects=False, **kwargs) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("washington_source_unavailable")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("washington_source_too_large")
    return Tree(data.decode("utf-8", errors="strict")).root


def _select(root):
    choices = [n for n in root.find("select") if n.attrs.get("name") == "ddlOrgNames"]
    if len(choices) != 1:
        raise ValueError("washington_filter_missing")
    options = [n for n in choices[0].find("option") if n.text() == AGENCY]
    if len(options) != 1 or not options[0].attrs.get("value", "").isdigit():
        raise ValueError("washington_agency_missing")
    return options[0]


def parse(root, today, agency_value):
    selected = _select(root)
    if "selected" not in selected.attrs or selected.attrs["value"] != agency_value:
        raise ValueError("washington_filter_not_applied")
    grids = [n for n in root.find("table") if n.attrs.get("id") == "DataGrid1"]
    if len(grids) != 1:
        raise ValueError("washington_grid_missing")
    grid = grids[0]
    # Any remaining paging link means this single-page adapter is incomplete.
    if any(re.search(r"Page|_ctl1\$_ctl", n.attrs.get("href", ""), re.I) for n in grid.find("a")):
        raise ValueError("washington_pagination_incomplete")
    current = date.fromisoformat(today)
    results = []
    for row in grid.find("tr"):
        links = [
            n
            for n in row.find("a")
            if re.fullmatch(r"Search_BidDetails\.aspx\?ID=\d+", n.attrs.get("href", ""))
        ]
        if not links:
            continue
        cells = [n for n in row.children if isinstance(n, Node) and n.tag == "td"]
        if len(cells) != 3 or len(links) != 1:
            raise ValueError("washington_invalid_row")
        # First column is labelled Close Date / Amendment Date; first date is closure.
        match = re.match(r"(\d{2}/\d{2}/\d{2})\b", cells[0].text())
        if not match:
            raise ValueError("washington_deadline_missing")
        deadline = datetime.strptime(match.group(1), "%m/%d/%y").date()
        evidence = cells[1].text()  # Exclude the third/contact column.
        if deadline <= current or not re.search(r"\b(RFP|RFI)\b", evidence):
            continue
        if re.search(r"\b(cancelled|canceled|withdrawn|closed|awarded)\b", evidence, re.I):
            continue
        if len(evidence) > 8000:
            raise ValueError("washington_evidence_too_large")
        link = links[0]
        url = urljoin(SOURCE, link.attrs["href"])
        item = {
            "state": "WA",
            "url": url,
            "source_url": SOURCE,
            "title": link.text(),
            "quote": f"Agency filter: {AGENCY}. Solicitation close date: {deadline.isoformat()}. {evidence}",
            "deadline_date": deadline.isoformat(),
            "coverage_scope": "OSPI public WEBS solicitations only",
        }
        item["notice_id"] = hashlib.sha256(
            json.dumps({"state": "WA", "url": url}, sort_keys=True).encode()
        ).hexdigest()
        item["hash"] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        results.append(item)
    if len(results) > 100:
        raise ValueError("washington_record_limit")
    return results


def fetch(client, source, today):
    if source.get("state") not in ("WA", "Washington") or source.get("url") != SOURCE:
        raise ValueError("invalid_washington_source")
    initial = _read(client, "GET")
    option = _select(initial)
    hidden = {
        n.attrs["name"]: n.attrs.get("value", "")
        for n in initial.find("input")
        if n.attrs.get("type", "").lower() == "hidden" and n.attrs.get("name", "").startswith("__")
    }
    if not all(k in hidden for k in ("__VIEWSTATE", "__EVENTVALIDATION")):
        raise ValueError("washington_form_missing")
    hidden.update(ddlOrgNames=option.attrs["value"], ImageButton1="Search")
    filtered = _read(client, "POST", data=hidden)
    return parse(filtered, today, option.attrs["value"])
