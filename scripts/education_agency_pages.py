"""Bounded public AL department / CA CCEE procurement adapters, not all districts."""

import hashlib
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

SOURCES = {
    "AL": "https://www.alabamaachieves.org/communication/legal/request-for-proposal/",
    "CA": "https://ccee-ca.org/request-for-proposal-collection/",
}
BUYERS = {
    "AL": "Alabama State Department of Education",
    "CA": "California Collaborative for Educational Excellence",
}
MAX_BYTES = 3_000_000
MAX_PAGES = 7


class Node:
    def __init__(self, tag="", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        if self.tag in ("script", "style", "nav", "footer"):
            return ""
        return " ".join(
            " ".join(c.text() if isinstance(c, Node) else c for c in self.children).split()
        )

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()


class Tree(HTMLParser):
    def __init__(self, body):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in (
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        ):
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _url(value, state):
    url = urljoin(SOURCES[state], value)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != urlsplit(SOURCES[state]).netloc
        or parsed.username
        or parsed.fragment
        or parsed.query
    ):
        raise ValueError("agency_link_outside_source")
    return url


def _deadline(text, today):
    # Deliberately never treats question, conference or posting dates as deadlines.
    values = re.findall(
        r"(?:Proposal Submission Deadline|Proposals? Due Date|Proposals? must be received no later than)(?:\s|:|by|on|\d|[.:()A-Za-z-]){0,65}?\b([A-Z][a-z]+ \d{1,2},? \d{4})",
        text,
    )
    try:
        dates = {datetime.strptime(v.replace(",", ""), "%B %d %Y").date() for v in values}
    except ValueError:
        return None
    if len(dates) != 1 or next(iter(dates)) <= date.fromisoformat(today):
        return None
    return next(iter(dates)).isoformat()


def _record(state, url, title, text, today):
    # Contact names/email are not necessary procurement evidence.
    if len(text) > 12000 or re.search(
        r"\b(cancelled|canceled|withdrawn|awarded|closed)\b", text, re.I
    ):
        return None
    deadline = _deadline(text, today)
    if not deadline or not re.search(
        r"\bRFP\b|\bRFI\b|request(?:ing)? (?:for )?proposals?", title + " " + text, re.I
    ):
        return None
    text = re.split(r"Point of Contact|Contact Person", text, flags=re.I)[0].strip()
    number = re.search(r"\b(?:RFP|RFI)\s+(?:ALSDE\s+)?\d{4}-\d+\b", title)
    item = {
        "state": state,
        "url": url,
        "source_url": SOURCES[state],
        "title": title,
        "quote": text,
        "notice_number": number.group() if number else None,
        "deadline_date": deadline,
        "buyer": BUYERS[state],
        "coverage_scope": BUYERS[state] + " notices only; not all school districts",
    }
    item["notice_id"] = _hash({"state": state, "url": url})
    item["hash"] = _hash(item)
    return item


def parse_alabama(body, today):
    tree = Tree(body)
    section = None
    recognized = False
    records = []
    for node in tree.root.walk():
        if node.tag == "h3" and "qode-title-holder" in node.attrs.get("class", "").split():
            section = node.text().lower()
            recognized |= section == "open"
        if (
            node.tag != "div"
            or "vc_grid-item" not in node.attrs.get("class", "").split()
            or section != "open"
        ):
            continue
        heading = next((n for n in node.walk() if n.tag == "h4"), None)
        link = next(
            (
                n
                for n in node.walk()
                if n.tag == "a" and "vc_gitem-link" in n.attrs.get("class", "").split()
            ),
            None,
        )
        if not heading or not link:
            raise ValueError("alabama_notice_structure_changed")
        title = heading.text()
        if re.search(r"Questions and Answers|Q\s*&\s*A", title, re.I):
            continue
        record = _record("AL", _url(link.attrs.get("href", ""), "AL"), title, node.text(), today)
        if record:
            records.append(record)
    if not recognized:
        raise ValueError("alabama_open_section_missing")
    return records


def california_links(body):
    section = None
    recognized = False
    links = {}
    for node in Tree(body).root.walk():
        if node.tag == "h3":
            section = node.text().upper()
            recognized |= section == "ACTIVE"
        if section == "ACTIVE" and node.tag == "a" and node.text():
            url = _url(node.attrs.get("href", ""), "CA")
            # The live page repeats a date-only anchor for one notice.
            if len(node.text()) > len(links.get(url, "")):
                links[url] = node.text()
    if not recognized:
        raise ValueError("california_active_section_missing")
    if len(links) > MAX_PAGES - 1:
        raise ValueError("california_page_budget_exceeded")
    return links


def parse_california_detail(body, url, title, today):
    _url(url, "CA")
    tree = Tree(body)
    sections = [n for n in tree.root.walk() if "post-content" in n.attrs.get("class", "").split()]
    if not sections:
        sections = [
            n
            for n in tree.root.walk()
            if n.tag == "article" and "hentry" in n.attrs.get("class", "").split()
        ]
    if len(sections) != 1:
        raise ValueError("california_detail_structure_missing")
    return _record("CA", url, title, sections[0].text(), today)


def _get(client, url):
    with client.stream(
        "GET",
        url,
        follow_redirects=False,
        timeout=20,
        headers={"User-Agent": "AIIA-Education-Source-Audit/1.0"},
    ) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("agency_source_unavailable")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("agency_source_too_large")
    body = data.decode("utf-8", errors="strict")
    visible = Tree(body).root.text()
    if re.search(
        r"request rejected|access denied|verify you are human|captcha challenge", visible, re.I
    ):
        raise ValueError("agency_source_challenge")
    return body


def fetch(client, source, today):
    state = {"Alabama": "AL", "California": "CA"}.get(source.get("state"), source.get("state"))
    if state not in SOURCES or source.get("url") != SOURCES[state]:
        raise ValueError("invalid_agency_source")
    body = _get(client, SOURCES[state])
    if state == "AL":
        return parse_alabama(body, today)
    records = []
    for url, title in california_links(body).items():
        record = parse_california_detail(_get(client, url), url, title, today)
        if record:
            records.append(record)
    return records
