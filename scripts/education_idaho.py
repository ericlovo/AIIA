"""Row-bound Idaho Department of Education current RFP/RFI opportunities."""

import re
from datetime import date, datetime
from urllib.parse import urlsplit

from scripts.education_agency_pages import Node, Tree, _hash

SOURCE = "https://www.sde.idaho.gov/contracting-opportunities/"
MAX_BYTES = 2_000_000
MAX_ROWS = 50


def parse(body, today):
    tree = Tree(body)
    if re.search(
        r"access denied|request rejected|verify you are human|captcha challenge",
        tree.root.text(),
        re.I,
    ):
        raise ValueError("idaho_source_challenge")
    section = None
    tables = []
    for node in tree.root.walk():
        if node.tag in ("h1", "h2"):
            section = node.text()
        if node.tag == "table" and section == "Current Opportunities":
            tables.append(node)
    if len(tables) != 1:
        raise ValueError("idaho_current_table_missing")
    rows = [n for n in tables[0].walk() if n.tag == "tr"]
    if not 1 <= len(rows) <= MAX_ROWS + 1:
        raise ValueError("idaho_row_budget")
    expected = ["Description", "Documentation", "Additional Information"]
    header = [n.text() for n in rows[0].children if isinstance(n, Node) and n.tag in ("td", "th")]
    if header != expected:
        raise ValueError("idaho_columns_changed")
    if any(
        n.tag == "a"
        and (n.text().lower() in ("next", "next page") or "Page$" in n.attrs.get("href", ""))
        for n in tables[0].walk()
    ):
        raise ValueError("idaho_unhandled_pagination")
    records = []
    seen = set()
    for row in rows[1:]:
        cells = [n for n in row.children if isinstance(n, Node) and n.tag in ("td", "th")]
        if len(cells) != 3:
            raise ValueError("idaho_row_shape_changed")
        # Inspect cancellation across the entire row before selecting safe evidence.
        if re.search(
            r"\b(cancelled|canceled|withdrawn|closed|awarded|expired)\b", row.text(), re.I
        ):
            continue
        title, document, information = (n.text() for n in cells)
        kind = {
            "Request for Information": "RFI",
            "Request for Proposal": "RFP",
            "Request for Proposals": "RFP",
            "RFI": "RFI",
            "RFP": "RFP",
        }.get(document)
        if kind is None:
            continue
        if re.search(
            r"\bgrant awards?\b|\bgrant applications?\b|\buniversity\b|\bcollege\b",
            row.text(),
            re.I,
        ):
            continue
        links = [n.attrs.get("href", "") for n in cells[1].walk() if n.tag == "a"]
        if len(links) != 1:
            raise ValueError("idaho_document_link_missing")
        url = links[0]
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.sde.idaho.gov"
            or parsed.username
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/wp-content/uploads/")
            or not parsed.path.lower().endswith(".pdf")
        ):
            raise ValueError("idaho_document_link_invalid")
        if url in seen:
            raise ValueError("idaho_duplicate_notice")
        seen.add(url)
        dates = re.findall(
            r"\bby\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})(?:\s+at\s+(\d{1,2}:\d{2}\s*(?:am|pm)\s+[A-Z]{2,4}))?",
            information,
        )
        if len(dates) != 1:
            raise ValueError("idaho_submission_deadline_unknown")
        stated, time = dates[0]
        if not re.search(
            r"Submit\s+(?:all\s+)?(?:required\s+)?(?:materials|proposals|responses)\b.{0,250}\bby\s+"
            + re.escape(stated),
            information,
            re.I,
        ):
            raise ValueError("idaho_submission_deadline_unknown")
        try:
            closing = datetime.strptime(stated.replace(",", ""), "%B %d %Y").date()
        except ValueError as exc:
            raise ValueError("idaho_submission_deadline_invalid") from exc
        if closing <= date.fromisoformat(today):
            continue
        if len(title) > 1000:
            raise ValueError("idaho_title_too_long")
        # Never retain the Additional Information cell containing personal email.
        deadline_text = stated + (" at " + time if time else "")
        quote = f"{title}. Documentation: {document}. Submit required materials by {deadline_text}."
        item = {
            "state": "ID",
            "url": url,
            "source_url": SOURCE,
            "title": title,
            "quote": quote,
            "notice_number": None,
            "deadline_date": closing.isoformat(),
            "buyer": "Idaho Department of Education",
            "coverage_scope": "Idaho Department of Education current opportunities only; not all school districts",
        }
        item["notice_id"] = _hash({"state": "ID", "url": url})
        item["hash"] = _hash(item)
        records.append(item)
    return records


def fetch(client, source, today):
    if source.get("state") not in ("ID", "Idaho") or source.get("url") != SOURCE:
        raise ValueError("invalid_idaho_source")
    with client.stream("GET", SOURCE, follow_redirects=False, timeout=20) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("idaho_source_unavailable")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError("idaho_source_too_large")
    return parse(content.decode("utf-8", errors="strict"), today)
