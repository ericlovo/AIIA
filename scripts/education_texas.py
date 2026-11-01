"""Public Texas ESBD search API, all open pages bounded to 600 rows/25 pages."""

import hashlib
import json
import re
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.parse import quote

SOURCE = "https://www.txsmartbuy.gov/esbd"
API = "https://www.txsmartbuy.gov/app/extensions/CPA/CPAMain/1.0.0/services/ESBD.Service.ss"
MAX_BYTES = 2_000_000
MAX_RECORDS = 600
MAX_PAGES = 25
TYPE = re.compile(r"\b(?:RFP|RFI)\b|\bRequest for (?:Proposals?|Information)\b", re.I)
BUYER = re.compile(r"\bISD\b|\bschools?\b|\bEducation Agency\b|\bEducation Service Center\b", re.I)


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _read(client, method, url, **kwargs):
    with client.stream(method, url, follow_redirects=False, timeout=25, **kwargs) as response:
        if response.status_code != 200:
            raise ValueError("texas_source_unavailable")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise ValueError("texas_source_too_large")
    return body.decode("utf-8", errors="strict")


def _digest(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def fetch(client, source, today):
    """No account needed. The official UI calls this read-only search via POST.

    status=1 returns both Posted and Addendum Posted. Raw /esbd HTML ignores
    query filters and contains historical records, so it is not used for paging.
    Same-day cutoff notices are withheld; dates and times come from detail pages.
    """
    if source.get("state") != "TX" or source.get("url") != SOURCE:
        raise ValueError("invalid_texas_source")
    current = date.fromisoformat(today)
    rows, total, size, seen = [], None, None, set()
    page = 1
    while True:
        data = json.loads(
            _read(client, "POST", API, json={"urlRoot": "esbd", "page": page, "status": "1"})
        )
        n, width, lines = (
            data.get("totalRecordsFound"),
            data.get("recordsPerPage"),
            data.get("lines"),
        )
        if (
            type(n) is not int
            or type(width) is not int
            or not 0 <= n <= MAX_RECORDS
            or not 1 <= width <= MAX_RECORDS
            or not isinstance(lines, list)
            or data.get("page") != page
        ):
            raise ValueError("texas_feed_incomplete")
        if total is None:
            total, size = n, width
        if (
            n != total
            or width != size
            or (total + size - 1) // size > MAX_PAGES
            or len(lines) != min(size, max(0, total - (page - 1) * size))
        ):
            raise ValueError("texas_feed_incomplete")
        for row in lines:
            key = row.get("internalid") if isinstance(row, dict) else None
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("texas_feed_incomplete")
            seen.add(key)
            rows.append(row)
        if len(rows) == total:
            break
        page += 1
    found = []
    for row in rows:
        fields = [
            "solicitationId",
            "title",
            "agencyName",
            "agencyNumber",
            "responseDue",
            "responseTime",
            "statusName",
        ]
        if any(not isinstance(row.get(k), str) for k in fields):
            raise ValueError("texas_row_schema_changed")
        number, title, agency = row["solicitationId"], row["title"], row["agencyName"]
        if (
            row["statusName"] not in ("Posted", "Addendum Posted")
            or row.get("cancelledDate")
            or not BUYER.search(agency)
            or re.search(r"\b(?:university|college|higher education)\b", agency, re.I)
        ):
            continue
        if source.get("buyer_scope") == "state_agencies" and re.search(
            r"\bISD\b|\bdistrict\b", agency, re.I
        ):
            continue
        try:
            deadline = datetime.strptime(row["responseDue"], "%m/%d/%Y").date()
            posted = datetime.strptime(row["postingDate"], "%m/%d/%Y").date()
        except (ValueError, KeyError, TypeError):
            continue
        if deadline <= current or posted > current or not number or len(number) > 200:
            continue
        url = SOURCE + "/" + quote(number, safe="")
        parser = Text()
        parser.feed(_read(client, "GET", url))
        detail = " ".join(" ".join(parser.parts).split())
        if (
            f"Solicitation ID: {number} " not in detail
            or f"Response Due Date: {row['responseDue']} " not in detail
            or f"Response Due Time: {row['responseTime']} " not in detail
            or f"Agency/Texas SmartBuy Member Number: {row['agencyNumber']} " not in detail
        ):
            raise ValueError("texas_detail_mismatch")
        if not any(f"Status: {status} " in detail for status in ("Posted", "Addendum Posted")):
            continue
        description = re.search(r"Solicitation Description:\s*(.*?)\s*Attachments", detail)
        evidence = title + " " + (description.group(1) if description else "")
        if not TYPE.search(evidence) or re.search(
            r"\b(?:cancelled|canceled|withdrawn|awarded|closed)\b", evidence, re.I
        ):
            continue
        payload = {
            "title": title,
            "agency": agency,
            "agency_number": row["agencyNumber"],
            "notice_number": number,
            "deadline_date": deadline.isoformat(),
            "deadline_time": row["responseTime"],
            "status": row["statusName"],
            "description": description.group(1) if description else "",
            "as_of": today,
        }
        item = {
            "state": "TX",
            "url": url,
            "source_url": SOURCE,
            "title": title,
            "notice_number": number,
            "deadline_date": deadline.isoformat(),
            "quote": json.dumps(payload, sort_keys=True),
            "coverage_scope": "Participating ESBD education buyers only; not exhaustive district coverage",
        }
        item["identity"] = _digest({"state": "TX", "url": url})
        item["hash"] = _digest(item)
        found.append(item)
    return found
