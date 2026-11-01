"""PA public eMarketplace Education-agency RFP/RFI searches; no district coverage."""

import re
from datetime import date, datetime
from urllib.parse import parse_qs, urljoin, urlsplit

from scripts.education_agency_pages import Node, Tree, _hash

SOURCE = "https://www.emarketplace.state.pa.us/Search.aspx"
PREFIX = "ctl00$MainBody$"
MAX_BYTES = 3_000_000
MAX_RECORDS = 500


def _tree(body):
    tree = Tree(body)
    if re.search(
        r"access denied|request rejected|verify you are human|captcha challenge",
        tree.root.text(),
        re.I,
    ):
        raise ValueError("pennsylvania_source_challenge")
    return tree


def form_fields(body):
    tree = _tree(body)
    forms = [n for n in tree.root.walk() if n.tag == "form" and n.attrs.get("id") == "aspnetForm"]
    if len(forms) != 1 or urljoin(SOURCE, forms[0].attrs.get("action", "")) != SOURCE:
        raise ValueError("pennsylvania_search_form_missing")
    fields = {}
    for n in forms[0].walk():
        name = n.attrs.get("name")
        if not name:
            continue
        if n.tag == "input" and (
            n.attrs.get("type") in ("hidden", "text", "search") or "checked" in n.attrs
        ):
            fields[name] = n.attrs.get("value", "")
        elif n.tag == "select":
            options = [o for o in n.walk() if o.tag == "option"]
            if not options:
                raise ValueError("pennsylvania_empty_select")
            selected = next((o for o in options if "selected" in o.attrs), options[0])
            fields[name] = selected.attrs.get("value", "")
            required = {
                "ddlAgency": ("12", "Department of Education"),
                "ddlRows": ("32767", "ALL"),
            }.get(name.removeprefix(PREFIX))
            if required and not any(
                o.attrs.get("value") == required[0] and o.text() == required[1] for o in options
            ):
                raise ValueError("pennsylvania_filter_changed")
    if not all(
        key in fields
        for key in (
            "__VIEWSTATE",
            "__EVENTVALIDATION",
            PREFIX + "ddlAgency",
            PREFIX + "ddlTypes",
            PREFIX + "ddlRows",
            PREFIX + "rdoArch",
        )
    ):
        raise ValueError("pennsylvania_form_fields_missing")
    return fields


def parse_results(body, kind, today):
    fields = form_fields(body)
    expected = {"ddlAgency": "12", "ddlTypes": kind, "ddlRows": "32767", "rdoArch": "0"}
    if kind not in ("RFP", "RFI") or any(fields.get(PREFIX + k) != v for k, v in expected.items()):
        raise ValueError("pennsylvania_filter_not_applied")
    tree = _tree(body)
    tables = [
        n
        for n in tree.root.walk()
        if n.tag == "table" and n.attrs.get("id") == "ctl00_MainBody_grdResults"
    ]
    if len(tables) != 1 or "List of Current Solicitations" not in tables[0].text():
        raise ValueError("pennsylvania_results_missing")
    table = tables[0]
    if any(n.tag == "a" and "Page$" in n.attrs.get("href", "") for n in table.walk()):
        raise ValueError("pennsylvania_results_paginated")
    if "No records were found matching the selected criteria" in table.text():
        if any(
            n.tag == "a" and "Solicitations.aspx" in n.attrs.get("href", "") for n in table.walk()
        ):
            raise ValueError("pennsylvania_ambiguous_empty_results")
        return []
    headers = None
    records = []
    scanned = 0
    seen = set()
    for tr in table.walk():
        if tr.tag != "tr":
            continue
        cells = [n for n in tr.children if isinstance(n, Node) and n.tag in ("td", "th")]
        texts = [n.text() for n in cells]
        if "Solicitation #" in texts:
            headers = texts
            continue
        if not headers or len(cells) != len(headers):
            continue
        row = dict(zip(headers, texts, strict=True))
        required = [
            "Solicitation #",
            "Types",
            "Solicitation Title",
            "Description",
            "Agency",
            "Solicitation Start Date",
            "Solicitation Due Date",
            "Status",
        ]
        if any(k not in row for k in required):
            raise ValueError("pennsylvania_columns_changed")
        scanned += 1
        if scanned > MAX_RECORDS:
            raise ValueError("pennsylvania_record_budget")
        number = row["Solicitation #"]
        if number in seen:
            raise ValueError("pennsylvania_duplicate_notice")
        seen.add(number)
        if row["Agency"] != "Department of Education" or row["Types"] != kind:
            raise ValueError("pennsylvania_record_filter_mismatch")
        if row["Status"] != "Open":
            continue
        try:
            opened = datetime.strptime(row["Solicitation Start Date"].split()[0], "%m/%d/%Y").date()
            deadline = datetime.strptime(row["Solicitation Due Date"].split()[0], "%m/%d/%Y").date()
        except (ValueError, IndexError) as exc:
            raise ValueError("pennsylvania_invalid_date") from exc
        if not opened <= date.fromisoformat(today) < deadline:
            continue
        title = row["Solicitation Title"]
        description = row["Description"]
        text = title + " " + description
        if re.search(r"\b(cancelled|canceled|withdrawn|closed|awarded)\b", text, re.I):
            continue
        if not re.search(
            r"\bk[- ]?12\b|\bschools?\b|\belementary\b|\bsecondary\b|\bstudent assessment\b",
            text,
            re.I,
        ):
            continue
        if re.search(
            r"\buniversity\b|\bcollege\b|\bhigher education\b|\bgrant awards?\b|\bgrant applications?\b",
            text,
            re.I,
        ):
            continue
        links = [n.attrs.get("href", "") for n in tr.walk() if n.tag == "a" and n.text() == number]
        if len(links) != 1:
            raise ValueError("pennsylvania_notice_link_missing")
        url = urljoin(SOURCE, links[0])
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.emarketplace.state.pa.us"
            or parsed.path != "/Solicitations.aspx"
            or parse_qs(parsed.query) != {"SID": [number]}
            or parsed.fragment
        ):
            raise ValueError("pennsylvania_notice_link_invalid")
        # Never emit Contact Person column; surrounding rows are never evidence.
        quote = f"{title}. {description}. Agency: {row['Agency']}. Type: {kind}. Status: Open. Solicitation due: {row['Solicitation Due Date']}"
        if len(quote) > 12000:
            continue
        item = {
            "state": "PA",
            "url": url,
            "source_url": SOURCE,
            "title": title,
            "quote": quote,
            "notice_number": number,
            "deadline_date": deadline.isoformat(),
            "buyer": row["Agency"],
            "coverage_scope": "Pennsylvania Department of Education only; not all school districts",
        }
        item["notice_id"] = _hash({"state": "PA", "url": url, "notice_number": number})
        item["hash"] = _hash(item)
        records.append(item)
    if headers is None or not scanned:
        raise ValueError("pennsylvania_record_structure_missing")
    return records


def _request(client, method, data=None):
    with client.stream(method, SOURCE, data=data, follow_redirects=False, timeout=20) as response:
        if (
            response.status_code != 200
            or "html" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("pennsylvania_source_unavailable")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError("pennsylvania_source_too_large")
    return content.decode("utf-8", errors="strict")


def fetch(client, source, today):
    if source.get("state") not in ("PA", "Pennsylvania") or source.get("url") != SOURCE:
        raise ValueError("invalid_pennsylvania_source")
    records = []
    # Four requests maximum: independent public form initialization + search per kind.
    for kind in ("RFP", "RFI"):
        fields = form_fields(_request(client, "GET"))
        fields.update(
            {
                PREFIX + "ddlAgency": "12",
                PREFIX + "ddlTypes": kind,
                PREFIX + "ddlRows": "32767",
                PREFIX + "rdoArch": "0",
                PREFIX + "btnSearch": "Search",
            }
        )
        records.extend(parse_results(_request(client, "POST", fields), kind, today))
    return records
