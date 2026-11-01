"""Optional national school technology/E-rate discovery, not general procurement.

No activation or model calls. Form 470 dates cannot establish an open RFP deadline.
"""

import hashlib
import json
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit

ENDPOINT = "https://opendata.usac.org/resource/jt8s-3q52.json"
STATES = frozenset(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ]
)
FIELDS = (
    "application_number",
    "service_request_id",
    "applicant_type",
    "fcc_form_470_status",
    "billed_entity_state",
    "rfp_documents",
    "form_pdf",
    "certified_date_time",
    "allowable_contract_date",
    "rfp_upload_date",
    "service_category",
    "service_type",
)
PAGE_SIZE = 250
MAX_PAGES = 8
MAX_BYTES = 1_000_000
LOOKBACK_DAYS = 90


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("usac_invalid_date")
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        raise ValueError("usac_invalid_date") from None


def _attachment(value, application):
    if not isinstance(value, dict) or not isinstance(value.get("url"), str):
        raise ValueError("usac_invalid_attachment")
    url = value["url"]
    parsed = urlsplit(url)
    if (
        len(url) > 2048
        or parsed.scheme != "https"
        or parsed.hostname != "publicdata.usac.org"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not parsed.path.startswith(f"/EPC/Prd/Form470/{application}/")
        or parsed.fragment
        or parsed.query
        or ".." in parsed.path
    ):
        raise ValueError("usac_invalid_attachment")
    return url


def _page(client, where, offset):
    params = {
        "$select": ",".join(FIELDS),
        "$where": where,
        "$order": "application_number,service_request_id,rfp_documents",
        "$limit": PAGE_SIZE,
        "$offset": offset,
    }
    with client.stream("GET", ENDPOINT, params=params, follow_redirects=False) as response:
        if (
            response.status_code != 200
            or "json" not in response.headers.get("content-type", "").lower()
        ):
            raise ValueError("usac_source_unavailable")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("usac_response_too_large")
    rows = json.loads(data)
    if not isinstance(rows, list) or len(rows) > PAGE_SIZE:
        raise ValueError("usac_invalid_page")
    return rows


def collect(client, state, today):
    """Return pending discovery records for one state, last 90 certification days.

    A full final page fails closed: callers must report unavailable/incomplete,
    never accept a partial result. URL validation does not verify document contents.
    """
    if state not in STATES:
        raise ValueError("usac_invalid_state")
    current = date.fromisoformat(today)
    cutoff = current - timedelta(days=LOOKBACK_DAYS)
    where = (
        f"billed_entity_state='{state}' AND fcc_form_470_status='Certified' "
        "AND applicant_type in ('School','School District') AND rfp_documents IS NOT NULL "
        f"AND certified_date_time >= '{cutoff.isoformat()}T00:00:00' "
        f"AND certified_date_time < '{(current + timedelta(days=1)).isoformat()}T00:00:00'"
    )
    results = {}
    for page in range(MAX_PAGES):
        rows = _page(client, where, page * PAGE_SIZE)
        for row in rows:
            if not isinstance(row, dict) or set(row) - set(FIELDS):
                raise ValueError("usac_invalid_record")
            if (
                row.get("billed_entity_state") != state
                or row.get("fcc_form_470_status") != "Certified"
                or row.get("applicant_type") not in ("School", "School District")
            ):
                raise ValueError("usac_scope_mismatch")
            application = row.get("application_number", "")
            service = row.get("service_request_id", "")
            if (
                not isinstance(application, str)
                or not application.isdigit()
                or not isinstance(service, str)
                or not service.isdigit()
            ):
                raise ValueError("usac_invalid_identity")
            certified = _timestamp(row.get("certified_date_time"))
            if not cutoff <= certified <= current:
                raise ValueError("usac_date_scope_mismatch")
            for key in ("allowable_contract_date", "rfp_upload_date"):
                if row.get(key) is not None:
                    _timestamp(row[key])
            attachment = _attachment(row.get("rfp_documents"), application)
            form = _attachment(row.get("form_pdf"), application)
            facts = {
                key: row[key]
                for key in FIELDS
                if key in row and key not in ("rfp_documents", "form_pdf")
            }
            facts.update(
                rfp_attachment_url=attachment,
                form_url=form,
                deadline="unknown; attachment contents not reviewed",
                allowable_contract_date_meaning="earliest permissible contracting date, not bid deadline",
                current_open_status="unknown",
            )
            item = {
                "state": state,
                "url": attachment,
                "source_url": ENDPOINT,
                "title": f"FCC Form 470 {application}, service request {service}",
                "quote": json.dumps(facts, sort_keys=True),
                "coverage_scope": "school technology/E-rate only",
                "review_status": "pending_document_review",
                "currently_open": None,
                "deadline_date": None,
                "alert_eligible": False,
                "attachment_verified": False,
                "notice_id": _hash(
                    {
                        "state": state,
                        "application": application,
                        "service": service,
                        "url": attachment,
                    }
                ),
            }
            item["hash"] = _hash(item)
            old = results.get(item["notice_id"])
            if old is not None and old != item:
                raise ValueError("usac_conflicting_record")
            results[item["notice_id"]] = item
        if len(rows) < PAGE_SIZE:
            return list(results.values())
    raise ValueError("usac_record_limit_incomplete")
