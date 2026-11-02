"""Segment query banks, named-organization prefilter, and yield-weighted rotation.

Importing this module starts no work and opens no network. Queries stay inside
the existing Google News RSS path. They do not crawl article pages.
"""

import hashlib
import json
import re

QUERY_BANK = {
    "healthcare": (
        {
            "id": "hc-ehr-event",
            "q": (
                '(hospital OR "health system") (EHR OR "electronic health record" OR Epic '
                'OR Cerner OR "Oracle Health") (migration OR replacement OR implements OR '
                '"goes live" OR "has selected")'
            ),
        },
        {
            "id": "hc-cio-hire",
            "q": (
                '(hospital OR "health system") ("chief information officer" OR CIO OR '
                '"chief technology officer" OR CTO) (named OR appointed OR hired OR joins)'
            ),
        },
        {
            "id": "hc-rfp",
            "q": (
                '(hospital OR "health system") (RFP OR "request for proposal") (EHR OR ERP '
                'OR "revenue cycle" OR "patient portal" OR "IT services")'
            ),
        },
        {
            "id": "hc-acquisition",
            "q": (
                '("health system" OR hospital) (acquires OR acquisition OR merger) '
                '(hospital OR clinic OR "medical group" OR "health system")'
            ),
        },
        {
            "id": "hc-expansion",
            "q": (
                '(hospital OR "health system") ("breaks ground" OR "new campus" OR '
                '"opens clinic" OR "new hospital" OR "outpatient center")'
            ),
        },
    ),
    "pe-technical-diligence": (
        {
            "id": "pe-software-acquisition",
            "q": (
                '"private equity" (acquires OR acquired OR "to acquire") (software OR '
                'technology OR "IT services" OR SaaS OR platform)'
            ),
        },
        {
            "id": "pe-platform-addon",
            "q": (
                '("private equity" OR "PE-backed") ("platform investment" OR '
                '"add-on acquisition" OR "portfolio company") (technology OR software)'
            ),
        },
        {
            "id": "pe-cto-hire",
            "q": (
                '"private equity" ("chief technology officer" OR CTO OR CIO OR '
                '"technology transformation") (appointed OR hired OR names)'
            ),
        },
        {
            "id": "pe-systems-change",
            "q": (
                '("PE-backed" OR "private equity") (ERP OR Salesforce OR Workday OR '
                '"system migration" OR "digital transformation")'
            ),
        },
    ),
    "alumni-associations": (
        {
            "id": "al-platform-event",
            "q": (
                '"alumni association" (platform OR website OR CRM OR membership OR '
                "engagement) (launches OR selects OR implements OR migrates OR redesign)"
            ),
        },
        {
            "id": "al-advancement-vendor",
            "q": (
                '("university advancement" OR "alumni relations" OR "alumni association") '
                "(Blackbaud OR Salesforce OR Ellucian OR Anthology) (selects OR implements "
                "OR launches OR migrates)"
            ),
        },
        {
            "id": "al-rfp",
            "q": (
                '"alumni association" (RFP OR "request for proposal" OR vendor) '
                "(website OR software OR membership OR fundraising OR platform)"
            ),
        },
        {
            "id": "al-named-associations",
            "q": (
                '("Harvard Alumni Association" OR "Stanford Alumni Association" OR '
                '"Penn Alumni" OR "Alumni Association of the University of Michigan" OR '
                '"Ohio State Alumni Association" OR "Texas Exes" OR "UCLA Alumni") '
                "(platform OR website OR membership OR fundraising OR CRM OR giving)"
            ),
        },
    ),
}

GENERIC_ORGANIZATIONS = {
    "hospital",
    "hospitals",
    "clinic",
    "clinics",
    "healthcare",
    "health care",
    "hospital expansion",
    "healthcare industry",
    "healthcare workers",
    "doctors",
    "nurses",
    "private equity",
    "private equity firms",
    "company",
    "companies",
    "fund",
    "alumni",
    "alumni association",
    "association",
    "university",
    "college",
    "industry",
    "market",
    "report",
    "analysis",
}

TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

ORG_SUFFIX = re.compile(
    r"\b(?:Inc|LLC|Ltd|Corp|Corporation|University|College|Hospital|Health|System|"
    r"Association|Foundation|Partners|Capital|Holdings|Clinic|Medical)\b",
    re.I,
)
PROPER_SPAN = re.compile(
    r"\b([A-Z][A-Za-z0-9&'.-]+(?:[ \-](?:[A-Z][A-Za-z0-9&'.-]+|and|of|the|for|at)){1,6})\b"
)


def queries_for(segment):
    bank = QUERY_BANK.get(segment)
    if not bank:
        raise ValueError("unknown_segment")
    return bank


def query_by_id(segment, query_id):
    for item in queries_for(segment):
        if item["id"] == query_id:
            return item
    return None


def default_query(segment):
    return queries_for(segment)[0]


def select_discovery_query(segment, yield_stats=None, query_id=None, slot=0):
    """Pick a query for this run. Unused queries stay in rotation; zero-yield decays."""
    chosen = query_by_id(segment, query_id) if query_id else None
    if chosen:
        return chosen
    bank = queries_for(segment)
    stats = {
        row["query_id"]: row
        for row in (yield_stats or [])
        if isinstance(row, dict) and row.get("segment") == segment
    }
    ranked = []
    for index, item in enumerate(bank):
        row = stats.get(item["id"]) or {}
        jobs = row.get("jobs")
        supported = row.get("supported")
        jobs = jobs if type(jobs) is int and jobs >= 0 else 0
        supported = supported if type(supported) is int and supported >= 0 else 0
        score = (supported + 0.5) / (jobs + 1)
        if jobs >= 4 and supported == 0:
            score *= 0.25
        ranked.append((score, 0 if jobs == 0 else 1, index, item))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    window = ranked[: max(2, (len(ranked) + 1) // 2)]
    return window[int(slot) % len(window)][3]


def _normalize_name(value):
    cleaned = re.sub(r"[^\w]+", " ", (value or "").casefold()).strip()
    return re.sub(r"^(the|a|an)\s+", "", cleaned)


def named_organization_present(source):
    """True when headline/snippet likely names an organization, not a generic theme."""
    if not isinstance(source, dict):
        return False
    text = " ".join(part for part in (source.get("title"), source.get("text")) if part)
    if not text.strip():
        return False
    publisher = _normalize_name(source.get("publisher", ""))
    for match in PROPER_SPAN.finditer(text):
        name = match.group(1).strip(" ,.;:|-")
        normalized = _normalize_name(name)
        words = [word for word in normalized.split() if word and word not in TITLE_STOPWORDS]
        if len(words) < 2 or normalized in GENERIC_ORGANIZATIONS:
            continue
        if publisher and (normalized == publisher or normalized in publisher):
            continue
        if ORG_SUFFIX.search(name) or len(words) >= 2:
            return True
    return False


def evidence_text(source, limit=1800):
    """Title plus RSS snippet, capped so a cold Qwen load stays inside the job wall."""
    if not isinstance(source, dict):
        return ""
    title = (source.get("title") or "").strip()
    snippet = (source.get("text") or "").strip()
    if snippet.startswith(title):
        combined = snippet
    elif snippet:
        combined = f"{title} {snippet}".strip() if title else snippet
    else:
        combined = title
    encoded = combined.encode("utf-8")
    if len(encoded) <= limit:
        return combined
    return encoded[:limit].decode("utf-8", errors="ignore").rstrip()


def attach_snippet(source, limit=1800):
    text = evidence_text(source, limit=limit)
    if not text:
        return source
    attached = dict(source)
    attached["text"] = text
    return attached


def discovery_slot(entity_id):
    digest = hashlib.sha256((entity_id or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def empty_yield_row(segment, query_id):
    return {
        "segment": segment,
        "query_id": query_id,
        "jobs": 0,
        "supported": 0,
        "rejected": 0,
        "failed": 0,
        "inconclusive": {},
    }


def apply_yield(row, outcome, reason=None):
    updated = {
        "segment": row["segment"],
        "query_id": row["query_id"],
        "jobs": int(row.get("jobs") or 0) + 1,
        "supported": int(row.get("supported") or 0),
        "rejected": int(row.get("rejected") or 0),
        "failed": int(row.get("failed") or 0),
        "inconclusive": dict(row.get("inconclusive") or {}),
    }
    if outcome == "supported":
        updated["supported"] += 1
    elif outcome == "rejected":
        updated["rejected"] += 1
    elif outcome == "failed":
        updated["failed"] += 1
    else:
        key = reason if isinstance(reason, str) and reason else "inconclusive"
        updated["inconclusive"][key] = updated["inconclusive"].get(key, 0) + 1
    return updated


def serialize_inconclusive(counts):
    return json.dumps(counts or {}, sort_keys=True, separators=(",", ":"))


def parse_inconclusive(raw):
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(key, str) and type(value) is int and value >= 0
    }
