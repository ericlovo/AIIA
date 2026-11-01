"""NYSED public procurements with a verified missing-intermediate chain repair.

The server omits its intermediate. Existing certifi roots remain the only trust
anchors: verify the pinned intermediate against them, then disable partial-chain
trust when supplying that chain certificate to an isolated TLS context.
"""

import hashlib
import json
import re
import shutil
import ssl
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import certifi
import httpx

SOURCE = "https://www.nysed.gov/funding-opportunities/competitive-service-contract-procurements"
INTERMEDIATE = "https://secure.globalsign.com/cacert/gsrsaovsslca2018.crt"
PIN = "b676ffa3179e8812093a1b5eafee876ae7a6aaf231078dad1bfb21cd2893764a"
MAX_BYTES = 2_000_000
MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
DATE = MONTHS + r"\s+\d{1,2},?\s+20\d{2}"


def _read(client, url, limit=MAX_BYTES):
    with client.stream("GET", url, follow_redirects=False, timeout=20) as response:
        if response.status_code != 200:
            raise ValueError("nysed_source_unavailable")
        result = bytearray()
        for chunk in response.iter_bytes():
            result.extend(chunk)
            if len(result) > limit:
                raise ValueError("nysed_source_too_large")
    return bytes(result)


def repaired_context(client):
    """No global store changes, no new root, no disabled TLS verification.

    Rotation of the official intermediate fails closed until its pin is reviewed.
    Requires openssl executable; works with OpenSSL's existing root-chain verifier.
    """
    der = _read(client, INTERMEDIATE, 100_000)
    if hashlib.sha256(der).hexdigest() != PIN:
        raise ValueError("nysed_intermediate_pin_mismatch")
    pem = ssl.DER_cert_to_PEM_cert(der)
    executable = shutil.which("openssl")
    if not executable:
        raise ValueError("nysed_openssl_unavailable")
    roots = certifi.where()
    with tempfile.TemporaryDirectory(prefix="nysed-chain-") as directory:
        path = Path(directory) / "intermediate.pem"
        path.write_text(pem)
        try:
            verified = subprocess.run(
                [executable, "verify", "-CAfile", roots, "-no-CApath", "-no-CAstore", str(path)],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("nysed_intermediate_verification_timeout") from None
        if verified.returncode != 0:
            raise ValueError("nysed_intermediate_untrusted")
    context = ssl.create_default_context(cafile=roots)
    context.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
    context.load_verify_locations(cadata=pem)
    if (
        context.verify_mode != ssl.CERT_REQUIRED
        or not context.check_hostname
        or context.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    ):
        raise ValueError("nysed_tls_verification_invalid")
    return context


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.links, self.link_text = [], [], []
        self.href, self.skip = None, 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.link_text = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if tag == "a" and self.href is not None:
            self.links.append((self.href, " ".join(" ".join(self.link_text).split())))
            self.href = None

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)
            if self.href is not None:
                self.link_text.append(data)

    @property
    def text(self):
        return " ".join(" ".join(self.parts).split())


def _page(client, url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.nysed.gov"
        or not parsed.path.startswith("/funding-opportunities/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("nysed_invalid_url")
    page = Page()
    page.feed(_read(client, url).decode("utf-8", errors="strict"))
    return page


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def qualify(title, url, text, today):
    """Only dates inside an explicit submission-deadline section qualify.

    Question, conference, contract-term and posting dates are not deadlines.
    Missing/ambiguous HTML deadlines and document-only notices stay withheld.
    """
    if re.search(r"\b(?:cancelled|canceled|withdrawn|awarded|closed)\b", text, re.I):
        return None
    if re.search(
        r"\b(?:college|university|libraries|library|geological|museum|adult literacy|rehabilitation)\b",
        title,
        re.I,
    ):
        return None
    section = re.search(
        r"Submission Deadline(?:\s*&\s*Instructions)?\s*(.*?)(?:NYSED Designated Contacts|RFP Documents|Get the Latest Updates)",
        text,
        re.I,
    )
    if not section:
        return None
    content = section.group(1)
    matches = re.findall(
        r"(?:must be received|must be submitted|bids? (?:are )?due|proposals? (?:are )?due)[^.]{0,200}?("
        + DATE
        + r")",
        content,
        re.I,
    )
    try:
        deadlines = {
            datetime.strptime(re.sub(",", "", d).title(), "%B %d %Y").date() for d in matches
        }
    except ValueError:
        return None
    if len(deadlines) != 1 or next(iter(deadlines)) <= date.fromisoformat(today):
        return None
    deadline = next(iter(deadlines)).isoformat()
    item = {
        "state": "NY",
        "url": url,
        "source_url": SOURCE,
        "title": title,
        "deadline_date": deadline,
        "quote": json.dumps(
            {
                "title": title,
                "agency": "New York State Education Department",
                "deadline_date": deadline,
                "submission_evidence": content[:1800],
            },
            sort_keys=True,
        ),
        "coverage_scope": "NYSED main-host HTML statewide agency procurements only; excludes ACCES off-host and document-only notices; not all New York school districts",
    }
    item["identity"] = _digest({"state": "NY", "url": url})
    item["hash"] = _digest(item)
    return item


def fetch(client, source, today):
    if source.get("state") != "NY" or source.get("url") != SOURCE:
        raise ValueError("invalid_nysed_source")
    context = repaired_context(client)
    with httpx.Client(
        verify=context, timeout=20, follow_redirects=False, trust_env=False
    ) as isolated:
        listing = _page(isolated, SOURCE)
        if "Competitive Service Contract Procurements" not in listing.text:
            raise ValueError("nysed_listing_unavailable")
        # Separate-host ACCES adult/vocational and document-only notices are
        # outside this adapter's explicitly limited main-host HTML coverage.
        entries = {
            urljoin(SOURCE, u): t
            for u, t in listing.links
            if re.match(r"^(?:RFP|RFI)\b", t, re.I)
            and urlsplit(urljoin(SOURCE, u)).netloc == "www.nysed.gov"
            and urlsplit(urljoin(SOURCE, u)).path.startswith("/funding-opportunities/")
            and not urlsplit(u).path.lower().endswith((".pdf", ".doc", ".docx"))
        }
        if not entries or len(entries) > 60:
            raise ValueError("nysed_listing_incomplete")

        def collect(pair):
            url, title = pair
            return qualify(title, url, _page(isolated, url).text, today)

        with ThreadPoolExecutor(max_workers=4) as executor:
            return [item for item in executor.map(collect, entries.items()) if item is not None]
