"""
URL fetcher for the research harness.

Fetches a URL with httpx and returns clean plain text. HTML is stripped with
the stdlib html.parser; PDFs (arXiv full texts and the like) are extracted
with pypdf. Content type is decided by the response header *and* the leading
bytes, so PDFs served as application/octet-stream are still recognized.
"""

import asyncio
import io
import logging
import re
from html.parser import HTMLParser

import httpx

logger = logging.getLogger("aiia.research.fetcher")

_SKIP_TAGS = frozenset(["script", "style", "nav", "header", "footer", "aside", "noscript"])
_BLOCK_TAGS = frozenset(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr"])

_PDF_MAGIC = b"%PDF"
_PDF_MAX_PAGES = 60  # bound work on large papers; truncation is noted in the text

_HEADERS = {
    "User-Agent": "AIIA-Research/0.5 (autonomous research harness)",
    "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _BLOCK_TAGS and self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        raw = re.sub(r" {2,}", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def pdf_to_text(data: bytes, max_pages: int = _PDF_MAX_PAGES) -> str:
    """Extract plain text from PDF bytes using pypdf.

    Imports pypdf lazily so the fetcher stays importable (and HTML/text
    fetching keeps working) even if pypdf is missing — in that case a PDF
    raises ValueError, which callers log as a research gap.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ValueError("PDF support requires pypdf (pip install pypdf)") from e

    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages
    parts: list[str] = []
    for page in pages[:max_pages]:
        try:
            parts.append((page.extract_text() or "").strip())
        except Exception as e:  # pragma: no cover - malformed page
            logger.warning(f"PDF page extraction failed: {e}")
    text = "\n\n".join(p for p in parts if p)
    if len(pages) > max_pages:
        text += f"\n\n[Truncated: first {max_pages} of {len(pages)} pages]"
    return text.strip()


def _looks_like_pdf(content: bytes, content_type: str, url: str) -> bool:
    return (
        content[:4] == _PDF_MAGIC
        or "application/pdf" in content_type
        or url.lower().split("?")[0].endswith(".pdf")
    )


def extract_text(content: bytes, content_type: str, url: str = "") -> str:
    """Turn raw response bytes into plain text based on type sniffing.

    Pure and synchronous so it can be unit-tested without httpx and run
    off the event loop (PDF parsing is CPU-bound). Raises ValueError for
    content types we don't handle.
    """
    ct = content_type.lower()
    if _looks_like_pdf(content, ct, url):
        return pdf_to_text(content)
    if "text/html" in ct or "xml" in ct:
        return html_to_text(content.decode("utf-8", "replace"))
    if "text/plain" in ct or not ct:
        return content.decode("utf-8", "replace")
    raise ValueError(f"Unsupported content type: {content_type}")


async def fetch_url(url: str, timeout: float = 15.0, max_chars: int = 80_000) -> tuple[str, str]:
    """
    Fetch a URL and return (canonical_url, plain_text). Raises on failure.
    """
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=_HEADERS
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    canonical_url = str(response.url)

    # Run extraction (HTML stripping / PDF parsing) off the event loop.
    text = await asyncio.to_thread(extract_text, response.content, content_type, canonical_url)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"

    logger.info(f"Fetched {canonical_url} [{content_type or '?'}]: {len(text)} chars")
    return canonical_url, text
