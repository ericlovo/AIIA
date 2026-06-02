"""
URL fetcher for the research harness.

Fetches a URL with httpx and strips HTML to clean plain text.
Uses stdlib html.parser — no extra dependencies.
"""

import logging
import re
from html.parser import HTMLParser

import httpx

logger = logging.getLogger("aiia.research.fetcher")

_SKIP_TAGS = frozenset(["script", "style", "nav", "header", "footer", "aside", "noscript"])
_BLOCK_TAGS = frozenset(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr"])

_HEADERS = {
    "User-Agent": "AIIA-Research/0.5 (autonomous research harness)",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
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


async def fetch_url(url: str, timeout: float = 15.0, max_chars: int = 80_000) -> tuple[str, str]:
    """
    Fetch a URL and return (canonical_url, plain_text). Raises on failure.
    """
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=_HEADERS) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if "text/html" in content_type:
        text = html_to_text(response.text)
    elif "text/plain" in content_type or not content_type:
        text = response.text
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"

    canonical_url = str(response.url)
    logger.info(f"Fetched {canonical_url}: {len(text)} chars")
    return canonical_url, text
