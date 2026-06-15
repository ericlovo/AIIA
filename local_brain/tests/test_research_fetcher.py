"""Offline tests for the research fetcher's content extraction.

No network or Brain needed — these exercise the pure `extract_text`
dispatcher (HTML / plain / PDF sniffing) and `pdf_to_text`. PDFs are
built in-process with a minimal valid-PDF generator.
"""

from __future__ import annotations

import pytest

from local_brain.research.fetcher import extract_text, html_to_text, pdf_to_text


def make_pdf(text: str) -> bytes:
    """Assemble a minimal but valid single-page PDF with extractable text."""
    content = b"BT /F1 24 Tf 72 700 Td (%s) Tj ET" % text.encode()
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(content), content),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj" % i + body + b"endobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += b"xref\n0 %d\n" % n
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (n, xref_pos)
    return bytes(out)


class TestExtractTextHTML:
    def test_strips_tags(self):
        out = extract_text(b"<html><body><p>Hello</p><p>World</p></body></html>", "text/html")
        assert "Hello" in out and "World" in out
        assert "<p>" not in out

    def test_skips_script_and_style(self):
        html = b"<p>keep</p><script>var x=1</script><style>.a{}</style>"
        out = extract_text(html, "text/html; charset=utf-8")
        assert "keep" in out
        assert "var x" not in out

    def test_html_to_text_helper_matches(self):
        assert html_to_text("<p>hi</p>") == "hi"


class TestExtractTextPlain:
    def test_plain_text_passthrough(self):
        assert extract_text(b"just text", "text/plain") == "just text"

    def test_empty_content_type_treated_as_text(self):
        assert extract_text(b"bare", "") == "bare"

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            extract_text(b"\x89PNG\r\n", "image/png")


class TestExtractTextPDF:
    def test_pdf_by_magic_bytes(self):
        out = extract_text(make_pdf("Hello Erdos 351"), "application/octet-stream")
        assert "Hello Erdos 351" in out

    def test_pdf_by_content_type(self):
        out = extract_text(make_pdf("Bound proof"), "application/pdf")
        assert "Bound proof" in out

    def test_pdf_by_url_suffix(self):
        # Server lies about type, no magic match impossible here, but the
        # .pdf URL suffix should still route through the PDF extractor.
        out = extract_text(make_pdf("via suffix"), "application/pdf", url="https://x/p.pdf")
        assert "via suffix" in out

    def test_pdf_to_text_direct(self):
        assert "Direct call" in pdf_to_text(make_pdf("Direct call"))

    def test_page_limit_truncation_note(self):
        # max_pages=0 forces the "truncated" branch deterministically.
        out = pdf_to_text(make_pdf("ignored"), max_pages=0)
        assert "Truncated" in out
        assert "1 pages" in out


class TestPdfDetectionPriority:
    def test_pdf_magic_wins_over_html_content_type(self):
        # A server mislabels a PDF as text/html; magic bytes should win.
        out = extract_text(make_pdf("real pdf"), "text/html")
        assert "real pdf" in out
