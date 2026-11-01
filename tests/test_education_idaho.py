import unittest

import httpx

from scripts import education_idaho as i


def row(
    date="November 11, 2026",
    extra="",
    href="https://www.sde.idaho.gov/wp-content/uploads/2026/10/literacy.pdf",
    doctype="Request for Information",
):
    return f'<tr><td>Idaho K-3 Literacy Coaching Services</td><td><a href="{href}">{doctype}</a></td><td>Submit all required materials to person@example.org by {date} at 4:00 pm MT {extra}</td></tr>'


def page(content=None):
    return (
        "<h2>Current Opportunities</h2><table><tr><th>Description</th><th>Documentation</th><th>Additional Information</th></tr>"
        + (row() if content is None else content)
        + "</table><h2>Contracting Terms &amp; Conditions</h2>"
    )


class IdahoTests(unittest.TestCase):
    def test_generic_rfi_link_uses_row_title(self):
        result = i.parse(page(), "2026-10-31")[0]
        self.assertEqual(result["title"], "Idaho K-3 Literacy Coaching Services")
        self.assertEqual(result["deadline_date"], "2026-11-11")
        self.assertIn("4:00 pm MT", result["quote"])
        self.assertNotIn("person@example.org", str(result))
        self.assertIsNone(result["notice_number"])

    def test_closed_full_row_and_expired(self):
        for r in [
            row(extra="Cancelled"),
            row(date="October 31, 2026"),
            row(doctype="Apply for Contract"),
        ]:
            self.assertEqual(i.parse(page(r), "2026-10-31"), [])

    def test_unknown_and_conflicting_dates_fail(self):
        for r in [row(date="TBD"), row(extra="extended by December 1, 2026")]:
            with self.assertRaises(ValueError):
                i.parse(page(r), "2026-10-31")

    def test_document_exact_host_and_scheme(self):
        for href in [
            "http://www.sde.idaho.gov/wp-content/uploads/a.pdf",
            "https://www.sde.idaho.gov.evil.example/wp-content/uploads/a.pdf",
            "https://www.sde.idaho.gov/wp-content/uploads/a.pdf?redirect=elsewhere",
        ]:
            with self.assertRaises(ValueError):
                i.parse(page(row(href=href)), "2026-10-31")

    def test_header_and_section_and_duplicates(self):
        for body in [
            page().replace("Current Opportunities", "Archived Opportunities"),
            page().replace("Documentation", "Other"),
            page(row() + row()),
            page(row() * 51),
        ]:
            with self.assertRaises(ValueError):
                i.parse(body, "2026-10-31")

    def test_sibling_rows_not_evidence(self):
        result = i.parse(
            page(row() + row(doctype="Apply for Contract", extra="Cancelled")), "2026-10-31"
        )
        self.assertEqual(len(result), 1)

    def test_fetch_one_bounded_page(self):
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=page())
            )
        ) as client:
            self.assertEqual(
                len(i.fetch(client, {"state": "Idaho", "url": i.SOURCE}, "2026-10-31")), 1
            )
        with (
            httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(302))) as client,
            self.assertRaises(ValueError),
        ):
            i.fetch(client, {"state": "ID", "url": i.SOURCE}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()


def test_question_deadline_is_not_submission_deadline():
    body = page().replace("Submit all required materials", "Submit questions")
    try:
        i.parse(body, "2026-11-01")
    except ValueError as exc:
        assert str(exc) == "idaho_submission_deadline_unknown"
    else:
        raise AssertionError("Question deadline must not qualify")
