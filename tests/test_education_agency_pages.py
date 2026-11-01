import unittest

import httpx

from scripts import education_agency_pages as a


def al(section="Open", deadline="November 2, 2026", extra=""):
    return f'<h3 class="qode-title-holder">{section}</h3><div class="vc_grid-item"><a class="vc_gitem-link" href="/2026/10/notice/"></a><h4>RFI 2026-05 K-5 Mathematics</h4><p>AMENDED Proposal Submission Deadline {deadline}</p>{extra}</div>'


def ca(url="/notice/"):
    return f'<h3>ACTIVE</h3><p><a href="{url}">School evaluation RFP</a></p><h3>ARCHIVE</h3><a href="/old/">Old RFP</a>'


def detail(deadline="November 5, 2026", extra=""):
    return f'<div class="post-content">Request for Proposal school evaluation. Proposals Due Date: {deadline} by 4PM {extra}<p>Point of Contact: Example Person</p></div>'


class AgencyTests(unittest.TestCase):
    def test_al_section_and_notice_binding(self):
        rows = a.parse_alabama(al() + al("Closed"), "2026-10-31")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["notice_number"], "RFI 2026-05")
        self.assertEqual(rows[0]["deadline_date"], "2026-11-02")
        self.assertEqual(rows[0]["url"], "https://www.alabamaachieves.org/2026/10/notice/")

    def test_al_closed_expired_unknown_and_contradictory(self):
        for content in [
            al(deadline="October 31, 2026"),
            al(deadline="TBD"),
            al(extra="Awarded"),
            al(extra="Proposal Submission Deadline December 5, 2026"),
        ]:
            self.assertEqual(a.parse_alabama(content, "2026-10-31"), [])
        self.assertEqual(
            a.parse_alabama('<h3 class="qode-title-holder">Open</h3>' + al("Closed"), "2026-10-31"),
            [],
        )

    def test_stable_identity_changed_evidence(self):
        first = a.parse_alabama(al(), "2026-10-31")[0]
        second = a.parse_alabama(al(deadline="November 3, 2026"), "2026-10-31")[0]
        self.assertEqual(first["notice_id"], second["notice_id"])
        self.assertNotEqual(first["hash"], second["hash"])

    def test_ca_active_only_and_details(self):
        links = a.california_links(ca())
        self.assertEqual(list(links), ["https://ccee-ca.org/notice/"])
        row = a.parse_california_detail(detail(), next(iter(links)), "School RFP", "2026-10-31")
        self.assertEqual(row["deadline_date"], "2026-11-05")
        self.assertIsNone(row["notice_number"])
        self.assertNotIn("Example Person", row["quote"])

    def test_ca_wordpress_page_variant(self):
        body = (
            detail()
            .replace('<div class="post-content">', '<article class="page hentry">')
            .replace("</div>", "</article>")
        )
        self.assertIsNotNone(
            a.parse_california_detail(body, "https://ccee-ca.org/notice/", "RFP", "2026-10-31")
        )

    def test_ca_date_and_rejection(self):
        for body in [
            detail("October 31, 2026"),
            detail("TBD"),
            detail(extra="Cancelled"),
            detail().replace("Proposals Due Date", "Questions Due Date"),
        ]:
            self.assertIsNone(
                a.parse_california_detail(
                    body, "https://ccee-ca.org/notice/", "School RFP", "2026-10-31"
                )
            )
        for body in [
            "<h3>Archive</h3>",
            ca("https://evil.example/notice"),
            "<h3>ACTIVE</h3>" + "".join(f'<a href="/{i}/">RFP{i}</a>' for i in range(7)),
        ]:
            with self.assertRaises(ValueError):
                a.california_links(body)

    def test_fetch_bounds_and_status(self):
        for status, headers, body in [
            (302, {"location": "https://evil.example"}, ""),
            (200, {"content-type": "text/html"}, "Access denied"),
            (200, {"content-type": "text/html"}, "x" * (a.MAX_BYTES + 1)),
            (200, {"content-type": "application/json"}, "{}"),
        ]:
            with (
                httpx.Client(
                    transport=httpx.MockTransport(
                        lambda req, status=status, headers=headers, body=body: httpx.Response(
                            status, headers=headers, text=body
                        )
                    )
                ) as client,
                self.assertRaises(ValueError),
            ):
                a.fetch(client, {"state": "AL", "url": a.SOURCES["AL"]}, "2026-10-31")

    def test_fetch_ca_two_pages_no_redirect(self):
        requests = []

        def handler(req):
            requests.append(req)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=ca() if len(requests) == 1 else detail(),
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(
                len(a.fetch(client, {"state": "California", "url": a.SOURCES["CA"]}, "2026-10-31")),
                1,
            )
        self.assertEqual(len(requests), 2)
        with self.assertRaises(ValueError):
            a.fetch(None, {"state": "AL", "url": a.SOURCES["CA"]}, "2026-10-31")

    def test_structure_missing_not_empty_success(self):
        with self.assertRaises(ValueError):
            a.parse_alabama("<p>Welcome</p>", "2026-10-31")
        with self.assertRaises(ValueError):
            a.parse_california_detail(
                "<p>Welcome</p>", "https://ccee-ca.org/notice/", "RFP", "2026-10-31"
            )


if __name__ == "__main__":
    unittest.main()


def test_cancellation_after_contact_not_stripped_before_validation():
    text = "RFP for K12 services. Proposal Submission Deadline: November 5, 2026. Point of Contact: Procurement. This solicitation is cancelled."
    assert (
        a._record("CA", "https://ccee-ca.org/notice/", "Services RFP", text, "2026-11-01") is None
    )
