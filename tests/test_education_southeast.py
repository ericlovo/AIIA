import unittest

import httpx

from scripts import education_southeast as s

TITLE = "RFP for Statewide Student Assessments for DOE"


def listing():
    return f'<table id="MainContent_GV_oppo"><tr><td>CURRENT OPPORTUNITIES <a href="Detail.aspx?id=457">{TITLE} $30,000,000.00 Agency: Office of State Procurement Funding Source(s): Department of Education Category: Education</a></td></tr></table>'


def detail(date="Nov 10, 2026", extra="", kind="RFP"):
    values = {
        "opportunity_name": TITLE,
        "fund": "Department of Education",
        "month": date,
        "Label14": kind,
        "agency": "Office of State Procurement",
        "Label15": "Click here for details.",
        "amount": "$30,000,000.00",
    }
    return (
        '<table id="MainContent_DV1">'
        + "".join(f'<span id="{s.PREFIX + k}">{v}</span>' for k, v in values.items())
        + extra
        + "</table>"
    )


class SoutheastTests(unittest.TestCase):
    def test_bound_record_no_amount(self):
        row = s.detail(detail(), "457", TITLE, "2026-10-31")
        self.assertEqual(row["deadline_date"], "2026-11-10")
        self.assertEqual(row["source_record_id"], "457")
        self.assertIsNone(row["notice_number"])
        self.assertNotIn("$", str(row))
        self.assertIn("subset", row["coverage_scope"])

    def test_stale_cancelled_and_wrong_type(self):
        for body in [
            detail("Oct 31, 2026"),
            detail(extra="<p>Cancelled</p>"),
            detail(kind="Grant"),
        ]:
            self.assertIsNone(s.detail(body, "457", TITLE, "2026-10-31"))

    def test_correlation_and_dates_fail(self):
        for body in [
            detail("TBD"),
            detail().replace(TITLE, "Other title"),
            detail().replace("Department of Education", "Other Agency"),
        ]:
            with self.assertRaises(ValueError):
                s.detail(body, "457", TITLE, "2026-10-31")

    def test_listing_exact_current_not_archived(self):
        self.assertEqual(s.listing(listing()), {"457": TITLE})
        for body in [
            listing().replace("CURRENT OPPORTUNITIES", "Archive"),
            listing().replace("Detail.aspx?id=457", "https://evil.example/Detail.aspx?id=457"),
            listing().replace("</table>", '<a href="javascript:Page$2">2</a></table>'),
        ]:
            with self.assertRaises(ValueError):
                s.listing(body)

    def test_fetch_two_bounded_pages(self):
        calls = []

        def handler(req):
            calls.append(req)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=listing() if len(calls) == 1 else detail(),
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(
                len(s.fetch(client, {"state": "LA", "url": s.SOURCE}, "2026-10-31")), 1
            )
        self.assertEqual(len(calls), 2)

    def test_redirect_or_wrong_source_fails(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(302, headers={"location": "https://evil.example"})
                )
            ) as client,
            self.assertRaises(ValueError),
        ):
            s.fetch(client, {"state": "LA", "url": s.SOURCE}, "2026-10-31")
        with self.assertRaises(ValueError):
            s.fetch(None, {"state": "GA", "url": s.SOURCE}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()
