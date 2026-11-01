import unittest

import httpx

from scripts import education_northeast as n


def listing(identity="123", page=None):
    pager = (
        f"<a href=\"javascript:__doPostBack('gvResults','Page${page}')\">{page}</a>" if page else ""
    )
    return f"""<h1>Search Results - Open State Bids</h1><input type="hidden" name="__VIEWSTATE" value="state"><input type="hidden" name="__EVENTVALIDATION" value="validation"><table id="gvResults"><tr><td><a href="javascript:openPrintView('BidPreview.aspx?BidID={identity}', 'Window{identity}');">School support</a> Agency of Education Close Date: 11/5/2026</td></tr><tr><td>{pager}</td></tr></table>"""


def detail(extra="", closing="11/5/2026 4:30 PM"):
    fields = {
        "lblBidTitle": "School support",
        "lblAuthorName": "Agency of Education",
        "lblBidCloseDate": closing,
        "lblBidDescription": "Request for Proposal (RFP) for school attendance support.",
        "lblSpecialInstructions": "",
        "lblBidType": "Job Contract",
        "lblContactName": "Example Person",
        "lblRFQNumber": "",
    }
    return "".join(f'<span id="{k}">{v}</span>' for k, v in fields.items()) + extra


class NortheastTests(unittest.TestCase):
    def test_notice_bound_and_contacts_excluded(self):
        row = n.detail(
            detail(), "123", {"title": "School support", "closing": "11/5/2026"}, "2026-10-31"
        )
        self.assertEqual(row["source_record_id"], "123")
        self.assertEqual(row["deadline_date"], "2026-11-05")
        self.assertIsNone(row["notice_number"])
        self.assertNotIn("Example Person", str(row))

    def test_cancelled_after_contacts_withheld(self):
        self.assertIsNone(
            n.detail(
                detail("<p>Cancelled</p>"),
                "123",
                {"title": "School support", "closing": "11/5/2026"},
                "2026-10-31",
            )
        )

    def test_deadline_title_agency_binding(self):
        for body in [
            detail(closing="11/6/2026"),
            detail().replace("Agency of Education", "University"),
            detail().replace("School support", "Other title"),
            detail(closing="TBD"),
        ]:
            with self.assertRaises(ValueError):
                n.detail(
                    body, "123", {"title": "School support", "closing": "11/5/2026"}, "2026-10-31"
                )

    def test_same_day_excluded(self):
        self.assertIsNone(
            n.detail(
                detail(), "123", {"title": "School support", "closing": "11/5/2026"}, "2026-11-05"
            )
        )

    def test_structure_and_urls_fail(self):
        for body in [
            "<p>Welcome</p>",
            listing().replace("BidPreview.aspx?BidID=123", "https://evil.example?BidID=123"),
            listing() + "Access denied",
        ]:
            with self.assertRaises(ValueError):
                n.listing(body)

    def test_complete_pagination(self):
        calls = []

        def handler(req):
            calls.append(req)
            body = (
                listing("123", 2)
                if len(calls) == 1
                else listing("124", 1)
                if len(calls) == 2
                else detail()
            )
            return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            rows = n.fetch(client, {"state": "VT", "url": n.SOURCE}, "2026-10-31")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r.method for r in calls], ["GET", "POST", "GET", "GET"])

    def test_repeated_page_fails(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(
                        200, headers={"content-type": "text/html"}, text=listing("123", 2)
                    )
                )
            ) as client,
            self.assertRaisesRegex(ValueError, "repeated_page"),
        ):
            n.fetch(client, {"state": "VT", "url": n.SOURCE}, "2026-10-31")

    def test_no_redirect_or_unrecognized_source(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(302, headers={"location": "https://evil.example"})
                )
            ) as client,
            self.assertRaises(ValueError),
        ):
            n.fetch(client, {"state": "VT", "url": n.SOURCE}, "2026-10-31")
        with self.assertRaises(ValueError):
            n.fetch(None, {"state": "ME", "url": n.SOURCE}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()
