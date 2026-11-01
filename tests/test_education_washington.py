import unittest

import httpx

from scripts import education_washington as wa


def page(selected=False, deadline="12/03/26", extra="", title="School meals", ref="RFP 2026-01"):
    return f"""<form><input type="hidden" name="__VIEWSTATE" value="fixture"><input type="hidden" name="__EVENTVALIDATION" value="fixture"><select name="ddlOrgNames"><option value="87" {"selected" if selected else ""}>{wa.AGENCY}</option></select><table id="DataGrid1"><tr><td>{deadline}<br>10/20/26</td><td><a href="Search_BidDetails.aspx?ID=123">{title}</a>Ref#:{ref} School food procurement</td><td>Excluded contact</td></tr>{extra}</table></form>"""


class WashingtonTests(unittest.TestCase):
    def test_live_shape_filter_and_dates(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=page(request.method == "POST")
            )

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            items = wa.fetch(client, {"state": "WA", "url": wa.SOURCE}, "2026-10-31")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["deadline_date"], "2026-12-03")
        self.assertNotIn("Excluded contact", items[0]["quote"])
        self.assertIn(b"ddlOrgNames=87", requests[1].content)
        self.assertEqual(
            items[0]["url"], "https://pr-webs-vendor.des.wa.gov/Search_BidDetails.aspx?ID=123"
        )

    def test_unapplied_filter_fails(self):
        with self.assertRaisesRegex(ValueError, "not_applied"):
            wa.parse(wa.Tree(page()).root, "2026-10-31", "87")

    def test_pagination_fails(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            wa.parse(
                wa.Tree(
                    page(
                        True,
                        extra="<tr><td><a href=\"javascript:__doPostBack('DataGrid1$_ctl1$_ctl1','')\">2</a></td></tr>",
                    )
                ).root,
                "2026-10-31",
                "87",
            )

    def test_closed_same_day_and_wrong_type_withheld(self):
        for changes in [
            {"deadline": "10/30/26"},
            {"deadline": "10/31/26"},
            {"ref": "RFQ 2026-01"},
            {"title": "Cancelled school meals"},
        ]:
            with self.subTest(changes=changes):
                self.assertEqual(
                    wa.parse(wa.Tree(page(True, **changes)).root, "2026-10-31", "87"), []
                )

    def test_missing_structure_fails(self):
        with self.assertRaises(ValueError):
            wa.parse(wa.Tree("<html>Access denied</html>").root, "2026-10-31", "87")

    def test_redirect_fails(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(302, headers={"location": "https://example.com"})
                )
            ) as c,
            self.assertRaises(ValueError),
        ):
            wa.fetch(c, {"state": "WA", "url": wa.SOURCE}, "2026-10-31")

    def test_identity_stable_on_evidence_change(self):
        a = wa.parse(wa.Tree(page(True)).root, "2026-10-31", "87")[0]
        b = wa.parse(wa.Tree(page(True, deadline="12/04/26")).root, "2026-10-31", "87")[0]
        self.assertEqual(a["notice_id"], b["notice_id"])
        self.assertNotEqual(a["hash"], b["hash"])
