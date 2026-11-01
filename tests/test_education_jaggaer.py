import unittest

import httpx

from scripts import education_jaggaer as j


def page(
    desc="Public school instructional assessment", closing="11/5/2026", number="PED-RFP-1", total=1
):
    return f"""<h1>StateOfMontana Public Bid Opportunities</h1><div>1-1 of {total} Results</div><table><tr><td>Open</td><td><a href="/apps/Router/ViewSourcingEvent?AuthToken=do-not-retain">{desc}</a> Open 10/1/2026, 2:00 PM MDT Close {closing}, 2:00 PM MST Type RFP Number {number} Contact Example Person example@example.org</td></tr></table>"""


class JaggaerTests(unittest.TestCase):
    def test_bound_record_no_ephemeral_token_or_contact(self):
        rows, meta = j.parse_page(page(), "MT", "2026-10-31")
        self.assertEqual(meta["records_checked"], 1)
        self.assertEqual(rows[0]["deadline_date"], "2026-11-05")
        self.assertEqual(rows[0]["notice_number"], "PED-RFP-1")
        self.assertNotIn("AuthToken", str(rows))
        self.assertNotIn("Example Person", str(rows))

    def test_expired_same_day_higher_ed_non_education_excluded(self):
        for body in [
            page(closing="10/31/2026"),
            page(desc="University business school study"),
            page(desc="Highway repair"),
        ]:
            self.assertEqual(j.parse_page(body, "MT", "2026-10-31")[0], [])

    def test_description_cancellation_and_withdrawal_withheld(self):
        for status in ("cancelled", "withdrawn"):
            with self.subTest(status=status):
                rows, meta = j.parse_page(
                    page(desc=f"Public school assessment. This solicitation is {status}."),
                    "MT", "2026-10-31",
                )
                self.assertEqual(rows, [])
                self.assertEqual(meta["records_checked"], 1)

    def test_workspace_counts_challenge_fail(self):
        for body in [
            page().replace("StateOfMontana", "StateOfNewMexico"),
            page().replace("1-1 of 1", "1-2 of 2"),
            page() + "Access denied",
            page().replace("11/5/2026", "13/5/2026"),
        ]:
            with self.assertRaises(ValueError):
                j.parse_page(body, "MT", "2026-10-31")

    def test_notice_ids_not_shared_portal_url(self):
        first = j.parse_page(page(), "MT", "2026-10-31")[0][0]
        second = j.parse_page(page(number="PED-RFP-2"), "MT", "2026-10-31")[0][0]
        self.assertNotEqual(first["notice_id"], second["notice_id"])

    def test_fetch_bounded_get(self):
        calls = []

        def handler(req):
            calls.append(req)
            return httpx.Response(200, headers={"content-type": "text/html"}, text=page())

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            rows = j.fetch(client, {"state": "MT", "url": j.SOURCES["MT"]}, "2026-10-31")
        self.assertEqual(len(rows), 1)
        self.assertEqual(calls[0].url.params["PageSize"], "200")
        self.assertEqual(calls[0].method, "GET")

    def test_short_page_not_complete_coverage(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(
                        200, headers={"content-type": "text/html"}, text=page(total=300)
                    )
                )
            ) as client,
            self.assertRaisesRegex(ValueError, "short_page"),
        ):
            j.fetch(client, {"state": "MT", "url": j.SOURCES["MT"]}, "2026-10-31")

    def test_redirect_wrong_source_rejected(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(302, headers={"location": "https://evil.example"})
                )
            ) as client,
            self.assertRaises(ValueError),
        ):
            j.fetch(client, {"state": "MT", "url": j.SOURCES["MT"]}, "2026-10-31")
        with self.assertRaises(ValueError):
            j.fetch(None, {"state": "MT", "url": "https://evil.example"}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()
