import unittest

import httpx

from scripts import education_newyork as ny


def notice(number="R1234", deadline="December 9, 2026", extra=""):
    return f"<h2>{number} - School tutoring</h2><ul><li>Proposal Due Date and Time: Wednesday, {deadline}, 4:00 P.M.</li></ul><p>Request for Proposals for public school tutoring.</p>{extra}"


class NewYorkTests(unittest.TestCase):
    def test_notice_boundaries_and_stable_id(self):
        first = ny.parse(notice() + notice("R1235"), "2026-10-31")
        self.assertEqual(len(first), 2)
        self.assertNotEqual(first[0]["notice_id"], first[1]["notice_id"])
        changed = ny.parse(notice(deadline="December 10, 2026"), "2026-10-31")[0]
        self.assertEqual(first[0]["notice_id"], changed["notice_id"])
        self.assertNotEqual(first[0]["hash"], changed["hash"])
        self.assertEqual(first[0]["deadline_date"], "2026-12-09")

    def test_closed_expired_same_day_indefinite_withheld(self):
        for body in [
            notice(extra="<p>Cancelled</p>"),
            notice(deadline="October 30, 2026"),
            notice(deadline="October 31, 2026"),
            "<h2>R1234 - Tutors</h2><p>RFP Open Indefinitely.</p>",
        ]:
            with self.subTest(body=body):
                self.assertEqual(ny.parse(body, "2026-10-31"), [])

    def test_other_deadlines_not_proposal_deadlines(self):
        body = notice().replace("Proposal Due Date and Time", "Q&A Due Date")
        self.assertEqual(ny.parse(body, "2026-10-31"), [])

    def test_contradictory_dates_withheld(self):
        body = notice(extra="<p>Proposal Due Date: December 10, 2026</p>")
        self.assertEqual(ny.parse(body, "2026-10-31"), [])

    def test_missing_structure_is_failure(self):
        with self.assertRaisesRegex(ValueError, "structure_missing"):
            ny.parse("<html>Welcome to our website</html>", "2026-10-31")

    def test_duplicate_notice_is_failure(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ny.parse(notice() + notice(), "2026-10-31")

    def test_fetch_checks(self):
        for status, headers, body in [
            (302, {"location": "https://example.org"}, ""),
            (200, {"content-type": "text/html"}, "Access denied"),
            (200, {"content-type": "text/html"}, "x" * (ny.MAX_BYTES + 1)),
            (200, {"content-type": "application/json"}, "{}"),
        ]:
            with (
                self.subTest(status=status, size=len(body)),
                httpx.Client(
                    transport=httpx.MockTransport(
                        lambda request, status=status, headers=headers, body=body: httpx.Response(
                            status, headers=headers, text=body
                        )
                    )
                ) as client,
                self.assertRaises(ValueError),
            ):
                ny.fetch(client, {"state": "NY", "url": ny.SOURCE}, "2026-10-31")

    def test_fetch_success(self):
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/html"}, text=notice()
                )
            )
        ) as client:
            self.assertEqual(
                len(ny.fetch(client, {"state": "New York", "url": ny.SOURCE}, "2026-10-31")), 1
            )

    def test_wrong_source_rejected_before_request(self):
        with self.assertRaisesRegex(ValueError, "invalid_newyork_source"):
            ny.fetch(None, {"state": "NY", "url": "https://example.org"}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()
