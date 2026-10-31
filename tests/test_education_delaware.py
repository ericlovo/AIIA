import importlib.util
import json
import unittest
from pathlib import Path

import httpx

spec = importlib.util.spec_from_file_location(
    "de", Path(__file__).parents[1] / "scripts/education_delaware.py"
)
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)


class DelawareTests(unittest.TestCase):
    def run_feed(self, rows, **overrides):
        payload = {"total": 1, "page": 1, "records": len(rows), "rows": rows, **overrides}

        def handler(request):
            self.assertEqual(request.url.host, "mmp.delaware.gov")
            if request.method == "POST":
                self.assertEqual(json.loads(request.content)["rows"], 100)
                return httpx.Response(200, json=payload)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<table><tr><td>SYN</td><td>Synthetic School District</td></tr><tr><td>UNI</td><td>Synthetic University</td></tr></table>",
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return de.fetch(client, {"state": "DE", "url": de.SOURCE}, "2026-10-31")

    def row(self, **changes):
        return {
            "Id": 123,
            "Title": "School tutoring services",
            "AgencyCode": "SYN",
            "Type": "RFP",
            "DeadlineDate": "2026-11-05",
            "OpenDate": "2026-10-15",
            **changes,
        }

    def test_exact_notice_provenance(self):
        item = self.run_feed([self.row()])[0]
        self.assertEqual(item["url"], de.BASE + "/Bids/Details/123")
        self.assertIn("Synthetic School District", item["quote"])
        changed = self.run_feed([self.row(DeadlineDate="2026-11-06")])[0]
        self.assertEqual(item["identity"], changed["identity"])
        self.assertNotEqual(item["hash"], changed["hash"])

    def test_closed_unknown_same_day_and_non_k12_withheld(self):
        for changes in [
            {"DeadlineDate": "2026-10-30"},
            {"DeadlineDate": "2026-10-31"},
            {"DeadlineDate": None},
            {"Type": None},
            {"Type": "RFQ"},
            {"AgencyCode": "UNI"},
            {"AgencyCode": "UNKNOWN"},
            {"OpenDate": "2026-11-01"},
        ]:
            with self.subTest(changes=changes):
                self.assertEqual(self.run_feed([self.row(**changes)]), [])

    def test_explicit_title_type_supported(self):
        self.assertEqual(
            len(self.run_feed([self.row(Type=None, Title="RFI school tutoring services")])), 1
        )

    def test_truncated_feed_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_feed([self.row()], records=101, total=2)

    def test_invalid_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid_row"):
            self.run_feed([self.row(Id=True)])

    def test_redirect_rejected(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(302, headers={"location": "https://example.com"})
                )
            ) as client,
            self.assertRaisesRegex(ValueError, "unavailable"),
        ):
            de.fetch(client, {"state": "DE", "url": de.SOURCE}, "2026-10-31")

    def test_response_size_bounded(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        200,
                        headers={"content-type": "application/json"},
                        content=b" " * (de.MAX_BYTES + 1),
                    )
                )
            ) as client,
            self.assertRaisesRegex(ValueError, "too_large"),
        ):
            de.fetch(client, {"state": "DE", "url": de.SOURCE}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()
