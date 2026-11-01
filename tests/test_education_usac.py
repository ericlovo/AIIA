import unittest
from unittest.mock import patch

import httpx

from scripts import education_usac as usac


def row(**changes):
    return {
        "application_number": "270000001",
        "service_request_id": "1",
        "applicant_type": "School District",
        "fcc_form_470_status": "Certified",
        "billed_entity_state": "WI",
        "rfp_documents": {"url": "https://publicdata.usac.org/EPC/Prd/Form470/270000001/1/rfp.pdf"},
        "form_pdf": {"url": "https://publicdata.usac.org/EPC/Prd/Form470/270000001/1/form.pdf"},
        "certified_date_time": "2026-10-20T00:00:00.000",
        "allowable_contract_date": "2026-11-17T00:00:00.000",
        **changes,
    }


class UsacTests(unittest.TestCase):
    def collect(self, rows):
        def handle(request):
            self.assertNotIn("contact_", request.url.params["$select"])
            self.assertNotIn("email", request.url.params["$select"])
            self.assertIn("billed_entity_state='WI'", request.url.params["$where"])
            return httpx.Response(200, json=rows)

        with httpx.Client(transport=httpx.MockTransport(handle)) as c:
            return usac.collect(c, "WI", "2026-10-31")

    def test_pending_not_open_or_deadline(self):
        item = self.collect([row()])[0]
        self.assertFalse(item["alert_eligible"])
        self.assertIsNone(item["currently_open"])
        self.assertIsNone(item["deadline_date"])
        self.assertIn("not bid deadline", item["quote"])
        self.assertFalse(item["attachment_verified"])

    def test_scope_and_untrusted_attachment_rejected(self):
        for changes in [
            {"applicant_type": "Library"},
            {"applicant_type": "Consortium"},
            {"billed_entity_state": "NY"},
            {"fcc_form_470_status": "Canceled"},
            {"certified_date_time": "2020-01-01"},
            {"rfp_documents": {"url": "https://evil.example/rfp.pdf"}},
            {"contact_name": "must not retain"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.collect([row(**changes)])

    def test_empty_and_duplicate(self):
        self.assertEqual(self.collect([]), [])
        self.assertEqual(len(self.collect([row(), row()])), 1)

    def test_state_injection_rejected(self):
        with self.assertRaises(ValueError):
            usac.collect(None, "WI' OR 1=1", "2026-10-31")

    def test_overflow_never_partial_success(self):
        with (
            patch.object(usac, "PAGE_SIZE", 1),
            patch.object(usac, "MAX_PAGES", 2),
            self.assertRaisesRegex(ValueError, "incomplete"),
        ):
            self.collect([row()])

    def test_pagination(self):
        offsets = []

        def handle(request):
            offset = int(request.url.params["$offset"])
            offsets.append(offset)
            return httpx.Response(200, json=[row()] if offset == 0 else [])

        with (
            patch.object(usac, "PAGE_SIZE", 1),
            httpx.Client(transport=httpx.MockTransport(handle)) as c,
        ):
            self.assertEqual(len(usac.collect(c, "WI", "2026-10-31")), 1)
        self.assertEqual(offsets, [0, 1])

    def test_http_redirect_and_oversize_fail(self):
        for response in [
            httpx.Response(302, headers={"location": "https://example.com"}),
            httpx.Response(200, text="no JSON"),
            httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b" " * (usac.MAX_BYTES + 1),
            ),
        ]:
            with (
                httpx.Client(
                    transport=httpx.MockTransport(lambda r, response=response: response)
                ) as c,
                self.assertRaises(ValueError),
            ):
                usac.collect(c, "WI", "2026-10-31")
