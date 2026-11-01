import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from scripts import education_additional as a


def ms(value):
    return (
        "/Date("
        + str(
            int(
                datetime.fromisoformat(value)
                .replace(tzinfo=ZoneInfo("America/Chicago"))
                .timestamp()
                * 1000
            )
        )
        + ")/"
    )


def iowa():
    return {
        "iTotalRecords": 1,
        "aaData": [
            {
                "ID": "00000000-0000-0000-0000-000000000001",
                "BidNumber": "ED-RFP-123",
                "AgencyID": a.IOWA_AGENCY,
                "AgencyName": "Education, Dept Of",
                "Solicitation": "School assessment RFP",
                "Description": "K-12 school assessment services",
                "EffectiveDate": ms("2026-10-01"),
                "ExpirationDate": ms("2026-11-05"),
                "IsActive": True,
                "Status": "Open",
                "AgencyContactName": "Example Person",
            }
        ],
    }


def indiana(title="School assessment RFP", deadline="11/5/2026", extra=""):
    headers = [
        "Event Name",
        "Agency",
        "Event ID",
        "Event Description",
        "Response Due By",
        "Contact",
    ]
    values = [
        title,
        "Education",
        "001230000000123",
        "K-12 school assessment services " + extra,
        deadline + " 3:00:00PM EST",
        "Example Person",
    ]
    return (
        '<table id="events-table"><tr>'
        + "".join("<th>" + h + "</th>" for h in headers)
        + "</tr><tr>"
        + "".join("<td>" + v + "</td>" for v in values)
        + "</tr></table>"
    )


class AdditionalTests(unittest.TestCase):
    def test_iowa_complete_bound_record(self):
        row = a.parse_iowa(iowa(), "2026-10-31")[0]
        self.assertEqual(row["deadline_date"], "2026-11-05")
        self.assertNotIn("Example Person", str(row))
        self.assertEqual(a.parse_iowa({"iTotalRecords": 0, "aaData": []}, "2026-10-31"), [])

    def test_iowa_incomplete_wrong_agency_and_invalid_dates(self):
        for change in ["count", "agency", "date"]:
            data = iowa()
            if change == "count":
                data["iTotalRecords"] = 2
            if change == "agency":
                data["aaData"][0]["AgencyID"] = "other"
            if change == "date":
                data["aaData"][0]["ExpirationDate"] = "TBD"
            with self.assertRaises(ValueError):
                a.parse_iowa(data, "2026-10-31")

    def test_iowa_closed_expired_and_grants(self):
        for key, value in [
            ("Status", "Closed"),
            ("ExpirationDate", ms("2026-10-31")),
            ("Description", "K-12 grant applications"),
            ("Description", "University school"),
        ]:
            data = iowa()
            data["aaData"][0][key] = value
            self.assertEqual(a.parse_iowa(data, "2026-10-31"), [])

    def test_indiana_bound_record(self):
        row = a.parse_indiana(indiana(), "2026-10-31")[0]
        self.assertEqual(row["notice_number"], "001230000000123")
        self.assertNotIn("Example Person", str(row))
        self.assertEqual(row["deadline_date"], "2026-11-05")

    def test_indiana_expired_cancelled_non_rfp(self):
        for body in [
            indiana(deadline="10/31/2026"),
            indiana(extra="Cancelled"),
            indiana(title="School assessment IFB"),
        ]:
            self.assertEqual(a.parse_indiana(body, "2026-10-31"), [])

    def test_indiana_structure_failures(self):
        for body in [
            "Welcome",
            indiana(deadline="TBD"),
            indiana() + "Access denied",
            indiana().replace("</table>", '<a href="Page$2">Next</a></table>'),
        ]:
            with self.assertRaises(ValueError):
                a.parse_indiana(body, "2026-10-31")

    def test_transport_exact_api_and_no_redirect(self):
        calls = []

        def handler(req):
            calls.append(req)
            return httpx.Response(200, json={"iTotalRecords": 0, "aaData": []})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(
                a.fetch(client, {"state": "IA", "url": a.SOURCES["IA"]}, "2026-10-31"), []
            )
        self.assertEqual(str(calls[0].url), a.IOWA_API)
        with (
            httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(302))) as client,
            self.assertRaises(ValueError),
        ):
            a.fetch(client, {"state": "IN", "url": a.SOURCES["IN"]}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()


def test_closed_expired_indiana_notice_is_withheld():
    for status in ("Closed", "Expired"):
        assert (
            a.parse_indiana(indiana(title=status + " RFP for K12 school services"), "2026-11-01")
            == []
        )


def test_iowa_nonstring_agency_rejected():
    data = iowa()
    data["aaData"][0]["AgencyID"] = None
    try:
        a.parse_iowa(data, "2026-11-01")
    except ValueError as exc:
        assert str(exc) == "iowa_agency_mismatch"
    else:
        raise AssertionError("Malformed agency must fail explicitly")
