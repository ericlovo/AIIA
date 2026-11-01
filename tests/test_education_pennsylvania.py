import unittest

import httpx

from scripts import education_pennsylvania as p


def form(kind="RFP", results=None, agency="12"):
    fields = '<input type="hidden" name="__VIEWSTATE" value="public-state"><input type="hidden" name="__EVENTVALIDATION" value="public-validation">'
    for name, value, label in [
        ("ddlAgency", agency, "Department of Education"),
        ("ddlTypes", kind, kind),
        ("ddlRows", "32767", "ALL"),
    ]:
        fields += f'<select name="{p.PREFIX + name}"><option selected value="{value}">{label}</option></select>'
    fields += f'<input type="radio" checked name="{p.PREFIX}rdoArch" value="0">'
    results = (
        results
        if results is not None
        else "<tr><td>No records were found matching the selected criteria</td></tr>"
    )
    return f'<form id="aspnetForm" action="./Search.aspx">{fields}<table id="ctl00_MainBody_grdResults"><caption>List of Current Solicitations</caption>{results}</table></form>'


def row(kind="RFP", deadline="11/5/2026", title="K-12 School assessment services", status="Open"):
    headers = [
        "Solicitation #",
        "Types",
        "Solicitation Title",
        "Description",
        "Agency",
        "Solicitation Start Date",
        "Solicitation Due Date",
        "Status",
        "Contact Person",
    ]
    values = [
        '<a href="Solicitations.aspx?SID=ED-123">ED-123</a>',
        kind,
        title,
        "Instructional assessment services",
        "Department of Education",
        "10/1/2026",
        deadline + " 4:00:00 PM",
        status,
        "Example Person",
    ]
    return (
        "<tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr><tr>"
        + "".join(f"<td>{v}</td>" for v in values)
        + "</tr>"
    )


class PennsylvaniaTests(unittest.TestCase):
    def test_bound_record(self):
        records = p.parse_results(form(results=row()), "RFP", "2026-10-31")
        self.assertEqual(records[0]["notice_number"], "ED-123")
        self.assertEqual(records[0]["deadline_date"], "2026-11-05")
        self.assertNotIn("Example Person", str(records))
        self.assertNotIn("__VIEWSTATE", str(records))

    def test_actual_empty(self):
        self.assertEqual(p.parse_results(form(), "RFP", "2026-10-31"), [])

    def test_filters_must_apply(self):
        for body in [
            form(kind="IFB"),
            form(agency="0"),
            form().replace('value="32767"', 'value="10"'),
            form(results=row(kind="IFB")),
        ]:
            with self.assertRaises(ValueError):
                p.parse_results(body, "RFP", "2026-10-31")

    def test_expired_unknown_higher_ed_grants_closed(self):
        for kwargs in [
            {"deadline": "10/31/2026"},
            {"title": "University business school evaluation"},
            {"title": "K-12 grant applications"},
            {"status": "Closed"},
        ]:
            self.assertEqual(p.parse_results(form(results=row(**kwargs)), "RFP", "2026-10-31"), [])
        with self.assertRaises(ValueError):
            p.parse_results(form(results=row(deadline="TBD")), "RFP", "2026-10-31")

    def test_pagination_and_challenge_rejected(self):
        for body in [
            form(results=row() + "<a href=\"javascript:__doPostBack('grid','Page$2')\">2</a>"),
            form() + "Access denied",
            form(results=row()).replace(
                "Solicitations.aspx?SID=ED-123", "https://evil.example/?SID=ED-123"
            ),
        ]:
            with self.assertRaises(ValueError):
                p.parse_results(body, "RFP", "2026-10-31")

    def test_bounded_public_search_transport(self):
        calls = []

        def handler(req):
            calls.append(req)
            kind = "RFI" if len(calls) > 2 else "RFP"
            return httpx.Response(200, headers={"content-type": "text/html"}, text=form(kind))

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(p.fetch(client, {"state": "PA", "url": p.SOURCE}, "2026-10-31"), [])
        self.assertEqual([r.method for r in calls], ["GET", "POST", "GET", "POST"])
        self.assertIn(b"ddlAgency=12", calls[1].content)
        self.assertIn(b"ddlRows=32767", calls[1].content)

    def test_redirect_and_wrong_source(self):
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda req: httpx.Response(302, headers={"location": "https://evil.example"})
                )
            ) as client,
            self.assertRaises(ValueError),
        ):
            p.fetch(client, {"state": "PA", "url": p.SOURCE}, "2026-10-31")
        with self.assertRaises(ValueError):
            p.fetch(None, {"state": "PA", "url": "https://evil.example"}, "2026-10-31")


if __name__ == "__main__":
    unittest.main()
