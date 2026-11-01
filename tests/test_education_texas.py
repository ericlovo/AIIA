"""Public ESBD contract tests: completeness, expiry and authoritative qualification."""

import json

import httpx
import pytest

from scripts.education_texas import SOURCE, fetch


def row(n, **changes):
    item = {
        "internalid": str(n),
        "title": "Consulting RFP",
        "solicitationId": f"RFP-{n}",
        "agencyName": "Example ISD - S1000",
        "agencyNumber": "S1000",
        "responseDue": "11/5/2026",
        "responseTime": "10:00 AM",
        "statusName": "Posted",
        "postingDate": "10/2/2026",
        "cancelledDate": "",
    }
    return {**item, **changes}


def detail(item):
    return (
        f"Solicitation ID: {item['solicitationId']} Status: {item['statusName']} "
        f"Response Due Date: {item['responseDue']} Response Due Time: {item['responseTime']} "
        f"Agency/Texas SmartBuy Member Number: {item['agencyNumber']} "
        "Solicitation Description: Request for Proposals for student services. Attachments"
    )


def run(handler):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return fetch(client, {"state": "TX", "url": SOURCE}, "2026-10-31")


def test_multiple_pages_detail_qualification_and_buyer_filter():
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, text=detail(row(2, title="Consulting")))
        p = json.loads(request.content)["page"]
        r = row(1, agencyName="Example University - 700") if p == 1 else row(2, title="Consulting")
        return httpx.Response(
            200, json={"page": p, "recordsPerPage": 1, "totalRecordsFound": 2, "lines": [r]}
        )

    found = run(handler)
    assert len(found) == 1
    assert found[0]["notice_number"] == "RFP-2"
    assert found[0]["deadline_date"] == "2026-11-05"
    assert len(requests) == 3
    assert all(json.loads(r.content)["status"] == "1" for r in requests if r.method == "POST")


@pytest.mark.parametrize(
    "fault", ["duplicate", "total_changed", "overflow", "wrong_page", "truncated"]
)
def test_incomplete_collection_fails(fault):
    def handler(request):
        p = json.loads(request.content)["page"]
        return httpx.Response(
            200,
            json={
                "page": 0 if fault == "wrong_page" else p,
                "recordsPerPage": 1,
                "totalRecordsFound": 601
                if fault == "overflow"
                else 3
                if fault == "total_changed" and p == 2
                else 2,
                "lines": [] if fault == "truncated" else [row(1 if fault == "duplicate" else p)],
            },
        )

    with pytest.raises(ValueError, match="incomplete"):
        run(handler)


@pytest.mark.parametrize(
    "changes",
    [
        {"responseDue": "10/31/2026"},
        {"statusName": "Awarded"},
        {"cancelledDate": "10/31/2026"},
        {"agencyName": "University School of Law - 777"},
        {"postingDate": "11/1/2026"},
    ],
)
def test_nonqualifying_rows_do_not_fetch_detail(changes):
    def handler(request):
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "page": 1,
                "recordsPerPage": 24,
                "totalRecordsFound": 1,
                "lines": [row(1, **changes)],
            },
        )

    assert run(handler) == []


def test_detail_identity_mismatch_fails():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text=detail(row(2)))
        return httpx.Response(
            200, json={"page": 1, "recordsPerPage": 24, "totalRecordsFound": 1, "lines": [row(1)]}
        )

    with pytest.raises(ValueError, match="detail_mismatch"):
        run(handler)


def test_verified_empty_result():
    assert (
        run(
            lambda _: httpx.Response(
                200, json={"page": 1, "recordsPerPage": 24, "totalRecordsFound": 0, "lines": []}
            )
        )
        == []
    )


@pytest.mark.parametrize("status", ["cancelled", "withdrawn"])
def test_description_cancellation_and_withdrawal_withheld(status):
    def handler(request):
        if request.method == "GET":
            body = detail(row(1)).replace(
                "student services.", f"student services. This solicitation is {status}."
            )
            return httpx.Response(200, text=body)
        return httpx.Response(200, json={
            "page": 1, "recordsPerPage": 25, "totalRecordsFound": 1, "lines": [row(1)],
        })
    assert run(handler) == []
