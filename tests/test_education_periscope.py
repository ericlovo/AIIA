"""Contract checks for public pagination, provenance and fail-closed qualification."""

import json

import httpx
import pytest

from scripts.education_periscope import BASES, PATH, TABLE, fetch


def row(
    n,
    title="Education consulting RFP",
    agency="Example School District",
    opening="11/19/2026 2:00 PM",
):
    cells = [str(n), str(n), agency, "", "", "", title, opening, "", "Sent", ""]
    cells[0] = (
        f'<a href="/bso/external/bidDetail.sda?docId={n}&amp;external=true&amp;parentUrl=close">{n}</a>'
    )
    return "<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"


def listing(rows, total):
    return (
        f'<form id="bidSearchResultsForm"><input type="hidden" name="javax.faces.ViewState" value="initial">'
        f'<input type="hidden" name="_csrf" value="public-session-token"></form>1-{min(25, total)} of {total}'
        + "".join(rows)
    )


def detail(n, bulletin="Request for Proposals for professional services."):
    return f"Bid Number: {n} Bulletin Desc: {bulletin} Ship-to Address: Example School District"


def test_paginates_and_qualifies_bulletin_without_title_type():
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            assert b"_first=25" in request.content
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f'<partial-response><changes><update id="{TABLE}"><![CDATA[{row(26, title="Consulting")}]]></update><extension type="args">{{"totalRecords":26}}</extension></changes></partial-response>',
            )
        if "bidDetail" in request.url.path:
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=detail(request.url.params["docId"])
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=listing([row(n, agency="Department of Transportation") for n in range(1, 26)], 26),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        found = fetch(client, {"state": "OR", "url": BASES["OR"] + PATH}, "2026-10-31")
    assert len(found) == 1
    assert found[0]["notice_number"] == "26"
    assert json.loads(found[0]["quote"])["bid_opening_date"] == "2026-11-19"
    assert "deadline_date" not in found[0]
    assert found[0]["alert_eligible"] is False
    assert len(requests) == 3


@pytest.mark.parametrize("fault", ["repeated", "changed_total", "oversize", "browser_check"])
def test_incomplete_or_interstitial_fails(fault):
    def handler(request):
        if fault == "browser_check":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="Please wait while checking your browser",
            )
        if request.method == "POST":
            n = 1 if fault == "repeated" else 26
            total = 27 if fault == "changed_total" else 26
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f'<partial-response><update id="{TABLE}"><![CDATA[{row(n)}]]></update><extension type="args">{{"totalRecords":{total}}}</extension></partial-response>',
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=listing([row(n) for n in range(1, 26)], 201 if fault == "oversize" else 26),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(ValueError):
        fetch(client, {"state": "OR", "url": BASES["OR"] + PATH}, "2026-10-31")


@pytest.mark.parametrize(
    "title,bulletin,opening,expected",
    [
        ("Consulting", "Formal RFP for consulting", "11/19/2026 2:00 PM", 1),
        ("Supplies IFB", "Invitation for Bid", "11/19/2026 2:00 PM", 0),
        ("Consulting RFP", "Canceled RFP", "11/19/2026 2:00 PM", 0),
        ("Consulting RFP", "RFP", "10/31/2026 2:00 PM", 0),
    ],
)
def test_type_and_expiry(title, bulletin, opening, expected):
    def handler(request):
        body = (
            detail(1, bulletin)
            if "bidDetail" in request.url.path
            else listing([row(1, title=title, opening=opening)], 1)
        )
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        found = fetch(client, {"state": "NV", "url": BASES["NV"] + PATH}, "2026-10-31")
    assert len(found) == expected


def test_untrusted_detail_host_fails():
    body = listing([row(1)], 1).replace(
        "/bso/external/bidDetail.sda", "https://attacker.example/bso/external/bidDetail.sda"
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={"content-type": "text/html"}, text=body)
            )
        ) as client,
        pytest.raises(ValueError, match="invalid_detail_link"),
    ):
        fetch(client, {"state": "NV", "url": BASES["NV"] + PATH}, "2026-10-31")


@pytest.mark.parametrize(
    "prefix",
    ["http://nevadaepro.com", "https://user@nevadaepro.com", "https://nevadaepro.com:8443"],
)
def test_insecure_detail_link_rejected(prefix):
    body = listing([row(1)], 1).replace(
        "/bso/external/bidDetail.sda", prefix + "/bso/external/bidDetail.sda"
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={"content-type": "text/html"}, text=body)
            )
        ) as client,
        pytest.raises(ValueError, match="invalid_detail_link"),
    ):
        fetch(client, {"state": "NV", "url": BASES["NV"] + PATH}, "2026-11-01")


@pytest.mark.parametrize(
    "state,total,success",
    [
        ("MA", 201, True),
        ("MA", 1000, True),
        ("MA", 1001, False),
        ("OR", 201, False),
    ],
)
def test_state_specific_complete_pagination_limits(state, total, success):
    from urllib.parse import parse_qs

    pages = []

    def handler(request):
        if request.method == "GET":
            pages.append(0)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=listing(
                    [row(n, agency="Department of Transportation") for n in range(25)], total
                ),
            )
        first = int(parse_qs(request.content.decode())[TABLE + "_first"][0])
        pages.append(first)
        fragment = "".join(
            row(n, agency="Department of Transportation")
            for n in range(first, min(first + 25, total))
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/xml"},
            text=f'<partial-response><update id="{TABLE}"><![CDATA[{fragment}]]></update><extension type="args">{{"totalRecords":{total}}}</extension></partial-response>',
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if success:
            assert fetch(client, {"state": state, "url": BASES[state] + PATH}, "2026-10-31") == []
            assert pages == list(range(0, total, 25))
        else:
            with pytest.raises(ValueError, match="incomplete"):
                fetch(client, {"state": state, "url": BASES[state] + PATH}, "2026-10-31")
            assert pages == [0]


def test_massachusetts_state_education_buyer_remains_pending():
    def handler(request):
        assert "bidDetail" not in request.url.path
        text = listing([row(1, agency="Department of Elementary and Secondary Education")], 1)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=text)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        found = fetch(client, {"state": "MA", "url": BASES["MA"] + PATH}, "2026-10-31")
    assert len(found) == 1
    assert found[0]["alert_eligible"] is False
    assert found[0]["qualification_status"] == "pending_document_review"
