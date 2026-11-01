import json

import pytest

from scripts.education_midwest import parse_nebraska

HEADER = [
    "Posted",
    "Description",
    "Category - NIGP Code",
    "Opening",
    "Type",
    "PCO/Buyer",
    "Solicitation Number",
    "Agency",
    "Updated",
]


def row(
    number="RFP1",
    agency="Nebraska Department of Education",
    opening="11/13/26",
    kind="Request for Proposal",
    url="/materiel/purchasing/RFP1/RFP1.html",
):
    cells = [
        "10/22/26",
        f'<a href="{url}">Supported Employment Services</a>',
        "",
        opening,
        kind,
        "",
        number,
        agency,
        "10/22/26",
    ]
    return "<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"


def table(rows):
    return (
        '<section id="current-bid-opportunities"><table><tr>'
        + "".join(f"<th>{x}</th>" for x in HEADER)
        + "</tr>"
        + rows
        + "</table></section>"
    )


def test_current_education_row_is_pending_not_alertable():
    result = parse_nebraska(table(row()), "2026-10-31")
    assert len(result) == 1
    assert result[0]["alert_eligible"] is False
    assert "deadline_date" not in result[0]
    assert json.loads(result[0]["quote"])["bid_opening_date"] == "2026-11-13"


def test_archive_and_award_rows_not_collected():
    body = (
        table(row(agency="Department of Transportation"))
        + '<section id="awarded"><table>'
        + row()
        + "</table></section>"
    )
    assert parse_nebraska(body, "2026-10-31") == []


@pytest.mark.parametrize(
    "change",
    [{"opening": "10/31/26"}, {"kind": "Invitation to Bid"}, {"agency": "State Purchasing Bureau"}],
)
def test_nonqualifying_rows_withheld(change):
    assert parse_nebraska(table(row(**change)), "2026-10-31") == []


@pytest.mark.parametrize(
    "body",
    [
        "Directory",
        table(row() + row()),
        table(row(url="http://das.nebraska.gov/materiel/purchasing/a.html")),
        table(row(url="https://attacker.example/materiel/purchasing/a.html")),
        table(row()).replace("Opening", "Deadline"),
    ],
)
def test_invalid_or_ambiguous_source_fails(body):
    with pytest.raises(ValueError):
        parse_nebraska(body, "2026-10-31")
