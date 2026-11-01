import httpx
import pytest

from scripts import education_westvirginia as w


def page(status="Open", kind="RFP", closing="2026-11-20T19:00:00Z", total=1):
    return f'''Displaying 1 - 1 of {total}<article class="c-search-result-card">
    <a href="/about-us/purchasing/bids/2026/10/school-support">School support RFP</a>
    <div class="c-field--name-field-solicitation-status">{status}</div>
    <div class="c-field--name-field-solicitation-type">{kind}</div>
    <div class="c-field--name-field-close-datetime"><time datetime="{closing}">November 20, 2026</time></div>
    </article>'''


def test_specific_open_card_and_source():
    result = w.parse(page(), "2026-11-01")
    assert len(result) == 1 and result[0]["deadline_date"] == "2026-11-20"
    assert result[0]["url"].startswith(w.SOURCE + "/")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "Closed"},
        {"status": "Awarded"},
        {"kind": "RFQ"},
        {"closing": "2026-11-01T19:00:00Z"},
    ],
)
def test_ineligible_withheld(kwargs):
    assert w.parse(page(**kwargs), "2026-11-01") == []


@pytest.mark.parametrize(
    "body",
    [
        "A directory",
        page(total=2),
        page().replace(
            "/about-us/purchasing/bids/2026", "http://wvde.us/about-us/purchasing/bids/2026"
        ),
    ],
)
def test_missing_incomplete_or_untrusted_structure_rejected(body):
    with pytest.raises(ValueError):
        w.parse(body, "2026-11-01")


def test_no_redirects_or_untrusted_sources():
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(302, headers={"location": "https://example.org"})
            )
        ) as client,
        pytest.raises(ValueError),
    ):
        w.fetch(client, {"state": "WV", "url": w.SOURCE}, "2026-11-01")
    with pytest.raises(ValueError):
        w.fetch(None, {"state": "WV", "url": "https://example.org"}, "2026-11-01")
