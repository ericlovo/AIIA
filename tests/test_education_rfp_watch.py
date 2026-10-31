import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "watch", Path(__file__).parents[1] / "scripts/education_rfp_watch.py"
)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
CONFIG = {"sources": [{"state": "WI", "url": "https://education.example.org/notices"}]}
PAGE = '<p>Currently open K12 school bus purchase, accepting proposals.</p><a href="/rfp/123">RFP 123 school buses</a><p>Submit proposals; deadline not specified.</p>'


def client_for(monkeypatch, *, uncertain=False, grants=False, missing_scope=False):
    for key, value in {
        "TYPESAFE_API_KEY": "private-jev",
        "AIIA_SLACK_BOT_TOKEN": "private-slack",
        "AIIA_SLACK_CHANNEL_ID": "C123456789",
        "AIIA_SLACK_TEAM_ID": "T123",
    }.items():
        monkeypatch.setenv(key, value)
    calls = []

    def transport(req):
        calls.append(req)
        if req.url.host == "education.example.org":
            assert "authorization" not in req.headers
            return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)
        if req.url.host == "api.typesafe.ai":
            data = json.loads(req.content)
            assert data["model"] == "jev-1.13.0"
            assert data["state"]["notice"]["hash"]
            answers = {k: {"type": "noul", "noul": 0.95} for k in w.QUESTIONS}
            if grants:
                answers["procurement_of_goods_services"]["noul"] = 0.1
            return httpx.Response(
                200,
                json={
                    "model": w.MODEL,
                    "answers": answers,
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            )
        method = req.url.path.rsplit("/", 1)[-1]
        if method == "auth.test":
            return httpx.Response(200, json={"ok": True, "team_id": "T123"})
        if method == "conversations.info":
            if missing_scope:
                return httpx.Response(200, json={"ok": False, "error": "missing_scope"})
            return httpx.Response(
                200, json={"ok": True, "channel": {"id": "C123456789", "is_member": True}}
            )
        assert method == "chat.postMessage"
        data = json.loads(req.content)
        assert data["channel"] == "C123456789" and data["mrkdwn"] is False
        assert not data["unfurl_links"]
        if uncertain:
            raise httpx.ReadTimeout("private-token")
        return httpx.Response(200, json={"ok": True, "ts": "123.456", "channel": "C123456789"})

    return httpx.Client(
        transport=httpx.MockTransport(transport), follow_redirects=False, trust_env=False
    ), calls


def execute(tmp_path, client, send=False, config=CONFIG):
    return w.run(
        config,
        tmp_path / "state.json",
        tmp_path / "out",
        send,
        client=client,
        today="2026-10-31",
        resolve=lambda _: None,
    )


def test_dryrun_cached_then_send_once(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch)
    result = execute(tmp_path, client)
    assert result["delivery"] == "dry-run" and result["review_matches"] == 1
    assert len(calls) == 2
    assert "Deadline: unknown/unverified" in (tmp_path / "out/digest.txt").read_text()
    assert execute(tmp_path, client, True)["delivery"] == "delivered"
    assert execute(tmp_path, client, True)["delivery"] == "already_delivered"
    assert sum(r.url.host == "api.typesafe.ai" for r in calls) == 1
    assert sum(r.url.path.endswith("chat.postMessage") for r in calls) == 1


def test_uncertain_slack_persisted_never_reposted(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch, uncertain=True)
    with pytest.raises(httpx.ReadTimeout):
        execute(tmp_path, client, True)
    state = json.loads((tmp_path / "state.json").read_text())
    assert next(iter(state["messages"].values()))["status"] == "pending"
    with pytest.raises(ValueError, match="manual_recovery"):
        execute(tmp_path, client, True)
    assert sum(r.url.path.endswith("chat.postMessage") for r in calls) == 1


def test_grants_withheld_and_fixedchannel_missing_scope_allowed(tmp_path, monkeypatch):
    client, _ = client_for(monkeypatch, grants=True, missing_scope=True)
    assert execute(tmp_path, client, True)["review_matches"] == 0


def test_wrong_team_stops_before_collecting(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch)
    monkeypatch.setenv("AIIA_SLACK_TEAM_ID", "OTHER")
    with pytest.raises(ValueError, match="workspace"):
        execute(tmp_path, client, True)
    assert len(calls) == 1


def test_directory_not_notice_and_mentions_inert():
    assert (
        w.candidates(
            CONFIG["sources"][0],
            '<a href="/rfps">RFPs</a>',
            CONFIG["sources"][0]["url"],
            {"education.example.org"},
        )
        == []
    )
    text = w.message(
        "today",
        [{"state": "WI", "title": "<!channel>", "quote": "<@U123>", "url": "https://example.org"}],
        1,
        ["AK"],
        4,
    )
    assert "<!channel>" not in text and "<@U123>" not in text
    assert "1/50" in text and "Unavailable: AK" in text


def test_crosshost_redirect_blocks():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={"location": "https://elsewhere.example/x"})
        )
    )
    with pytest.raises(ValueError, match="cross_host"):
        w.fetch(
            client,
            "https://education.example.org",
            {"education.example.org", "elsewhere.example"},
            lambda _: None,
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://education.example.org",
        "https://u:p@education.example.org",
        "https://127.0.0.1",
        "https://education.example.org:99",
    ],
)
def test_urls_rejected(url):
    with pytest.raises(ValueError):
        w.allowed_url(url, {"education.example.org", "127.0.0.1"})


def test_daily_cap_persisted(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch)
    w.save(
        tmp_path / "state.json",
        {
            "version": 1,
            "cache": {},
            "deliveries": {},
            "daily": {"2026-10-31": {"calls": 20, "input_tokens": 0, "output_tokens": 0}},
        },
    )
    assert execute(tmp_path, client)["deferred"] == 1
    assert len(calls) == 1


def test_unavailable_source_not_zero_results(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    result = execute(tmp_path, client)
    assert result["coverage_states"] == 0
    assert result["unavailable_sources"] == ["Wisconsin"]
    assert "Unavailable: Wisconsin" in (tmp_path / "out/digest.txt").read_text()


def test_malformed_judgment_counted_once_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-jev")
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "education.example.org":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)
        return httpx.Response(
            200, json={"model": w.MODEL, "answers": {k: "malformed" for k in w.QUESTIONS}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = execute(tmp_path, client)
    second = execute(tmp_path, client)
    assert first["judgment_unavailable"] == second["judgment_unavailable"] == 1
    assert second["daily_usage"]["calls"] == 1
    assert sum(r.url.host == "api.typesafe.ai" for r in calls) == 1


def test_content_change_has_new_key():
    source = CONFIG["sources"][0]
    first = w.candidates(source, PAGE, source["url"], {"education.example.org"})
    changed = w.candidates(
        source, PAGE.replace("123", "124"), source["url"], {"education.example.org"}
    )
    assert first[0]["hash"] != changed[0]["hash"]


def test_daily_current_status_rejudged_but_delivered_skipped(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch)
    execute(tmp_path, client)
    result = w.run(
        CONFIG,
        tmp_path / "state.json",
        tmp_path / "out",
        client=client,
        today="2026-11-01",
        resolve=lambda _: None,
    )
    assert result["evaluated_candidate_count"] == 1
    assert sum(r.url.host == "api.typesafe.ai" for r in calls) == 2
    w.run(
        CONFIG,
        tmp_path / "state.json",
        tmp_path / "out",
        True,
        client=client,
        today="2026-11-01",
        resolve=lambda _: None,
    )
    result = w.run(
        CONFIG,
        tmp_path / "state.json",
        tmp_path / "out",
        client=client,
        today="2026-11-02",
        resolve=lambda _: None,
    )
    assert result["evaluated_candidate_count"] == 0
    assert sum(r.url.host == "api.typesafe.ai" for r in calls) == 2
    assert result["states_checked"] == ["Wisconsin"]
    assert "Alabama" in result["states_not_onboarded"]
    assert result["source_candidate_counts"][0]["candidates"] == 1
    assert (tmp_path / "out/summary.md").exists()


@pytest.mark.parametrize(
    "body",
    ["<title>Request Rejected</title>", "<h1>Access denied</h1>", "<div>captcha challenge</div>"],
)
def test_waf_200_is_unavailable(tmp_path, body):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/html"}, text=body)
        )
    )
    result = execute(tmp_path, client)
    assert result["coverage_states"] == 0
    assert result["unavailable_sources"] == ["Wisconsin"]


def test_wrong_slack_receipt_channel_stays_pending(tmp_path, monkeypatch):
    client, _ = client_for(monkeypatch)
    original = w.slack_json

    def wrong(client, method, token, **kwargs):
        result = original(client, method, token, **kwargs)
        if method == "chat.postMessage":
            result["channel"] = "C999999999"
        return result

    monkeypatch.setattr(w, "slack_json", wrong)
    with pytest.raises(ValueError, match="receipt"):
        execute(tmp_path, client, True)
    assert json.loads((tmp_path / "out/summary.json").read_text())["delivery"] == "pending"


def test_closed_heading_far_above_table_is_excluded():
    body = '<h2>Closed procurements</h2><p>Background</p><p>Dates</p><p>History</p><h3>School goods</h3><table><tr><td><a href="/old">RFP 123 school buses</a></td><td>Historic record</td></tr></table>'
    source = CONFIG["sources"][0]
    assert w.candidates(source, body, source["url"], {"education.example.org"}) == []


def test_row_heading_boundaries_and_stable_notice_identity():
    source = CONFIG["sources"][0]
    body = '<h2>Open procurements</h2><table><tr><td><a href="/rfp/123">RFP 123 school buses</a></td><td>Due November 30</td></tr><tr><td>Unrelated neighbor</td></tr></table>'
    first = w.candidates(source, body, source["url"], {"education.example.org"})[0]
    assert "Open procurements" in first["quote"] and "Due November 30" in first["quote"]
    assert "Unrelated neighbor" not in first["quote"]
    next_row = w.candidates(
        source,
        body.replace("Unrelated neighbor", "Changed neighbor"),
        source["url"],
        {"education.example.org"},
    )[0]
    assert first["hash"] == next_row["hash"]
    revised = w.candidates(
        source, body.replace("November 30", "December 30"), source["url"], {"education.example.org"}
    )[0]
    assert first["notice_id"] == revised["notice_id"]
    assert first["hash"] != revised["hash"]


def test_changed_evidence_same_delivered_url_never_resends(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch)
    execute(tmp_path, client, True)
    original = w.candidates

    def changed(*args):
        items = original(*args)
        for item in items:
            item["quote"] += " Changed neighboring information."
            item["hash"] = w.digest(item)
        return items

    monkeypatch.setattr(w, "candidates", changed)
    result = w.run(
        CONFIG,
        tmp_path / "state.json",
        tmp_path / "out",
        True,
        client=client,
        today="2026-11-01",
        resolve=lambda _: None,
    )
    assert result["review_matches"] == 0
    assert result["evaluated_candidate_count"] == 0
    assert sum(r.url.host == "api.typesafe.ai" for r in calls) == 1
    post = [json.loads(r.content) for r in calls if r.url.path.endswith("chat.postMessage")][-1]
    assert "Review notice" not in post["text"]


def test_legacy_delivered_state_requires_migration(tmp_path, monkeypatch):
    client, calls = client_for(monkeypatch)
    w.save(
        tmp_path / "state.json",
        {"version": 1, "cache": {}, "deliveries": {"old-evidence-hash": "delivered"}, "daily": {}},
    )
    with pytest.raises(ValueError, match="migration"):
        execute(tmp_path, client)
    assert not calls


def test_delaware_adapter_integrates_stable_delivery_id(tmp_path, monkeypatch):
    def adapter(client, source, today):
        assert source["state"] == "Delaware"
        return [
            {
                "state": "DE",
                "url": "https://mmp.delaware.gov/Bids/Details/42",
                "source_url": source["url"],
                "title": "RFP school buses",
                "quote": "Open K12 purchase",
                "hash": "adapter-evidence",
                "identity": "adapter-identity",
            }
        ]

    monkeypatch.setattr(w, "fetch_delaware", adapter)
    config = {
        "sources": [{"state": "DE", "url": "https://mmp.delaware.gov/Bids/", "adapter": "delaware"}]
    }
    c, calls = client_for(monkeypatch)
    with c:
        result = w.run(
            config,
            tmp_path / "state.json",
            tmp_path,
            False,
            client=c,
            today="2026-10-31",
            resolve=lambda _: None,
        )
    assert result["states_checked"] == ["Delaware"]
    assert result["evaluated_candidate_count"] == 1
    request = next(json.loads(r.content) for r in calls if r.url.host == "api.typesafe.ai")
    assert request["state"]["notice"]["notice_id"] == w.digest(
        {"state": "Delaware", "url": "https://mmp.delaware.gov/Bids/Details/42"}
    )
