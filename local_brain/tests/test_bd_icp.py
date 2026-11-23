import asyncio
import copy
import json
import logging
from functools import wraps

import pytest

from local_brain.command_center import bd_pipeline as pipeline
from local_brain.command_center.bd_cadence import next_query_for
from local_brain.command_center.bd_conductor import CONFIG_PATH, BDConductor
from local_brain.command_center.bd_icp import (
    SUPPRESSED_REASON,
    activate,
    deactivate,
    load_bot_profile,
    matches_suppression,
    normalize_org_name,
    query_variants,
    sits_ratio_label,
)
from local_brain.command_center.bd_service import BDService


def async_test(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return run


SYNTHETIC_ICP = {
    "version": 1,
    "customer": "Acme Widgets",
    "bot_ids": ["healthcare"],
    "updated": "2026-11-10",
    "source": "operator",
    "buyer_types": {
        "hot": ["health systems", "community hospitals", "county providers"],
        "downweight": ["retail clinics"],
    },
    "employer_sectors": ["healthcare providers"],
    "employer_size": "mid-market",
    "geography": {
        "priority_1": ["Wisconsin", "Example County"],
        "priority_2": ["Midwest"],
        "lower": ["West Coast"],
    },
    "products": {
        "BENEFITS": {
            "name": "Benefits administration",
            "value_share": 0.4,
            "typical_ratio_of_sits": 0.2,
        }
    },
    "channels": {
        "treat_as": "channel/partner, not end buyer",
        "known": ["brokers", "co-ops", "health plans"],
    },
    "signals": ["expansion", "hiring", "new clinic"],
    "stages_note": "Prefer expansion and hiring over brand mentions.",
}

SYNTHETIC_SUPPRESSION = {
    "version": 1,
    "customer": "Acme Widgets",
    "bot_ids": ["healthcare"],
    "reason": "existing customer pipeline",
    "updated": "2026-11-10",
    "organizations": ["Acme Widgets, Inc.", "Example County Hospital"],
    "notes": "synthetic fixture",
}

SUPPRESSED_ORG_NAMES = ("Example County Hospital", "Acme Widgets, Inc.")


def write_profile(root, bot_id="healthcare", icp=SYNTHETIC_ICP, suppression=SYNTHETIC_SUPPRESSION):
    folder = root / bot_id
    folder.mkdir(parents=True, exist_ok=True)
    if icp is not None:
        (folder / "profile.icp.json").write_text(json.dumps(icp), encoding="utf-8")
    if suppression is not None:
        (folder / "profile.suppression.json").write_text(json.dumps(suppression), encoding="utf-8")
    return folder


def job(**changes):
    return {
        "tenant_id": "synthetic-test",
        "segment": "healthcare",
        "mode": "replay",
        "bot_id": "healthcare",
        **changes,
    }


def _gap_clear_source(name, extra="opens a new outpatient clinic expansion"):
    title = f"{name} {extra}"
    source, extraction, judgments = pipeline.synthetic_fixture("healthcare")
    source = {
        **source,
        "title": title,
        "url": f"https://example.org/{name.replace(' ', '-').lower()}",
    }
    extraction = {
        **extraction,
        "organization": name,
        "quote": title,
        "signal": extra,
    }
    return source, extraction, judgments


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Acme Widgets, Inc.", "acme widgets"),
        ("ACME WIDGETS LLC", "acme widgets"),
        ("Acme Widgets Co.", "acme widgets"),
        ("Acme Widgets Ltd", "acme widgets"),
        ("Acme & Widgets", "acme and widgets"),
        ("Acme Widgets (Holdings)", "acme widgets"),
        ("Acme Widgets - Madison", "acme widgets"),
        ("The Acme Widgets", "acme widgets"),
        ("Example County Hospital, Ltd.", "example county hospital"),
        ("Example County Government", "example county"),
        ("Example County Govt", "example county"),
        ("County of Example", "example county"),
        ("City of Example", "example city"),
        ("Town of Example", "example town"),
        ("Village of Example", "example village"),
        ("Example County Department", "example county"),
        ("Example County Dept", "example county"),
        ("Example County Board of Commissioners", "example county"),
        ("Example County Authority", "example county"),
        ("Example County Office", "example county"),
        ("Northwind Foods Corp", "northwind foods"),
        ("Contoso Health, Inc.", "contoso health"),
        ("Fabrikam Paper LLC", "fabrikam paper"),
        ("Example Area School District", "example school district"),
        ("Example School District", "example school district"),
        ("County Government", "county"),
        ("Board of Commissioners", "board of commissioners"),
        ("The Department", "department"),
    ],
)
def test_org_name_normalization_cases(raw, expected):
    assert normalize_org_name(raw) == expected


@pytest.mark.parametrize(
    "candidate,suppressed,match",
    [
        ("Acme Widgets, Inc.", "ACME WIDGETS LLC", True),
        ("Acme Widgets (Holdings)", "Acme Widgets", True),
        ("Acme Widgets - Madison", "Acme Widgets, Inc.", True),
        ("Acme & Sons Co.", "Acme and Sons", True),
        ("Example County Hospital System", "Example County Hospital", True),
        ("Example County Government", "Example County", True),
        ("County of Example", "Example County", True),
        ("City of Example", "Example City", True),
        ("Town of Example", "Example Town", True),
        ("Village of Example", "Example Village", True),
        ("Example County Board of Commissioners", "Example County", True),
        ("Example County Department", "Example County", True),
        ("Example County Dept", "Example County", True),
        ("Northwind Foods Corp", "Northwind Foods", True),
        ("Contoso Health, Inc.", "Contoso Health", True),
        ("Fabrikam Paper LLC", "Fabrikam Paper", True),
        ("Example Area School District", "Example School District", True),
        ("Example County Govt", "Example County", True),
        ("Orchard Health", "Acme Widgets, Inc.", False),
        ("County Hospital", "Example County Hospital", False),
        ("Acme Widgets of Example County", "Acme Widgets", False),
    ],
)
def test_suppression_match_is_conservative(candidate, suppressed, match):
    assert matches_suppression(candidate, [suppressed]) is match


@pytest.mark.parametrize(
    "candidate,suppressed",
    [
        # Health-system name vs a similarly worded different company.
        ("Example Health Systems", "Example Paper"),
        ("Contoso Health", "Contoso Paper"),
        ("Contoso Health", "Northwind Foods"),
        # Paper/manufacturing company vs a county entity with the same place word.
        ("Fabrikam Paper", "Fabrikam County"),
        ("Example Paper", "Example County"),
        ("Northwind Foods", "Northwind County"),
        # Share only a first token; one-token core must not match a longer core.
        ("Northwind", "Northwind Foods"),
        ("Contoso", "Contoso Health"),
        ("Acme Widgets", "Acme Paper"),
        ("Fabrikam", "Fabrikam Paper"),
        # Names that reduce to only generic words must not match a real org.
        ("County Government", "Example County"),
        ("Board of Commissioners", "Example County"),
        ("The Department", "Example County Department"),
        ("Government", "Example County Government"),
        ("Authority Group", "Example County Authority"),
    ],
)
def test_suppression_match_rejects_shared_token_near_misses(candidate, suppressed):
    assert matches_suppression(candidate, [suppressed]) is False
    assert matches_suppression(suppressed, [candidate]) is False


def test_missing_dir_is_absent_and_does_not_shape_query(tmp_path):
    profile = load_bot_profile("healthcare", directory=tmp_path / "missing")
    assert profile.status == "absent"
    assert profile.loaded is False
    assert profile.suppression_count == 0
    assert (
        pipeline.discovery_query("healthcare", profile)
        == pipeline.SEGMENTS["healthcare"] + " when:14d"
    )
    assert pipeline.discovery_query("healthcare") == pipeline.SEGMENTS["healthcare"] + " when:14d"


def test_files_for_other_bot_are_ignored(tmp_path):
    other = dict(SYNTHETIC_ICP, bot_ids=["alumni-associations"])
    write_profile(
        tmp_path,
        icp=other,
        suppression=dict(SYNTHETIC_SUPPRESSION, bot_ids=["alumni-associations"]),
    )
    profile = load_bot_profile("healthcare", directory=tmp_path)
    assert profile.status == "absent"
    assert profile.loaded is False


def test_typical_ratio_of_sits_accepts_number_or_range(tmp_path):
    write_profile(tmp_path)
    single = load_bot_profile("healthcare", directory=tmp_path)
    assert single.status == "ok"
    assert "20% of SITS" in single.summary()
    ranged = copy.deepcopy(SYNTHETIC_ICP)
    ranged["products"]["BENEFITS"]["typical_ratio_of_sits"] = [0.143, 0.175]
    write_profile(tmp_path, icp=ranged)
    profile = load_bot_profile("healthcare", directory=tmp_path)
    assert profile.status == "ok"
    assert profile.loaded is True
    assert sits_ratio_label([0.143, 0.175]) == "14-18% of SITS"
    assert "14-18% of SITS" in profile.summary()
    assert "Acme Widgets" not in profile.summary()
    one_item = copy.deepcopy(SYNTHETIC_ICP)
    one_item["products"]["BENEFITS"]["typical_ratio_of_sits"] = [0.2]
    write_profile(tmp_path, icp=one_item)
    listed = load_bot_profile("healthcare", directory=tmp_path)
    assert listed.status == "ok"
    assert "20% of SITS" in listed.summary()
    omitted = copy.deepcopy(SYNTHETIC_ICP)
    del omitted["products"]["BENEFITS"]["typical_ratio_of_sits"]
    write_profile(tmp_path, icp=omitted)
    bare = load_bot_profile("healthcare", directory=tmp_path)
    assert bare.status == "ok"
    assert "of SITS" not in bare.summary()


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "0.2",
        "high",
        [[0.143, 0.175]],
        [0.143, 0.175, 0.2],
        [],
        {"low": 0.143, "high": 0.175},
        1.5,
        -0.1,
        [True, 0.2],
        ["0.143", "0.175"],
        [0.143, [0.175]],
    ],
)
def test_typical_ratio_of_sits_rejects_invalid_forms(tmp_path, caplog, value):
    icp = copy.deepcopy(SYNTHETIC_ICP)
    icp["products"]["BENEFITS"]["typical_ratio_of_sits"] = value
    write_profile(tmp_path, icp=icp)
    with caplog.at_level(logging.WARNING, logger="local_brain.command_center.bd_icp"):
        profile = load_bot_profile("healthcare", directory=tmp_path)
    assert profile.status == "error"
    assert profile.loaded is False
    assert "key=products" in caplog.text
    assert "0.143" not in caplog.text
    assert "0.175" not in caplog.text
    assert "Benefits administration" not in caplog.text


def test_malformed_files_fail_closed(tmp_path, caplog):
    folder = tmp_path / "healthcare"
    folder.mkdir()
    (folder / "broken.icp.json").write_text("{not-json", encoding="utf-8")
    profile = load_bot_profile("healthcare", directory=tmp_path)
    assert profile.status == "error"
    assert profile.loaded is False
    assert profile.icp is None
    assert profile.organizations == ()
    assert profile.public_fields()["icp_status"] == "error"
    assert profile.public_fields()["icp_error"] == "malformed_icp_profile"
    assert "Acme Widgets" not in caplog.text
    junk = dict(SYNTHETIC_ICP, version=2, buyer_types={"hot": ["SHOULD_NOT_SHAPE_QUERY"]})
    (folder / "broken.icp.json").write_text(json.dumps(junk), encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="local_brain.command_center.bd_icp"):
        profile = load_bot_profile("healthcare", directory=tmp_path)
    assert profile.status == "error"
    assert "key=version" in caplog.text
    assert "SHOULD_NOT_SHAPE_QUERY" not in caplog.text
    assert "SHOULD_NOT_SHAPE_QUERY" not in pipeline.discovery_query("healthcare", profile)
    assert query_variants(profile) == [("base", "")]


def test_config_key_and_env_discovery(tmp_path, monkeypatch):
    write_profile(tmp_path)
    monkeypatch.delenv("AIIA_BD_ICP_DIR", raising=False)
    via_config = load_bot_profile("healthcare", config={"icp_dir": str(tmp_path)})
    assert via_config.loaded is True
    assert via_config.customer == "Acme Widgets"
    other = tmp_path / "env-root"
    write_profile(other)
    monkeypatch.setenv("AIIA_BD_ICP_DIR", str(other))
    via_env = load_bot_profile("healthcare", config={"icp_dir": str(tmp_path / "unused")})
    assert via_env.loaded is True
    assert via_env.suppression_count == 2


def test_loaded_profile_shapes_query_and_prompts(tmp_path):
    write_profile(tmp_path)
    profile = load_bot_profile("healthcare", directory=tmp_path)
    queries = [pipeline.discovery_query("healthcare", profile, turn) for turn in range(3)]
    for query in queries:
        assert query.startswith(pipeline.SEGMENTS["healthcare"])
        assert '(Wisconsin OR "Example County" OR Midwest)' in query
        assert "West Coast" not in query
        assert "retail clinics" not in query
    assert "health systems" in queries[1]
    assert '"healthcare providers"' in queries[2]
    summary = profile.summary()
    assert "Prefer buyer types" in summary
    assert "Downweight lanes" in summary
    assert "channel/partner" in summary
    assert "Benefits administration" in summary
    assert "20% of SITS" in summary
    assert "Acme Widgets" not in summary
    assert "Example County Hospital" not in summary
    assert len(summary) < 2400
    token = activate(profile)
    try:
        source, extraction, _ = pipeline.synthetic_fixture("healthcare")
        qwen = pipeline.qwen_payload(source)
        jev = pipeline.jev_payload(source, extraction, "healthcare")
        assert "Prefer buyer types" in qwen["messages"][0]["content"]
        assert jev["state"]["icp_targeting"] == summary
        assert "Example County Hospital" not in json.dumps(qwen)
        assert "Example County Hospital" not in json.dumps(jev)
        assert "Acme Widgets" not in json.dumps(qwen)
        assert "Acme Widgets" not in json.dumps(jev)
        total = pipeline.token_reservation(qwen, 512) + pipeline.token_reservation(jev, 1024)
        assert total < 8000
    finally:
        deactivate(token)


def test_icp_dir_config_validation(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["icp_dir"] = ""
    with pytest.raises(ValueError, match="icp_dir"):
        BDConductor(tmp_path / "bad-icp.sqlite", config=config)
    config["icp_dir"] = str(tmp_path)
    BDConductor(tmp_path / "ok-icp.sqlite", config=config)


@async_test
async def test_suppressed_org_skips_jev_and_never_writes_candidate(tmp_path, monkeypatch):
    write_profile(tmp_path)
    monkeypatch.setenv("AIIA_BD_ICP_DIR", str(tmp_path))
    source, extraction, _ = _gap_clear_source("Acme Widgets, Inc.")
    judged = []

    async def retrieve(segment, rotation=0):
        return [source]

    async def local(source):
        return extraction, 10

    async def forbidden(*args, **kwargs):
        judged.append(True)
        raise AssertionError("suppressed organization must not call Jev")

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", forbidden)
    result = await pipeline.run_pipeline(job(mode="live"))
    assert judged == []
    assert result["packets"] == []
    assert result["reason"] == SUPPRESSED_REASON
    assert result["withheld"][0]["reason"] == SUPPRESSED_REASON
    assert result["withheld"][0]["organization"] == "Acme Widgets, Inc."
    assert result["withheld"][0]["headline"] == source["title"]
    assert result["reviewed_evidence"] == [pipeline.evidence_key(source)]
    assert "customer pipeline" in next_query_for(SUPPRESSED_REASON).lower()


@async_test
async def test_qwen_screen_skips_suppressed_org_and_sends_another(tmp_path, monkeypatch):
    write_profile(tmp_path)
    monkeypatch.setenv("AIIA_BD_ICP_DIR", str(tmp_path))
    blocked, blocked_ex, _ = _gap_clear_source("Example County Hospital")
    allowed, allowed_ex, judgments = _gap_clear_source("Orchard Health System")
    judged = []

    async def retrieve(segment, rotation=0):
        return [blocked, allowed]

    async def local(source):
        if source["url"] == blocked["url"]:
            return blocked_ex, 10
        return allowed_ex, 11

    async def judge(source, extracted, segment):
        judged.append(extracted["organization"])
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", judge)
    result = await pipeline.run_pipeline(job(mode="live"))
    assert judged == ["Orchard Health System"]
    assert result["packets"][0]["organization"] == "Orchard Health System"
    reasons = {item["reason"] for item in result["withheld"]}
    assert SUPPRESSED_REASON in reasons


@async_test
async def test_no_files_live_path_still_proposes(monkeypatch):
    source, extraction, judgments = _gap_clear_source("Orchard Health System")
    judged = []

    async def retrieve(segment, rotation=0):
        return [source]

    async def local(source):
        return extraction, 10

    async def judge(source, extracted, segment):
        judged.append(1)
        return judgments

    monkeypatch.setattr(pipeline, "retrieve_sources", retrieve)
    monkeypatch.setattr(pipeline, "extract_ready", local)
    monkeypatch.setattr(pipeline, "judge", judge)
    result = await pipeline.run_pipeline(job(mode="live"))
    assert judged == [1]
    assert result["packets"][0]["organization"] == "Orchard Health System"


def test_snapshot_icp_fields_and_never_leaks_suppression_names(tmp_path, monkeypatch):
    write_profile(tmp_path)
    monkeypatch.setenv("AIIA_BD_ICP_DIR", str(tmp_path))
    config = json.loads(CONFIG_PATH.read_text())
    store = BDConductor(tmp_path / "jobs.sqlite", config=config, clock=lambda: 1_800_000_000.0)
    service = BDService(store)
    snapshot = service.snapshot()
    blob = json.dumps(snapshot)
    for name in SUPPRESSED_ORG_NAMES:
        assert name not in blob
    healthcare = next(bot for bot in snapshot["bots"] if bot["id"] == "healthcare")
    assert healthcare["icp_loaded"] is True
    assert healthcare["icp_customer"] == "Acme Widgets"
    assert healthcare["icp_status"] == "ok"
    assert healthcare["suppression_count"] == 2
    assert healthcare["suppressed_withholds_today"] == 0
    assert "organizations" not in healthcare
    alumni = next(bot for bot in snapshot["bots"] if bot["id"] == "alumni-associations")
    assert alumni["icp_loaded"] is False
    assert alumni["icp_status"] == "absent"
    assert alumni["suppression_count"] == 0


def test_snapshot_malformed_fail_closed_and_suppressed_withholds_today(tmp_path, monkeypatch):
    folder = tmp_path / "healthcare"
    folder.mkdir()
    (folder / "broken.icp.json").write_text("{", encoding="utf-8")
    monkeypatch.setenv("AIIA_BD_ICP_DIR", str(tmp_path))
    now = [1_800_000_000.0]
    config = json.loads(CONFIG_PATH.read_text())
    store = BDConductor(tmp_path / "jobs.sqlite", config=config, clock=lambda: now[0])
    service = BDService(store)
    snapshot = service.snapshot()
    healthcare = next(bot for bot in snapshot["bots"] if bot["id"] == "healthcare")
    assert healthcare["icp_loaded"] is False
    assert healthcare["icp_status"] == "error"
    assert healthcare["icp_error"] == "malformed_icp_profile"
    assert healthcare["icp_customer"] is None
    assert healthcare["suppression_count"] == 0
    assert "Example County Hospital" not in json.dumps(snapshot)

    async def run():
        queued = store.enqueue("healthcare", "synthetic-healthcare", "one", "research")
        leased = store.lease("worker")
        store.start(queued["id"], leased["lease_token"])
        store.complete(
            queued["id"],
            leased["lease_token"],
            {
                "packets": [],
                "reason": SUPPRESSED_REASON,
                "withheld": [
                    {
                        "evidence_key": "a" * 64,
                        "reason": SUPPRESSED_REASON,
                        "organization": "Synthetic Orchard",
                        "headline": "Synthetic Orchard expands",
                        "source_url": "https://example.org/synthetic",
                        "quote": "Synthetic Orchard expands",
                    }
                ],
            },
        )

    asyncio.run(run())
    counted = service.snapshot()
    healthcare = next(bot for bot in counted["bots"] if bot["id"] == "healthcare")
    assert healthcare["suppressed_withholds_today"] == 1
    assert "Example County Hospital" not in json.dumps(counted)


def test_conductor_does_not_crash_when_icp_dir_is_a_file(tmp_path):
    target = tmp_path / "not-a-dir"
    target.write_text("x", encoding="utf-8")
    profile = load_bot_profile("healthcare", directory=target)
    assert profile.status == "error"
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text()))
    config["icp_dir"] = str(target)
    runtime = BDConductor(tmp_path / "jobs.sqlite", config=config)
    assert runtime.snapshot()["mode"] == "replay"


def test_icp_variants_rotate_one_refinement_at_a_time(tmp_path):
    write_profile(tmp_path)
    profile = load_bot_profile("healthcare", directory=tmp_path)
    labels = [pipeline.discovery_plan("healthcare", profile, turn)[0] for turn in range(4)]
    assert labels == ["geography", "geography+buyers", "geography+sectors", "geography"]
    # Requiring buyers and sectors together is the over-constrained query this replaces.
    for turn in range(3):
        query = pipeline.discovery_query("healthcare", profile, turn)
        assert not ("health systems" in query and "healthcare providers" in query)
        assert query.endswith(" when:14d")
    # A bad rotation value falls back to the first variant instead of failing the job.
    for bad in (-1, 1.5, "2", None, True):
        assert pipeline.discovery_plan("healthcare", profile, bad)[0] == "geography"


def test_icp_variants_without_geography_or_refinements(tmp_path):
    write_profile(
        tmp_path,
        icp={
            **SYNTHETIC_ICP,
            "geography": {"priority_1": [], "priority_2": [], "lower": ["West Coast"]},
        },
    )
    profile = load_bot_profile("healthcare", directory=tmp_path)
    assert [label for label, _ in query_variants(profile)] == ["buyers", "sectors"]
    only_geo = tmp_path / "geo"
    write_profile(
        only_geo,
        icp={
            **SYNTHETIC_ICP,
            "buyer_types": {"hot": [], "downweight": []},
            "employer_sectors": [],
        },
    )
    profile = load_bot_profile("healthcare", directory=only_geo)
    assert query_variants(profile) == [("geography", '(Wisconsin OR "Example County" OR Midwest)')]


def test_long_icp_terms_drop_whole_terms_and_keep_queries_balanced(tmp_path):
    long_icp = {
        **SYNTHETIC_ICP,
        "buyer_types": {
            "hot": [
                "integrated regional health systems",
                "independent community hospitals",
                "county behavioral health providers",
                "federally qualified health centers",
                "critical access hospitals",
                "multi-specialty physician groups",
            ],
            "downweight": [],
        },
        "employer_sectors": ["healthcare providers", "senior living operators"],
        "geography": {
            "priority_1": ["Wisconsin", "Minnesota", "Iowa", "Illinois"],
            "priority_2": ["Midwest", "Great Lakes region", "Upper Midwest"],
            "lower": [],
        },
    }
    write_profile(tmp_path, icp=long_icp)
    profile = load_bot_profile("healthcare", directory=tmp_path)
    variants = dict(query_variants(profile))
    geography = '(Wisconsin OR Minnesota OR Iowa OR Illinois OR Midwest OR "Great Lakes region" OR "Upper Midwest")'
    assert variants["geography"] == geography
    for extra in variants.values():
        assert len(extra) <= 280
        assert extra.startswith(geography)
        assert extra.count("(") == extra.count(")")
        assert extra.count('"') % 2 == 0
    # The buyer group lost trailing terms to fit, never part of a term.
    buyers = variants["geography+buyers"]
    assert '"integrated regional health systems"' in buyers
    assert "multi-specialty" not in buyers
    assert buyers.endswith(")")
