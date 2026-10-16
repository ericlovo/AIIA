import json

import pytest

from scripts.benchmark_studio_models import main, measure


def fake_api(calls):
    def api(base, path, payload=None):
        calls.append((base, path, payload))
        if path == "/api/tags":
            return {"models": [{"name": "test:1b"}]}
        if path == "/api/version":
            return {"version": "test-version"}
        if path == "/api/show":
            return {"model_info": {"general.architecture": "test"}}
        if path == "/api/ps":
            return {"models": []}
        return {
            "message": {"content": "synthetic"},
            "eval_count": 10,
            "eval_duration": 1_000_000_000,
        }

    return api


def test_inventory_never_runs_inference(capsys):
    calls = []
    assert main([], api=fake_api(calls)) == 0
    assert [row[1] for row in calls] == ["/api/tags", "/api/ps", "/api/version"]


def test_opt_in_uses_only_selected_installed_model(capsys):
    calls = []
    main(["--run", "--model", "test:1b"], api=fake_api(calls))
    chats = [row[2] for row in calls if row[1] == "/api/chat"]
    assert len(chats) == 2
    assert all(p["model"] == "test:1b" and p["think"] is False for p in chats)
    assert all("keep_alive" not in p for p in chats)
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[1]["decode_tokens_per_second"] == 10
    assert output[1]["resident_before"] is False


@pytest.mark.parametrize("args", [["--run"], ["--run", "--model", "not-installed"]])
def test_invalid_model_never_infers(args):
    calls = []
    with pytest.raises(SystemExit):
        main(args, api=fake_api(calls))
    assert all(row[1] != "/api/chat" for row in calls)


def test_remote_endpoint_rejected_without_network():
    calls = []
    with pytest.raises(SystemExit):
        main(["--url", "https://example.com"], api=fake_api(calls))
    assert not calls


def test_invalid_measurements_are_unknown():
    result = measure({"eval_count": True, "eval_duration": -1, "total_duration": "1"}, 10)
    assert result["eval_count"] is None
    assert result["decode_tokens_per_second"] is None
    assert result["total_ms"] is None


def test_inventory_reports_provider_metadata_without_inference(capsys):
    calls = []

    def api(base, path, payload=None):
        calls.append(path)
        if path == "/api/version":
            return {"version": "0.test"}
        model = {
            "name": "test:1b",
            "digest": "abc",
            "size": 100,
            "details": {"parameter_size": "1B", "quantization_level": "Q4_K_M"},
        }
        if path == "/api/ps":
            model.update(context_length=8192, size_vram=90)
        return {"models": [model]}

    main([], api=api)
    inventory = json.loads(capsys.readouterr().out)["inventory"]
    assert inventory["version"] == "0.test"
    assert inventory["installed"][0]["digest"] == "abc"
    assert inventory["installed"][0]["details"]["quantization_level"] == "Q4_K_M"
    assert "size_vram" not in inventory["installed"][0]
    assert inventory["loaded"][0]["context_length"] == 8192
    assert inventory["loaded"][0]["size_vram"] == 90
    assert calls == ["/api/tags", "/api/ps", "/api/version"]


@pytest.mark.parametrize(
    "name,tag,shown",
    [
        ("test:cloud", {}, {}),
        ("test-large-cloud", {}, {}),
        ("alias:latest", {"remote_host": "cloud.example"}, {}),
        ("alias:latest", {"remote_model": "remote"}, {}),
        ("alias:latest", {}, {"remote_host": "cloud.example"}),
        ("alias:latest", {}, {"remote_model": "remote"}),
        ("alias:latest", {}, {}),
    ],
)
def test_cloud_or_unknown_model_never_infers(name, tag, shown):
    calls = []

    def api(base, path, payload=None):
        calls.append(path)
        if path == "/api/tags":
            return {"models": [{"name": name, **tag}]}
        if path == "/api/ps":
            return {"models": []}
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/show":
            return shown
        raise AssertionError("inference must not run")

    with pytest.raises(SystemExit):
        main(["--run", "--model", name], api=api)
    assert "/api/chat" not in calls
