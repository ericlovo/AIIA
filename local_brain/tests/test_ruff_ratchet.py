import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ruff_ratchet.py"
spec = importlib.util.spec_from_file_location("ruff_ratchet", SCRIPT)
ratchet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ratchet)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "local_brain").mkdir()
    (tmp_path / "scripts/ruff-baseline.json").write_text(json.dumps({"B006": 1}))
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nignore = ["B006", "E501", "B008"]\n'
    )
    return tmp_path


def response(rows, code=1):
    return SimpleNamespace(returncode=code, stdout=json.dumps(rows), stderr="fixture error")


@pytest.mark.parametrize("count,expected", [(0, 0), (1, 0), (2, 1)])
def test_count_ceiling(project, count, expected):
    result = response([{"code": "B006", "count": count}], int(count > 0))
    assert ratchet.check(project, lambda *a, **k: result) == expected


def test_zero_statistics_pass(project):
    assert ratchet.check(project, lambda *a, **k: response([], 0)) == 0


def test_reduction_in_one_rule_cannot_offset_another(project):
    (project / "scripts/ruff-baseline.json").write_text(json.dumps({"B006": 1, "F841": 2}))
    (project / "pyproject.toml").write_text('[tool.ruff.lint]\nignore = ["B006", "F841"]\n')
    result = response([{"code": "B006", "count": 2}, {"code": "F841", "count": 0}])
    assert ratchet.check(project, lambda *a, **k: result) == 1


def test_new_ignore_requires_baseline(project):
    (project / "pyproject.toml").write_text('[tool.ruff.lint]\nignore = ["B006", "E402"]\n')
    assert ratchet.check(project) == 2


@pytest.mark.parametrize(
    "result",
    [
        response([], 2),
        response([], 1),
        response({"count": 1}),
        response([{"code": "B006", "count": True}]),
        response([{"code": "B006", "count": -1}]),
        response([{"code": "E402", "count": 1}]),
        response([{"code": "B006", "count": 1}] * 2),
        SimpleNamespace(returncode=1, stdout="not JSON", stderr=""),
    ],
)
def test_bad_tool_results_fail_closed(project, result):
    assert ratchet.check(project, lambda *a, **k: result) == 2


def test_missing_executable_fails_closed(project):
    def missing(*args, **kwargs):
        raise FileNotFoundError("fixture")

    assert ratchet.check(project, missing) == 2


@pytest.mark.parametrize("baseline", [{"B006": True}, {"B006": -1}, [], {"F841": 1}])
def test_invalid_baseline(project, baseline):
    (project / "scripts/ruff-baseline.json").write_text(json.dumps(baseline))
    assert ratchet.check(project) == 2


def test_actual_ruff_detects_new_ignored_violation(project):
    source = project / "local_brain/example.py"
    source.write_text("def first(items=[]):\n    return items\n")
    assert ratchet.check(project) == 0
    source.write_text(source.read_text() + "\ndef second(items=[]):\n    return items\n")
    # Normal lint ignores B006. The ratchet must explicitly enable it again.
    normal = subprocess.run(
        [ratchet.sys.executable, "-m", "ruff", "check", "local_brain"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert normal.returncode == 0
    assert ratchet.check(project) == 1
