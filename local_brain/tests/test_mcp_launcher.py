import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/aiia-mcp", repo / "scripts/aiia-mcp")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "AIIA_AIRGAP": "1",
    }
    return repo, env


def fake_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then exit "${PREFLIGHT_EXIT:-0}"; fi\n'
        'printf "%s\\n" "$PWD" "$EQ_BRAIN_DATA_DIR" "$AIIA_AIRGAP" "$0" "$@"\n'
    )
    path.chmod(0o755)


def run(repo, env):
    return subprocess.run(
        ["/bin/sh", str(repo / "scripts/aiia-mcp")],
        cwd=repo.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_venv_preferred_and_default_data_dir(launcher):
    repo, env = launcher
    python = repo / ".venv/bin/python3"
    fake_python(python)
    result = run(repo, env)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        str(repo),
        f"{env['HOME']}/.aiia/eq_data",
        "1",
        str(python),
        "-m",
        "local_brain.mcp_server",
    ]


def test_path_fallback_preserves_explicit_data_dir(launcher):
    repo, env = launcher
    python = repo / "tools/python3"
    fake_python(python)
    env["PATH"] = f"{python.parent}{os.pathsep}{env['PATH']}"
    env["EQ_BRAIN_DATA_DIR"] = str(repo / "custom data")
    result = run(repo, env)
    assert result.returncode == 0
    assert result.stdout.splitlines()[1:4] == [env["EQ_BRAIN_DATA_DIR"], "1", str(python)]


def test_missing_dependencies_fail_without_environment_disclosure(launcher):
    repo, env = launcher
    fake_python(repo / ".venv/bin/python3")
    env.update(PREFLIGHT_EXIT="1", TYPESAFE_API_KEY="fixture-secret")
    result = run(repo, env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert len(result.stderr.splitlines()) == 1
    assert "pip install -e ." in result.stderr
    assert "fixture-secret" not in result.stderr


def test_config_uses_repo_relative_launcher():
    config = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["aiia"]
    assert config == {"command": "sh", "args": ["./scripts/aiia-mcp"]}


def test_missing_python_is_actionable(launcher):
    repo, env = launcher
    tools = repo / "empty-path"
    tools.mkdir()
    (tools / "dirname").symlink_to(shutil.which("dirname"))
    env["PATH"] = str(tools)
    result = run(repo, env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Install Python 3.10+" in result.stderr
    assert len(result.stderr.splitlines()) == 1


def test_missing_home_requires_explicit_data_dir(launcher):
    repo, env = launcher
    fake_python(repo / ".venv/bin/python3")
    del env["HOME"]
    result = run(repo, env)
    assert result.returncode == 1
    assert "Set HOME or EQ_BRAIN_DATA_DIR" in result.stderr
