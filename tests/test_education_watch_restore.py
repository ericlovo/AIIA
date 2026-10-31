import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "restore", Path(__file__).parents[1] / "scripts/education_watch_restore.py"
)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_RUN_ID", "22")
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "2")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    return tmp_path


def test_first_run(runtime, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "1")
    monkeypatch.setattr(
        restore, "gh", lambda *args: json.dumps({"workflow_runs": [{"id": 22, "run_number": 1}]})
    )
    restore.main()


def test_deleted_history_is_not_first_run(runtime, monkeypatch):
    monkeypatch.setattr(restore, "gh", lambda *args: json.dumps({"workflow_runs": []}))
    with pytest.raises(RuntimeError, match="previous_run_unavailable"):
        restore.main()


def test_rerun_blocks(runtime, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    with pytest.raises(RuntimeError, match="rerun_requires_manual"):
        restore.main()


def test_failed_previous_blocks_even_if_artifact_exists(runtime, monkeypatch):
    monkeypatch.setattr(
        restore,
        "gh",
        lambda *args: json.dumps(
            {
                "workflow_runs": [
                    {"id": 21, "run_number": 1, "status": "completed", "conclusion": "failure"}
                ]
            }
        ),
    )
    with pytest.raises(RuntimeError, match="previous_run_failed"):
        restore.main()


def test_success_requires_state(runtime, monkeypatch):
    def fake(*args):
        if args[0] == "api":
            return json.dumps(
                {
                    "workflow_runs": [
                        {"id": 21, "run_number": 1, "status": "completed", "conclusion": "success"}
                    ]
                }
            )
        return ""

    monkeypatch.setattr(restore, "gh", fake)
    with pytest.raises(RuntimeError, match="previous_ledger_missing"):
        restore.main()
