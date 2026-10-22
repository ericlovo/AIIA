import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/case_collision_guard.py"
spec = importlib.util.spec_from_file_location("case_collision_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def listed(*paths: str):
    payload = b"\0".join(path.encode() for path in paths) + b"\0"
    return SimpleNamespace(returncode=0, stdout=payload, stderr=b"")


def test_morning_note_style_module_stems_collide():
    found = guard.find_collisions(
        [
            "dashboard/src/console/MorningNote.tsx",
            "dashboard/src/console/morningNote.ts",
            "dashboard/src/console/MorningNote.css",
        ]
    )
    assert found == [
        (
            "module",
            [
                "dashboard/src/console/MorningNote.tsx",
                "dashboard/src/console/morningNote.ts",
            ],
        )
    ]


def test_same_path_different_case_collides():
    found = guard.find_collisions(["README.md", "readme.md"])
    assert found == [("path", ["README.md", "readme.md"])]


def test_distinct_stems_and_directories_are_clean():
    assert (
        guard.find_collisions(
            [
                "dashboard/src/console/MorningNote.tsx",
                "dashboard/src/console/morningNoteModel.ts",
                "dashboard/src/console/MorningNote.css",
                "dashboard/tests/morningNote.test.ts",
            ]
        )
        == []
    )


def test_check_fails_on_collision():
    assert guard.check(runner=lambda *a, **k: listed("Foo.ts", "foo.tsx")) == 1


def test_check_passes_when_clean():
    assert (
        guard.check(
            runner=lambda *a, **k: listed(
                "dashboard/src/console/MorningNote.tsx",
                "dashboard/src/console/morningNoteModel.ts",
            )
        )
        == 0
    )


def test_git_failure_fails_closed():
    result = SimpleNamespace(returncode=128, stdout=b"", stderr=b"not a git repo")
    assert guard.check(runner=lambda *a, **k: result) == 2


def test_repo_tree_is_clean():
    assert guard.check() == 0


@pytest.mark.parametrize(
    "paths",
    [
        ["a/foo.ts", "b/foo.ts"],
        ["src/foo.ts", "src/foo.test.ts"],
    ],
)
def test_non_colliding_pairs(paths):
    assert guard.find_collisions(paths) == []
