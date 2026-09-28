import sys
from importlib.metadata import version
from pathlib import Path

from local_brain.__version__ import __version__

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_package_version_has_one_python_source():
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert "version" not in config["project"]
    assert "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "local_brain.__version__.__version__"
    }


def test_installed_metadata_matches_runtime_version():
    assert version("aiia") == __version__
