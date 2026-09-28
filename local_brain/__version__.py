"""Single source of truth for the AIIA package version.

Python package metadata, the CLI and backend version responses use this module.
Setuptools reads the literal below through pyproject.toml's dynamic version.
The dashboard's npm package version is separate. Bump here on Python releases
and update CHANGELOG.md + git tag; reinstall to refresh installed metadata.
"""

__version__ = "0.7.0"
