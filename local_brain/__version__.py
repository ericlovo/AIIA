"""Single source of truth for the AIIA package version.

All runtime components (local_api, command_center, dashboard) and the
pyproject.toml read their version from this module. Bump here on release
and update CHANGELOG.md + git tag.
"""

__version__ = "0.4.0"
