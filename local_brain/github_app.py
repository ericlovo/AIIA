"""Read-only GitHub App credentials, status probe, and API helper.

Agent Studio's "GitHub read" tool is deliberately fail-closed. Agents get
remote GitHub access only when a dedicated GitHub App install is configured
with read-only repository permissions. The human `gh` CLI / user OAuth
token on the Mini is never an agent credential — even if `gh auth status`
succeeds.

See docs/GITHUB-APP-READONLY.md for the owner runbook.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("aiia.github_app")

GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_CONFIG_REL = Path(".aiia") / "github-app.json"
DEFAULT_KEY_REL = Path(".aiia") / "github-app.pem"
_READ_METHODS = frozenset({"GET", "HEAD"})

# Env keys — never GH_TOKEN / GITHUB_TOKEN (those are user/PAT credentials).
ENV_APP_ID = "AIIA_GITHUB_APP_ID"
ENV_INSTALLATION_ID = "AIIA_GITHUB_APP_INSTALLATION_ID"
ENV_PRIVATE_KEY = "AIIA_GITHUB_APP_PRIVATE_KEY"
ENV_PRIVATE_KEY_PATH = "AIIA_GITHUB_APP_PRIVATE_KEY_PATH"
ENV_CONFIG = "AIIA_GITHUB_APP_CONFIG"
ENV_REPOS = "AIIA_GITHUB_APP_REPOS"


@dataclass(frozen=True)
class GitHubAppCredentials:
    """Material needed to mint a GitHub App installation token."""

    app_id: str
    installation_id: str
    private_key_pem: str
    repositories: tuple[str, ...] = ()


def github_resource_status(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    cli_user_present: bool | None = None,
) -> dict[str, Any]:
    """Probe Agent Studio's GitHub-read resource — never uses `gh` auth.

    Returns a JSON-serializable dict:

        status: connected | disconnected | not_configured
        mode:   read_only
        source: github_app | none
        cli_user_present: bool  (diagnostic only; does not flip status)
        detail: str
        app_id / installation_id / repositories when known (never secrets)
    """
    env_map = env if env is not None else os.environ
    home_path = home if home is not None else Path.home()
    cli = probe_gh_cli_user() if cli_user_present is None else bool(cli_user_present)

    creds = load_credentials(env=env_map, home=home_path)
    if creds is None:
        return {
            "status": "not_configured",
            "mode": "read_only",
            "source": "none",
            "cli_user_present": cli,
            "detail": (
                "GitHub App credentials are not configured. See docs/GITHUB-APP-READONLY.md."
            ),
        }

    ok, err = validate_private_key(creds.private_key_pem)
    base = {
        "mode": "read_only",
        "source": "github_app",
        "cli_user_present": cli,
        "app_id": creds.app_id,
        "installation_id": creds.installation_id,
    }
    if creds.repositories:
        base["repositories"] = list(creds.repositories)

    if not ok:
        base["status"] = "disconnected"
        base["detail"] = err or "GitHub App private key is not usable"
        return base

    base["status"] = "connected"
    base["detail"] = "read-only GitHub App"
    return base


def agent_github_read_prompt(status: Mapping[str, Any] | None = None) -> str:
    """System-prompt fragment for agents that have the GitHub read tool."""
    probe = dict(status) if status is not None else github_resource_status()
    state = probe.get("status")
    if state == "connected":
        install = probe.get("installation_id") or "unknown"
        repos = probe.get("repositories") or []
        allow = (
            " Allowlist: " + ", ".join(str(r) for r in repos) + "."
            if repos
            else " Scope is the repositories this App is installed on."
        )
        return (
            "GitHub read access is connected via a read-only GitHub App "
            f"(installation {install}).{allow} "
            "Allowed: read repository contents, metadata, issues, and pull "
            "requests that the App's permissions grant. "
            "Forbidden: writes, comments, reviews, merges, pushes, releases, "
            "or using the human `gh` CLI / user OAuth token. "
            "Mounted local clones are a separate tool (Repository read); do "
            "not conflate them with remote GitHub API data."
        )
    detail = probe.get("detail") or "GitHub App is not configured"
    return (
        "GitHub read access is not connected "
        f"({probe.get('status', 'not_configured')}: {detail}). "
        "Do not claim GitHub data until the owner installs a read-only "
        "GitHub App for AIIA (see docs/GITHUB-APP-READONLY.md). "
        "The human GitHub CLI session on this machine is not an agent credential."
    )


def probe_gh_cli_user(timeout: float = 1.0) -> bool:
    """True when `gh auth status` succeeds. Never reads or returns the token."""
    gh = shutil.which("gh")
    if not gh:
        return False
    try:
        result = subprocess.run(
            [gh, "auth", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GH_PROMPT_DISABLED": "1"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def load_credentials(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> GitHubAppCredentials | None:
    """Load App id + installation id + PEM. None if any piece is missing.

    Resolution (env wins over file for each field):

    1. ``AIIA_GITHUB_APP_*`` environment variables
    2. JSON at ``AIIA_GITHUB_APP_CONFIG`` or ``~/.aiia/github-app.json``
    3. Default PEM path ``~/.aiia/github-app.pem`` when ids are known
    """
    env_map = env if env is not None else os.environ
    home_path = home if home is not None else Path.home()
    file_cfg = _read_file_config(env_map, home_path)

    app_id = _first(
        env_map.get(ENV_APP_ID),
        file_cfg.get("app_id"),
        file_cfg.get("appId"),
    )
    installation_id = _first(
        env_map.get(ENV_INSTALLATION_ID),
        file_cfg.get("installation_id"),
        file_cfg.get("installationId"),
    )
    pem = _load_private_key_pem(env_map, file_cfg, home_path)
    repos = _load_repositories(env_map, file_cfg)

    if not app_id or not installation_id or not pem:
        return None
    return GitHubAppCredentials(
        app_id=app_id,
        installation_id=installation_id,
        private_key_pem=pem,
        repositories=repos,
    )


def validate_private_key(pem: str) -> tuple[bool, str | None]:
    """Return (ok, error). Loads the PEM so a header-only blob is not 'connected'."""
    text = _normalize_pem(pem)
    if "BEGIN" not in text or "PRIVATE KEY" not in text:
        return False, "private key is not a PEM private key"
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        load_pem_private_key(text.encode(), password=None)
    except ImportError:
        return False, "cryptography is not installed; cannot validate GitHub App key"
    except Exception:
        return False, "private key is not a loadable RSA PEM"
    return True, None


def mint_app_jwt(creds: GitHubAppCredentials, *, now: int | None = None) -> str:
    """Mint a short-lived GitHub App JWT (RS256). Never logs the key or JWT."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    issued = int(time.time() if now is None else now)
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(
        json.dumps(
            {"iat": issued - 60, "exp": issued + 540, "iss": str(creds.app_id)},
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    key = serialization.load_pem_private_key(creds.private_key_pem.encode(), password=None)
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{_b64url(signature)}"


class GitHubReadClient:
    """Installation-token client for read-only GitHub API calls.

    TODO(agent-tools): invoke GET helpers from the Agent Studio run path
    once tools can call out. Until then this is the sanctioned way to
    mint an installation token — never ``gh`` user auth, never a PAT.
    """

    def __init__(
        self,
        creds: GitHubAppCredentials | None = None,
        *,
        http: Any | None = None,
        env: Mapping[str, str] | None = None,
        home: Path | None = None,
    ):
        self._creds = creds if creds is not None else load_credentials(env=env, home=home)
        self._http = http
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def configured(self) -> bool:
        return self._creds is not None

    def get_installation_token(self) -> str:
        """Exchange the App JWT for an installation access token."""
        if self._creds is None:
            raise RuntimeError("GitHub App is not configured. See docs/GITHUB-APP-READONLY.md.")
        now = time.time()
        if self._token and now < (self._token_expires_at - 60):
            return self._token

        jwt = mint_app_jwt(self._creds)
        url = f"{GITHUB_API}/app/installations/{self._creds.installation_id}/access_tokens"
        response = self._request_http(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
        )
        token = str(response.get("token") or "")
        if not token:
            raise RuntimeError("GitHub App installation token response had no token")
        expires_at = _parse_github_expiry(response.get("expires_at"))
        self._token = token
        self._token_expires_at = expires_at
        return token

    def request(self, method: str, path: str) -> Any:
        """Authenticated GitHub API call. Write methods are refused."""
        verb = method.upper()
        if verb not in _READ_METHODS:
            raise PermissionError(f"GitHub App client is read-only; refusing {verb}")
        if self._creds is None:
            raise RuntimeError("GitHub App is not configured. See docs/GITHUB-APP-READONLY.md.")
        url = _github_api_url(path)
        token = self.get_installation_token()
        return self._request_http(
            verb,
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
        )

    def list_installation_repos(self) -> list[str]:
        """GET /installation/repositories — namespaced owner/repo slugs.

        TODO(agent-tools): inject this snapshot into the agent prompt when
        a live call is cheap enough for the Studio poll path.
        """
        payload = self.request("GET", "/installation/repositories")
        repos = payload.get("repositories") if isinstance(payload, dict) else None
        if not isinstance(repos, list):
            return []
        names = []
        for repo in repos:
            if isinstance(repo, dict) and repo.get("full_name"):
                names.append(str(repo["full_name"]))
        return names

    def _request_http(self, method: str, url: str, headers: dict[str, str]) -> Any:
        http = self._http
        if http is not None:
            handler = getattr(http, method.lower())
            response = handler(url, headers=headers, timeout=20.0)
            return _coerce_http_payload(response)

        import httpx

        with httpx.Client(timeout=20.0) as client:
            response = client.request(method, url, headers=headers)
            response.raise_for_status()
            return response.json()


def _read_file_config(env: Mapping[str, str], home: Path) -> dict[str, Any]:
    raw_path = (env.get(ENV_CONFIG) or "").strip()
    path = Path(raw_path).expanduser() if raw_path else home / DEFAULT_CONFIG_REL
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("GitHub App config unreadable at %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _load_private_key_pem(
    env: Mapping[str, str], file_cfg: Mapping[str, Any], home: Path
) -> str | None:
    inline = _first(env.get(ENV_PRIVATE_KEY), file_cfg.get("private_key"))
    if inline:
        return _normalize_pem(inline)

    path_raw = _first(
        env.get(ENV_PRIVATE_KEY_PATH),
        file_cfg.get("private_key_path"),
        file_cfg.get("privateKeyPath"),
    )
    if not path_raw:
        default_pem = home / DEFAULT_KEY_REL
        path_raw = str(default_pem) if default_pem.is_file() else None
    if not path_raw:
        return None
    path = Path(path_raw).expanduser()
    if not path.is_file():
        return None
    try:
        return _normalize_pem(path.read_text())
    except OSError as exc:
        logger.warning("GitHub App private key unreadable at %s: %s", path, exc)
        return None


def _load_repositories(env: Mapping[str, str], file_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    raw = env.get(ENV_REPOS)
    if raw and raw.strip():
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    listed = file_cfg.get("repositories") or file_cfg.get("repos") or []
    if isinstance(listed, str):
        return tuple(part.strip() for part in listed.split(",") if part.strip())
    if isinstance(listed, list):
        return tuple(str(item).strip() for item in listed if str(item).strip())
    return ()


def _first(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_pem(raw: str) -> str:
    text = raw.strip()
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _github_api_url(path: str) -> str:
    """Resolve a GitHub API path. Absolute URLs must be api.github.com over HTTPS."""
    if path.startswith(("http://", "https://")):
        parsed = urlparse(path)
        if parsed.scheme != "https" or parsed.netloc != "api.github.com":
            raise PermissionError("GitHub App client only calls api.github.com")
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{GITHUB_API}{path}"


def _parse_github_expiry(raw: Any) -> float:
    if isinstance(raw, str) and raw:
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return stamp.timestamp()
        except ValueError:
            pass
    return time.time() + 3600


def _coerce_http_payload(response: Any) -> Any:
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    if hasattr(response, "json"):
        payload = response.json()
        return payload() if callable(payload) else payload
    if isinstance(response, dict):
        return response
    raise RuntimeError("unexpected GitHub HTTP response type")
