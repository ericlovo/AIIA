"""GitHub App read-only status probe + installation-token helper.

No live GitHub or `gh` CLI required — credentials and HTTP are injected.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_brain import github_app
from local_brain.github_app import (
    GitHubAppCredentials,
    GitHubReadClient,
    agent_github_read_prompt,
    github_resource_status,
    load_credentials,
    mint_app_jwt,
    probe_gh_cli_user,
    validate_private_key,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


@pytest.fixture
def app_env(rsa_pem: str, isolated_aiia_dir: str) -> dict[str, str]:
    return {
        "AIIA_GITHUB_APP_ID": "123456",
        "AIIA_GITHUB_APP_INSTALLATION_ID": "789012",
        "AIIA_GITHUB_APP_PRIVATE_KEY": rsa_pem,
        "HOME": str(Path(isolated_aiia_dir).parent),
    }


@pytest.fixture(autouse=True)
def _clear_github_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "AIIA_GITHUB_APP_ID",
        "AIIA_GITHUB_APP_INSTALLATION_ID",
        "AIIA_GITHUB_APP_PRIVATE_KEY",
        "AIIA_GITHUB_APP_PRIVATE_KEY_PATH",
        "AIIA_GITHUB_APP_CONFIG",
        "AIIA_GITHUB_APP_REPOS",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


# ----------------------------------------------------------------------------
# Status probe
# ----------------------------------------------------------------------------


def test_no_app_is_not_configured(isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    status = github_resource_status(env={}, home=home, cli_user_present=False)
    assert status["status"] == "not_configured"
    assert status["mode"] == "read_only"
    assert status["source"] == "none"
    assert status["cli_user_present"] is False
    assert "GITHUB-APP-READONLY" in status["detail"]
    assert "token" not in json.dumps(status).lower()
    assert "BEGIN" not in json.dumps(status)


def test_app_configured_via_env_is_connected(app_env: dict[str, str], isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    status = github_resource_status(env=app_env, home=home, cli_user_present=False)
    assert status["status"] == "connected"
    assert status["mode"] == "read_only"
    assert status["source"] == "github_app"
    assert status["app_id"] == "123456"
    assert status["installation_id"] == "789012"
    assert "private_key" not in status
    assert "BEGIN" not in json.dumps(status)


def test_app_configured_via_files_is_connected(rsa_pem: str, isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    pem_path = home / ".aiia" / "github-app.pem"
    cfg_path = home / ".aiia" / "github-app.json"
    pem_path.write_text(rsa_pem)
    pem_path.chmod(0o600)
    cfg_path.write_text(
        json.dumps(
            {
                "app_id": "42",
                "installation_id": "99",
                "private_key_path": str(pem_path),
                "repositories": ["ericlovo/AIIA", "ericlovo/aiia-console"],
            }
        )
    )
    cfg_path.chmod(0o600)

    status = github_resource_status(env={}, home=home, cli_user_present=False)
    assert status["status"] == "connected"
    assert status["source"] == "github_app"
    assert status["repositories"] == ["ericlovo/AIIA", "ericlovo/aiia-console"]


def test_gh_cli_user_alone_does_not_connect(isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    hosts = home / ".config" / "gh" / "hosts.yml"
    hosts.parent.mkdir(parents=True)
    hosts.write_text("github.com:\n  oauth_token: gho_this_must_never_connect_agents\n")

    status = github_resource_status(
        env={"GH_TOKEN": "gho_human-session", "GITHUB_TOKEN": "gho_human-session"},
        home=home,
        cli_user_present=True,
    )
    assert status["status"] == "not_configured"
    assert status["source"] == "none"
    assert status["cli_user_present"] is True
    dumped = json.dumps(status)
    assert "gho_" not in dumped
    assert "oauth_token" not in dumped


def test_gh_cli_present_does_not_override_connected_app(
    app_env: dict[str, str], isolated_aiia_dir: str
):
    home = Path(isolated_aiia_dir).parent
    status = github_resource_status(env=app_env, home=home, cli_user_present=True)
    assert status["status"] == "connected"
    assert status["source"] == "github_app"
    assert status["cli_user_present"] is True


def test_invalid_pem_is_disconnected_not_connected(isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    # Assemble at runtime so the tree never contains PEM armor (gitleaks).
    kind = "RSA PRIVATE KEY"
    env = {
        "AIIA_GITHUB_APP_ID": "1",
        "AIIA_GITHUB_APP_INSTALLATION_ID": "2",
        "AIIA_GITHUB_APP_PRIVATE_KEY": (
            f"-----BEGIN {kind}-----\nnot-a-real-key\n-----END {kind}-----"
        ),
    }
    status = github_resource_status(env=env, home=home, cli_user_present=True)
    assert status["status"] == "disconnected"
    assert status["source"] == "github_app"
    assert status["cli_user_present"] is True
    assert status["status"] != "connected"


def test_partial_env_is_not_configured(isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    status = github_resource_status(
        env={"AIIA_GITHUB_APP_ID": "1"},
        home=home,
        cli_user_present=False,
    )
    assert status["status"] == "not_configured"
    assert load_credentials(env={"AIIA_GITHUB_APP_ID": "1"}, home=home) is None


def test_env_overrides_file_config(rsa_pem: str, isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    cfg = home / ".aiia" / "github-app.json"
    cfg.write_text(json.dumps({"app_id": "file-id", "installation_id": "file-inst"}))
    status = github_resource_status(
        env={
            "AIIA_GITHUB_APP_ID": "env-id",
            "AIIA_GITHUB_APP_INSTALLATION_ID": "env-inst",
            "AIIA_GITHUB_APP_PRIVATE_KEY": rsa_pem,
        },
        home=home,
        cli_user_present=False,
    )
    assert status["app_id"] == "env-id"
    assert status["installation_id"] == "env-inst"


def test_default_pem_path_used_when_ids_in_json(rsa_pem: str, isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    (home / ".aiia" / "github-app.pem").write_text(rsa_pem)
    (home / ".aiia" / "github-app.json").write_text(
        json.dumps({"app_id": "7", "installation_id": "8"})
    )
    creds = load_credentials(env={}, home=home)
    assert creds is not None
    assert creds.app_id == "7"


# ----------------------------------------------------------------------------
# Prompt copy
# ----------------------------------------------------------------------------


def test_prompt_forbids_claims_when_not_configured():
    text = agent_github_read_prompt(
        {"status": "not_configured", "detail": "GitHub App credentials are not configured."}
    )
    assert "Do not claim GitHub data" in text
    assert "not an agent credential" in text
    assert "re-authenticates the local GitHub CLI" not in text


def test_prompt_describes_read_only_when_connected():
    text = agent_github_read_prompt(
        {
            "status": "connected",
            "installation_id": "789012",
            "repositories": ["ericlovo/AIIA"],
        }
    )
    assert "read-only GitHub App" in text
    assert "installation 789012" in text
    assert "ericlovo/AIIA" in text
    assert "Forbidden" in text
    assert "`gh` CLI" in text


# ----------------------------------------------------------------------------
# gh CLI probe (diagnostic only)
# ----------------------------------------------------------------------------


def test_probe_gh_cli_user_false_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(github_app.shutil, "which", lambda _name: None)
    assert probe_gh_cli_user() is False


def test_probe_gh_cli_user_true_on_zero_exit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(github_app.shutil, "which", lambda _name: "/usr/bin/gh")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="Logged in to github.com", stderr="")

    monkeypatch.setattr(github_app.subprocess, "run", fake_run)
    assert probe_gh_cli_user() is True


def test_probe_gh_cli_user_false_on_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(github_app.shutil, "which", lambda _name: "/usr/bin/gh")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="not logged in")

    monkeypatch.setattr(github_app.subprocess, "run", fake_run)
    assert probe_gh_cli_user() is False


# ----------------------------------------------------------------------------
# JWT + read-only client
# ----------------------------------------------------------------------------


def test_validate_private_key_accepts_rsa(rsa_pem: str):
    ok, err = validate_private_key(rsa_pem)
    assert ok is True
    assert err is None


def test_mint_app_jwt_round_trip(rsa_pem: str):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    creds = GitHubAppCredentials(
        app_id="123456",
        installation_id="1",
        private_key_pem=rsa_pem,
    )
    token = mint_app_jwt(creds, now=1_700_000_000)
    header_b64, payload_b64, signature_b64 = token.split(".")
    payload = json.loads(_b64url_decode(payload_b64))
    assert payload["iss"] == "123456"
    assert payload["exp"] - payload["iat"] == 600

    key = serialization.load_pem_private_key(rsa_pem.encode(), password=None)
    key.public_key().verify(
        _b64url_decode_bytes(signature_b64),
        f"{header_b64}.{payload_b64}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_installation_token_and_read_request(rsa_pem: str):
    creds = GitHubAppCredentials("11", "22", rsa_pem)
    http = _FakeGitHubHttp(
        post_payload={
            "token": "ghs_install_only",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        get_payload={
            "repositories": [{"full_name": "ericlovo/AIIA"}, {"full_name": "ericlovo/mindmoor"}]
        },
    )
    client = GitHubReadClient(creds, http=http)
    assert client.get_installation_token() == "ghs_install_only"
    names = client.list_installation_repos()
    assert names == ["ericlovo/AIIA", "ericlovo/mindmoor"]
    assert http.posts and "/app/installations/22/access_tokens" in http.posts[0]
    assert http.gets and http.gets[0].endswith("/installation/repositories")


def test_read_client_refuses_writes(rsa_pem: str):
    client = GitHubReadClient(GitHubAppCredentials("1", "2", rsa_pem), http=_FakeGitHubHttp())
    with pytest.raises(PermissionError, match="read-only"):
        client.request("POST", "/repos/ericlovo/AIIA/issues")
    with pytest.raises(PermissionError, match="read-only"):
        client.request("PATCH", "/repos/ericlovo/AIIA")
    assert client._http.posts == []
    assert client._http.gets == []


def test_read_client_refuses_non_github_hosts(rsa_pem: str):
    client = GitHubReadClient(GitHubAppCredentials("1", "2", rsa_pem), http=_FakeGitHubHttp())
    with pytest.raises(PermissionError, match="api.github.com"):
        client.request("GET", "https://api.github.com.evil.example/repos/x")
    with pytest.raises(PermissionError, match="api.github.com"):
        client.request("GET", "https://example.com/repos/x")


def test_server_no_longer_hardcodes_disconnected():
    text = Path("local_brain/command_center/server.py").read_text()
    assert "github_resource_status" in text
    assert "agent_github_read_prompt" in text
    assert '{"status": "disconnected", "mode": "read_only"}' not in text
    assert "re-authenticates the local GitHub CLI" not in text


def test_read_client_unconfigured_raises(isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    client = GitHubReadClient(env={}, home=home, http=_FakeGitHubHttp())
    assert client.configured is False
    with pytest.raises(RuntimeError, match="not configured"):
        client.get_installation_token()


def test_escaped_pem_in_env(rsa_pem: str, isolated_aiia_dir: str):
    home = Path(isolated_aiia_dir).parent
    escaped = rsa_pem.replace("\n", "\\n")
    creds = load_credentials(
        env={
            "AIIA_GITHUB_APP_ID": "1",
            "AIIA_GITHUB_APP_INSTALLATION_ID": "2",
            "AIIA_GITHUB_APP_PRIVATE_KEY": escaped,
        },
        home=home,
    )
    assert creds is not None
    assert "BEGIN" in creds.private_key_pem
    assert "\n" in creds.private_key_pem


# ----------------------------------------------------------------------------
# HTTP stub
# ----------------------------------------------------------------------------


class _FakeGitHubHttp:
    def __init__(self, post_payload: dict | None = None, get_payload: dict | None = None):
        self.post_payload = post_payload or {}
        self.get_payload = get_payload or {}
        self.posts: list[str] = []
        self.gets: list[str] = []

    def post(self, url: str, headers=None, timeout=None):
        self.posts.append(url)
        return SimpleNamespace(
            json=lambda: self.post_payload,
            raise_for_status=lambda: None,
        )

    def get(self, url: str, headers=None, timeout=None):
        self.gets.append(url)
        return SimpleNamespace(
            json=lambda: self.get_payload,
            raise_for_status=lambda: None,
        )


def _b64url_decode(segment: str) -> str:
    return _b64url_decode_bytes(segment).decode()


def _b64url_decode_bytes(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))
