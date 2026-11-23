"""Cloudflare Access JWT check on the Command Center.

Tokens are signed with RSA keys generated here and checked against a JWKS
served by a fake fetcher. Nothing in this file touches the network: the real
fetcher's HTTP client is replaced with one that fails the test.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from local_brain.command_center import access_auth
from local_brain.command_center.access_auth import (
    CloudflareAccessMiddleware,
    JWKSCache,
    JWKSUnavailable,
    load_config,
)

pytestmark = pytest.mark.access_gate

TEAM = "aiia-test"
ISSUER = f"https://{TEAM}.cloudflareaccess.com"
CERTS_URL = f"{ISSUER}/cdn-cgi/access/certs"
AUD = "4714c1358e65fe4b408ad6d432a5f878f08194bdb4752441fd56faefa9b2b6f2"
KID = "key-current"
PUBLIC_URL = "https://aiia.example.test"
REMOTE_PEER = ("203.0.113.7", 40000)
LOOPBACK_PEER = ("127.0.0.1", 40000)
EVENTS = "/api/integrations/slack/events"
COMMANDS = "/api/integrations/slack/commands"


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwk(key: rsa.RSAPrivateKey, kid: str) -> dict[str, Any]:
    entry = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    return {**entry, "kid": kid, "alg": "RS256", "use": "sig"}


class FakeJWKS:
    """Stands in for the Cloudflare certs endpoint and counts fetches."""

    def __init__(self, *entries: dict[str, Any]) -> None:
        self.document: Any = {"keys": list(entries)}
        self.urls: list[str] = []
        self.error: Exception | None = None

    async def __call__(self, url: str) -> Any:
        self.urls.append(url)
        if self.error:
            raise self.error
        return self.document


def token(key: rsa.RSAPrivateKey, kid: str = KID, **overrides: Any) -> str:
    now = int(time.time())
    claims = {
        "aud": [AUD],
        "iss": ISSUER,
        "iat": now - 10,
        "exp": now + 600,
        "sub": "user-1",
        "email": "owner@example.test",
        **overrides,
    }
    claims = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("test tried to fetch the JWKS over the network")

    monkeypatch.setattr(httpx, "AsyncClient", refuse)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CF_ACCESS_TEAM_DOMAIN", TEAM)
    monkeypatch.setenv("CF_ACCESS_AUD", AUD)


@pytest.fixture
def jwks(signing_key: rsa.RSAPrivateKey) -> FakeJWKS:
    return FakeJWKS(jwk(signing_key, KID))


@pytest.fixture
def app(jwks: FakeJWKS) -> FastAPI:
    app = FastAPI()
    app.add_middleware(CloudflareAccessMiddleware, fetch_jwks=jwks)

    @app.get("/api/ping")
    async def ping(request: Request) -> dict[str, Any]:
        return {"email": getattr(request.state, "access_claims", {}).get("email")}

    @app.post("/api/approve")
    async def approve() -> dict[str, bool]:
        return {"approved": True}

    @app.post(EVENTS)
    @app.post(COMMANDS)
    async def slack() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/integrations/slack/status")
    async def slack_status() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_text("hello")
        await socket.close()

    return app


@pytest.fixture
def remote(app: FastAPI) -> TestClient:
    """A caller that is not on this machine."""
    return TestClient(app, base_url=PUBLIC_URL, client=REMOTE_PEER)


@pytest.fixture
def tunnel(app: FastAPI) -> TestClient:
    """What cloudflared sends: loopback peer, public Host, edge headers."""
    return TestClient(
        app,
        base_url=PUBLIC_URL,
        client=LOOPBACK_PEER,
        headers={"Cf-Connecting-Ip": "198.51.100.9", "Cf-Ray": "8f1a-ORD"},
    )


@pytest.fixture
def local(app: FastAPI) -> TestClient:
    """A process on the same machine calling the listener directly."""
    return TestClient(app, base_url="http://localhost:8200", client=LOOPBACK_PEER)


def bearer(value: str) -> dict[str, str]:
    return {"Cf-Access-Jwt-Assertion": value}


def assert_auth_failure(response: httpx.Response, reason: str) -> None:
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert response.headers["cache-control"] == "no-store"
    assert "cloudflare-access" in response.headers["www-authenticate"]
    assert response.json() == {
        "detail": "authentication_required",
        "reason": reason,
        "auth": "cloudflare_access",
    }


# ── Token checks ─────────────────────────────────────────────


def test_valid_token_is_accepted_and_claims_reach_the_route(
    configured, tunnel, jwks, signing_key
) -> None:
    response = tunnel.get("/api/ping", headers=bearer(token(signing_key)))
    assert response.status_code == 200
    assert response.json() == {"email": "owner@example.test"}
    assert jwks.urls == [CERTS_URL]


def test_valid_token_from_a_non_loopback_peer_is_accepted(configured, remote, signing_key) -> None:
    assert remote.get("/api/ping", headers=bearer(token(signing_key))).status_code == 200


def test_bad_signature_is_rejected(configured, tunnel, other_key) -> None:
    # Right key id, right claims, signed by a key Cloudflare never published.
    response = tunnel.get("/api/ping", headers=bearer(token(other_key)))
    assert_auth_failure(response, "invalid_token")


def test_tampered_payload_is_rejected(configured, tunnel, signing_key) -> None:
    header, _, signature = token(signing_key).split(".")
    forged = base64.urlsafe_b64encode(
        json.dumps({"aud": [AUD], "iss": ISSUER, "iat": 1, "exp": int(time.time()) + 9999}).encode()
    ).rstrip(b"=")
    response = tunnel.get("/api/ping", headers=bearer(f"{header}.{forged.decode()}.{signature}"))
    assert_auth_failure(response, "invalid_token")


def test_wrong_audience_is_rejected(configured, tunnel, signing_key) -> None:
    # A token Access issued for a different application on the same team.
    response = tunnel.get("/api/ping", headers=bearer(token(signing_key, aud=["other-app"])))
    assert_auth_failure(response, "invalid_token")


def test_wrong_issuer_is_rejected(configured, tunnel, signing_key) -> None:
    value = token(signing_key, iss="https://someone-else.cloudflareaccess.com")
    assert_auth_failure(tunnel.get("/api/ping", headers=bearer(value)), "invalid_token")


def test_expired_token_is_rejected(configured, tunnel, signing_key) -> None:
    now = int(time.time())
    value = token(signing_key, iat=now - 7200, exp=now - 3600)
    assert_auth_failure(tunnel.get("/api/ping", headers=bearer(value)), "token_expired")


@pytest.mark.parametrize("claim", ["exp", "iat", "iss", "aud"])
def test_token_missing_a_required_claim_is_rejected(configured, tunnel, signing_key, claim) -> None:
    value = token(signing_key, **{claim: None})
    assert_auth_failure(tunnel.get("/api/ping", headers=bearer(value)), "invalid_token")


def test_missing_header_is_rejected(configured, tunnel, remote, jwks) -> None:
    assert_auth_failure(tunnel.get("/api/ping"), "missing_token")
    assert_auth_failure(remote.post("/api/approve"), "missing_token")
    assert jwks.urls == []


def test_unsigned_and_hmac_tokens_are_rejected(configured, tunnel, signing_key) -> None:
    now = int(time.time())
    claims = {"aud": [AUD], "iss": ISSUER, "iat": now, "exp": now + 600}
    unsigned = jwt.encode(claims, None, algorithm="none", headers={"kid": KID})
    assert_auth_failure(tunnel.get("/api/ping", headers=bearer(unsigned)), "invalid_token")

    # Algorithm confusion: HS256 keyed with the public key an attacker can read.
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    signing_input = (
        b64(json.dumps({"alg": "HS256", "kid": KID}).encode())
        + "."
        + b64(json.dumps(claims).encode())
    )
    secret = json.dumps(jwk(signing_key, KID)).encode()
    mac = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    response = tunnel.get("/api/ping", headers=bearer(f"{signing_input}.{b64(mac)}"))
    assert_auth_failure(response, "invalid_token")


@pytest.mark.parametrize("value", ["garbage", "a.b.c", "x" * 9000])
def test_malformed_token_is_rejected(configured, tunnel, value) -> None:
    assert_auth_failure(tunnel.get("/api/ping", headers=bearer(value)), "invalid_token")


def test_unknown_key_id_is_rejected(configured, tunnel, signing_key) -> None:
    value = token(signing_key, kid="key-nobody-published")
    assert_auth_failure(tunnel.get("/api/ping", headers=bearer(value)), "invalid_token")


# ── Slack exemption ──────────────────────────────────────────


@pytest.mark.parametrize("path", [EVENTS, COMMANDS])
def test_signed_slack_paths_skip_the_token_check(configured, remote, tunnel, path) -> None:
    assert remote.post(path).status_code == 200
    assert tunnel.post(path).status_code == 200


def test_slack_paths_are_exempt_even_with_access_unconfigured(remote) -> None:
    assert remote.post(EVENTS).status_code == 200
    assert remote.post(COMMANDS).status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", EVENTS),
        ("POST", EVENTS + "/"),
        ("POST", EVENTS + "/../../approve"),
        ("POST", "/api/integrations/slack/eventsx"),
        ("GET", "/api/integrations/slack/status"),
        ("POST", "/api/approve"),
    ],
)
def test_nothing_else_is_exempt(configured, remote, method, path) -> None:
    assert_auth_failure(remote.request(method, path), "missing_token")


def test_real_command_center_gates_everything_but_the_slack_paths(
    isolated_studio_stores,
) -> None:
    from local_brain.command_center import server

    assert [m.cls for m in server.app.user_middleware].count(CloudflareAccessMiddleware) == 1
    client = TestClient(server.app, base_url=PUBLIC_URL, client=REMOTE_PEER)
    for method, path in [
        ("GET", "/"),
        ("GET", "/api/health"),
        ("GET", "/api/integrations/slack/status"),
        ("POST", "/api/actions/some-id/approve"),
        ("POST", "/api/agents/some-id/run"),
    ]:
        assert_auth_failure(client.request(method, path), "access_not_configured")
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws"):
        pass
    # The Slack paths get past the gate and are refused by their own check.
    for path in (EVENTS, COMMANDS):
        response = client.post(path, content=b"{}")
        assert response.status_code == 503
        assert response.json() == {"detail": "slack_capture_not_configured"}


# ── Unconfigured: loopback only ──────────────────────────────


def test_loopback_is_allowed_with_vars_unset_and_says_so_loudly(local, jwks, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="aiia.command_center.access"):
        assert local.get("/api/ping").status_code == 200
        assert local.post("/api/approve").status_code == 200
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # once per process, not once per request
    assert "UNAUTHENTICATED loopback request allowed" in warnings[0]
    assert "CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD are not set" in warnings[0]
    assert jwks.urls == []


def test_ipv6_loopback_is_allowed_with_vars_unset(app) -> None:
    client = TestClient(app, base_url="http://[::1]:8200", client=("::1", 40000))
    assert client.get("/api/ping").status_code == 200


def test_non_loopback_fails_closed_with_vars_unset(remote, signing_key, caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="aiia.command_center.access"):
        assert_auth_failure(remote.get("/api/ping"), "access_not_configured")
        # A real, valid token does not help: there is nothing to check it against.
        response = remote.get("/api/ping", headers=bearer(token(signing_key)))
        assert_auth_failure(response, "access_not_configured")
    assert "cannot be authenticated" in caplog.text


def test_tunnel_traffic_fails_closed_with_vars_unset(tunnel, app) -> None:
    # cloudflared connects from localhost. The edge headers and the public
    # Host are what keep it from being mistaken for a local caller.
    assert_auth_failure(tunnel.get("/api/ping"), "access_not_configured")
    public_host = TestClient(app, base_url=PUBLIC_URL, client=LOOPBACK_PEER)
    assert_auth_failure(public_host.get("/api/ping"), "access_not_configured")


@pytest.mark.parametrize(
    "header",
    ["Cf-Access-Jwt-Assertion", "Cf-Connecting-Ip", "Cf-Ray", "X-Forwarded-For", "Forwarded"],
)
def test_any_proxy_header_disqualifies_loopback(local, header) -> None:
    response = local.get("/api/ping", headers={header: "198.51.100.9"})
    assert_auth_failure(response, "access_not_configured")


@pytest.mark.parametrize(
    "env",
    [
        {"CF_ACCESS_TEAM_DOMAIN": TEAM},
        {"CF_ACCESS_AUD": AUD},
        {"CF_ACCESS_TEAM_DOMAIN": "attacker.example/x", "CF_ACCESS_AUD": AUD},
    ],
)
def test_partial_or_malformed_config_fails_closed(monkeypatch, remote, signing_key, jwks, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    response = remote.get("/api/ping", headers=bearer(token(signing_key)))
    assert_auth_failure(response, "access_not_configured")
    assert jwks.urls == []


# ── Configured: local callers and WebSockets ─────────────────


def test_direct_loopback_needs_no_token_when_configured(configured, local, jwks, caplog) -> None:
    # The CLI, MCP server and scheduled scripts on the Mini call :8200 directly.
    with caplog.at_level(logging.WARNING, logger="aiia.command_center.access"):
        assert local.get("/api/ping").status_code == 200
    assert caplog.records == []
    assert jwks.urls == []


def test_websocket_handshake_is_gated(configured, tunnel, local, signing_key) -> None:
    with pytest.raises(WebSocketDisconnect), tunnel.websocket_connect("/ws"):
        pass
    bad = bearer("garbage")
    with pytest.raises(WebSocketDisconnect), tunnel.websocket_connect("/ws", headers=bad):
        pass
    with tunnel.websocket_connect("/ws", headers=bearer(token(signing_key))) as socket:
        assert socket.receive_text() == "hello"
    # Starlette's test client always dials ws://testserver unless given a full URL.
    with local.websocket_connect("ws://localhost:8200/ws") as socket:
        assert socket.receive_text() == "hello"


# ── JWKS cache ───────────────────────────────────────────────


def test_jwks_is_fetched_once_and_reused(configured, tunnel, jwks, signing_key) -> None:
    for _ in range(3):
        assert tunnel.get("/api/ping", headers=bearer(token(signing_key))).status_code == 200
    # Forged key ids inside the refresh window must not trigger more fetches.
    for kid in ("forged-1", "forged-2"):
        tunnel.get("/api/ping", headers=bearer(token(signing_key, kid=kid)))
    assert jwks.urls == [CERTS_URL]


def test_unreachable_jwks_is_a_503_not_an_auth_failure(
    configured, tunnel, jwks, signing_key
) -> None:
    jwks.error = httpx.ConnectError("no route")
    response = tunnel.get("/api/ping", headers=bearer(token(signing_key)))
    assert response.status_code == 503
    assert response.json() == {"detail": "access_keys_unavailable"}
    assert "www-authenticate" not in response.headers


@pytest.mark.parametrize("document", [{}, {"keys": []}, {"keys": [{"kty": "EC", "kid": "k"}]}, []])
def test_jwks_without_usable_keys_is_a_503(configured, tunnel, jwks, signing_key, document) -> None:
    jwks.document = document
    response = tunnel.get("/api/ping", headers=bearer(token(signing_key)))
    assert response.status_code == 503


async def test_cache_refetches_for_rotated_keys_and_keeps_old_keys_on_failure(
    signing_key, other_key
) -> None:
    now = [1000.0]
    source = FakeJWKS(jwk(signing_key, KID))
    cache = JWKSCache(CERTS_URL, source, clock=lambda: now[0])

    assert await cache.key_for(KID) is not None
    source.document = {"keys": [jwk(signing_key, KID), jwk(other_key, "key-next")]}
    assert await cache.key_for("key-next") is None  # inside the rate limit: no refetch
    assert len(source.urls) == 1
    now[0] += access_auth.JWKS_MIN_REFRESH_SECONDS
    assert await cache.key_for("key-next") is not None  # rotation picked up
    assert await cache.key_for(KID) is not None
    assert len(source.urls) == 2

    # Past the TTL with Cloudflare unreachable: keep serving the keys in hand.
    now[0] += access_auth.JWKS_TTL_SECONDS
    source.error = httpx.ConnectError("no route")
    assert await cache.key_for(KID) is not None
    assert len(source.urls) == 3
    assert await cache.key_for(KID) is not None  # failure backoff: no new fetch
    assert len(source.urls) == 3


async def test_cache_with_no_keys_raises_and_backs_off() -> None:
    now = [0.0]
    source = FakeJWKS()
    source.error = httpx.ConnectError("no route")
    cache = JWKSCache(CERTS_URL, source, clock=lambda: now[0])
    for _ in range(3):
        with pytest.raises(JWKSUnavailable):
            await cache.key_for(KID)
    assert len(source.urls) == 1
    now[0] += access_auth.JWKS_FAILURE_BACKOFF_SECONDS
    with pytest.raises(JWKSUnavailable):
        await cache.key_for(KID)
    assert len(source.urls) == 2


# ── Configuration parsing ────────────────────────────────────


@pytest.mark.parametrize(
    "team",
    [TEAM, f"{TEAM}.cloudflareaccess.com", f"https://{TEAM}.cloudflareaccess.com/", " AIIA-Test "],
)
def test_team_domain_forms_resolve_to_the_same_issuer(team) -> None:
    config, problem = load_config({"CF_ACCESS_TEAM_DOMAIN": team, "CF_ACCESS_AUD": AUD})
    assert problem == ""
    assert config is not None
    assert config.issuer == ISSUER
    assert config.certs_url == CERTS_URL
    assert config.audience == AUD


@pytest.mark.parametrize(
    "team",
    [
        "evil.example",
        "a.b",
        "team.cloudflareaccess.com.evil.example",
        "team/path",
        "-team",
        "te am",
    ],
)
def test_team_domain_must_be_a_cloudflare_access_host(team) -> None:
    config, problem = load_config({"CF_ACCESS_TEAM_DOMAIN": team, "CF_ACCESS_AUD": AUD})
    assert config is None
    assert "CF_ACCESS_TEAM_DOMAIN" in problem


def test_startup_log_is_loud_when_unconfigured_and_plain_when_configured(
    monkeypatch, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="aiia.command_center.access"):
        access_auth.log_startup_mode()
        assert caplog.records[-1].levelno == logging.WARNING
        assert "NOT CONFIGURED" in caplog.text
        assert "WITHOUT authentication" in caplog.text
        caplog.clear()
        monkeypatch.setenv("CF_ACCESS_TEAM_DOMAIN", TEAM)
        monkeypatch.setenv("CF_ACCESS_AUD", AUD)
        access_auth.log_startup_mode()
        assert caplog.records[-1].levelno == logging.INFO
        assert ISSUER in caplog.text
