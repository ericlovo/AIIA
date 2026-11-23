"""Cloudflare Access JWT verification for the Command Center.

Remote access reaches the Command Center through a Cloudflare tunnel whose
hostname is protected by Cloudflare Access. Access signs every request it lets
through and passes the token in the ``Cf-Access-Jwt-Assertion`` header. This
middleware checks that token in the app, so a misconfigured or deleted Access
policy fails closed instead of exposing the approve endpoints.

Who must present a token:

* Every HTTP request and WebSocket handshake that is not a *direct loopback*
  call. Direct loopback means all three hold: the TCP peer is a loopback
  address, the ``Host`` header names loopback, and the request carries none of
  the headers a proxy adds. cloudflared connects from localhost, so the peer
  address alone cannot tell the tunnel from a local caller. The forwarding
  headers and the public ``Host`` can, and a remote caller cannot remove them.
* Direct loopback calls are allowed without a token. The CLI, the MCP server,
  the Brain and the scheduled scripts on the same machine call :8200 this way,
  and the listener is bound to loopback (see ``server.bind_host``).
* The two Slack webhook paths are exempt because ``slack_capture`` verifies
  Slack's HMAC signature on them. Nothing else is exempt.

Configuration: ``CF_ACCESS_TEAM_DOMAIN`` (``myteam`` or
``myteam.cloudflareaccess.com``) and ``CF_ACCESS_AUD`` (the Access application's
audience tag). When either is missing or malformed, nothing can be verified, so
every request that needs a token is refused.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

logger = logging.getLogger("aiia.command_center.access")

TOKEN_HEADER = b"cf-access-jwt-assertion"

# Signature-verified in slack_capture.verified_body. Exact POST matches only.
SLACK_SIGNED_PATHS = frozenset(
    {
        "/api/integrations/slack/events",
        "/api/integrations/slack/commands",
    }
)

# Any of these means the request came through a proxy (the tunnel), not from a
# process on this machine. Cloudflare sets them at the edge on every request.
PROXY_HEADERS = frozenset(
    {
        TOKEN_HEADER,
        b"cf-connecting-ip",
        b"cf-connecting-ipv6",
        b"cf-ray",
        b"cf-visitor",
        b"cf-ipcountry",
        b"cdn-loop",
        b"true-client-ip",
        b"x-forwarded-for",
        b"x-forwarded-host",
        b"x-forwarded-proto",
        b"x-real-ip",
        b"forwarded",
    }
)

LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

AUTH_DETAIL = "authentication_required"
KEYS_UNAVAILABLE_DETAIL = "access_keys_unavailable"

JWKS_TTL_SECONDS = 3600.0
# An unknown key id triggers at most one refetch per minute, so a flood of
# forged tokens cannot turn this server into a request generator.
JWKS_MIN_REFRESH_SECONDS = 60.0
JWKS_FAILURE_BACKOFF_SECONDS = 10.0
JWKS_FETCH_TIMEOUT_SECONDS = 5.0
CLOCK_SKEW_SECONDS = 30
MAX_TOKEN_BYTES = 8192

_TEAM_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ACCESS_SUFFIX = ".cloudflareaccess.com"

FetchJWKS = Callable[[str], Awaitable[Any]]


@dataclass(frozen=True)
class AccessConfig:
    team_host: str
    audience: str

    @property
    def issuer(self) -> str:
        return f"https://{self.team_host}"

    @property
    def certs_url(self) -> str:
        return f"{self.issuer}/cdn-cgi/access/certs"


def load_config(env: Mapping[str, str] | None = None) -> tuple[AccessConfig | None, str]:
    """Read the two env vars. Returns ``(config, problem)``; one of them is empty."""
    env = os.environ if env is None else env
    team = env.get("CF_ACCESS_TEAM_DOMAIN", "").strip().lower()
    audience = env.get("CF_ACCESS_AUD", "").strip()
    if not team and not audience:
        return None, "CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD are not set"
    if not team or not audience:
        missing = "CF_ACCESS_TEAM_DOMAIN" if not team else "CF_ACCESS_AUD"
        return None, f"{missing} is not set"
    team = team.removeprefix("https://").rstrip("/")
    label = team.removesuffix(_ACCESS_SUFFIX)
    # The label is the only part taken from the environment, so the JWKS fetch
    # can only ever go to a cloudflareaccess.com host.
    if not _TEAM_LABEL.fullmatch(label):
        return None, "CF_ACCESS_TEAM_DOMAIN is not a Cloudflare Access team domain"
    return AccessConfig(team_host=f"{label}{_ACCESS_SUFFIX}", audience=audience), ""


def log_startup_mode() -> None:
    """Say at startup, loudly when unprotected, how remote requests are treated."""
    config, problem = load_config()
    if config:
        logger.info(
            "Cloudflare Access verification ON: issuer %s. Direct loopback calls "
            "and the two signed Slack paths skip the token check.",
            config.issuer,
        )
        return
    logger.warning(
        "\n%s\nCLOUDFLARE ACCESS VERIFICATION IS NOT CONFIGURED (%s).\n"
        "Direct loopback calls are allowed WITHOUT authentication (local dev).\n"
        "Every other request, including everything through the tunnel, gets 401.\n"
        "Set CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD to enable remote access.\n%s",
        "=" * 72,
        problem,
        "=" * 72,
    )


async def fetch_jwks(url: str) -> Any:
    async with httpx.AsyncClient(
        timeout=JWKS_FETCH_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


class JWKSUnavailable(Exception):
    """No signing keys could be loaded, so no token can be checked."""


class JWKSCache:
    """The team's public signing keys, by key id.

    Refetched when older than the TTL and, rate-limited, when a token names a
    key id we have not seen (Cloudflare rotates keys). If a refetch fails and
    older keys are in hand, those keep being used: they are public keys, and
    refusing every request during a Cloudflare blip would lock the owner out.
    With no keys at all the caller gets ``JWKSUnavailable``.
    """

    def __init__(
        self,
        url: str,
        fetch: FetchJWKS = fetch_jwks,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._fetch = fetch
        self._clock = clock
        self._keys: dict[str, Any] | None = None
        self._fetched_at = 0.0
        self._last_attempt: float | None = None
        self._last_failed = False
        self._lock = asyncio.Lock()

    async def key_for(self, kid: str) -> Any | None:
        async with self._lock:
            now = self._clock()
            stale = self._keys is None or now - self._fetched_at >= JWKS_TTL_SECONDS
            rotated = (
                self._keys is not None
                and kid not in self._keys
                and self._waited(now, JWKS_MIN_REFRESH_SECONDS)
            )
            if stale or rotated:
                await self._refresh(now)
            if self._keys is None:
                raise JWKSUnavailable(self._url)
            return self._keys.get(kid)

    def _waited(self, now: float, seconds: float) -> bool:
        return self._last_attempt is None or now - self._last_attempt >= seconds

    async def _refresh(self, now: float) -> None:
        if self._last_failed and not self._waited(now, JWKS_FAILURE_BACKOFF_SECONDS):
            return
        self._last_attempt = now
        try:
            keys = _signing_keys(await self._fetch(self._url))
        except Exception as exc:  # network, HTTP status, JSON or key parsing
            self._last_failed = True
            logger.error(
                "Could not load Cloudflare Access signing keys from %s (%s: %s); %s",
                self._url,
                type(exc).__name__,
                exc,
                "using the previous keys" if self._keys else "no keys available",
            )
            return
        self._last_failed = False
        self._keys = keys
        self._fetched_at = now


def _signing_keys(document: Any) -> dict[str, Any]:
    keys: dict[str, Any] = {}
    entries = document.get("keys") if isinstance(document, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("kty") != "RSA":
            continue
        kid = entry.get("kid")
        if not isinstance(kid, str) or not kid:
            continue
        try:
            keys[kid] = jwt.PyJWK.from_dict(entry, algorithm="RS256").key
        except jwt.PyJWTError:
            continue
    if not keys:
        raise ValueError("JWKS document has no usable RSA keys")
    return keys


class TokenRejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def verify_token(token: str, config: AccessConfig, keys: JWKSCache) -> dict[str, Any]:
    """Return the claims of a valid Access token or raise ``TokenRejected``."""
    if not token or len(token) > MAX_TOKEN_BYTES:
        raise TokenRejected("invalid_token")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise TokenRejected("invalid_token") from exc
    kid = header.get("kid")
    # Pinning RS256 here and in decode() rules out "none" and HMAC-with-the-
    # public-key downgrade tricks.
    if header.get("alg") != "RS256" or not isinstance(kid, str) or not kid:
        raise TokenRejected("invalid_token")
    key = await keys.key_for(kid)
    if key is None:
        raise TokenRejected("invalid_token")
    try:
        return jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=config.audience,
            issuer=config.issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenRejected("token_expired") from exc
    except jwt.PyJWTError as exc:
        raise TokenRejected("invalid_token") from exc


def _is_loopback_address(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def _hostname(host_header: str) -> str:
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if ":" in host else host


def is_direct_loopback(scope: Mapping[str, Any]) -> bool:
    """True only for a call made on this machine, straight to the listener."""
    client = scope.get("client")
    if not client or not _is_loopback_address(str(client[0])):
        return False
    hosts = []
    for name, value in scope.get("headers") or ():
        name = name.lower()
        if name in PROXY_HEADERS:
            return False
        if name == b"host":
            hosts.append(value.decode("latin-1"))
    # Requiring a loopback Host also stops DNS rebinding: a hostile page that
    # points its own domain at 127.0.0.1 still sends its own name in Host.
    return len(hosts) == 1 and _hostname(hosts[0]) in LOOPBACK_HOSTNAMES


class CloudflareAccessMiddleware:
    """ASGI middleware. Plain ASGI so it also covers WebSocket handshakes."""

    def __init__(self, app: Any, fetch_jwks: FetchJWKS = fetch_jwks) -> None:
        self.app = app
        self._fetch_jwks = fetch_jwks
        self._config: AccessConfig | None = None
        self._keys: JWKSCache | None = None
        self._warned_open_loopback = False

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in SLACK_SIGNED_PATHS
        ):
            await self.app(scope, receive, send)
            return

        config, problem = load_config()
        if is_direct_loopback(scope):
            if config is None and not self._warned_open_loopback:
                self._warned_open_loopback = True
                logger.warning(
                    "UNAUTHENTICATED loopback request allowed (%s %s): %s. "
                    "This is the local-dev path; remote requests are refused.",
                    scope.get("method", "WS"),
                    scope.get("path"),
                    problem,
                )
            await self.app(scope, receive, send)
            return

        if config is None:
            logger.error(
                "Refused %s %s: %s, so the request cannot be authenticated.",
                scope.get("method", "WS"),
                scope.get("path"),
                problem,
            )
            await _reject(scope, send, 401, AUTH_DETAIL, "access_not_configured")
            return

        if config != self._config or self._keys is None:
            self._config = config
            self._keys = JWKSCache(config.certs_url, self._fetch_jwks)
        tokens = [value for name, value in scope["headers"] if name.lower() == TOKEN_HEADER]
        if len(tokens) != 1:
            await _reject(scope, send, 401, AUTH_DETAIL, "missing_token")
            return
        try:
            claims = await verify_token(tokens[0].decode("latin-1"), config, self._keys)
        except TokenRejected as exc:
            logger.warning(
                "Refused %s %s: Cloudflare Access token rejected (%s).",
                scope.get("method", "WS"),
                scope.get("path"),
                exc.reason,
            )
            await _reject(scope, send, 401, AUTH_DETAIL, exc.reason)
            return
        except JWKSUnavailable:
            # Not the caller's fault and not evidence about their session, so
            # this is a 503 and the dashboard keeps its generic error state.
            await _reject(scope, send, 503, KEYS_UNAVAILABLE_DETAIL, "")
            return
        scope.setdefault("state", {})["access_claims"] = claims
        await self.app(scope, receive, send)


async def _reject(scope: Any, send: Any, status: int, detail: str, reason: str) -> None:
    if scope["type"] == "websocket":
        # Closing before accept makes the server answer the handshake with 403.
        await send({"type": "websocket.close", "code": 1008, "reason": detail})
        return
    payload: dict[str, str] = {"detail": detail}
    headers = [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]
    if status == 401:
        payload.update(reason=reason, auth="cloudflare_access")
        headers.append((b"www-authenticate", b'Bearer realm="cloudflare-access"'))
    body = json.dumps(payload).encode()
    headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
