"""
A2A protocol client — discovery + task delegation against any A2A endpoint.

The router in aiia/a2a/router.py exposes the server side. This
client speaks the same protocol: registry queries by tag, agent card fetch,
and JSON-RPC SendMessage with text-only parts.

Designed so a planner agent (or Claude Code via an MCP bridge) can:
    1. Discover what's available on the Mini (or any peer)
    2. Read an agent's card to understand inputs and tags
    3. Send a task and receive the resulting Task envelope

The client never imports the local registry — it's pure HTTP, so it works
against the Mini from any machine on the Tailscale network and against a
peer Mini if you ever run more than one.

Usage:
    async with A2AClient("http://mac-mini.tail:8100") as client:
        agents = await client.list_agents(tags=["scope:global"])
        result = await client.send_message(
            agent_id="aiia-ask",
            text="Summarize the last week of decisions.",
        )
        print(result.answer_text())
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from local_brain.a2a.schema import (
    AgentCard,
    JsonRpcRequest,
    JsonRpcResponse,
    Task,
)

logger = logging.getLogger("aplora.a2a.client")

DEFAULT_TIMEOUT_SECONDS = 30.0


class A2AClientError(RuntimeError):
    """Raised when an A2A call fails — network, HTTP, or JSON-RPC level."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass
class AgentSummary:
    """Lightweight registry listing — what /a2a/registry/agents returns per agent."""

    agent_id: str
    name: str
    description: str
    tags: list[str]
    card_url: str
    rpc_url: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSummary:
        return cls(
            agent_id=data["agent_id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=list(data.get("tags") or []),
            card_url=data.get("card_url", ""),
            rpc_url=data.get("rpc_url", ""),
        )


@dataclass
class TaskResult:
    """Wraps a completed Task with a few ergonomic helpers for callers."""

    task: Task

    @property
    def state(self) -> str:
        return self.task.status.state.value

    @property
    def is_completed(self) -> bool:
        return self.state == "TASK_STATE_COMPLETED"

    def answer_text(self, separator: str = "\n\n") -> str:
        """
        Concatenate text parts across all artifacts.

        Most agents return a single text artifact, but the spec allows
        multiple — this gives callers a single string regardless.
        """
        chunks: list[str] = []
        for artifact in self.task.artifacts:
            for part in artifact.parts:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        return separator.join(chunks)


class A2AClient:
    """
    Async A2A client. Use as an async context manager to share an httpx.AsyncClient
    across multiple calls; one-shot calls also work via the convenience methods.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._http: httpx.AsyncClient | None = http_client
        self._owns_http = http_client is None

    async def __aenter__(self) -> A2AClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    # ─── Discovery ────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """GET /a2a/health — server-side liveness + agent count."""
        client = await self._client()
        try:
            resp = await client.get(f"{self._base_url}/a2a/health")
        except httpx.RequestError as exc:
            raise A2AClientError(f"network error contacting {self._base_url}: {exc}") from exc
        self._raise_for_http(resp)
        return resp.json()

    async def list_agents(
        self,
        *,
        tags: Iterable[str] | None = None,
        require_all: bool = False,
    ) -> list[AgentSummary]:
        """
        GET /a2a/registry/agents — discover agents, optionally filtered by tag.

        - tags=None returns every agent
        - require_all=False (default): match any of the requested tags (OR)
        - require_all=True: agent must carry every tag (AND)
        """
        params: list[tuple[str, str]] = []
        if tags:
            for tag in tags:
                params.append(("tag", tag))
        if require_all:
            params.append(("require_all", "true"))

        client = await self._client()
        try:
            resp = await client.get(
                f"{self._base_url}/a2a/registry/agents",
                params=params,
            )
        except httpx.RequestError as exc:
            raise A2AClientError(f"network error contacting {self._base_url}: {exc}") from exc
        self._raise_for_http(resp)
        body = resp.json()
        return [AgentSummary.from_dict(a) for a in body.get("agents", [])]

    async def get_agent_card(self, agent_id: str) -> AgentCard:
        """
        GET /a2a/agents/{agent_id}/.well-known/agent.json — full spec-compliant card.
        """
        client = await self._client()
        try:
            resp = await client.get(
                f"{self._base_url}/a2a/agents/{agent_id}/.well-known/agent.json"
            )
        except httpx.RequestError as exc:
            raise A2AClientError(f"network error contacting {self._base_url}: {exc}") from exc
        self._raise_for_http(resp)
        return AgentCard.model_validate(resp.json())

    # ─── Task delegation ──────────────────────────────────────────────

    async def send_message(
        self,
        *,
        agent_id: str,
        text: str,
        role: str = "user",
        request_id: str | None = None,
        method: str = "SendMessage",
    ) -> TaskResult:
        """
        POST /a2a/agents/{agent_id}/rpc — send a text message and await the resulting Task.

        The server wraps the executor's output as an Artifact on a Task with
        TASK_STATE_COMPLETED (or FAILED, in which case we still return the task
        so callers can inspect status.message).

        Args:
            agent_id: Target agent registered on the remote registry
            text: Message body — single text part is sent
            role: Message role, defaults to "user"
            request_id: JSON-RPC request id, auto-generated if omitted
            method: JSON-RPC method name. Defaults to "SendMessage"; the router
                also accepts "message/send" and "tasks/send" for spec drift.
        """
        if not text or not text.strip():
            raise ValueError("send_message requires non-empty text")

        rpc_id = request_id or f"req-{uuid4()}"
        rpc_req = JsonRpcRequest(
            id=rpc_id,
            method=method,
            params={
                "message": {
                    "role": role,
                    "parts": [{"text": text, "kind": "text"}],
                }
            },
        )

        client = await self._client()
        try:
            resp = await client.post(
                f"{self._base_url}/a2a/agents/{agent_id}/rpc",
                json=rpc_req.model_dump(by_alias=True, exclude_none=True),
            )
        except httpx.RequestError as exc:
            raise A2AClientError(f"network error contacting {self._base_url}: {exc}") from exc

        # 4xx still carries a JSON-RPC error envelope; let _parse_rpc handle it
        if resp.status_code >= 500:
            self._raise_for_http(resp)

        try:
            body = resp.json()
        except ValueError as exc:
            raise A2AClientError(
                f"invalid JSON in A2A response (status {resp.status_code}): {exc}",
                status_code=resp.status_code,
            ) from exc

        return self._parse_rpc_response(body, status_code=resp.status_code)

    # ─── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _raise_for_http(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise A2AClientError(
                f"HTTP {resp.status_code} from A2A endpoint: {detail}",
                status_code=resp.status_code,
            )

    @staticmethod
    def _parse_rpc_response(body: dict[str, Any], *, status_code: int) -> TaskResult:
        try:
            rpc_resp = JsonRpcResponse.model_validate(body)
        except Exception as exc:
            raise A2AClientError(
                f"malformed JSON-RPC response envelope: {exc}",
                status_code=status_code,
            ) from exc

        if rpc_resp.error is not None:
            raise A2AClientError(
                f"A2A agent returned error {rpc_resp.error.code}: {rpc_resp.error.message}",
                code=rpc_resp.error.code,
                status_code=status_code,
            )

        result = rpc_resp.result or {}
        task_payload = result.get("task")
        if not isinstance(task_payload, dict):
            raise A2AClientError(
                "A2A response missing result.task",
                status_code=status_code,
            )
        try:
            task = Task.model_validate(task_payload)
        except Exception as exc:
            raise A2AClientError(
                f"malformed task payload in A2A response: {exc}",
                status_code=status_code,
            ) from exc
        return TaskResult(task=task)
