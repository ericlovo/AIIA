"""
FastAPI router that exposes the A2A registry over HTTP.

Endpoints:
    GET  /a2a/agents/{agent_id}/.well-known/agent.json
    POST /a2a/agents/{agent_id}/rpc
    GET  /a2a/registry/agents
    GET  /a2a/registry/agents/{agent_id}
    GET  /a2a/health

The router is instantiated via build_router(registry) so the registry is
injected rather than imported. That keeps the a2a module independent of
local_api and makes the router unit-testable with a stub registry.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from local_brain.a2a.registry import AgentRegistry
from local_brain.a2a.schema import (
    Artifact,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

logger = logging.getLogger("aplora.a2a.router")

JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


def build_router(registry: AgentRegistry) -> APIRouter:
    router = APIRouter(prefix="/a2a", tags=["a2a"])

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "online",
            "agents_registered": len(registry),
            "protocol": "A2A JSONRPC 1.0",
        }

    @router.get("/registry/agents")
    async def list_agents(
        tag: list[str] | None = Query(default=None),
        require_all: bool = Query(default=False),
    ) -> dict[str, Any]:
        matches = registry.query(tags=tag, require_all=require_all)
        return {
            "count": len(matches),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "name": a.card.name,
                    "description": a.card.description,
                    "tags": a.card.all_tags(),
                    "card_url": f"/a2a/agents/{a.agent_id}/.well-known/agent.json",
                    "rpc_url": f"/a2a/agents/{a.agent_id}/rpc",
                }
                for a in matches
            ],
        }

    @router.get("/registry/agents/{agent_id}")
    async def get_agent(agent_id: str) -> dict[str, Any]:
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"unknown agent: {agent_id}")
        return agent.card.model_dump(by_alias=True, exclude_none=True)

    @router.get("/agents/{agent_id}/.well-known/agent.json")
    async def agent_card(agent_id: str) -> JSONResponse:
        agent = registry.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"unknown agent: {agent_id}")
        return JSONResponse(
            content=agent.card.model_dump(by_alias=True, exclude_none=True),
            media_type="application/json",
        )

    @router.post("/agents/{agent_id}/rpc")
    async def agent_rpc(agent_id: str, request: Request) -> JSONResponse:
        agent = registry.get(agent_id)
        if agent is None:
            return _rpc_error(
                None,
                JSONRPC_METHOD_NOT_FOUND,
                f"unknown agent: {agent_id}",
                status_code=404,
            )

        try:
            payload = await request.json()
        except Exception:
            return _rpc_error(None, JSONRPC_INVALID_PARAMS, "request body must be JSON")

        try:
            rpc_req = JsonRpcRequest(**payload)
        except Exception as exc:
            return _rpc_error(
                payload.get("id") if isinstance(payload, dict) else None,
                JSONRPC_INVALID_PARAMS,
                f"invalid JSON-RPC envelope: {exc}",
            )

        if rpc_req.method not in ("SendMessage", "message/send", "tasks/send"):
            return _rpc_error(
                rpc_req.id,
                JSONRPC_METHOD_NOT_FOUND,
                f"unsupported method: {rpc_req.method}",
            )

        try:
            message = _extract_message(rpc_req.params)
        except ValueError as exc:
            return _rpc_error(rpc_req.id, JSONRPC_INVALID_PARAMS, str(exc))

        task = Task(status=TaskStatus(state=TaskState.SUBMITTED))
        task.history.append(message)
        task.status = TaskStatus(state=TaskState.WORKING)

        try:
            result = await agent.executor.execute(message)
        except Exception as exc:
            logger.exception("executor failed for agent %s", agent_id)
            task.status = TaskStatus(state=TaskState.FAILED, message=str(exc))
            response = JsonRpcResponse(
                id=rpc_req.id,
                result={"task": task.model_dump(by_alias=True, exclude_none=True)},
            )
            return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))

        task.artifacts.append(
            Artifact(
                name=result.artifact_name,
                description=result.artifact_description,
                parts=result.parts,
            )
        )
        task.status = TaskStatus(state=TaskState.COMPLETED)

        response = JsonRpcResponse(
            id=rpc_req.id,
            result={"task": task.model_dump(by_alias=True, exclude_none=True)},
        )
        return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))

    return router


def _extract_message(params: dict[str, Any]) -> Message:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    message_payload = params.get("message")
    if not isinstance(message_payload, dict):
        raise ValueError("params.message is required and must be an object")

    role = message_payload.get("role", "user")
    raw_parts = message_payload.get("parts") or []
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("params.message.parts must be a non-empty array")

    text_parts: list[TextPart] = []
    for raw in raw_parts:
        if not isinstance(raw, dict):
            raise ValueError("each part must be an object")
        text = raw.get("text")
        if not isinstance(text, str):
            raise ValueError("MVP only supports text parts with a 'text' field")
        text_parts.append(TextPart(text=text))

    return Message(
        role=role,
        parts=text_parts,
        message_id=message_payload.get("messageId", message_payload.get("message_id", None))
        or Message.model_fields["message_id"].default_factory(),
        context_id=message_payload.get("contextId") or message_payload.get("context_id"),
    )


def _rpc_error(
    request_id: Any,
    code: int,
    message: str,
    *,
    status_code: int = 200,
) -> JSONResponse:
    response = JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(code=code, message=message),
    )
    return JSONResponse(
        content=response.model_dump(by_alias=True, exclude_none=True),
        status_code=status_code,
    )
