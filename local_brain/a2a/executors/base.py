"""
AgentExecutor protocol — the minimal contract every A2A agent implements.

An executor takes an incoming Message and returns a list of Parts that will
be wrapped as an Artifact on the resulting Task. Keeping the surface this
small means the router owns all protocol machinery (task state, JSON-RPC
envelope, error translation), and executors can focus on doing their job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

from local_brain.a2a.schema import Message, Part


@dataclass
class ExecutorResult:
    """What an executor returns: parts to attach to the task artifact."""

    parts: List[Part]
    artifact_name: str = "result"
    artifact_description: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AgentExecutor(Protocol):
    async def execute(self, message: Message) -> ExecutorResult: ...
