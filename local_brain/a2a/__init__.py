"""
AIIA A2A (Agent-to-Agent) protocol module.

Exposes internal agents (CLI tools, AIIA, platform services) as A2A-compliant
endpoints that can be discovered and invoked by any A2A client. Mounted onto
the Local Brain FastAPI app on :8100.

Public surface:
    AgentCard, AgentSkill, Message, Part, Task, TaskState  — schema
    AgentRegistry                                          — discovery
    AgentExecutor, SubprocessExecutor, AIIAExecutor,
        AIIAMemoryExecutor                                 — executors
    build_router                                           — FastAPI integration
    register_default_agents                                — startup bootstrap
    A2AClient, A2AClientError, AgentSummary, TaskResult    — outbound client
"""

from local_brain.a2a.bootstrap import register_default_agents
from local_brain.a2a.client import (
    A2AClient,
    A2AClientError,
    AgentSummary,
    TaskResult,
)
from local_brain.a2a.executors.aiia_executor import AIIAExecutor
from local_brain.a2a.executors.aiia_memory_executor import AIIAMemoryExecutor
from local_brain.a2a.executors.base import AgentExecutor
from local_brain.a2a.executors.subprocess_executor import SubprocessExecutor
from local_brain.a2a.registry import AgentRegistry
from local_brain.a2a.router import build_router
from local_brain.a2a.schema import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Task,
    TaskState,
    TaskStatus,
)

__all__ = [
    "AgentCard",
    "AgentSkill",
    "AgentCapabilities",
    "AgentInterface",
    "Message",
    "Part",
    "Task",
    "TaskState",
    "TaskStatus",
    "Artifact",
    "AgentRegistry",
    "AgentExecutor",
    "SubprocessExecutor",
    "AIIAExecutor",
    "AIIAMemoryExecutor",
    "build_router",
    "register_default_agents",
    "A2AClient",
    "A2AClientError",
    "AgentSummary",
    "TaskResult",
]
