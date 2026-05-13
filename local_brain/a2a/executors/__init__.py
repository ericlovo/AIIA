"""A2A executors — the glue between an incoming Message and the work that produces Artifacts."""

from local_brain.a2a.executors.aiia_executor import AIIAExecutor
from local_brain.a2a.executors.aiia_memory_executor import AIIAMemoryExecutor
from local_brain.a2a.executors.base import AgentExecutor, ExecutorResult
from local_brain.a2a.executors.subprocess_executor import SubprocessExecutor

__all__ = [
    "AgentExecutor",
    "ExecutorResult",
    "SubprocessExecutor",
    "AIIAExecutor",
    "AIIAMemoryExecutor",
]
