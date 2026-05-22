"""
Bootstrap — constructs the default A2A agent registry for the Local Brain.

Called from local_api.py startup. Any dependencies the executors need
(Ollama client, AIIA getter) are passed in explicitly so this module
does not import local_api — preventing circular imports and keeping the
dependency arrow pointed one way.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from local_brain.a2a.executors.aiia_executor import AIIAExecutor
from local_brain.a2a.executors.aiia_memory_executor import AIIAMemoryExecutor
from local_brain.a2a.executors.aiia_status_executor import AIIAStatusExecutor
from local_brain.a2a.executors.subprocess_executor import SubprocessExecutor
from local_brain.a2a.registry import AgentRegistry
from local_brain.a2a.schema import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)

logger = logging.getLogger("aplora.a2a.bootstrap")

AIIAGetter = Callable[[], Awaitable[Any]]


def _find_brain_cli() -> str | None:
    """Locate the brain CLI, with a fallback to the standard install location.

    shutil.which() depends on PATH, which under launchd on macOS doesn't
    include the user's home directory unless explicitly set in the plist.
    The brain CLI is conventionally installed at ~/aplora-local-brain/brain
    by the Mini setup scripts, so fall back to that absolute path if the
    PATH lookup fails. Without this fallback, the three brain-CLI-dependent
    A2A agents (brain-status, review-security, generate-report) silently
    skip registration when the brain service is started by launchd.
    """
    found = shutil.which("brain")
    if found:
        return found
    fallback = Path.home() / "aplora-local-brain" / "brain"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    return None


def register_default_agents(
    *,
    base_url: str = "http://localhost:8100",
    aiia_getter: AIIAGetter | None = None,
) -> AgentRegistry:
    """
    Build a fresh registry and populate it with the default agent set.

    Args:
        base_url: The externally-visible URL where this Local Brain is
            reachable. Used to stamp AgentInterface.url so cards advertise
            the right endpoint. Should be the Tailscale URL in production
            (e.g., http://mac-mini.tail:8100).
        aiia_getter: Async callable that resolves to the AIIA singleton.
            Typically local_api._ensure_aiia. If None, the AIIA agent is
            skipped with a log warning.
    """
    registry = AgentRegistry()

    _maybe_register_brain_status(registry, base_url)
    _maybe_register_review_security(registry, base_url)
    _maybe_register_generate_report(registry, base_url)
    _register_run_tests(registry, base_url)

    if aiia_getter is not None:
        _register_aiia_ask(registry, base_url, aiia_getter)
        _register_aiia_remember(registry, base_url, aiia_getter)
        _register_aiia_search(registry, base_url, aiia_getter)
        _register_aiia_status(registry, base_url, aiia_getter)
        _register_aiia_log_story(registry, base_url, aiia_getter)
    else:
        logger.warning("aiia_getter not provided; aiia-* agents will not be registered")

    logger.info("A2A registry bootstrapped with %d agents", len(registry))
    return registry


def _maybe_register_brain_status(registry: AgentRegistry, base_url: str) -> None:
    brain_path = _find_brain_cli()
    if brain_path is None:
        logger.warning(
            "brain CLI not found on PATH or at ~/aplora-local-brain/brain; "
            "skipping brain-status agent"
        )
        return

    agent_id = "brain-status"
    card = AgentCard(
        name="Local Brain Status",
        description=(
            "Reports health of Aplora local infrastructure: Ollama, Brain API, "
            "Command Center, and recent nightly automation jobs."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="check-services",
                name="Check service health",
                description=(
                    "Returns status of Ollama, Brain API, Command Center, "
                    "and recent nightly jobs (security scan, memory sync, daily report)."
                ),
                tags=["layer:dev-tools", "scope:global", "domain:infrastructure"],
                examples=[
                    "Is the Mini healthy?",
                    "Run brain status and report any warnings.",
                ],
            ),
        ],
    )
    executor = SubprocessExecutor(
        command=[brain_path, "status"],
        timeout_seconds=15.0,
        artifact_name="brain-status-output",
    )
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s (%s)", agent_id, brain_path)


def _register_aiia_ask(
    registry: AgentRegistry,
    base_url: str,
    aiia_getter: AIIAGetter,
) -> None:
    agent_id = "aiia-ask"
    card = AgentCard(
        name="AIIA",
        description=(
            "AIIA (AI Information Architecture) — persistent AI teammate on the "
            "Mac Mini. Answers questions using ChromaDB knowledge, structured "
            "memory, and the local LLM. Every task routed here gets AIIA's full "
            "context, not a raw LLM call."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="ask",
                name="Ask AIIA",
                description=(
                    "Natural-language question answering backed by AIIA's knowledge "
                    "base (5,500+ indexed repo docs) and structured memory. Local LLM."
                ),
                tags=[
                    "layer:aiia",
                    "scope:global",
                    "domain:knowledge",
                    "domain:memory",
                ],
                examples=[
                    "What did I decide about the deployment strategy last quarter?",
                    "Summarize the guardrails architecture for our data pipelines.",
                    "What did we learn from the last incident retrospective?",
                ],
            ),
        ],
    )
    executor = AIIAExecutor(aiia_getter=aiia_getter)
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s", agent_id)


def _register_aiia_remember(
    registry: AgentRegistry,
    base_url: str,
    aiia_getter: AIIAGetter,
) -> None:
    agent_id = "aiia-remember"
    card = AgentCard(
        name="AIIA Memory Writer",
        description=(
            "Writes a fact into AIIA's persistent memory. The fact is stored "
            "in structured memory, indexed in the knowledge store, synced to "
            "the cloud backup, and mirrored into the Obsidian vault. Prefix "
            "your message with 'category:' to route it (decisions, patterns, "
            "lessons, team_knowledge, agents, sessions, wip)."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="remember",
                name="Remember a fact",
                description=(
                    "Persist a fact to AIIA's memory across all future sessions. "
                    "Returns a confirmation with the routed category."
                ),
                tags=[
                    "layer:aiia",
                    "scope:global",
                    "domain:memory",
                    "action:write",
                ],
                examples=[
                    "decisions: We pinned ChromaDB to 0.5.23 for compatibility.",
                    "patterns: Path-scoped rules live in .claude/rules/<glob>.md.",
                    "Eric prefers tables in markdown over bullet lists.",
                ],
            ),
        ],
    )
    executor = AIIAMemoryExecutor(aiia_getter=aiia_getter, mode="remember")
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s", agent_id)


def _register_aiia_search(
    registry: AgentRegistry,
    base_url: str,
    aiia_getter: AIIAGetter,
) -> None:
    agent_id = "aiia-search"
    card = AgentCard(
        name="AIIA Knowledge Search",
        description=(
            "Vector-searches AIIA's knowledge store and returns the top "
            "matching document chunks. No LLM reasoning — much faster than "
            "aiia-ask. Use for raw document discovery; use aiia-ask when "
            "you need a synthesized answer."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="search",
                name="Search knowledge store",
                description=(
                    "Returns the top N matching chunks from AIIA's vector "
                    "store. Each result includes source, score, and a "
                    "truncated text preview."
                ),
                tags=[
                    "layer:aiia",
                    "scope:global",
                    "domain:knowledge",
                    "domain:memory",
                    "action:read",
                ],
                examples=[
                    "tenant configuration",
                    "Dockerfile patterns for Render deployment",
                    "rate limiter implementation",
                ],
            ),
        ],
    )
    executor = AIIAMemoryExecutor(aiia_getter=aiia_getter, mode="search")
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s", agent_id)


def _register_run_tests(registry: AgentRegistry, base_url: str) -> None:
    """
    Run the project test suite as an A2A agent.

    Uses sys.executable so we always invoke pytest with the same Python
    (and therefore same venv) the Local Brain itself is running in. Avoids
    PATH dependency on a 'pytest' shim.
    """
    agent_id = "run-tests"
    card = AgentCard(
        name="Test Suite Runner",
        description=(
            "Runs the project pytest suite via the Local Brain's Python. "
            "Stops on first failure and returns a short traceback so a "
            "calling agent can decide whether to investigate or proceed."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="run-pytest",
                name="Run pytest",
                description=(
                    "Executes pytest with -x --tb=short. Returns stdout. "
                    "The input message is currently ignored — this is the "
                    "MVP shape; later we can route the message to a -k filter."
                ),
                tags=[
                    "layer:dev-tools",
                    "scope:global",
                    "domain:testing",
                    "action:execute",
                ],
                examples=[
                    "Run the test suite.",
                    "Are tests green?",
                ],
            ),
        ],
    )
    executor = SubprocessExecutor(
        command=[sys.executable, "-m", "pytest", "-x", "--tb=short"],
        timeout_seconds=300.0,
        artifact_name="pytest-output",
    )
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s", agent_id)


def _maybe_register_review_security(registry: AgentRegistry, base_url: str) -> None:
    """
    Wraps `brain scan -q` (the quick security scan: secrets + deps).
    Skipped if the brain CLI is not findable.
    """
    brain_path = _find_brain_cli()
    if brain_path is None:
        logger.warning(
            "brain CLI not found on PATH or at ~/aplora-local-brain/brain; "
            "skipping review-security agent"
        )
        return

    agent_id = "review-security"
    card = AgentCard(
        name="Security Review",
        description=(
            "Runs the quick security scan suite (trufflehog secrets + trivy "
            "dependency CVEs). Returns the summary report. Use the slower "
            "full scan offline; this one is for CI-style spot checks."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="quick-scan",
                name="Quick security scan",
                description=(
                    "Runs `brain scan -q` — secrets + dependency CVEs only, "
                    "skipping the heavier SAST and container passes."
                ),
                tags=[
                    "layer:dev-tools",
                    "scope:global",
                    "domain:security",
                    "action:execute",
                ],
                examples=[
                    "Are there any new secrets in the tree?",
                    "Quick security check.",
                ],
            ),
        ],
    )
    executor = SubprocessExecutor(
        command=[brain_path, "scan", "-q"],
        timeout_seconds=300.0,
        artifact_name="security-scan-output",
    )
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s (%s)", agent_id, brain_path)


def _maybe_register_generate_report(registry: AgentRegistry, base_url: str) -> None:
    """
    Wraps `brain report` (today's shipped code report).
    Skipped if the brain CLI is not findable.
    """
    brain_path = _find_brain_cli()
    if brain_path is None:
        logger.warning(
            "brain CLI not found on PATH or at ~/aplora-local-brain/brain; "
            "skipping generate-report agent"
        )
        return

    agent_id = "generate-report"
    card = AgentCard(
        name="Daily Code Report",
        description=(
            "Generates today's shipped-code report — git log analysis grouped "
            "by product, with commit summaries and author attribution. Use "
            "this when you need to know what changed in the last 24 hours."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="todays-report",
                name="Today's shipped code",
                description=(
                    "Runs `brain report` and returns the generated summary. "
                    "Input message currently ignored — future versions can "
                    "route a date string to `brain report YYYY-MM-DD`."
                ),
                tags=[
                    "layer:dev-tools",
                    "scope:global",
                    "domain:reporting",
                    "action:read",
                ],
                examples=[
                    "What did we ship today?",
                    "Summarize the last 24 hours of commits.",
                ],
            ),
        ],
    )
    executor = SubprocessExecutor(
        command=[brain_path, "report"],
        timeout_seconds=120.0,
        artifact_name="daily-report-output",
    )
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s (%s)", agent_id, brain_path)


def _register_aiia_status(
    registry: AgentRegistry,
    base_url: str,
    aiia_getter: AIIAGetter,
) -> None:
    """
    AIIA status dashboard — knowledge counts, memory stats, model info.
    No LLM call — just metadata reads. Fast and deterministic.
    """
    agent_id = "aiia-status"
    card = AgentCard(
        name="AIIA Status",
        description=(
            "Returns AIIA's current health: identity, model, knowledge doc "
            "count, memory stats by category, and Supermemory connection. "
            "No LLM call — instant metadata read."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="status",
                name="AIIA health and stats",
                description=(
                    "Returns identity, model, knowledge doc count, memory "
                    "breakdown by category, and connection statuses."
                ),
                tags=[
                    "layer:aiia",
                    "scope:global",
                    "domain:monitoring",
                    "action:read",
                ],
                examples=[
                    "How many docs does AIIA have?",
                    "AIIA status check",
                    "How many memories are stored?",
                ],
            ),
        ],
    )
    executor = AIIAStatusExecutor(aiia_getter=aiia_getter)
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s", agent_id)


def _register_aiia_log_story(
    registry: AgentRegistry,
    base_url: str,
    aiia_getter: AIIAGetter,
) -> None:
    """
    Log a story/idea to the roadmap backlog via AIIA memory.

    Uses the memory executor in "remember" mode with category routing.
    Input format: "project: Story title — description"
    The memory executor handles category parsing via the "category: text"
    wire format.
    """
    agent_id = "aiia-log-story"
    card = AgentCard(
        name="AIIA Story Logger",
        description=(
            "Logs a feature idea, bug, or task to the roadmap backlog via "
            "AIIA memory. Write stories as: 'project: title — description'. "
            "AIIA deduplicates against existing backlog items."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Aplora AI", url="https://aplora.ai"),
        supported_interfaces=[AgentInterface(url=f"{base_url}/a2a/agents/{agent_id}/rpc")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="log-story",
                name="Log a backlog story",
                description=(
                    "Persists a story/idea to the roadmap backlog via AIIA's "
                    "structured memory. Category is auto-set to 'project'."
                ),
                tags=[
                    "layer:aiia",
                    "scope:global",
                    "domain:planning",
                    "action:write",
                ],
                examples=[
                    "project: Add WebAuthn passkey support as alternative 2FA factor",
                    "project: CK Marketing — automate monthly scorecard PDF generation",
                ],
            ),
        ],
    )
    executor = AIIAMemoryExecutor(
        aiia_getter=aiia_getter,
        mode="remember",
        default_category="project",
        default_source="a2a-story",
    )
    registry.register(agent_id, card, executor)
    logger.info("registered A2A agent: %s", agent_id)
