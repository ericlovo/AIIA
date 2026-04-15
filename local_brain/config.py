"""
Local Brain Configuration

All settings for the Mac Mini intelligence node.
Configured via environment variables with sensible defaults.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific local model role."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    description: str = ""


@dataclass
class Gemma4Capabilities:
    """Feature capabilities for Gemma 4 family models.

    Populated from the routing model name at config load time. Used by
    higher-level modules (SmartConductor, A2A executors, voice handler)
    to decide whether to exercise Gemma 4-specific code paths or fall
    back to the generic JSON-parsing approach.

    Note on native_function_calling:
        Gemma 4 E4B advertises native function calling support, and
        Ollama forwards tool schemas to the model. In practice, smoke
        tests against `gemma4:e4b` via Ollama show the model returns
        empty content with no `tool_calls` emitted — the Modelfile
        template for E4B tool-use appears incomplete upstream as of
        April 2026. Until Ollama ships proper tool-use templating for
        this model family, native_function_calling defaults to False
        regardless of the model name. Operators can force-enable it
        via AIIA_NATIVE_TOOLS_ENABLED=true once upstream is fixed.

    The V1 JSON routing path (what SmartConductor uses today) is
    reliable and does not depend on native tool calling, so the
    default-off posture is safe.
    """

    native_function_calling: bool = False
    native_audio_input: bool = False
    thinking_mode: bool = False
    native_system_prompt: bool = False

    @staticmethod
    def detect(model_name: str) -> "Gemma4Capabilities":
        """Detect capabilities for a given Gemma 4 model variant.

        native_function_calling additionally requires the
        AIIA_NATIVE_TOOLS_ENABLED env var to be explicitly set to
        "true" — see class docstring for why.
        """
        lower = model_name.lower()
        is_e4b = "e4b" in lower
        is_gemma4 = lower.startswith("gemma4") or lower.startswith("gemma-4")
        native_tools_opt_in = os.getenv("AIIA_NATIVE_TOOLS_ENABLED", "false").lower() == "true"
        return Gemma4Capabilities(
            native_function_calling=is_e4b and native_tools_opt_in,
            native_audio_input=is_e4b,
            thinking_mode=is_gemma4,
            native_system_prompt=is_gemma4,
        )


@dataclass
class AutonomyConfig:
    """
    Phase 2 autonomy settings. All features gated behind explicit flags.
    Ships disabled — set AIIA_AUTONOMY_LEVEL=phase2 to enable.

    Environment variables:
        AIIA_AUTONOMY_LEVEL: "phase1" (default) or "phase2" (enables proactive features)
        AIIA_AUTONOMY_MAX_SEVERITY: Max severity auto-downgraded from GATED (default: low)
        AIIA_BUSINESS_HOURS_TZ: IANA timezone for business-hours check (default: UTC)
        AIIA_PROACTIVE_HEALTH_CHECK_URL: Optional prod health probe before proactive actions
        AIIA_SERVICES_CONFIG: Path to JSON file listing services for SelfHealingMonitor
    """

    # Master switch — "phase1" keeps current behavior, "phase2" enables proactive features
    level: str = "phase1"

    # Proactive story execution — auto-decomposes and queues P0/P1 stories outside business hours
    proactive_story_execution: bool = False
    proactive_priorities: List[str] = field(default_factory=lambda: ["P0", "P1"])
    proactive_min_confidence: float = 0.85
    proactive_business_hours_start: int = 9  # 9am local
    proactive_business_hours_end: int = 18  # 6pm local
    proactive_business_hours_tz: str = "UTC"
    # Optional URL probed before proactive execution. None = skip the gate.
    proactive_health_check_url: Optional[str] = None

    # GATED auto-downgrade — promote stale low-severity GATED actions to SUPERVISED
    gated_downgrade_enabled: bool = False
    gated_downgrade_hours: int = 48
    gated_downgrade_max_severity: str = "low"

    # Self-healing production monitor — detects unhealthy services, attempts known fixes
    self_healing_enabled: bool = False
    self_healing_check_seconds: int = 300  # 5 minutes
    self_healing_max_attempts: int = 2
    # Services to monitor — list of {"name": str, "url": str}. Loaded from
    # the JSON file at AIIA_SERVICES_CONFIG. Empty by default: operator
    # must opt in to monitor anything.
    monitored_services: List[Dict[str, str]] = field(default_factory=list)

    # Memory quality loop — local consolidation + dedup + ChromaDB promotion
    memory_quality_enabled: bool = False
    memory_quality_interval_hours: int = 6
    memory_quality_threshold: float = 0.80
    memory_quality_max_promotions: int = 50

    # Hard safety boundaries — never overridden by autonomy
    forbidden_files: List[str] = field(
        default_factory=lambda: [
            ".env",
            "secrets.json",
            "*.pem",
            "*.key",
            "render.yaml",
            "docker-compose.prod.yml",
        ]
    )
    forbidden_actions: List[str] = field(
        default_factory=lambda: [
            "delete_database",
            "modify_production_secrets",
            "push_to_main_without_pr",
            "disable_auth",
        ]
    )


@dataclass
class LocalBrainConfig:
    """
    Configuration for the Local Brain running on Mac Mini.

    Environment variables:
        LOCAL_LLM_URL: Ollama base URL (default: http://localhost:11434)
        LOCAL_BRAIN_PORT: Port for Local Brain API (default: 8100)
        LOCAL_BRAIN_HOST: Host to bind to (default: 0.0.0.0)
        LOCAL_BRAIN_API_KEY: API key for securing the Local Brain endpoint
        LOCAL_ROUTING_MODEL: Model for smart conductor routing (default: llama3.1:8b)
        LOCAL_TASK_MODEL: Model for general tasks (default: llama3.1:8b)
        LOCAL_DEEP_MODEL: Model for deep reasoning — nightly workers (default: deepseek-r1:14b)
        LOCAL_EMBED_MODEL: Model for embeddings (default: nomic-embed-text)
        EQ_BRAIN_ENABLED: Enable EQ Brain persistent memory (default: true)
        EQ_BRAIN_DATA_DIR: Directory for EQ Brain data (default: ~/.aiia/eq_data)
        EXECUTION_ENABLED: Enable execution engine (default: false)
        EXECUTION_POLL_INTERVAL: Seconds between polling for approved actions (default: 15)
        EXECUTION_MAX_TIMEOUT: Max seconds per action execution (default: 600)
        EXECUTION_MAX_RETRIES: Retry count for failed executions (default: 2)
        EXECUTION_AUTO_COMMIT: Auto-commit fixes to aiia/* branches (default: false)
        EXECUTION_BRANCH_PREFIX: Git branch prefix (default: aiia/)
        EXECUTION_DATA_DIR: Data dir (default: ~/.aiia/eq_data/execution)
        CLAUDE_CODE_PATH: Path to claude CLI binary (default: claude)
        EXECUTION_MAX_CONCURRENT: Max concurrent subprocesses (default: 1)
        EXECUTION_MAX_FILES_PER_ACTION: Safety limit on files per action (default: 20)
        EXECUTION_SUPERVISED_COUNTDOWN: Seconds before supervised actions execute (default: 30)
    """

    # Ollama connection
    ollama_url: str = ""
    ollama_timeout: float = 120.0  # Local models can take a moment on first load

    # Local Brain API server
    api_host: str = "0.0.0.0"  # nosec B104
    api_port: int = 8100
    api_key: Optional[str] = None  # Set to require auth from production backend

    # Model assignments — which model handles what
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Gemma 4 capability flags (populated in __post_init__ based on
    # the routing model name). Consumers consult this instead of
    # inspecting the model name themselves.
    primary_capabilities: Gemma4Capabilities = field(default_factory=Gemma4Capabilities)

    # Phase 2 autonomy configuration (gated by AIIA_AUTONOMY_LEVEL)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)

    # AIIA — persistent AI teammate (knowledge + memory)
    eq_brain_enabled: bool = True  # Config key kept for backward compat
    eq_brain_data_dir: str = ""  # Set from env or default to ~/.aiia/eq_data
    eq_brain_collection: str = "aiia_knowledge"

    # Supermemory cloud sync — SDK removed (April 2026), hardcoded off
    supermemory_enabled: bool = False
    supermemory_timeout: float = 8.0

    # Hybrid cloud memory — disabled with Supermemory removal
    hybrid_cloud_enabled: bool = False
    hybrid_cloud_timeout: float = 8.0

    # Metered sync tuning
    sync_quality_gate: int = 3  # env: SYNC_QUALITY_GATE (min score to sync)
    sync_daily_budget: int = 200_000  # env: SYNC_DAILY_BUDGET
    sync_project_excluded_sources: str = (
        "health_journal,code_health,test_run,security_scan"  # env: SYNC_PROJECT_EXCLUDED
    )

    # Recursive inference engine (Phase 4 — RLM-inspired)
    recursive_max_iterations: int = 15  # env: RECURSIVE_MAX_ITERATIONS
    recursive_max_depth: int = 3  # env: RECURSIVE_MAX_DEPTH
    recursive_token_budget: int = 50_000  # env: RECURSIVE_TOKEN_BUDGET
    recursive_temperature: float = 0.15  # Low temp for reliable JSON output

    # Obsidian vault sync (VaultWriter) — optional knowledge vault export
    # Resolution in local_brain/vault_paths.py — set OBSIDIAN_VAULT_DIR to
    # override the ~/Documents/AIIA → ~/.aiia/vault fallback chain.
    vault_dir: str = ""
    auto_file_queries: bool = True  # File substantive AIIA answers to wiki/

    # Feature flags
    smart_routing_enabled: bool = True  # Use local LLM for conductor routing
    summarization_enabled: bool = True  # Handle summarization locally
    memory_extraction_enabled: bool = True  # Extract memories locally
    pii_scanning_enabled: bool = True  # PII/PHI detection locally
    embeddings_enabled: bool = True  # Generate embeddings locally

    # Execution engine
    execution_enabled: bool = False  # Off by default
    execution_poll_interval: int = 15  # seconds between checking for approved actions
    execution_max_timeout: int = 600  # 10 min max per action execution
    execution_max_retries: int = 2  # retry failed executions
    execution_auto_commit: bool = False  # auto-commit fixes to aiia/* branches
    execution_branch_prefix: str = "aiia/"  # git branch naming
    execution_data_dir: str = ""  # set in __post_init__
    claude_code_path: str = "claude"  # path to claude CLI binary
    execution_max_concurrent: int = 1  # max concurrent subprocesses
    execution_max_files_per_action: int = 20  # safety: max files an action can touch
    execution_supervised_countdown: int = 30  # seconds before supervised actions execute

    def __post_init__(self):
        """Load from environment variables."""
        self.ollama_url = self.ollama_url or os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        self.api_host = os.getenv("LOCAL_BRAIN_HOST", self.api_host)
        self.api_port = int(os.getenv("LOCAL_BRAIN_PORT", str(self.api_port)))
        self.api_key = os.getenv("LOCAL_BRAIN_API_KEY", self.api_key)

        # EQ Brain
        self.eq_brain_enabled = os.getenv("EQ_BRAIN_ENABLED", "true").lower() == "true"
        self.eq_brain_data_dir = os.getenv(
            "EQ_BRAIN_DATA_DIR",
            os.path.expanduser("~/.aiia/eq_data"),
        )
        self.eq_brain_collection = os.getenv("EQ_BRAIN_COLLECTION", self.eq_brain_collection)

        # Supermemory cloud sync — disabled by default (optional integration)
        self.supermemory_enabled = os.getenv("SUPERMEMORY_ENABLED", "false").lower() == "true"
        self.supermemory_timeout = float(
            os.getenv("SUPERMEMORY_TIMEOUT", str(self.supermemory_timeout))
        )

        # Hybrid cloud memory
        self.hybrid_cloud_enabled = os.getenv("HYBRID_CLOUD_ENABLED", "false").lower() == "true"
        self.hybrid_cloud_timeout = float(
            os.getenv("HYBRID_CLOUD_TIMEOUT", str(self.hybrid_cloud_timeout))
        )

        # Obsidian vault — resolution centralized in local_brain/vault_paths.py.
        # Honors OBSIDIAN_VAULT_DIR; falls back to ~/Documents/AIIA if it
        # exists, else ~/.aiia/vault. See vault_paths.vault_dir() for details.
        from local_brain.vault_paths import vault_dir

        self.vault_dir = str(vault_dir())
        self.auto_file_queries = os.getenv("AUTO_FILE_QUERIES", "true").lower() == "true"

        # Metered sync tuning
        self.sync_quality_gate = int(os.getenv("SYNC_QUALITY_GATE", str(self.sync_quality_gate)))
        self.sync_daily_budget = int(os.getenv("SYNC_DAILY_BUDGET", str(self.sync_daily_budget)))
        self.sync_project_excluded_sources = os.getenv(
            "SYNC_PROJECT_EXCLUDED", self.sync_project_excluded_sources
        )

        # Recursive inference engine
        self.recursive_max_iterations = int(
            os.getenv("RECURSIVE_MAX_ITERATIONS", str(self.recursive_max_iterations))
        )
        self.recursive_max_depth = int(
            os.getenv("RECURSIVE_MAX_DEPTH", str(self.recursive_max_depth))
        )
        self.recursive_token_budget = int(
            os.getenv("RECURSIVE_TOKEN_BUDGET", str(self.recursive_token_budget))
        )

        # Execution engine
        _exec_enabled = os.getenv("EXECUTION_ENABLED", "false").lower()
        self.execution_enabled = _exec_enabled in ("true", "1")
        self.execution_poll_interval = int(
            os.getenv("EXECUTION_POLL_INTERVAL", str(self.execution_poll_interval))
        )
        self.execution_max_timeout = int(
            os.getenv("EXECUTION_MAX_TIMEOUT", str(self.execution_max_timeout))
        )
        self.execution_max_retries = int(
            os.getenv("EXECUTION_MAX_RETRIES", str(self.execution_max_retries))
        )
        _auto_commit = os.getenv("EXECUTION_AUTO_COMMIT", "false").lower()
        self.execution_auto_commit = _auto_commit in ("true", "1")
        self.execution_branch_prefix = os.getenv(
            "EXECUTION_BRANCH_PREFIX", self.execution_branch_prefix
        )
        self.execution_data_dir = os.getenv(
            "EXECUTION_DATA_DIR",
            os.path.join(
                os.path.expanduser("~"),
                ".aiia",
                "eq_data",
                "execution",
            ),
        )
        self.claude_code_path = os.getenv("CLAUDE_CODE_PATH", self.claude_code_path)
        self.execution_max_concurrent = int(
            os.getenv(
                "EXECUTION_MAX_CONCURRENT",
                str(self.execution_max_concurrent),
            )
        )
        self.execution_max_files_per_action = int(
            os.getenv(
                "EXECUTION_MAX_FILES_PER_ACTION",
                str(self.execution_max_files_per_action),
            )
        )
        self.execution_supervised_countdown = int(
            os.getenv(
                "EXECUTION_SUPERVISED_COUNTDOWN",
                str(self.execution_supervised_countdown),
            )
        )
        self.execution_max_concurrent = int(
            os.getenv(
                "EXECUTION_MAX_CONCURRENT",
                str(self.execution_max_concurrent),
            )
        )
        self.execution_max_files_per_action = int(
            os.getenv(
                "EXECUTION_MAX_FILES_PER_ACTION",
                str(self.execution_max_files_per_action),
            )
        )
        self.execution_supervised_countdown = int(
            os.getenv(
                "EXECUTION_SUPERVISED_COUNTDOWN",
                str(self.execution_supervised_countdown),
            )
        )

        # Default model assignments
        if not self.models:
            routing_model = os.getenv("LOCAL_ROUTING_MODEL", "llama3.1:8b-instruct-q8_0")
            task_model = os.getenv("LOCAL_TASK_MODEL", "llama3.1:8b-instruct-q8_0")
            embed_model = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")
            deep_model = os.getenv("LOCAL_DEEP_MODEL", "deepseek-r1:14b")

            self.models = {
                "routing": ModelConfig(
                    model_name=routing_model,
                    temperature=0.1,  # Low temp for consistent classification
                    max_tokens=256,  # Routing responses are short
                    description="Smart Conductor — intent classification and routing",
                ),
                "task": ModelConfig(
                    model_name=task_model,
                    temperature=0.7,
                    max_tokens=4096,
                    description="General task completion — summarization, extraction",
                ),
                "embed": ModelConfig(
                    model_name=embed_model,
                    description="Text embeddings for RAG and similarity search",
                ),
                "pii": ModelConfig(
                    model_name=routing_model,  # Same model, different prompt
                    temperature=0.0,  # Deterministic for compliance
                    max_tokens=512,
                    description="PII/PHI detection and classification",
                ),
                "deep": ModelConfig(
                    model_name=deep_model,
                    temperature=0.6,
                    max_tokens=8192,
                    description="Deep reasoning — consolidation, code review, briefings (nightly)",
                ),
            }

        # Phase 2: Autonomy config from env
        autonomy_level = os.getenv("AIIA_AUTONOMY_LEVEL", "phase1")
        is_phase2 = autonomy_level == "phase2"

        monitored_services: List[Dict[str, str]] = []
        services_config_path = os.getenv("AIIA_SERVICES_CONFIG")
        if services_config_path and os.path.exists(services_config_path):
            try:
                with open(services_config_path) as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    monitored_services = [
                        s for s in loaded if isinstance(s, dict) and "name" in s and "url" in s
                    ]
            except (json.JSONDecodeError, OSError):
                # Malformed config — fall back to empty list, monitor will skip
                pass

        self.autonomy = AutonomyConfig(
            level=autonomy_level,
            proactive_story_execution=is_phase2,
            gated_downgrade_enabled=is_phase2,
            self_healing_enabled=is_phase2,
            memory_quality_enabled=is_phase2,
            gated_downgrade_max_severity=os.getenv("AIIA_AUTONOMY_MAX_SEVERITY", "low"),
            proactive_business_hours_tz=os.getenv("AIIA_BUSINESS_HOURS_TZ", "UTC"),
            proactive_health_check_url=os.getenv("AIIA_PROACTIVE_HEALTH_CHECK_URL"),
            monitored_services=monitored_services,
        )

        # Detect Gemma 4 capabilities from the routing model name.
        # If the operator uses llama3.1 (default) this returns the
        # all-False default Gemma4Capabilities and consumers behave
        # as if no native features are available, which matches how
        # SmartConductor already uses JSON parsing today.
        routing_model_name = self.models.get("routing", ModelConfig("")).model_name or ""
        self.primary_capabilities = Gemma4Capabilities.detect(routing_model_name)


# Singleton
_config: Optional[LocalBrainConfig] = None


def get_config() -> LocalBrainConfig:
    """Get or create the Local Brain config singleton."""
    global _config
    if _config is None:
        _config = LocalBrainConfig()
    return _config
