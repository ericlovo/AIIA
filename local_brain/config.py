"""
Local Brain Configuration

All settings for the Mac Mini intelligence node.
Configured via environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a specific local model role."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    description: str = ""


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
    api_key: str | None = None  # Set to require auth from production backend

    # Model assignments — which model handles what
    models: dict[str, ModelConfig] = field(default_factory=dict)

    # AIIA — persistent AI teammate (knowledge + memory)
    eq_brain_enabled: bool = True  # Config key kept for backward compat
    eq_brain_data_dir: str = ""  # Set from env or default to ~/.aiia/eq_data
    eq_brain_collection: str = "aiia_knowledge"

    # Recursive inference engine (Phase 4 — RLM-inspired)
    recursive_max_iterations: int = 15  # env: RECURSIVE_MAX_ITERATIONS
    recursive_max_depth: int = 3  # env: RECURSIVE_MAX_DEPTH
    recursive_token_budget: int = 50_000  # env: RECURSIVE_TOKEN_BUDGET
    recursive_temperature: float = 0.15  # Low temp for reliable JSON output

    # Obsidian vault sync (VaultWriter) — optional knowledge vault export
    vault_dir: str = ""  # Set from env OBSIDIAN_VAULT_DIR or default to ~/.aiia/vault
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

        # Obsidian vault — optional knowledge export
        _vault_default = os.path.expanduser("~/.aiia/vault")
        self.vault_dir = os.getenv("OBSIDIAN_VAULT_DIR", _vault_default)
        self.auto_file_queries = os.getenv("AUTO_FILE_QUERIES", "true").lower() == "true"

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


# Singleton
_config: LocalBrainConfig | None = None


def get_config() -> LocalBrainConfig:
    """Get or create the Local Brain config singleton."""
    global _config
    if _config is None:
        _config = LocalBrainConfig()
    return _config


@dataclass
class AutonomyConfig:
    """
    Phase 2 autonomy settings — feature-flagged background loops.

    Ships disabled: `level` defaults to "phase1" (no autonomous action) and
    every loop has its own off-by-default switch on top of that. Set
    `level="phase2"` (env: AIIA_AUTONOMY_LEVEL) plus the per-loop flag to
    arm a loop. Direct construction stays pure (no env reads) so tests can
    pin exact values; use `AutonomyConfig.from_env()` for the env-driven
    instance the runners build.
    """

    # Master switch — "phase1" (disabled) or "phase2" (loops may run)
    level: str = "phase1"

    # Proactive story execution (proactive_executor.py)
    proactive_story_execution: bool = False
    proactive_business_hours_tz: str = "UTC"
    proactive_business_hours_start: int = 9  # hour [0-23], inclusive
    proactive_business_hours_end: int = 18  # hour [0-23], exclusive
    proactive_health_check_url: str | None = None
    proactive_priorities: list[str] = field(default_factory=lambda: ["P0", "P1"])

    # Gated → supervised downgrade (gated_downgrade.py)
    gated_downgrade_enabled: bool = False
    gated_downgrade_hours: int = 24  # staleness cutoff before downgrade
    gated_downgrade_max_severity: str = "low"  # only downgrade up to this severity

    # Self-healing service monitor (self_healing.py)
    self_healing_enabled: bool = False
    monitored_services: list[str] = field(default_factory=list)

    # Memory quality loop (memory_quality.py)
    memory_quality_enabled: bool = False
    memory_quality_threshold: float = 0.7  # min score to promote to knowledge
    memory_quality_max_promotions: int = 10  # budget per cycle

    # Autonomous research loop (research_loop.py)
    research_enabled: bool = False
    research_max_topics_per_cycle: int = 3  # how many topics to work per cycle
    research_sessions_per_topic: int = 1  # sessions to run on each topic

    @classmethod
    def from_env(cls) -> "AutonomyConfig":
        """Build a config from environment variables, defaults otherwise."""

        def _flag(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            return raw.lower() in ("true", "1", "yes") if raw is not None else default

        services_raw = os.getenv("AIIA_SERVICES_CONFIG", "")
        services = [s.strip() for s in services_raw.split(",") if s.strip()]

        return cls(
            level=os.getenv("AIIA_AUTONOMY_LEVEL", cls.level),
            proactive_story_execution=_flag(
                "AIIA_PROACTIVE_STORY_EXECUTION", cls.proactive_story_execution
            ),
            proactive_business_hours_tz=os.getenv(
                "AIIA_BUSINESS_HOURS_TZ", cls.proactive_business_hours_tz
            ),
            proactive_health_check_url=os.getenv("AIIA_HEALTH_CHECK_URL"),
            gated_downgrade_enabled=_flag(
                "AIIA_GATED_DOWNGRADE_ENABLED", cls.gated_downgrade_enabled
            ),
            self_healing_enabled=_flag("AIIA_SELF_HEALING_ENABLED", cls.self_healing_enabled),
            monitored_services=services,
            memory_quality_enabled=_flag(
                "AIIA_MEMORY_QUALITY_ENABLED", cls.memory_quality_enabled
            ),
            research_enabled=_flag("AIIA_RESEARCH_ENABLED", cls.research_enabled),
            research_max_topics_per_cycle=int(
                os.getenv("AIIA_RESEARCH_MAX_TOPICS", cls.research_max_topics_per_cycle)
            ),
            research_sessions_per_topic=int(
                os.getenv("AIIA_RESEARCH_SESSIONS_PER_TOPIC", cls.research_sessions_per_topic)
            ),
        )
