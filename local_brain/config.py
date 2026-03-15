"""
Local Brain Configuration

All settings for the Mac Mini intelligence node.
Configured via environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


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
        LOCAL_TASK_MODEL: Model for general tasks (default: deepseek-r1)
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

    # AIIA — persistent AI teammate (knowledge + memory)
    eq_brain_enabled: bool = True  # Config key kept for backward compat
    eq_brain_data_dir: str = (
        ""  # Set from env or default to ~/.aiia/eq_data
    )
    eq_brain_collection: str = "aiia_knowledge"

    # Supermemory cloud sync
    supermemory_enabled: bool = True  # env: SUPERMEMORY_ENABLED
    supermemory_timeout: float = 8.0  # env: SUPERMEMORY_TIMEOUT

    # Hybrid cloud memory (AIIA ask() queries Supermemory in parallel)
    hybrid_cloud_enabled: bool = True  # env: HYBRID_CLOUD_ENABLED
    hybrid_cloud_timeout: float = 8.0  # env: HYBRID_CLOUD_TIMEOUT

    # Metered sync tuning
    sync_quality_gate: int = 3  # env: SYNC_QUALITY_GATE (min score to sync)
    sync_daily_budget: int = 200_000  # env: SYNC_DAILY_BUDGET
    sync_project_excluded_sources: str = "health_journal,code_health,test_run,security_scan"  # env: SYNC_PROJECT_EXCLUDED

    # Recursive inference engine (Phase 4 — RLM-inspired)
    recursive_max_iterations: int = 15  # env: RECURSIVE_MAX_ITERATIONS
    recursive_max_depth: int = 3  # env: RECURSIVE_MAX_DEPTH
    recursive_token_budget: int = 50_000  # env: RECURSIVE_TOKEN_BUDGET
    recursive_temperature: float = 0.15  # Low temp for reliable JSON output

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
        self.ollama_url = self.ollama_url or os.getenv(
            "LOCAL_LLM_URL", "http://localhost:11434"
        )
        self.api_host = os.getenv("LOCAL_BRAIN_HOST", self.api_host)
        self.api_port = int(os.getenv("LOCAL_BRAIN_PORT", str(self.api_port)))
        self.api_key = os.getenv("LOCAL_BRAIN_API_KEY", self.api_key)

        # EQ Brain
        self.eq_brain_enabled = os.getenv("EQ_BRAIN_ENABLED", "true").lower() == "true"
        self.eq_brain_data_dir = os.getenv(
            "EQ_BRAIN_DATA_DIR",
            os.path.expanduser("~/.aiia/eq_data"),
        )
        self.eq_brain_collection = os.getenv(
            "EQ_BRAIN_COLLECTION", self.eq_brain_collection
        )

        # Supermemory
        self.supermemory_enabled = (
            os.getenv("SUPERMEMORY_ENABLED", "true").lower() == "true"
        )
        self.supermemory_timeout = float(
            os.getenv("SUPERMEMORY_TIMEOUT", str(self.supermemory_timeout))
        )

        # Hybrid cloud memory
        self.hybrid_cloud_enabled = (
            os.getenv("HYBRID_CLOUD_ENABLED", "true").lower() == "true"
        )
        self.hybrid_cloud_timeout = float(
            os.getenv("HYBRID_CLOUD_TIMEOUT", str(self.hybrid_cloud_timeout))
        )

        # Metered sync tuning
        self.sync_quality_gate = int(
            os.getenv("SYNC_QUALITY_GATE", str(self.sync_quality_gate))
        )
        self.sync_daily_budget = int(
            os.getenv("SYNC_DAILY_BUDGET", str(self.sync_daily_budget))
        )
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
            os.getenv(
                "EXECUTION_POLL_INTERVAL", str(self.execution_poll_interval)
            )
        )
        self.execution_max_timeout = int(
            os.getenv(
                "EXECUTION_MAX_TIMEOUT", str(self.execution_max_timeout)
            )
        )
        self.execution_max_retries = int(
            os.getenv(
                "EXECUTION_MAX_RETRIES", str(self.execution_max_retries)
            )
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
        self.claude_code_path = os.getenv(
            "CLAUDE_CODE_PATH", self.claude_code_path
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
            routing_model = os.getenv(
                "LOCAL_ROUTING_MODEL", "llama3.1:8b-instruct-q8_0"
            )
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
                    model_name=routing_model,  # Same small model, different prompt
                    temperature=0.0,  # Deterministic for compliance
                    max_tokens=512,
                    description="PII/PHI detection and classification",
                ),
                "deep": ModelConfig(
                    model_name=deep_model,
                    temperature=0.6,  # DeepSeek R1 optimal — chain-of-thought is self-correcting
                    max_tokens=8192,
                    description="Deep reasoning — consolidation, code review, briefings (nightly)",
                ),
            }


# Singleton
_config: Optional[LocalBrainConfig] = None


def get_config() -> LocalBrainConfig:
    """Get or create the Local Brain config singleton."""
    global _config
    if _config is None:
        _config = LocalBrainConfig()
    return _config
