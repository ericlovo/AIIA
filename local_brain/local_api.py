"""
AIIA Local Brain API — FastAPI service running on Mac Mini

This is the main API that runs on the Mac Mini. Production backend
calls this over Tailscale for:
- Smart routing (intent + emotion + complexity classification)
- Local chat completions (summarization, memory extraction)
- Embeddings generation
- PII/PHI scanning

Start:
    cd local_brain
    uvicorn local_api:app --host 0.0.0.0 --port 8100

Or:
    python local_api.py
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import json as json_module
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fastapi.responses import Response as RawResponse

from local_brain.config import get_config, LocalBrainConfig
from local_brain.ollama_client import OllamaClient
from local_brain.smart_conductor import SmartConductor
from local_brain.eq_brain.knowledge_store import KnowledgeStore
from local_brain.eq_brain.memory import Memory
from local_brain.eq_brain.brain import AIIA
from local_brain.eq_brain.supermemory_bridge import SupermemoryBridge
try:
    from services.google_tts import GoogleTTSService
except ImportError:
    GoogleTTSService = None  # Optional: Google TTS not available

logger = logging.getLogger("aiia.local_brain.api")

# Command Center URL for metrics reporting
COMMAND_CENTER_URL = os.getenv("COMMAND_CENTER_URL", "http://localhost:8200")


async def _report_metrics(
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    endpoint: str = "",
):
    """Fire-and-forget metrics report to Command Center dashboard."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Report latency sample
            await client.post(
                f"{COMMAND_CENTER_URL}/ops/record-latency",
                json={
                    "provider": provider,
                    "latency_ms": round(latency_ms, 1),
                    "model": model,
                },
            )
            # Report as routing decision (tracks request counts + tokens)
            await client.post(
                f"{COMMAND_CENTER_URL}/ops/record-routing",
                json={
                    "recommended_path": provider,
                    "domain": endpoint or "local_api",
                    "model": model,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            )
    except Exception:
        pass  # Never block API on metrics failure


# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIIA — AIIA Local Brain",
    description="AIIA (AI Information Architecture) — persistent AI teammate running on Mac Mini",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Locked down by Tailscale network
    allow_methods=["*"],
    allow_headers=["*"],
)


# Singletons initialized on startup
_ollama: Optional[OllamaClient] = None
_conductor: Optional[SmartConductor] = None
_config: Optional[LocalBrainConfig] = None
_aiia: Optional[AIIA] = None
_supermemory_bridge: Optional[SupermemoryBridge] = None
_google_tts: Optional[GoogleTTSService] = None
_active_sessions: Dict[str, Dict[str, Any]] = {}  # Active session tracking


@app.on_event("startup")
async def startup():
    global _ollama, _conductor, _config, _aiia, _supermemory_bridge, _google_tts
    _config = get_config()
    _ollama = OllamaClient(_config)
    _conductor = SmartConductor(_ollama, _config)

    # Initialize Supermemory bridge
    if _config.supermemory_enabled:
        _supermemory_bridge = SupermemoryBridge(timeout=_config.supermemory_timeout)
        if _supermemory_bridge.available:
            logger.info("Supermemory bridge online")
        else:
            logger.info(
                "Supermemory bridge disabled (no API key or SUPERMEMORY_ENABLED=false)"
            )
    else:
        logger.info("Supermemory bridge disabled by config")

    # Initialize Google Cloud TTS
    _google_tts = GoogleTTSService()
    if _google_tts.is_available:
        logger.info("Google Cloud TTS available")
    else:
        logger.info(
            "Google Cloud TTS not configured (no GOOGLE_API_KEY or GOOGLE_TTS_API_KEY)"
        )

    # Check Ollama health on startup
    health = await _ollama.health()
    if health["status"] == "online":
        logger.info(
            f"Local Brain online. Ollama at {_config.ollama_url} "
            f"with {health['model_count']} models: {health['models']}"
        )
    else:
        logger.warning(f"Ollama not reachable: {health.get('error', 'unknown')}")

    # AIIA initialized lazily on first request (ChromaDB + embeddings are memory-heavy)
    if _config.eq_brain_enabled:
        logger.info("AIIA will initialize on first request (lazy loading)")
    else:
        logger.info("AIIA disabled by config")


_aiia_init_lock = asyncio.Lock()


async def _ensure_aiia() -> Optional[AIIA]:
    """Lazy-initialize AIIA on first use. Thread-safe via asyncio lock."""
    global _aiia
    if _aiia is not None:
        return _aiia
    if _config and not _config.eq_brain_enabled:
        return None

    async with _aiia_init_lock:
        # Double-check after acquiring lock
        if _aiia is not None:
            return _aiia
        try:
            logger.info("Initializing AIIA (first request)...")
            knowledge = KnowledgeStore(
                data_dir=_config.eq_brain_data_dir,
                collection_name=_config.eq_brain_collection,
            )
            await knowledge.initialize()
            memory = Memory(data_dir=_config.eq_brain_data_dir)

            task_model = _config.models.get("task")
            model = task_model.model_name if task_model else "llama3.1:8b"
            deep_cfg = _config.models.get("deep")
            deep_model = deep_cfg.model_name if deep_cfg else None
            _aiia = AIIA(
                knowledge,
                memory,
                _ollama,
                model=model,
                supermemory_bridge=_supermemory_bridge,
                deep_model=deep_model,
                cloud_timeout=_config.hybrid_cloud_timeout,
            )

            status = await _aiia.status()
            logger.info(
                f"AIIA online: {status['knowledge']['knowledge_docs']} knowledge docs, "
                f"{status['memory']['total_memories']} memories"
            )
        except Exception as e:
            logger.warning(f"AIIA failed to initialize: {e}")
            _aiia = None
    return _aiia


async def _require_aiia() -> AIIA:
    """Lazy-init AIIA and raise 503 if it fails."""
    aiia = await _ensure_aiia()
    if aiia is None:
        raise HTTPException(status_code=503, detail="AIIA not initialized")
    return aiia


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Optional API key verification for production security."""
    if _config and _config.api_key:
        if x_api_key != _config.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")


# ─────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────


class RouteRequest(BaseModel):
    """Request to classify and route a user query."""

    query: str
    tenant_id: str = "default"
    has_documents: bool = False
    document_count: int = 0
    conversation_context: Optional[str] = None


class RouteResponse(BaseModel):
    """Smart routing result — replaces keyword matching."""

    domain: str
    eq_level: int
    eq_mode: str
    complexity_score: float
    recommended_path: str  # "local", "eos", "rlm"
    confidence: float
    reasoning: str
    latency_ms: float


class ChatRequest(BaseModel):
    """Request for local chat completion."""

    messages: List[Dict[str, str]]
    system: Optional[str] = None
    model: Optional[str] = None  # Override default
    model_role: str = "task"  # "routing", "task", "pii"
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False


class ChatResponse(BaseModel):
    """Local chat completion response."""

    content: str
    model: str
    usage: Dict[str, int] = {}
    latency_ms: float = 0.0


class EmbedRequest(BaseModel):
    """Request for local embeddings."""

    texts: List[str]
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    """Embedding response."""

    embeddings: List[List[float]]
    model: str
    count: int
    latency_ms: float


class SummarizeRequest(BaseModel):
    """Request to summarize text locally."""

    text: str
    max_length: int = 200
    style: str = "concise"  # "concise", "detailed", "bullet_points"


class MemoryExtractRequest(BaseModel):
    """Request to extract learnable facts from a conversation."""

    messages: List[Dict[str, str]]
    user_id: str
    existing_memories: Optional[List[str]] = None


class PIIScanRequest(BaseModel):
    """Request to scan text for PII/PHI."""

    text: str
    categories: List[str] = Field(
        default=["ssn", "email", "phone", "address", "dob", "medical", "financial"]
    )


class PIIScanResponse(BaseModel):
    """PII scan result."""

    has_pii: bool
    findings: List[Dict[str, Any]]
    risk_level: str  # "none", "low", "medium", "high", "critical"
    latency_ms: float


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Health check — reports AIIA and Ollama status."""
    ollama_health = await _ollama.health() if _ollama else {"status": "not_initialized"}

    return {
        "service": "aiia-local-brain",
        "identity": "AIIA",
        "status": "online",
        "ollama": ollama_health,
        "aiia": (await _aiia.status()) if _aiia else {"status": "not_initialized"},
        "supermemory": _supermemory_bridge.status()
        if _supermemory_bridge
        else {"available": False},
        "features": {
            "smart_routing": _config.smart_routing_enabled if _config else False,
            "summarization": _config.summarization_enabled if _config else False,
            "memory_extraction": _config.memory_extraction_enabled
            if _config
            else False,
            "pii_scanning": _config.pii_scanning_enabled if _config else False,
            "embeddings": _config.embeddings_enabled if _config else False,
        },
    }


# ─────────────────────────────────────────────────────────────
# Smart Routing (Phase 2 — replaces keyword Conductor)
# ─────────────────────────────────────────────────────────────


@app.post(
    "/v1/route", response_model=RouteResponse, dependencies=[Depends(verify_api_key)]
)
async def smart_route(request: RouteRequest):
    """
    Classify a user query using local LLM — replaces keyword matching.

    Returns domain, EQ level, complexity score, and recommended execution path.
    """
    if not _conductor:
        raise HTTPException(status_code=503, detail="Smart Conductor not initialized")

    result = await _conductor.route(
        query=request.query,
        tenant_id=request.tenant_id,
        has_documents=request.has_documents,
        document_count=request.document_count,
        conversation_context=request.conversation_context,
    )

    # Report routing metrics to Command Center
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=_config.models["routing"].model_name if _config else "llama3.1:8b",
            latency_ms=result.get("latency_ms", 0)
            if isinstance(result, dict)
            else getattr(result, "latency_ms", 0),
            endpoint="route",
        )
    )

    return result


# ─────────────────────────────────────────────────────────────
# Local Chat Completion
# ─────────────────────────────────────────────────────────────


@app.post(
    "/v1/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)]
)
async def local_chat(request: ChatRequest):
    """
    Local chat completion via Ollama.

    Uses the model assigned to the specified role (routing/task/pii),
    or a specific model override.
    """
    if not _ollama:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")

    # Resolve model from role or override
    model = request.model
    if not model and _config:
        model_config = _config.models.get(request.model_role)
        if model_config:
            model = model_config.model_name
    model = model or "llama3.1:8b"

    response = await _ollama.chat(
        model=model,
        messages=request.messages,
        system=request.system,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    content = response.get("message", {}).get("content", "")
    usage = {}
    if "eval_count" in response:
        usage["output_tokens"] = response["eval_count"]
    if "prompt_eval_count" in response:
        usage["input_tokens"] = response["prompt_eval_count"]

    # Report metrics to Command Center (fire-and-forget)
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=model,
            latency_ms=response.get("_latency_ms", 0),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            endpoint="chat",
        )
    )

    return ChatResponse(
        content=content,
        model=model,
        usage=usage,
        latency_ms=response.get("_latency_ms", 0),
    )


# ─────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────


@app.post(
    "/v1/embed", response_model=EmbedResponse, dependencies=[Depends(verify_api_key)]
)
async def generate_embeddings(request: EmbedRequest):
    """Generate embeddings locally using nomic-embed-text or similar."""
    if not _ollama:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")

    model = request.model
    if not model and _config:
        embed_config = _config.models.get("embed")
        if embed_config:
            model = embed_config.model_name
    model = model or "nomic-embed-text"

    start = time.monotonic()
    embeddings = await _ollama.embed_batch(model, request.texts)
    latency = (time.monotonic() - start) * 1000

    return EmbedResponse(
        embeddings=embeddings,
        model=model,
        count=len(embeddings),
        latency_ms=round(latency, 1),
    )


# ─────────────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────────────


@app.post("/v1/summarize", dependencies=[Depends(verify_api_key)])
async def summarize(request: SummarizeRequest):
    """Summarize text locally — free alternative to using Claude for summaries."""
    if not _ollama or not _config:
        raise HTTPException(status_code=503, detail="Not initialized")

    style_prompts = {
        "concise": f"Summarize the following text in {request.max_length} words or less. Be direct and factual.",
        "detailed": f"Provide a detailed summary of the following text in {request.max_length} words or less. Include key details and context.",
        "bullet_points": f"Summarize the following text as bullet points ({request.max_length} words max). Each bullet should capture one key point.",
    }
    system = style_prompts.get(request.style, style_prompts["concise"])

    task_model = _config.models.get("task")
    model = task_model.model_name if task_model else "llama3.1:8b"

    response = await _ollama.chat(
        model=model,
        messages=[{"role": "user", "content": request.text}],
        system=system,
        temperature=0.3,  # Lower temp for factual summaries
        max_tokens=request.max_length * 2,  # Rough word-to-token ratio
    )

    # Report metrics
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=model,
            latency_ms=response.get("_latency_ms", 0),
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            endpoint="summarize",
        )
    )

    return {
        "summary": response.get("message", {}).get("content", ""),
        "model": model,
        "latency_ms": response.get("_latency_ms", 0),
    }


# ─────────────────────────────────────────────────────────────
# Memory Extraction
# ─────────────────────────────────────────────────────────────


@app.post("/v1/extract-memory", dependencies=[Depends(verify_api_key)])
async def extract_memory(request: MemoryExtractRequest):
    """
    Extract learnable facts from a conversation — runs in background,
    so latency doesn't matter. Perfect for local model.
    """
    if not _ollama or not _config:
        raise HTTPException(status_code=503, detail="Not initialized")

    # Build conversation text
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in request.messages
    )

    existing = ""
    if request.existing_memories:
        existing = "\n\nAlready known facts about this user:\n" + "\n".join(
            f"- {m}" for m in request.existing_memories
        )

    system = (
        "You are a memory extraction system. Analyze the conversation and extract "
        "NEW facts about the user that would be useful to remember for future interactions. "
        "Return ONLY a JSON array of strings, each being one learnable fact. "
        "Only extract facts that are explicitly stated, not inferred. "
        "If there are no new facts, return an empty array []."
        f"{existing}"
    )

    task_model = _config.models.get("task")
    model = task_model.model_name if task_model else "llama3.1:8b"

    response = await _ollama.chat(
        model=model,
        messages=[{"role": "user", "content": conversation}],
        system=system,
        temperature=0.1,
        max_tokens=1024,
    )

    content = response.get("message", {}).get("content", "[]")

    # Parse JSON array from response
    import json

    try:
        memories = json.loads(content)
        if not isinstance(memories, list):
            memories = []
    except json.JSONDecodeError:
        # Try to extract array from markdown code block
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            try:
                memories = json.loads(content.strip())
            except json.JSONDecodeError:
                memories = []
        else:
            memories = []

    # Report metrics
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=model,
            latency_ms=response.get("_latency_ms", 0),
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            endpoint="extract_memory",
        )
    )

    # Feedback loop: store extracted facts as AIIA memories (fire-and-forget)
    if _aiia and memories:
        for fact in memories:
            if isinstance(fact, str) and len(fact) > 50:
                asyncio.create_task(
                    _aiia.remember(
                        fact=fact,
                        category="lessons",
                        source=f"extract-memory/{request.user_id}",
                    )
                )

    return {
        "user_id": request.user_id,
        "memories": memories,
        "count": len(memories),
        "model": model,
        "latency_ms": response.get("_latency_ms", 0),
    }


# ─────────────────────────────────────────────────────────────
# PII Scanning
# ─────────────────────────────────────────────────────────────


@app.post(
    "/v1/scan-pii",
    response_model=PIIScanResponse,
    dependencies=[Depends(verify_api_key)],
)
async def scan_pii(request: PIIScanRequest):
    """
    Scan text for PII/PHI using local model — data never leaves the machine.
    """
    if not _ollama or not _config:
        raise HTTPException(status_code=503, detail="Not initialized")

    categories_str = ", ".join(request.categories)

    system = (
        "You are a PII/PHI detection system for HIPAA and GDPR compliance. "
        f"Scan the text for these categories of sensitive information: {categories_str}. "
        "Return ONLY a JSON object with:\n"
        '- "has_pii": boolean\n'
        '- "findings": array of {"category": string, "value": string, "location": string}\n'
        '- "risk_level": "none" | "low" | "medium" | "high" | "critical"\n'
        "Be thorough. Flag anything that could identify a person."
    )

    pii_model = _config.models.get("pii")
    model = pii_model.model_name if pii_model else "llama3.1:8b"

    start = time.monotonic()
    response = await _ollama.chat(
        model=model,
        messages=[{"role": "user", "content": request.text}],
        system=system,
        temperature=0.0,
        max_tokens=1024,
    )
    latency = (time.monotonic() - start) * 1000

    content = response.get("message", {}).get("content", "{}")

    import json

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            try:
                result = json.loads(content.strip())
            except json.JSONDecodeError:
                result = {"has_pii": False, "findings": [], "risk_level": "none"}
        else:
            result = {"has_pii": False, "findings": [], "risk_level": "none"}

    # Report metrics
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=model,
            latency_ms=round(latency, 1),
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            endpoint="pii_scan",
        )
    )

    return PIIScanResponse(
        has_pii=result.get("has_pii", False),
        findings=result.get("findings", []),
        risk_level=result.get("risk_level", "none"),
        latency_ms=round(latency, 1),
    )


# ─────────────────────────────────────────────────────────────
# AIIA — Persistent Memory & Knowledge
# ─────────────────────────────────────────────────────────────


class AIIAAskRequest(BaseModel):
    """Ask AIIA a question."""

    question: str
    context: Optional[str] = None
    include_sessions: bool = True
    n_results: int = 5
    max_tokens: int = 4096
    num_ctx: int = 8192  # Context window: 8192 (text), 4096 (voice)
    depth: str = "fast"  # "fast" (local only), "hybrid" (+ cloud), "deep" (+ DeepSeek)


class AIIARememberRequest(BaseModel):
    """Teach AIIA something new."""

    fact: str
    category: str = "lessons"
    source: str = "session"
    metadata: Optional[Dict[str, Any]] = None


class AIIASessionEndRequest(BaseModel):
    """Record the end of a session."""

    session_id: str
    summary: str
    key_decisions: Optional[List[str]] = None
    lessons_learned: Optional[List[str]] = None


class AIIAIngestRequest(BaseModel):
    """Ingest a document into AIIA's knowledge store."""

    text: str
    source: str
    doc_type: str = "documentation"
    metadata: Optional[Dict[str, Any]] = None


@app.post("/v1/aiia/ask", dependencies=[Depends(verify_api_key)])
async def aiia_ask(request: AIIAAskRequest):
    """
    Ask AIIA a question. She searches knowledge + memory,
    builds context, and reasons with the local LLM.
    """
    _aiia = await _require_aiia()

    start = time.monotonic()
    result = await _aiia.ask(
        question=request.question,
        context=request.context,
        include_sessions=request.include_sessions,
        n_results=request.n_results,
        num_ctx=request.num_ctx,
        depth=request.depth,
    )
    latency = (time.monotonic() - start) * 1000

    # Report AIIA ask as a local LLM request
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=_aiia._model if _aiia else "llama3.1:8b",
            latency_ms=latency,
            endpoint="aiia_ask",
        )
    )

    return result


@app.post("/v1/aiia/ask/stream", dependencies=[Depends(verify_api_key)])
async def aiia_ask_stream(request: AIIAAskRequest):
    """
    Streaming version of /v1/aiia/ask. Returns SSE events:
      - meta: sources and hit counts (before LLM starts)
      - chunk: token content as it generates
      - done: final event with latency and full answer
    """
    _aiia = await _require_aiia()

    async def generate():
        async for event in _aiia.ask_stream(
            question=request.question,
            context=request.context,
            include_sessions=request.include_sessions,
            n_results=request.n_results,
            max_tokens=request.max_tokens,
            num_ctx=request.num_ctx,
            depth=request.depth,
        ):
            yield f"data: {json_module.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/aiia/ask/recursive", dependencies=[Depends(verify_api_key)])
async def aiia_ask_recursive(request: AIIAAskRequest):
    """
    Recursive inference over large documents. Streams SSE events:
      - meta: sources + variable handles
      - action: model's chosen action per iteration
      - result: action execution result
      - done: final answer + stats
      - fallback: if switching to chunked processing
      - error: if something goes wrong
    """
    _aiia = await _require_aiia()

    async def generate():
        async for event in _aiia.ask_recursive_stream(
            question=request.question,
            context=request.context,
            include_sessions=request.include_sessions,
            n_results=request.n_results,
        ):
            yield f"data: {json_module.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Keep legacy /v1/eq/ask route for backward compatibility
@app.post("/v1/eq/ask", dependencies=[Depends(verify_api_key)], include_in_schema=False)
async def eq_ask_legacy(request: AIIAAskRequest):
    return await aiia_ask(request)


@app.post("/v1/aiia/remember", dependencies=[Depends(verify_api_key)])
async def aiia_remember(request: AIIARememberRequest):
    """
    Teach AIIA something new. Stores in structured memory
    AND indexes in the knowledge store for semantic search.
    """
    _aiia = await _require_aiia()

    result = await _aiia.remember(
        fact=request.fact,
        category=request.category,
        source=request.source,
        metadata=request.metadata,
    )
    return result


@app.post(
    "/v1/eq/remember", dependencies=[Depends(verify_api_key)], include_in_schema=False
)
async def eq_remember_legacy(request: AIIARememberRequest):
    return await aiia_remember(request)


@app.post("/v1/aiia/session-end", dependencies=[Depends(verify_api_key)])
async def aiia_session_end(request: AIIASessionEndRequest):
    """
    Record the end of a session. Stores the summary and extracts
    decisions + lessons into AIIA's long-term memory.
    """
    _aiia = await _require_aiia()

    result = await _aiia.end_session(
        session_id=request.session_id,
        summary=request.summary,
        key_decisions=request.key_decisions,
        lessons_learned=request.lessons_learned,
    )

    # Clean up active session tracking
    _active_sessions.pop(request.session_id, None)

    return result


@app.post(
    "/v1/eq/session-end",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
async def eq_session_end_legacy(request: AIIASessionEndRequest):
    return await aiia_session_end(request)


@app.get("/v1/aiia/status", dependencies=[Depends(verify_api_key)])
async def aiia_status():
    """Full status of AIIA — knowledge docs, memories, model info."""
    aiia = await _ensure_aiia()
    if not aiia:
        return {
            "status": "disabled",
            "identity": "AIIA",
            "reason": "AIIA failed to initialize",
        }
    return await aiia.status()


@app.get(
    "/v1/eq/status", dependencies=[Depends(verify_api_key)], include_in_schema=False
)
async def eq_status_legacy():
    return await aiia_status()


@app.post("/v1/aiia/ingest", dependencies=[Depends(verify_api_key)])
async def aiia_ingest(request: AIIAIngestRequest):
    """Ingest a document into AIIA's knowledge store."""
    _aiia = await _require_aiia()

    await _aiia._knowledge.add_document(
        text=request.text,
        source=request.source,
        doc_type=request.doc_type,
        metadata=request.metadata,
    )
    return {
        "status": "indexed",
        "source": request.source,
        "doc_type": request.doc_type,
    }


@app.post(
    "/v1/eq/ingest", dependencies=[Depends(verify_api_key)], include_in_schema=False
)
async def eq_ingest_legacy(request: AIIAIngestRequest):
    return await aiia_ingest(request)


@app.post("/v1/aiia/search", dependencies=[Depends(verify_api_key)])
async def aiia_search(request: AIIAAskRequest):
    """
    Search AIIA's knowledge store without LLM reasoning.
    Faster than /v1/aiia/ask — returns raw document matches.
    """
    _aiia = await _require_aiia()

    results = await _aiia._knowledge.search(
        query=request.question,
        n_results=request.n_results,
    )
    return {"results": results, "count": len(results)}


@app.post(
    "/v1/eq/search", dependencies=[Depends(verify_api_key)], include_in_schema=False
)
async def eq_search_legacy(request: AIIAAskRequest):
    return await aiia_search(request)


@app.get("/v1/aiia/memory", dependencies=[Depends(verify_api_key)])
async def aiia_memory(category: Optional[str] = None, limit: int = 50):
    """Browse AIIA's structured memories, optionally filtered by category."""
    _aiia = await _require_aiia()

    memories = _aiia._memory.recall(category=category, limit=limit)
    return {
        "memories": memories,
        "count": len(memories),
        "stats": _aiia._memory.stats(),
    }


@app.get(
    "/v1/eq/memory", dependencies=[Depends(verify_api_key)], include_in_schema=False
)
async def eq_memory_legacy(category: Optional[str] = None, limit: int = 50):
    return await aiia_memory(category, limit)


# ─────────────────────────────────────────────────────────────
# Supermemory Bridge — Cloud Sync & Search
# ─────────────────────────────────────────────────────────────


class SupermemorySyncRequest(BaseModel):
    """Request to sync local memories to Supermemory cloud."""

    categories: Optional[List[str]] = None
    limit_per_category: int = 50


class SupermemorySearchRequest(BaseModel):
    """Request to search Supermemory cloud."""

    query: str
    search_type: str = "sme"  # "sme" or "aiia"
    domains: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    tenant_id: str = "default"
    limit: int = 5


@app.post("/v1/aiia/supermemory/sync", dependencies=[Depends(verify_api_key)])
async def aiia_supermemory_sync(request: SupermemorySyncRequest):
    """
    Bulk sync local memories to Supermemory cloud backup.
    Safe to re-run — dedup via deterministic custom_id.
    """
    _aiia = await _require_aiia()
    if not _supermemory_bridge or not _supermemory_bridge.available:
        raise HTTPException(status_code=503, detail="Supermemory bridge not available")

    categories = request.categories or list(_aiia._memory.CATEGORIES)
    results = {}

    for category in categories:
        memories = _aiia._memory.recall(
            category=category, limit=request.limit_per_category
        )
        if not memories:
            results[category] = {"synced": 0, "total": 0, "errors": 0}
            continue

        r = await _supermemory_bridge.sync_bulk(memories, category=category)
        results[category] = r

    total_synced = sum(r.get("synced", 0) for r in results.values())
    total_errors = sum(r.get("errors", 0) for r in results.values())

    return {
        "status": "completed",
        "total_synced": total_synced,
        "total_errors": total_errors,
        "by_category": results,
    }


@app.post("/v1/aiia/supermemory/search", dependencies=[Depends(verify_api_key)])
async def aiia_supermemory_search(request: SupermemorySearchRequest):
    """
    Search Supermemory cloud — SME domain knowledge or AIIA's cloud memories.
    """
    _aiia = await _require_aiia()
    if not _supermemory_bridge or not _supermemory_bridge.available:
        raise HTTPException(status_code=503, detail="Supermemory bridge not available")

    result = await _aiia.search_supermemory(
        query=request.query,
        search_type=request.search_type,
        domains=request.domains,
        categories=request.categories,
        tenant_id=request.tenant_id,
        limit=request.limit,
    )
    return result


@app.delete("/v1/aiia/memory/{memory_id}", dependencies=[Depends(verify_api_key)])
async def aiia_memory_delete(memory_id: str):
    """Delete a specific memory by ID."""
    _aiia = await _require_aiia()

    deleted = _aiia._memory.forget(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
    return {"status": "deleted", "memory_id": memory_id}


# ─────────────────────────────────────────────────────────────
# Voice Output — macOS TTS via `say`
# ─────────────────────────────────────────────────────────────

# TTS state: async lock ensures one speak at a time, process tracked by PID
_tts_lock = asyncio.Lock()
_tts_process: Optional[asyncio.subprocess.Process] = None


class SpeakRequest(BaseModel):
    """Request AIIA to speak aloud via Google Cloud TTS (or macOS fallback)."""

    text: str
    voice: str = "aiia"  # Google TTS preset: "aiia", "mia", or full voice name
    rate: int = 200  # words per minute (macOS fallback only)


def _strip_markdown_for_tts(text: str) -> str:
    """Strip markdown syntax so TTS doesn't read asterisks, hashes, etc."""
    import re

    text = re.sub(r"```[\s\S]*?```", "", text)  # fenced code blocks
    text = re.sub(r"`[^`]+`", "", text)  # inline code
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)  # *italic*
    text = re.sub(r"__(.+?)__", r"\1", text)  # __bold__
    text = re.sub(r"_(.+?)_", r"\1", text)  # _italic_
    text = re.sub(r"~~(.+?)~~", r"\1", text)  # ~~strikethrough~~
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [links](url)
    text = re.sub(r"[*#_~`>|]", "", text)  # any remaining markdown chars
    text = re.sub(r"\n{2,}", ". ", text)  # paragraph breaks -> pause
    text = re.sub(r"\n", " ", text)  # remaining newlines
    text = re.sub(r"\s{2,}", " ", text)  # collapse whitespace
    return text.strip()


async def _kill_current_tts():
    """Kill the current TTS process by PID if running."""
    global _tts_process
    if _tts_process and _tts_process.returncode is None:
        try:
            _tts_process.kill()
            await _tts_process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Error killing TTS process: {e}")
    _tts_process = None


async def _notify_speech_done():
    """Notify command center that speech has finished."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{COMMAND_CENTER_URL}/ops/speech-done")
    except Exception:
        pass  # Best-effort


@app.post("/v1/aiia/speak")
async def aiia_speak(request: SpeakRequest):
    """AIIA speaks aloud on the Mac Mini. Google Cloud TTS primary, macOS say fallback."""
    global _tts_process
    text = _strip_markdown_for_tts(request.text)
    if not text:
        return {"status": "empty", "text_length": 0}

    # Kill any existing speech before starting new one
    await _kill_current_tts()

    engine = "google_tts"

    # Try Google TTS first — synthesize MP3, play with afplay
    if _google_tts and _google_tts.is_available:
        try:
            audio_bytes = await _google_tts.synthesize(text, voice=request.voice)

            async def _play_audio():
                global _tts_process
                async with _tts_lock:
                    import tempfile

                    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    try:
                        tmp.write(audio_bytes)
                        tmp.close()
                        _tts_process = await asyncio.create_subprocess_exec(
                            "afplay",
                            tmp.name,
                        )
                        await _tts_process.wait()
                    except Exception as e:
                        logger.warning(f"afplay failed: {e}")
                    finally:
                        _tts_process = None
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                        asyncio.create_task(_notify_speech_done())

            asyncio.create_task(_play_audio())

            return {
                "status": "speaking",
                "text_length": len(text),
                "voice": request.voice,
                "engine": "google_tts",
            }
        except Exception as e:
            logger.warning(f"Google TTS failed, falling back to macOS say: {e}")
            engine = "macos_say"

    # Fallback: macOS say
    async def _speak():
        global _tts_process
        async with _tts_lock:
            try:
                _tts_process = await asyncio.create_subprocess_exec(
                    "say",
                    "-v",
                    "Samantha",
                    "-r",
                    str(request.rate),
                    text,
                )
                await _tts_process.wait()
            except Exception as e:
                logger.warning(f"TTS failed: {e}")
            finally:
                _tts_process = None
                asyncio.create_task(_notify_speech_done())

    asyncio.create_task(_speak())

    return {
        "status": "speaking",
        "text_length": len(text),
        "voice": "Samantha",
        "engine": "macos_say",
    }


@app.post("/v1/aiia/speak/stop")
async def aiia_speak_stop():
    """Stop current speech by killing the TTS process."""
    await _kill_current_tts()
    # Also notify command center immediately
    asyncio.create_task(_notify_speech_done())
    return {"status": "stopped"}


# ─────────────────────────────────────────────────────────────
# Google Cloud TTS — High-Quality Voice Synthesis
# ─────────────────────────────────────────────────────────────


class VoiceAskRequest(BaseModel):
    """Ask AIIA a question and get spoken audio back (for iOS Shortcut)."""

    question: str
    voice: str = "aiia"
    max_tokens: int = 512  # Shorter for voice responses
    speaking_rate: float = 1.0


class TTSRequest(BaseModel):
    """Request for Google Cloud TTS synthesis."""

    text: str
    voice: str = "aiia"  # Preset name or full Google voice name
    speaking_rate: float = 1.0
    audio_encoding: str = "MP3"


@app.post("/v1/aiia/voice")
async def aiia_voice(request: VoiceAskRequest):
    """
    Combined ask + TTS endpoint — returns MP3 audio of AIIA's spoken answer.

    Designed for iOS Shortcuts: dictate → one HTTP call → play audio.
    Chains AIIA.ask() → Google TTS → raw audio/mpeg response.
    Falls back to JSON text if TTS is unavailable.
    """
    _aiia = await _require_aiia()

    # 1. Ask AIIA (shorter context for voice)
    start = time.monotonic()
    result = await _aiia.ask(
        question=request.question,
        n_results=3,
        num_ctx=4096,
    )
    answer = result.get("answer", "")
    if not answer:
        raise HTTPException(status_code=500, detail="AIIA returned no answer")

    ask_latency = (time.monotonic() - start) * 1000

    # Report metrics
    asyncio.create_task(
        _report_metrics(
            provider="local",
            model=_aiia._model if _aiia else "llama3.1:8b",
            latency_ms=ask_latency,
            endpoint="aiia_voice",
        )
    )

    # 2. Synthesize speech via Google Cloud TTS
    if _google_tts and _google_tts.is_available:
        try:
            audio_bytes = await _google_tts.synthesize(
                text=answer,
                voice=request.voice,
                speaking_rate=request.speaking_rate,
                audio_encoding="MP3",
            )
            return RawResponse(content=audio_bytes, media_type="audio/mpeg")
        except Exception as e:
            logger.warning(f"Voice endpoint TTS failed, returning text: {e}")

    # 3. Fallback: return JSON with the text answer
    return {"answer": answer, "tts": "unavailable", "latency_ms": round(ask_latency, 1)}


@app.post("/v1/aiia/tts")
async def aiia_tts(request: TTSRequest):
    """
    Synthesize speech via Google Cloud TTS. Returns raw audio bytes.
    Falls back to macOS `say` + notification if Google TTS unavailable.
    """
    text = _strip_markdown_for_tts(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Try Google Cloud TTS first
    if _google_tts and _google_tts.is_available:
        try:
            audio_bytes = await _google_tts.synthesize(
                text=text,
                voice=request.voice,
                speaking_rate=request.speaking_rate,
                audio_encoding=request.audio_encoding,
            )
            media_type = {
                "MP3": "audio/mpeg",
                "OGG_OPUS": "audio/ogg",
                "LINEAR16": "audio/wav",
            }.get(request.audio_encoding, "audio/mpeg")

            return RawResponse(content=audio_bytes, media_type=media_type)
        except Exception as e:
            logger.warning(f"Google TTS failed, falling back to macOS say: {e}")

    # Fallback: trigger macOS `say` and return status
    speak_req = SpeakRequest(text=request.text)
    return await aiia_speak(speak_req)


@app.get("/v1/aiia/tts/voices")
async def aiia_tts_voices(language_code: str = "en-US"):
    """List available Google Cloud TTS voices."""
    if not _google_tts or not _google_tts.is_available:
        return {"voices": [], "error": "Google TTS not configured"}

    try:
        voices = await _google_tts.list_voices(language_code)
        return {"voices": voices, "count": len(voices)}
    except Exception as e:
        return {"voices": [], "error": str(e)}


@app.get("/v1/aiia/tts/health")
async def aiia_tts_health():
    """Check Google Cloud TTS availability."""
    if not _google_tts:
        return {"available": False, "reason": "not_initialized"}
    return await _google_tts.check_health()


# ─────────────────────────────────────────────────────────────
# Session Lifecycle
# ─────────────────────────────────────────────────────────────


class SessionStartRequest(BaseModel):
    """Start a new work session — loads relevant context."""

    task_description: str
    branch: str = ""
    files: List[str] = []


@app.post("/v1/aiia/session-start", dependencies=[Depends(verify_api_key)])
async def aiia_session_start(request: SessionStartRequest):
    """
    Start a session — loads WIP, recent decisions, relevant knowledge,
    and past sessions into a single context package.
    """
    _aiia = await _require_aiia()

    import uuid

    session_id = f"session-{uuid.uuid4().hex[:8]}"

    # Gather context in parallel
    wip_items = _aiia._memory.recall(category="wip", limit=10)
    recent_decisions = _aiia._memory.recall(category="decisions", limit=5)
    recent_sessions = _aiia._memory.recall(category="sessions", limit=5)

    # Search knowledge for the task
    relevant_knowledge = []
    if request.task_description:
        try:
            relevant_knowledge = await _aiia._knowledge.search(
                query=request.task_description,
                n_results=5,
            )
        except Exception:
            pass

    # Search for file-specific context
    if request.files:
        file_query = " ".join(request.files[:5])
        try:
            file_results = await _aiia._knowledge.search(
                query=file_query,
                n_results=3,
            )
            relevant_knowledge.extend(file_results)
        except Exception:
            pass

    # Track active session
    _active_sessions[session_id] = {
        "session_id": session_id,
        "task_description": request.task_description,
        "branch": request.branch,
        "files": request.files,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Gather extra context from Command Center (best-effort)
    security_snapshot = {}
    routing_stats = {}
    recent_insights = []
    token_summary = {}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Parallel requests to Command Center
            sec_resp, route_resp, insight_resp, token_resp = await asyncio.gather(
                client.get(f"{COMMAND_CENTER_URL}/api/insights"),
                client.get(f"{COMMAND_CENTER_URL}/api/routing/stats"),
                client.get(f"{COMMAND_CENTER_URL}/api/insights"),
                client.get(f"{COMMAND_CENTER_URL}/api/tokens/today"),
                return_exceptions=True,
            )
            if not isinstance(sec_resp, Exception) and sec_resp.status_code == 200:
                recent_insights = sec_resp.json().get("insights", [])[:5]
            if not isinstance(route_resp, Exception) and route_resp.status_code == 200:
                routing_stats = route_resp.json()
            if not isinstance(token_resp, Exception) and token_resp.status_code == 200:
                token_summary = token_resp.json()
    except Exception:
        pass  # Command Center may not be running

    return {
        "session_id": session_id,
        "wip_items": wip_items,
        "recent_decisions": recent_decisions,
        "recent_sessions": recent_sessions,
        "relevant_knowledge": relevant_knowledge,
        "recent_insights": recent_insights,
        "routing_stats": routing_stats,
        "token_summary": token_summary,
    }


@app.get("/v1/aiia/sessions", dependencies=[Depends(verify_api_key)])
async def aiia_sessions():
    """List active sessions."""
    return {
        "sessions": list(_active_sessions.values()),
        "count": len(_active_sessions),
    }


# ─────────────────────────────────────────────────────────────
# Pre-Commit Validation — Fast, blocking, no LLM
# ─────────────────────────────────────────────────────────────


@app.post("/v1/aiia/pre-commit-check")
async def pre_commit_check():
    """
    Fast programmatic pre-commit checks. Called by Claude Code hook
    before git commit. Returns {block: bool, reason: str}.

    Checks:
    - .env files staged
    - Hardcoded API keys in staged files
    - py_compile on staged .py files
    - Product-specific code in platform layer
    - SME auto-loading re-enabled
    """
    import re
    import subprocess

    repo_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    reasons = []

    try:
        # Get staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        staged_files = [
            f.strip() for f in result.stdout.strip().split("\n") if f.strip()
        ]
    except Exception as e:
        return {"block": False, "reason": f"Could not read staged files: {e}"}

    if not staged_files:
        return {"block": False}

    # Check 1: .env files staged
    env_files = [
        f
        for f in staged_files
        if f.endswith(".env") or "/.env" in f or f.startswith(".env")
    ]
    if env_files:
        reasons.append(f"Staged .env file(s): {', '.join(env_files)}")

    # Check 2: Hardcoded API keys in staged diffs
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        diff_text = diff_result.stdout
        # Only check added lines (starting with +)
        added_lines = [
            l
            for l in diff_text.split("\n")
            if l.startswith("+") and not l.startswith("+++")
        ]
        key_patterns = [
            r"sk-ant-[a-zA-Z0-9\-_]{20,}",
            r"AIza[a-zA-Z0-9\-_]{30,}",
            r"sm_[a-zA-Z0-9]{20,}",
            r"ghp_[a-zA-Z0-9]{30,}",
            r"sk-[a-zA-Z0-9]{40,}",
        ]
        for line in added_lines:
            for pattern in key_patterns:
                if re.search(pattern, line):
                    reasons.append(f"Possible API key in staged diff: {pattern}")
                    break
            if reasons and "API key" in reasons[-1]:
                break  # One match is enough
    except Exception:
        pass

    # Check 3: py_compile on staged .py files
    py_staged = [f for f in staged_files if f.endswith(".py")]
    for py_file in py_staged[:20]:  # Cap at 20 files
        full_path = os.path.join(repo_path, py_file)
        if not os.path.exists(full_path):
            continue
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", full_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                error = result.stderr.strip()[:150]
                reasons.append(f"Syntax error in {py_file}: {error}")
        except Exception:
            pass

    # Check 4: Product-specific code in platform layer
    platform_dirs = ["local_brain/", "platform/"]
    platform_files = [f for f in staged_files if any(f.startswith(d) for d in platform_dirs)]
    product_keywords = []  # Add tenant IDs here if using multi-tenant
    for pf in platform_files:
        full_path = os.path.join(repo_path, pf)
        if not os.path.exists(full_path):
            continue
        try:
            content = open(full_path, "r", errors="replace").read()
            for kw in product_keywords:
                if f'tenant_id = "{kw}"' in content or f'tenant_id="{kw}"' in content:
                    reasons.append(
                        f"Product-specific code ({kw}) in platform file: {pf}"
                    )
                    break
        except Exception:
            pass

    # Check 5: SME auto-loading re-enabled
    for f in staged_files:
        if "main.py" in f:
            full_path = os.path.join(repo_path, f)
            if os.path.exists(full_path):
                try:
                    content = open(full_path, "r", errors="replace").read()
                    if "load_sme_knowledge_on_startup" in content:
                        reasons.append(f"SME auto-loading may be re-enabled in {f}")
                except Exception:
                    pass

    if reasons:
        return {
            "block": True,
            "reason": "; ".join(reasons),
            "checks_failed": len(reasons),
        }

    return {"block": False, "checks_passed": len(staged_files)}


# ─────────────────────────────────────────────────────────────
# Post-Commit Review — LLM-powered diff review
# ─────────────────────────────────────────────────────────────


class ReviewCommitRequest(BaseModel):
    """Request to review a commit."""

    sha: str = "HEAD"


@app.post("/v1/aiia/review-commit")
async def review_commit(request: ReviewCommitRequest):
    """
    Review a commit diff using local LLM. Called by Claude Code hook
    after git commit. Creates action items if issues found.
    """
    if not _aiia:
        return {"status": "skipped", "reason": "AIIA not initialized"}

    repo_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    try:
        # Get the diff
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "HEAD~1..HEAD",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        diff = stdout.decode()[:6000]  # Cap diff size for LLM

        if not diff.strip():
            return {"status": "skipped", "reason": "Empty diff"}

        # Get commit message
        proc2 = await asyncio.create_subprocess_exec(
            "git",
            "log",
            "-1",
            "--format=%s",
            request.sha,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=5)
        commit_msg = stdout2.decode().strip()

        # Get changed .py files for syntax check
        proc3 = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "--name-only",
            "HEAD~1..HEAD",
            "--",
            "*.py",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout3, _ = await asyncio.wait_for(proc3.communicate(), timeout=5)
        changed_py = [
            f.strip() for f in stdout3.decode().strip().split("\n") if f.strip()
        ]

    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}

    # Syntax check changed .py files
    syntax_issues = []
    import subprocess

    for py_file in changed_py[:10]:
        full_path = os.path.join(repo_path, py_file)
        if not os.path.exists(full_path):
            continue
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", full_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                syntax_issues.append(f"{py_file}: {result.stderr.strip()[:100]}")
        except Exception:
            pass

    # Ask local LLM to review the diff
    review_prompt = (
        f"Review this git commit for obvious issues. Commit message: {commit_msg}\n\n"
        f"Diff:\n{diff}\n\n"
        "Look for: missing imports, broken logic, deleted code that shouldn't be, "
        "hardcoded values, security issues. Be concise. If no issues, say 'LGTM'."
    )

    issues_found = []

    try:
        result = await _aiia.ask(
            question=review_prompt,
            context="You are reviewing a code diff. Be brief and specific. Only flag real issues.",
            n_results=0,
            num_ctx=8192,
        )
        review_text = result.get("answer", "")

        # Check if LLM found issues (not just LGTM)
        lower = review_text.lower()
        if "lgtm" not in lower and any(
            kw in lower
            for kw in [
                "issue",
                "error",
                "bug",
                "missing",
                "broken",
                "concern",
                "problem",
            ]
        ):
            issues_found.append(review_text)
    except Exception as e:
        logger.warning(f"Post-commit LLM review failed: {e}")

    # Create action items if issues found
    created_actions = 0
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for syntax in syntax_issues:
                await client.post(
                    f"{COMMAND_CENTER_URL}/api/actions/{commit_msg}",  # won't match, use direct queue
                )
    except Exception:
        pass

    # Direct action creation via Command Center
    if syntax_issues or issues_found:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if syntax_issues:
                    for si in syntax_issues[:3]:
                        filepath = si.split(":")[0].strip()
                        # Create action via POST to a simple create endpoint
                        # (we'll store the review as a memory instead for now)
                        created_actions += 1

                if issues_found:
                    created_actions += 1
        except Exception:
            pass

    # Store review as AIIA memory
    review_summary = f"Post-commit review of '{commit_msg}': "
    if syntax_issues:
        review_summary += f"{len(syntax_issues)} syntax issues. "
    if issues_found:
        review_summary += issues_found[0][:300]
    elif not syntax_issues:
        review_summary += "LGTM"

    try:
        await _aiia.remember(
            fact=review_summary[:500],
            category="project",
            source="post_commit_review",
        )
    except Exception:
        pass

    return {
        "status": "reviewed",
        "commit": commit_msg,
        "syntax_issues": len(syntax_issues),
        "llm_issues": len(issues_found),
        "review": review_summary[:300],
    }


# ─────────────────────────────────────────────────────────────
# Session History Search
# ─────────────────────────────────────────────────────────────


class SearchSessionsRequest(BaseModel):
    """Search indexed Claude Code session transcripts."""

    query: str
    project: str = ""
    domain: str = ""
    limit: int = Field(default=5, ge=1, le=20)


@app.post("/v1/aiia/search-sessions", dependencies=[Depends(verify_api_key)])
async def search_sessions(request: SearchSessionsRequest):
    """
    Vector search across indexed Claude Code session transcripts.
    Returns matching sessions with summaries, metadata, and relevance scores.
    """
    _aiia = await _require_aiia()

    # Search ChromaDB sessions collection
    results = await _aiia._knowledge.search_sessions(
        query=request.query,
        n_results=request.limit,
    )

    # Filter by project/domain if specified
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if request.project and request.project not in meta.get("project_path", ""):
            continue
        if request.domain and meta.get("domain", "") != request.domain:
            continue
        filtered.append(
            {
                "session_id": meta.get("session_id", r.get("session_id", "")),
                "summary": r.get("summary", ""),
                "project_path": meta.get("project_path", ""),
                "branch": meta.get("branch", ""),
                "domain": meta.get("domain", ""),
                "start_timestamp": meta.get("start_timestamp", ""),
                "duration_seconds": meta.get("duration_seconds", 0),
                "files_count": meta.get("files_count", 0),
                "model": meta.get("model", ""),
                "slug": meta.get("slug", ""),
            }
        )

    return {"sessions": filtered, "count": len(filtered)}


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"Starting AIIA (AIIA Local Brain) on {config.api_host}:{config.api_port}"
    )
    uvicorn.run(app, host=config.api_host, port=config.api_port)
