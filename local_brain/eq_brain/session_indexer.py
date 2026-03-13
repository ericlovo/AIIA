"""
Session Indexer — Parses Claude Code transcripts into searchable knowledge.

Nightly pipeline that:
1. Scans ~/.claude/projects/ for JSONL session transcripts
2. Extracts structured metadata (files, tools, errors, commits) via heuristics
3. Enriches with LLM summaries via local model ($0)
4. Stores in ChromaDB sessions collection + AIIA structured memory

First run indexes all files (~5 min). Nightly incremental: 0-3 new sessions, ~30s.

Schedule: Daily at 5:30am via com.aiia.sessionindex launchd agent
"""

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiia.session_indexer")

# Message types to skip — noise that adds no value
SKIP_TYPES = {"progress", "queue-operation", "file-history-snapshot"}


@dataclass
class SessionRecord:
    """Structured extraction from a Claude Code JSONL transcript."""

    session_id: str = ""
    project_path: str = ""
    start_timestamp: str = ""
    end_timestamp: str = ""
    duration_seconds: float = 0.0
    branch: str = ""
    slug: str = ""
    model: str = ""
    files_touched: List[str] = field(default_factory=list)
    tools_used: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    user_intent: str = ""
    commit_messages: List[str] = field(default_factory=list)
    message_count: int = 0
    # LLM-enriched fields (Phase 3)
    summary: str = ""
    decisions: List[str] = field(default_factory=list)
    solutions: List[Dict[str, str]] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    domain: str = ""


class JSONLParser:
    """Stream-parses Claude Code JSONL transcripts line by line."""

    @staticmethod
    def parse_file(path: str):
        """Yield parsed dicts from a JSONL file, skipping noise types."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        msg_type = obj.get("type", "")
                        if msg_type in SKIP_TYPES:
                            continue
                        yield obj
                    except json.JSONDecodeError:
                        if line_num <= 3:
                            continue  # First few lines sometimes malformed
                        logger.debug(f"Skipping malformed line {line_num} in {path}")
        except (OSError, IOError) as e:
            logger.error(f"Cannot read {path}: {e}")

    @staticmethod
    def extract_text_content(msg: dict) -> str:
        """Extract visible text from a message, skipping thinking blocks."""
        message = msg.get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)

        return ""

    @staticmethod
    def extract_tool_calls(msg: dict) -> List[Dict[str, Any]]:
        """Extract tool_use blocks from an assistant message."""
        message = msg.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return []

        tools = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append(
                    {
                        "name": block.get("name", ""),
                        "id": block.get("id", ""),
                        "input": block.get("input", {}),
                    }
                )
        return tools

    @staticmethod
    def extract_tool_results(msg: dict) -> List[Dict[str, Any]]:
        """Extract tool_result blocks from a user message."""
        message = msg.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return []

        results = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Flatten text blocks in result content
                    texts = []
                    for rb in result_content:
                        if isinstance(rb, dict) and rb.get("type") == "text":
                            texts.append(rb.get("text", ""))
                    result_content = "\n".join(texts)
                results.append(
                    {
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": str(result_content)[:500],
                        "is_error": block.get("is_error", False),
                    }
                )
        return results


class SessionExtractor:
    """Heuristic extraction from JSONL transcripts — no LLM needed."""

    @staticmethod
    def extract(jsonl_path: str) -> SessionRecord:
        """Parse a JSONL file and extract structured metadata."""
        record = SessionRecord()
        filename = os.path.basename(jsonl_path)

        # Session ID from filename UUID
        uuid_match = re.match(r"([0-9a-f-]{36})", filename)
        record.session_id = uuid_match.group(1) if uuid_match else filename

        tool_counter = Counter()
        files_set = set()
        timestamps = []
        first_user_msg = None
        commit_msgs = []

        for msg in JSONLParser.parse_file(jsonl_path):
            msg_type = msg.get("type", "")
            role = msg.get("message", {}).get("role", "")

            # Extract metadata from first typed message
            if not record.branch and msg.get("gitBranch"):
                record.branch = msg["gitBranch"]
            if not record.slug and msg.get("slug"):
                record.slug = msg["slug"]
            if not record.project_path and msg.get("cwd"):
                record.project_path = msg["cwd"]
            if not record.model:
                model = msg.get("message", {}).get("model", "")
                if model:
                    record.model = model

            # Timestamps
            ts = msg.get("timestamp", "")
            if ts:
                timestamps.append(ts)

            # Count messages
            if msg_type in ("user", "assistant"):
                record.message_count += 1

            # First user message = intent
            if msg_type == "user" and role == "user" and first_user_msg is None:
                text = JSONLParser.extract_text_content(msg)
                if text and not text.startswith("{") and len(text) > 10:
                    first_user_msg = text[:500]

            # Tool calls from assistant messages
            if msg_type == "assistant":
                for tool in JSONLParser.extract_tool_calls(msg):
                    name = tool["name"]
                    tool_counter[name] += 1

                    # Files from Edit/Write tool calls
                    inp = tool.get("input", {})
                    if name in ("Edit", "Write", "NotebookEdit"):
                        fp = inp.get("file_path", "")
                        if fp:
                            files_set.add(fp)
                    elif name == "Read":
                        fp = inp.get("file_path", "")
                        if fp:
                            files_set.add(fp)

                    # Commit messages from Bash
                    if name == "Bash":
                        cmd = inp.get("command", "")
                        if "git commit" in cmd:
                            # Extract -m "message" or heredoc message
                            m_match = re.search(
                                r'(?:git commit.*?-m\s+["\'])(.*?)(?:["\'])',
                                cmd,
                                re.DOTALL,
                            )
                            if m_match:
                                commit_msgs.append(m_match.group(1).strip()[:200])
                            else:
                                # heredoc pattern
                                hd_match = re.search(
                                    r"<<['\"]?EOF['\"]?\n(.*?)EOF",
                                    cmd,
                                    re.DOTALL,
                                )
                                if hd_match:
                                    commit_msgs.append(hd_match.group(1).strip()[:200])

            # Errors from tool results
            if msg_type == "user":
                for result in JSONLParser.extract_tool_results(msg):
                    if result.get("is_error"):
                        record.errors.append(result["content"][:200])

        # Finalize
        record.files_touched = sorted(files_set)
        record.tools_used = dict(tool_counter.most_common())
        record.user_intent = first_user_msg or ""
        record.commit_messages = commit_msgs

        if timestamps:
            record.start_timestamp = timestamps[0]
            record.end_timestamp = timestamps[-1]
            try:
                start = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
                end = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
                record.duration_seconds = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass

        return record


class KnowledgeExtractor:
    """LLM-powered enrichment of session records via local model ($0)."""

    ENRICHMENT_PROMPT = """You are analyzing a Claude Code session transcript summary. Extract structured knowledge.

## Session Info
- Project: {project_path}
- Branch: {branch}
- Duration: {duration}
- User Intent: {user_intent}
- Files Modified: {files}
- Errors Encountered: {errors}
- Commits: {commits}
- Tools Used: {tools}

## Instructions

Produce a JSON object with these fields:
1. "summary": 2-3 sentence summary of what was accomplished
2. "decisions": Array of key technical decisions made (strings, max 5)
3. "solutions": Array of {{\"error\": \"...\", \"fix\": \"...\"}} for problems solved (max 5)
4. "patterns": Array of reusable patterns or conventions established (strings, max 3)
5. "domain": One of: platform, backend, marketing, sales, local-brain, security, devops, other

Return ONLY valid JSON. No markdown fences, no explanation."""

    def __init__(self, ollama_client, model: str = "qwen2.5:7b"):
        self._ollama = ollama_client
        self._model = model

    async def enrich_session(self, record: SessionRecord) -> SessionRecord:
        """Add LLM-generated summary, decisions, solutions, patterns, and domain."""
        # Build compact prompt from heuristic extractions
        duration_str = (
            f"{record.duration_seconds / 60:.0f} min"
            if record.duration_seconds
            else "unknown"
        )
        files_str = ", ".join(record.files_touched[:15]) or "none"
        errors_str = "; ".join(record.errors[:5]) or "none"
        commits_str = "; ".join(record.commit_messages[:5]) or "none"
        tools_str = (
            ", ".join(f"{k}:{v}" for k, v in list(record.tools_used.items())[:10])
            or "none"
        )

        prompt = self.ENRICHMENT_PROMPT.format(
            project_path=record.project_path or "unknown",
            branch=record.branch or "unknown",
            duration=duration_str,
            user_intent=record.user_intent[:400] or "not captured",
            files=files_str,
            errors=errors_str,
            commits=commits_str,
            tools=tools_str,
        )

        try:
            result = await self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )

            raw = result.get("message", {}).get("content", "")
            parsed = self._parse_json_response(raw)

            if parsed:
                record.summary = parsed.get("summary", "")[:500]
                record.decisions = parsed.get("decisions", [])[:5]
                record.solutions = parsed.get("solutions", [])[:5]
                record.patterns = parsed.get("patterns", [])[:3]
                record.domain = parsed.get("domain", "other")
            else:
                logger.warning(
                    f"Failed to parse LLM response for {record.session_id} "
                    f"({len(raw)} chars)"
                )
                record.summary = f"Session on {record.branch or 'unknown'}: {record.user_intent[:200]}"
                record.domain = "other"

        except Exception as e:
            logger.warning(f"LLM enrichment failed for {record.session_id}: {e}")
            # Graceful fallback — session still indexed with heuristics only
            record.summary = (
                f"Session on {record.branch or 'unknown'}: {record.user_intent[:200]}"
            )
            record.domain = "other"

        return record

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
        """3-level fallback JSON parser (same pattern as MemoryConsolidator)."""
        # Level 1: direct parse
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Level 2: markdown fence
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # Level 3: substring extraction
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(raw[first_brace : last_brace + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return None


class SessionIndexer:
    """Pipeline orchestrator — scans, extracts, enriches, stores."""

    def __init__(
        self,
        knowledge_store,
        memory,
        ollama_client=None,
        data_dir: str = "",
        enrichment_model: str = "qwen2.5:7b",
        supermemory_bridge=None,
    ):
        self._knowledge = knowledge_store
        self._memory = memory
        self._ollama = ollama_client
        self._enrichment_model = enrichment_model
        self._bridge = supermemory_bridge

        self._state_dir = os.path.join(data_dir, "session_index")
        os.makedirs(self._state_dir, exist_ok=True)

        self._state_path = os.path.join(self._state_dir, "state.json")
        self._index_path = os.path.join(self._state_dir, "index.json")

    # ─── State Management ──────────────────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self, state: Dict[str, Any]):
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_index(self) -> Dict[str, Any]:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"sessions": {}}

    def _save_index(self, index: Dict[str, Any]):
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2, default=str)

    # ─── Phase 1: File Scanner ─────────────────────────────────

    def scan_for_new_files(self, force: bool = False) -> List[str]:
        """Walk ~/.claude/projects/ for JSONL files, return those needing indexing."""
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            logger.warning(f"Claude projects dir not found: {claude_dir}")
            return []

        state = self._load_state()
        new_files = []

        for project_dir in claude_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                path_str = str(jsonl_file)
                current_mtime = jsonl_file.stat().st_mtime

                if force:
                    new_files.append(path_str)
                    continue

                prev = state.get(path_str)
                if not prev or prev.get("mtime") != current_mtime:
                    new_files.append(path_str)

        logger.info(
            f"Scanned {claude_dir}: {len(new_files)} files to index"
            + (" (force=True)" if force else "")
        )
        return new_files

    # ─── Phase 2+3: Extract + Enrich ───────────────────────────

    async def process_file(self, path: str) -> SessionRecord:
        """Extract metadata (Phase 2) and enrich with LLM (Phase 3)."""
        # Phase 2: heuristic extraction
        record = SessionExtractor.extract(path)

        # Phase 3: LLM enrichment (if client available)
        if self._ollama:
            extractor = KnowledgeExtractor(self._ollama, model=self._enrichment_model)
            record = await extractor.enrich_session(record)
        else:
            # Fallback summary from heuristics
            record.summary = (
                f"Session on {record.branch or 'unknown'}: {record.user_intent[:200]}"
            )
            record.domain = "other"

        return record

    # ─── Phase 4: Store ────────────────────────────────────────

    async def store_session(self, record: SessionRecord):
        """Store in ChromaDB sessions collection + AIIA structured memory."""
        # Build searchable document text
        doc_text = f"{record.summary}\n\nIntent: {record.user_intent[:300]}"
        if record.commit_messages:
            doc_text += f"\nCommits: {'; '.join(record.commit_messages[:5])}"
        if record.files_touched:
            doc_text += f"\nFiles: {', '.join(record.files_touched[:20])}"

        # Metadata for ChromaDB (must be flat primitives)
        metadata = {
            "session_id": record.session_id,
            "project_path": record.project_path or "",
            "branch": record.branch or "",
            "domain": record.domain or "other",
            "start_timestamp": record.start_timestamp or "",
            "end_timestamp": record.end_timestamp or "",
            "duration_seconds": record.duration_seconds,
            "message_count": record.message_count,
            "files_count": len(record.files_touched),
            "errors_count": len(record.errors),
            "commits_count": len(record.commit_messages),
            "model": record.model or "",
            "slug": record.slug or "",
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "session-index",
        }

        # Store in ChromaDB sessions collection
        try:
            await self._knowledge.add_session(
                summary=doc_text,
                session_id=f"idx:{record.session_id}",
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"ChromaDB store failed for {record.session_id}: {e}")

        # Auto-feed decisions to AIIA memory
        source_tag = f"session-index:{record.session_id}"
        for decision in record.decisions:
            if decision and len(decision) > 10:
                self._memory.remember(
                    fact=decision,
                    category="decisions",
                    source=source_tag,
                )

        # Auto-feed solutions as lessons
        for solution in record.solutions:
            if isinstance(solution, dict):
                error = solution.get("error", "")
                fix = solution.get("fix", "")
                if error and fix:
                    self._memory.remember(
                        fact=f"Problem: {error} -> Fix: {fix}",
                        category="lessons",
                        source=source_tag,
                    )

        # Auto-feed patterns
        for pattern in record.patterns:
            if pattern and len(pattern) > 10:
                self._memory.remember(
                    fact=pattern,
                    category="patterns",
                    source=source_tag,
                )

        # Cloud sync — push session to Supermemory (fire-and-forget, failure-safe)
        if self._bridge and self._bridge.available and record.summary:
            try:
                await self._bridge.sync_session(
                    session_id=record.session_id,
                    summary=record.summary,
                    decisions=record.decisions,
                    lessons=[
                        f"Problem: {s.get('error', '')} -> Fix: {s.get('fix', '')}"
                        for s in record.solutions
                        if isinstance(s, dict) and s.get("error") and s.get("fix")
                    ],
                )
                logger.info(f"Cloud synced session {record.session_id}")
            except Exception as e:
                logger.warning(
                    f"Cloud sync failed for session {record.session_id}: {e}"
                )

    # ─── Pipeline Orchestrator ─────────────────────────────────

    async def run(self, force: bool = False) -> Dict[str, Any]:
        """Full pipeline: scan -> extract -> enrich -> store -> report."""
        start = time.monotonic()
        state = self._load_state()
        index = self._load_index()

        # Phase 1: Scan
        new_files = self.scan_for_new_files(force=force)

        if not new_files:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "files_scanned": 0,
                "files_indexed": 0,
                "total_sessions": len(index.get("sessions", {})),
                "total_latency_ms": round(elapsed, 1),
                "errors": [],
            }

        indexed = 0
        errors = []
        decisions_added = 0
        lessons_added = 0
        patterns_added = 0

        for path in new_files:
            try:
                logger.info(f"Processing: {os.path.basename(path)}")

                # Phase 2+3: Extract + Enrich
                record = await self.process_file(path)

                # Phase 4: Store
                await self.store_session(record)

                # Update state
                state[path] = {
                    "mtime": os.path.getmtime(path),
                    "session_id": record.session_id,
                    "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }

                # Update index
                index["sessions"][record.session_id] = asdict(record)

                indexed += 1
                decisions_added += len(record.decisions)
                lessons_added += len(record.solutions)
                patterns_added += len(record.patterns)

            except Exception as e:
                logger.error(f"Failed to process {path}: {e}", exc_info=True)
                errors.append(f"{os.path.basename(path)}: {e}")

        # Save state + index
        self._save_state(state)
        self._save_index(index)

        elapsed = (time.monotonic() - start) * 1000
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "files_scanned": len(new_files),
            "files_indexed": indexed,
            "total_sessions": len(index.get("sessions", {})),
            "decisions_added": decisions_added,
            "lessons_added": lessons_added,
            "patterns_added": patterns_added,
            "total_latency_ms": round(elapsed, 1),
            "errors": errors,
        }

        logger.info(
            f"Indexing complete: {indexed}/{len(new_files)} files, "
            f"{decisions_added} decisions, {lessons_added} lessons, "
            f"{patterns_added} patterns in {elapsed:.0f}ms"
        )

        return report
