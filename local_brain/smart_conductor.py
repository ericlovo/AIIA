"""
Smart Conductor — Local LLM replaces keyword matching for query routing.

This is Phase 2 of the Local Brain architecture. Instead of 350 lines of
keyword lists and regex patterns in conductor.py, a local 8B model does
genuine intent classification with:
- Domain detection (finance, legal, compliance, estate, marketing, etc.)
- EQ level assessment (Fibonacci 1-21)
- Complexity scoring (simple → RLM-worthy)
- Routing recommendation (local, eos, rlm)

The local model runs on Mac Mini via Ollama — zero cost, sub-second latency.

Fallback: If local model is unavailable, returns a "fallback" response that
tells the production Conductor to use its existing keyword matching.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from local_brain.ollama_client import OllamaClient
from local_brain.config import LocalBrainConfig

logger = logging.getLogger("aiia.local_brain.conductor")


# The classification prompt — this is the brain of the routing system.
# A local 8B model can handle structured classification with high accuracy.
ROUTING_SYSTEM_PROMPT = """You are AIIA's routing intelligence. Your job is to classify user queries
so they reach the right agent with the right emotional tone.

Analyze the query and return a JSON object with exactly these fields:

{
  "domain": one of ["finance", "legal", "compliance", "document", "memory", "crisis", "estate", "marketing", "social", "general"],
  "eq_level": integer from Fibonacci sequence [1, 2, 3, 5, 8, 13, 21],
  "eq_mode": one of ["analyst", "guide", "supporter", "advocate"],
  "complexity_score": float 0.0 to 1.0,
  "recommended_path": one of ["local", "eos", "rlm"],
  "confidence": float 0.0 to 1.0,
  "reasoning": brief explanation (1 sentence max)
}

Classification rules:

DOMAIN:
- finance: revenue, profit, billing, expenses, budgets, P&L, Rule of Thirds, financial metrics
- legal: cases, lawsuits, settlements, litigation, contracts, court matters
- compliance: HIPAA, GDPR, PII, privacy, security audits, risk assessment
- document: file uploads, parsing, extraction, spreadsheets
- memory: "remember", "last time", "we discussed", user preferences
- crisis: suicide, self-harm, emergency, 911 — ALWAYS level 13+ ADVOCATE
- estate: wills, trusts, probate, inheritance, beneficiaries, estate tax
- marketing: brand strategy, campaigns, funnels, content strategy, SEO
- social: social media, Instagram, LinkedIn, TikTok, posts, engagement
- general: everything else, greetings, small talk, unclear intent

EQ LEVEL (Fibonacci emotional scaling):
- 1-2: Neutral, task-focused → ANALYST mode
- 3-5: Some emotional content, needs guidance → GUIDE mode
- 5-8: Significant emotional need, distress → SUPPORTER mode
- 13-21: Crisis, self-harm, emergency → ADVOCATE mode (always flag for human review)

COMPLEXITY:
- 0.0-0.3: Simple question, greeting, single lookup → "local" can handle
- 0.3-0.6: Moderate, needs domain knowledge → "eos" (single Claude call)
- 0.6-1.0: Multi-step analysis, comparisons, documents → "rlm" (agentic loop)

Signals that increase complexity:
- Multiple domains in one query
- Comparative/analytical language (compare, contrast, benchmark, forecast)
- Multi-step instructions (first... then... finally...)
- Document analysis with specific asks
- Explicit depth requests (deep dive, comprehensive, thorough)

Return ONLY the JSON object. No markdown, no explanation outside the JSON."""


class SmartConductor:
    """
    LLM-powered query router running on local Mac Mini.

    Replaces keyword matching in conductor.py with genuine intent understanding.
    Falls back gracefully if Ollama is unavailable.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        config: LocalBrainConfig,
    ):
        self.ollama = ollama
        self.config = config
        self._routing_model = config.models.get("routing")

    async def route(
        self,
        query: str,
        tenant_id: str = "default",
        has_documents: bool = False,
        document_count: int = 0,
        conversation_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify and route a user query using local LLM.

        Args:
            query: User's message
            tenant_id: Which tenant this is for (affects domain defaults)
            has_documents: Whether documents are attached
            document_count: Number of attached documents
            conversation_context: Optional recent conversation for context

        Returns:
            Dict with domain, eq_level, complexity_score, recommended_path, etc.
        """
        start = time.monotonic()

        # Build the classification prompt
        user_prompt = self._build_user_prompt(
            query, tenant_id, has_documents, document_count, conversation_context
        )

        try:
            model = (
                self._routing_model.model_name if self._routing_model else "llama3:8b"
            )
            temperature = (
                self._routing_model.temperature if self._routing_model else 0.1
            )
            max_tokens = self._routing_model.max_tokens if self._routing_model else 256

            response = await self.ollama.chat(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                system=ROUTING_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.get("message", {}).get("content", "")
            result = self._parse_routing_response(content)

            latency = (time.monotonic() - start) * 1000
            result["latency_ms"] = round(latency, 1)

            logger.info(
                f"Smart route: domain={result['domain']}, "
                f"eq={result['eq_level']}, "
                f"complexity={result['complexity_score']}, "
                f"path={result['recommended_path']}, "
                f"latency={latency:.0f}ms"
            )

            return result

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning(f"Smart Conductor failed ({latency:.0f}ms): {e}")
            return self._fallback_response(query, latency)

    def _build_user_prompt(
        self,
        query: str,
        tenant_id: str,
        has_documents: bool,
        document_count: int,
        conversation_context: Optional[str],
    ) -> str:
        """Build the classification prompt with all available signals."""
        parts = [f"Query: {query}"]

        if tenant_id:
            parts.append(f"Tenant: {tenant_id}")
        if has_documents:
            parts.append(f"Documents attached: {document_count}")
        if conversation_context:
            # Truncate context to keep prompt small for fast routing
            ctx = conversation_context[:500]
            parts.append(f"Recent conversation:\n{ctx}")

        return "\n".join(parts)

    def _parse_routing_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response, with fallback extraction."""
        # Try direct JSON parse
        try:
            result = json.loads(content)
            return self._validate_result(result)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in content:
            try:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                result = json.loads(json_str.strip())
                return self._validate_result(result)
            except (json.JSONDecodeError, IndexError):
                pass

        # Try finding JSON object in the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(content[start:end])
                return self._validate_result(result)
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse routing response: {content[:200]}")
        return self._default_result()

    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the parsed routing result."""
        valid_domains = [
            "finance",
            "legal",
            "compliance",
            "document",
            "memory",
            "crisis",
            "estate",
            "marketing",
            "social",
            "general",
        ]
        valid_modes = ["analyst", "guide", "supporter", "advocate"]
        valid_paths = ["local", "eos", "rlm"]
        fibonacci = [1, 2, 3, 5, 8, 13, 21]

        # Normalize domain
        domain = result.get("domain", "general")
        if domain not in valid_domains:
            domain = "general"

        # Normalize EQ level to nearest Fibonacci
        eq_level = result.get("eq_level", 1)
        if eq_level not in fibonacci:
            eq_level = min(fibonacci, key=lambda x: abs(x - eq_level))

        # Normalize EQ mode
        eq_mode = result.get("eq_mode", "analyst")
        if eq_mode not in valid_modes:
            if eq_level >= 13:
                eq_mode = "advocate"
            elif eq_level >= 5:
                eq_mode = "supporter"
            elif eq_level >= 3:
                eq_mode = "guide"
            else:
                eq_mode = "analyst"

        # Normalize complexity
        complexity = float(result.get("complexity_score", 0.3))
        complexity = max(0.0, min(1.0, complexity))

        # Normalize path
        path = result.get("recommended_path", "eos")
        if path not in valid_paths:
            if complexity >= 0.6:
                path = "rlm"
            elif complexity <= 0.3:
                path = "local"
            else:
                path = "eos"

        return {
            "domain": domain,
            "eq_level": eq_level,
            "eq_mode": eq_mode,
            "complexity_score": round(complexity, 3),
            "recommended_path": path,
            "confidence": float(result.get("confidence", 0.8)),
            "reasoning": result.get("reasoning", "Classified by local model"),
            "latency_ms": 0.0,  # Filled in by caller
        }

    def _default_result(self) -> Dict[str, Any]:
        """Default routing when parsing fails entirely."""
        return {
            "domain": "general",
            "eq_level": 1,
            "eq_mode": "analyst",
            "complexity_score": 0.3,
            "recommended_path": "eos",
            "confidence": 0.5,
            "reasoning": "Default routing (parse failure)",
            "latency_ms": 0.0,
        }

    def _fallback_response(self, query: str, latency: float) -> Dict[str, Any]:
        """
        Fallback when local model is unavailable.

        Returns a signal that tells the production Conductor to use
        its existing keyword matching instead.
        """
        return {
            "domain": "general",
            "eq_level": 1,
            "eq_mode": "analyst",
            "complexity_score": 0.3,
            "recommended_path": "eos",
            "confidence": 0.0,  # Zero confidence = use keyword fallback
            "reasoning": "Local model unavailable — use keyword conductor",
            "latency_ms": round(latency, 1),
        }
