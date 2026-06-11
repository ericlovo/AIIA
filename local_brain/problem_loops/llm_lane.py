"""LLM lanes for the problem loop — literature first.

Design rule: the model is never trusted on its own say-so.
  - Citations are verified BY CODE: we fetch each URL and require the verbatim
    quote to appear in the page text. A hallucinated source fails mechanically.
  - The refuter independently re-derives the status without seeing the prover's
    sources or reasoning; the arbiter requires agreement.
  - Promotion requires: parsed output + >= MIN_VERIFIED_CITATIONS code-verified
    citations from independent domains + refuter agreement. Otherwise nothing
    is written to memory (the iron rule).

The SDK adapter (SDKClient) lazy-imports claude_agent_sdk so this module — and
all its logic — is fully testable with a fake client where the SDK or an API
key is unavailable.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger("aiia.problem_loops.llm")

PROMPTS_DIR = Path(__file__).parent / "prompts"
MIN_VERIFIED_CITATIONS = 2
MAX_QUOTE_WORDS = 60  # tolerate slightly over the prompt's 40
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()


# ── LLM client protocol + SDK adapter ───────────────────────────


class SDKClient:
    """Real adapter over claude_agent_sdk.query (the story_runner pattern).

    Web tools enabled so the prover/refuter can actually search the literature.
    Lazy import: constructing this without the SDK raises a clear error.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: int = 20):
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "claude_agent_sdk is not installed (and an ANTHROPIC_API_KEY is "
                "required at runtime). The literature lane runs on the Mini; "
                "use --dry-run with the fake client for local testing."
            ) from e
        self.model = model
        self.max_turns = max_turns

    async def complete(self, prompt: str, budget_usd: float) -> dict[str, Any]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        text_parts: list[str] = []
        cost = 0.0
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=self.model,
                max_turns=self.max_turns,
                max_budget_usd=budget_usd,
                allowed_tools=["WebSearch", "WebFetch"],
                disallowed_tools=["Edit", "Write", "Bash"],
                permission_mode="acceptEdits",
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
        return {"text": "\n".join(text_parts), "cost_usd": cost}


# ── Output parsing ──────────────────────────────────────────────


def parse_json_block(text: str) -> dict[str, Any] | None:
    """Extract the last fenced ```json block (the required output contract)."""
    fences = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    for raw in reversed(fences):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    # tolerate a bare JSON object as a fallback
    try:
        start = text.index("{")
        return json.loads(text[start:])
    except (ValueError, json.JSONDecodeError):
        return None


# ── Code-verified citations ─────────────────────────────────────


def _default_fetch(url: str) -> str:
    import httpx

    resp = httpx.get(
        url,
        timeout=15,
        follow_redirects=True,
        headers={"User-Agent": "AIIA-problem-loop/0.1 (citation verification)"},
    )
    resp.raise_for_status()
    return resp.text


def _normalize(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # strip tags
    text = re.sub(r"&[a-z#0-9]+;", " ", text)  # entities → space
    return re.sub(r"\s+", " ", text).lower().strip()


def verify_citations(
    citations: list[dict[str, Any]],
    fetch: Callable[[str], str] = _default_fetch,
) -> list[dict[str, Any]]:
    """Mechanically verify each citation: fetch the URL, find the verbatim quote.

    Returns the citation list annotated with `verified` + `failure` fields.
    Never raises — a fetch error is just an unverified citation.
    """
    results = []
    for c in citations or []:
        out = dict(c)
        out["verified"] = False
        out["failure"] = None
        url, quote = c.get("url", ""), c.get("quote", "")
        if not url or not quote:
            out["failure"] = "missing url or quote"
        elif len(quote.split()) > MAX_QUOTE_WORDS:
            out["failure"] = f"quote too long (> {MAX_QUOTE_WORDS} words)"
        else:
            try:
                page = _normalize(fetch(url))
                if _normalize(quote) in page:
                    out["verified"] = True
                else:
                    out["failure"] = "quote not found on page"
            except Exception as e:
                out["failure"] = f"fetch failed: {e.__class__.__name__}"
        results.append(out)
    return results


def _independent_domains(verified: list[dict[str, Any]]) -> int:
    domains = set()
    for c in verified:
        if c["verified"]:
            host = urlparse(c.get("url", "")).netloc.removeprefix("www.")
            if host:
                domains.add(host)
    return len(domains)


# ── The literature cycle ────────────────────────────────────────


async def literature_cycle(
    problem: dict[str, Any],
    llm: Any,
    budget_usd: float = 1.0,
    fetch: Callable[[str], str] = _default_fetch,
) -> dict[str, Any]:
    """Prover surveys → citations code-verified → refuter independently re-derives
    → arbiter promotes only on full agreement + mechanical verification."""
    pid = problem["id"]

    # PROVER
    prover_prompt = load_prompt("literature_prover").format(
        problem_id=pid,
        name=problem["name"],
        statement=problem["statement"],
        status_prior=problem["status"],
        status_confidence=problem.get("status_confidence", "unknown"),
    )
    prover_raw = await llm.complete(prover_prompt, budget_usd=budget_usd * 0.55)
    prover = parse_json_block(prover_raw["text"])
    cost = prover_raw.get("cost_usd", 0.0)

    record: dict[str, Any] = {
        "problem_id": pid,
        "mode": "literature",
        "rung_reached": "status",
        "promoted": False,
        "cost_usd": cost,
        "prover": prover,
    }
    if not prover or "status" not in prover:
        record["failure"] = "prover output unparseable"
        logger.warning(f"[arbiter] {pid}: prover output unparseable — not promoted")
        return record

    # CODE VERIFICATION of prover citations
    checked = verify_citations(prover.get("citations", []), fetch=fetch)
    record["citations_checked"] = checked
    n_domains = _independent_domains(checked)
    citations_ok = n_domains >= MIN_VERIFIED_CITATIONS

    # REFUTER — independent derivation; sees only the claim, not the sources
    refuter_prompt = load_prompt("literature_refuter").format(
        problem_id=pid,
        name=problem["name"],
        statement=problem["statement"],
        prover_status=prover["status"],
        prover_frontier=prover.get("frontier_summary", ""),
    )
    refuter_raw = await llm.complete(refuter_prompt, budget_usd=budget_usd * 0.45)
    refuter = parse_json_block(refuter_raw["text"])
    cost += refuter_raw.get("cost_usd", 0.0)
    record["cost_usd"] = cost
    record["refuter"] = refuter

    refuter_agrees = bool(refuter and refuter.get("status") == prover["status"])

    # ARBITER
    promoted = citations_ok and refuter_agrees
    record["promoted"] = promoted
    record["arbiter"] = {
        "verified_citation_domains": n_domains,
        "citations_ok": citations_ok,
        "refuter_agrees": refuter_agrees,
    }
    if promoted:
        record["claim"] = (
            f"{problem['name']} ({pid}): status verified as "
            f"'{prover['status']}' — {prover.get('frontier_summary', '')} "
            f"[{n_domains} code-verified independent sources; refuter concurs]"
        )
        if prover.get("proposed_registry_update"):
            record["proposed_registry_update"] = prover["proposed_registry_update"]
    else:
        logger.warning(
            f"[arbiter] {pid}: NOT promoted — citations_ok={citations_ok} "
            f"(verified domains: {n_domains}), refuter_agrees={refuter_agrees}"
        )
    return record
