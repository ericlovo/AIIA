"""
Distillation — turn a raw transcript into a structured journal markdown note.

Routes through whichever chat provider has its API key configured (env vars):
Anthropic → OpenAI → Groq. The prompt is journaling-specific: prose-forward,
first person, reads like a continuation of the speaker's voice rather than a
meeting summary.

Mirrors the JS-side prompt in aiia-console/src/journal/distill.ts so the
desktop and mobile flows produce structurally identical notes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime

import httpx

from local_brain.sanction import log_tokens_bg

logger = logging.getLogger("aiia.journal.distiller")


class DistillationError(RuntimeError):
    """Raised when no provider is available or the upstream call fails."""


@dataclass(frozen=True)
class DistillationInput:
    transcript: str
    started_at: datetime
    duration_seconds: int
    transcription_provider: str = "groq"


@dataclass(frozen=True)
class DistillationResult:
    markdown: str
    provider: str
    model: str


# Default model per provider — tuned for fast, prose-good output suited to
# journaling. Anthropic is preferred because Claude writes journal-shaped
# prose particularly well.
_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-4-5"),
    "openai": ("OPENAI_API_KEY", "gpt-5-mini"),
    "groq": ("GROQ_API_KEY", "llama-3.3-70b-versatile"),
}

_PROVIDER_ORDER: tuple[str, ...] = ("anthropic", "openai", "groq")


def _pick_provider() -> tuple[str, str] | None:
    """Return (provider, model) for the first configured provider, or None."""
    for provider in _PROVIDER_ORDER:
        env_key, default_model = _PROVIDER_DEFAULTS[provider]
        if os.getenv(env_key, "").strip():
            return provider, default_model
    return None


def _system_prompt(inp: DistillationInput) -> str:
    started_iso = inp.started_at.isoformat()
    started_date = inp.started_at.date().isoformat()
    return "\n".join(
        [
            "You are a thoughtful journaling partner.",
            "The user just finished speaking aloud in a journaling session.",
            "Your job is to distill the raw transcript into a Markdown note they",
            "will read back later — something that feels personal, captures what",
            "they're actually working through, and reads as a continuation of",
            "their voice rather than a meeting summary.",
            "",
            "Output ONLY the markdown, no preamble. Begin with YAML frontmatter,",
            "then the body. Use serif-friendly prose, not bullet-lists-everywhere",
            "style. If a section has nothing to say, OMIT it — do not write",
            "placeholder bullets.",
            "",
            "Required frontmatter keys (use these exact values, do not invent):",
            f"  date: {started_date}",
            f"  started: {started_iso}",
            f"  duration_seconds: {inp.duration_seconds}",
            f"  transcription: {inp.transcription_provider}",
            "  aiia_managed: true",
            "  tags: [journal, session]",
            "",
            "Body structure (markdown headings):",
            "  # {Title — 4-7 words capturing the essence; no period}",
            "  {2-3 sentence opening that sets the scene}",
            "  ## What I'm working through",
            "  {2-4 paragraphs of prose, first person.}",
            "  ## Threads",
            "  - {3-7 bullets, each one line, naming a topic the session touched}",
            "  ## Decisions",
            "  - {only if decisions were actually made}",
            "  ## Open questions",
            "  - {1-3 questions still unresolved that deserve coming back to}",
        ]
    )


def _user_prompt(transcript: str) -> str:
    return f"Raw transcript follows. Distill it.\n\n---\n{transcript}\n---"


async def _distill_anthropic(
    system: str,
    user: str,
    *,
    model: str,
    timeout: float,
) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
    if resp.status_code >= 400:
        raise DistillationError(f"Anthropic returned HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text.strip():
        raise DistillationError("Anthropic returned empty content")
    usage = data.get("usage", {})
    log_tokens_bg(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), task="journal-distill")
    return text


async def _distill_openai(
    system: str,
    user: str,
    *,
    model: str,
    timeout: float,
) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
    if resp.status_code >= 400:
        raise DistillationError(f"OpenAI returned HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise DistillationError("OpenAI returned no choices")
    return choices[0].get("message", {}).get("content", "")


async def _distill_groq(
    system: str,
    user: str,
    *,
    model: str,
    timeout: float,
) -> str:
    # OpenAI-compatible. Same payload shape; different URL + key.
    api_key = os.environ["GROQ_API_KEY"]
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
    if resp.status_code >= 400:
        raise DistillationError(f"Groq returned HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise DistillationError("Groq returned no choices")
    return choices[0].get("message", {}).get("content", "")


_DISPATCH = {
    "anthropic": _distill_anthropic,
    "openai": _distill_openai,
    "groq": _distill_groq,
}


async def distill(
    inp: DistillationInput,
    *,
    timeout_seconds: float = 120.0,
) -> DistillationResult:
    """Distill the transcript into a journal markdown note.

    Picks the first configured provider in order [anthropic, openai, groq].
    Raises DistillationError if none are configured.
    """
    pick = _pick_provider()
    if pick is None:
        raise DistillationError(
            "No chat provider configured for distillation. Set one of "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY."
        )
    provider, model = pick
    handler = _DISPATCH[provider]
    system = _system_prompt(inp)
    user = _user_prompt(inp.transcript)
    markdown = await handler(system, user, model=model, timeout=timeout_seconds)
    if not markdown.strip():
        raise DistillationError(f"{provider} returned empty markdown")
    logger.info(
        "distilled session: provider=%s model=%s out_chars=%d in_chars=%d",
        provider,
        model,
        len(markdown),
        len(inp.transcript),
    )
    return DistillationResult(markdown=markdown, provider=provider, model=model)
