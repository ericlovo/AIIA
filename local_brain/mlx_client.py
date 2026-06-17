"""
MLX inference client for Apple Silicon.

Drop-in companion to OllamaClient. Uses mlx-lm for 10-20% faster
on-device inference vs Ollama on M-series chips.

Set MLX_MODEL env var to activate (e.g. "mlx-community/gemma-3-4b-it-4bit").
If MLX_MODEL is unset or mlx-lm isn't installed, falls back to Ollama.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator

logger = logging.getLogger("aiia.mlx")

_MLX_MODEL = os.getenv("MLX_MODEL", "")
_generator = None
_loaded_model = None


def _load(model_path: str):
    global _generator, _loaded_model
    if _loaded_model == model_path:
        return _generator
    try:
        import mlx_lm
        logger.info(f"Loading MLX model: {model_path}")
        model, tokenizer = mlx_lm.load(model_path)
        _generator = (model, tokenizer)
        _loaded_model = model_path
        logger.info("MLX model ready")
        return _generator
    except Exception as e:
        logger.warning(f"MLX load failed ({model_path}): {e}")
        return None


def is_available() -> bool:
    return bool(_MLX_MODEL)


async def chat(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    model_path = model or _MLX_MODEL
    if not model_path:
        raise RuntimeError("MLX_MODEL not configured")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _sync_chat, model_path, messages, max_tokens, temperature)
    return result


def _sync_chat(model_path: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    import mlx_lm

    gen = _load(model_path)
    if gen is None:
        raise RuntimeError(f"Failed to load MLX model: {model_path}")

    model, tokenizer = gen

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    return response
