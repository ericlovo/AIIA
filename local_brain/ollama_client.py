"""
Ollama Client — Direct interface to the local LLM runtime.

Wraps Ollama's REST API with:
- Chat completions (with system prompt support)
- Embeddings generation
- Model management (list, pull, health check)
- Streaming support for real-time responses

This is the low-level client. Higher-level services (smart conductor,
summarizer, etc.) use this client internally.
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from local_brain.config import LocalBrainConfig, get_config

logger = logging.getLogger("aiia.local_brain.ollama")


class OllamaClient:
    """
    Async client for Ollama's REST API.

    Usage:
        client = OllamaClient()
        response = await client.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": "Hello"}],
            system="You are AIIA.",
        )
        print(response["message"]["content"])
    """

    def __init__(self, config: Optional[LocalBrainConfig] = None):
        self.config = config or get_config()
        self.base_url = self.config.ollama_url
        self.timeout = self.config.ollama_timeout

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        num_ctx: int = 8192,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Ollama.

        Args:
            model: Model name (e.g., "llama3:8b")
            messages: Chat messages [{"role": "user/assistant/system", "content": "..."}]
            system: System prompt (prepended as system message)
            temperature: Response randomness
            max_tokens: Max response tokens
            stream: Whether to stream the response
            num_ctx: Context window size (8192 default, use 4096 for voice)
            timeout: Override default timeout (seconds) — use for slow models like DeepSeek R1

        Returns:
            Ollama chat response dict with message, usage info, timing
        """
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(messages)

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
                "num_batch": 512,
                "num_gpu": 99,
            },
        }

        start = time.monotonic()

        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.monotonic() - start) * 1000

        logger.debug(
            f"Ollama [{model}]: "
            f"in={data.get('prompt_eval_count', 0)}, "
            f"out={data.get('eval_count', 0)}, "
            f"latency={latency_ms:.0f}ms"
        )

        data["_latency_ms"] = latency_ms
        return data

    async def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        num_ctx: int = 8192,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from Ollama, yielding content chunks.

        Yields:
            Content string chunks as they arrive
        """
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(messages)

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
                "num_batch": 512,
                "num_gpu": 99,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        import json

                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            return

    async def embed(
        self,
        model: str,
        text: str,
    ) -> List[float]:
        """
        Generate embeddings for text using a local embedding model.

        Args:
            model: Embedding model name (e.g., "nomic-embed-text")
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()

        return data.get("embedding", [])

    async def embed_batch(
        self,
        model: str,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            model: Embedding model name
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed(model, text)
            embeddings.append(embedding)
        return embeddings

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models available in Ollama."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
        return data.get("models", [])

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from the Ollama registry.

        Args:
            model: Model name to pull (e.g., "llama3:8b")

        Returns:
            True if successful
        """
        logger.info(f"Pulling model: {model}")
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
            )
            response.raise_for_status()
        logger.info(f"Model pulled: {model}")
        return True

    async def health(self) -> Dict[str, Any]:
        """
        Check Ollama health and return system status.

        Returns:
            Dict with status, models, and system info
        """
        try:
            start = time.monotonic()
            models = await self.list_models()
            latency = (time.monotonic() - start) * 1000

            model_names = [m.get("name", "unknown") for m in models]

            return {
                "status": "online",
                "url": self.base_url,
                "models": model_names,
                "model_count": len(models),
                "latency_ms": round(latency, 1),
            }
        except httpx.ConnectError:
            return {
                "status": "offline",
                "url": self.base_url,
                "error": "Cannot connect to Ollama. Is it running?",
                "models": [],
            }
        except Exception as e:
            return {
                "status": "error",
                "url": self.base_url,
                "error": str(e),
                "models": [],
            }
