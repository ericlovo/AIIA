"""Standalone loopback probe sent to the Mini over SSH stdin."""

import json

import httpx


def run_mini_loopback(model, operation, prompt):
    """Check the airgap before querying the local Qwen model."""
    with httpx.Client(timeout=90, trust_env=False, follow_redirects=False) as client:
        response = client.get("http://127.0.0.1:8100/health", timeout=10)
        response.raise_for_status()
        airgap = response.json()["airgap"]
        if (
            airgap.get("enabled") is not True
            or airgap.get("egress", {}).get("typesafe.signals") != "disabled"
        ):
            raise ValueError("airgap_required")
        if operation == "health":
            return airgap
        response = client.get("http://127.0.0.1:11434/api/ps", timeout=10)
        response.raise_for_status()
        if response.json().get("models") != []:
            raise ValueError("local_runtime_busy")
        response = client.post("http://127.0.0.1:11434/api/show", json={"model": model}, timeout=10)
        response.raise_for_status()
        info = response.json()
        if (
            info.get("remote_host")
            or info.get("remote_model")
            or info.get("details", {}).get("format") != "gguf"
        ):
            raise ValueError("local_gguf_required")
        response = client.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "keep_alive": 0,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 256, "num_ctx": 2048},
            },
        )
        response.raise_for_status()
        result = response.json()
    if (
        result.get("done") is not True
        or result.get("done_reason") != "stop"
        or result.get("model") != model
    ):
        raise ValueError("incomplete_local_generation")
    return {
        "extraction": json.loads(result["message"]["content"]),
        "model": result["model"],
        "eval_count": result.get("eval_count"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "total_duration": result.get("total_duration"),
    }
