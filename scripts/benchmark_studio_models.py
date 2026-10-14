#!/usr/bin/env python3
"""Inventory local Ollama by default; --run explicitly enables synthetic inference."""

import argparse
import json
import time
import urllib.request
from urllib.parse import urlsplit

PROMPT = (
    "Summarize this synthetic incident in exactly three bullets: a scheduled import "
    "failed because its input file was absent; retry succeeded after the file arrived; "
    "no records were lost. Do not add facts."
)


def request(base, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": "application/json"}
    )

    # Local endpoint only, without proxy environment or redirects to other hosts.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=180 if payload is not None else 5) as response:
        return json.load(response)


def measure(raw, elapsed_ms):
    result = {"wall_ms": round(elapsed_ms, 2), "done_reason": raw.get("done_reason")}
    for key in (
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "prompt_eval_count",
        "eval_count",
    ):
        value = raw.get(key)
        value = value if type(value) is int and 0 <= value <= 2**63 - 1 else None
        output_key = key.replace("_duration", "_ms")
        result[output_key] = (
            value / 1_000_000 if value is not None and key.endswith("_duration") else value
        )
    duration, tokens = result["eval_ms"], result["eval_count"]
    result["decode_tokens_per_second"] = (
        tokens * 1000 / duration if duration and tokens is not None else None
    )
    return result


def main(argv=None, api=request):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--run", action="store_true", help="Opt in to local synthetic inference")
    parser.add_argument(
        "--model", action="append", default=[], help="Exact installed name; repeat to compare"
    )
    parser.add_argument("--num-ctx", type=int, choices=(4096, 8192, 32768), default=4096)
    parser.add_argument("--max-tokens", type=int, choices=(128, 256, 512, 1200), default=256)
    args = parser.parse_args(argv)
    parsed = urlsplit(args.url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        parser.error(
            "--url must be a loopback HTTP Ollama endpoint; use an SSH tunnel for the Mini"
        )
    base = args.url.rstrip("/")
    tags, running = api(base, "/api/tags"), api(base, "/api/ps")
    names = {model["name"] for model in tags["models"]}

    def metadata(model):
        result = {
            key: model[key]
            for key in ("name", "digest", "size", "context_length", "size_vram")
            if key in model
        }
        details = model.get("details")
        if isinstance(details, dict):
            result["details"] = {
                key: details[key]
                for key in ("family", "parameter_size", "quantization_level")
                if key in details
            }
        return result

    version = api(base, "/api/version")
    inventory = {
        "version": version.get("version"),
        "installed": [
            metadata(model) for model in sorted(tags["models"], key=lambda item: item["name"])
        ],
        "loaded": [metadata(model) for model in running["models"]],
    }
    print(json.dumps({"inventory": inventory}))
    if not args.run:
        return 0
    if not args.model or any(model not in names for model in args.model):
        parser.error("--run requires exact installed --model names; no models are downloaded")
    selected = list(dict.fromkeys(args.model))
    installed = {item["name"]: item for item in tags["models"]}

    def remote(metadata):
        return any(metadata.get(key) for key in ("remote_host", "remote_model"))

    # Preflight every selection before any inference, including locally renamed aliases.
    for model in selected:
        if model.lower().endswith((":cloud", "-cloud")) or remote(installed[model]):
            parser.error("cloud-backed models are not allowed in this local benchmark")
        shown = api(base, "/api/show", {"model": model})
        if not isinstance(shown, dict) or remote(shown):
            parser.error("model metadata does not confirm local inference")
        info = shown.get("model_info")
        if not isinstance(info, dict) or not info.get("general.architecture"):
            parser.error("local model architecture is unavailable; refusing inference")
    for model in selected:
        for attempt in range(2):
            before = api(base, "/api/ps")
            resident = any(item["name"] == model for item in before["models"])
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": args.max_tokens,
                    "num_ctx": args.num_ctx,
                },
            }
            started = time.monotonic()
            raw = api(base, "/api/chat", payload)
            elapsed = (time.monotonic() - started) * 1000
            print(
                json.dumps(
                    {
                        "model": model,
                        "attempt": attempt + 1,
                        "resident_before": resident,
                        "num_ctx": args.num_ctx,
                        "max_tokens": args.max_tokens,
                        **measure(raw, elapsed),
                        "content": raw.get("message", {}).get("content", ""),
                    }
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
