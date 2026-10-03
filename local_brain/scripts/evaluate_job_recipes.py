"""Opt-in local model evaluation; synthetic inputs, no Studio records or verdicts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from local_brain.command_center.agent_prompts import agent_system_prompt, assignment_prompt

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "local_brain/tests/fixtures/repository_job_quality.json"
MODEL = "qwen3:8b"
OLLAMA = "http://127.0.0.1:11434"
HEADINGS = ["## Evidence", "## Findings", "## Next action"]


def prepare_cases() -> list[dict[str, Any]]:
    exported = subprocess.run(
        ["node", "--experimental-strip-types", "dashboard/scripts/export-job-recipes.mjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    jobs = {job["recipe_id"]: job for job in json.loads(exported.stdout)}
    prepared = []
    for case in json.loads(FIXTURES.read_text())["cases"]:
        job = jobs[case["recipe_id"]]
        agent = job["agent"]
        prepared.append(
            {
                **case,
                "request": {
                    "model": MODEL,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": agent["temperature"],
                        "num_predict": agent["max_tokens"],
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": agent_system_prompt(agent, [case["repository_context"]]),
                        },
                        {"role": "user", "content": assignment_prompt(job["assignment"])},
                    ],
                },
            }
        )
    return prepared


def format_checks(text: str, required_refs: list[str]) -> dict[str, bool]:
    sections: dict[str, list[str]] = {}
    current = ""
    seen = []
    extra_text = False
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = line
            seen.append(line)
            sections.setdefault(line, [])
        elif current:
            sections[current].append(line)
        else:
            extra_text = True
    bullets = all(
        1 <= len(sections.get(heading, [])) <= limit
        and all(line.startswith("- ") for line in sections.get(heading, []))
        for heading, limit in zip(HEADINGS, [3, 2, 1], strict=True)
    )
    return {
        "word_limit": 0 < len(text.split()) <= 220,
        "three_sections": seen == HEADINGS and not extra_text,
        "bullet_limits": bullets,
        "required_refs_present": all(ref in text for ref in required_refs),
    }


def require_local_model(client: httpx.Client) -> None:
    response = client.post(f"{OLLAMA}/api/show", json={"model": MODEL})
    response.raise_for_status()
    model = response.json()
    if (
        model.get("remote_host")
        or model.get("remote_model")
        or model.get("details", {}).get("format") != "gguf"
    ):
        raise ValueError("Evaluation requires the locally installed GGUF model; no cloud model.")


def evaluate(client: httpx.Client, case: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"{OLLAMA}/api/chat", json=case["request"])
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
        raise ValueError("Invalid model response")
    text = payload["message"].get("content", "")
    if not isinstance(text, str):
        raise ValueError("Invalid model response content")
    checks = format_checks(text, case["required_refs"])
    checks["generation_complete"] = (
        payload.get("done") is True and payload.get("done_reason") == "stop"
    )
    checks["expected_model"] = payload.get("model") == MODEL
    duration = payload.get("total_duration")
    return {
        "id": case["id"],
        "recipe_id": case["recipe_id"],
        "output": text,
        "words": len(text.split()),
        "checks": checks,
        "input_tokens": payload.get("prompt_eval_count"),
        "output_tokens": payload.get("eval_count"),
        "duration_ms": duration / 1_000_000 if type(duration) is int and duration >= 0 else None,
        "done_reason": payload.get("done_reason"),
        "review_expectations": case["review_expectations"],
        "human_review": "pending",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Run serial local inference; otherwise prepare only"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for requests and results"
    )
    args = parser.parse_args(argv)
    cases = prepare_cases()
    if not cases:
        parser.error("At least one evaluation case is required.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "requests.json").write_text(json.dumps(cases, indent=2) + "\n")
    if not args.run:
        print(
            f"Prepared {len(cases)} cases. No model called. Add --run with a new output directory."
        )
        return 0
    report: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "cases": [],
    }
    # No proxy inheritance, redirects, retries, model pulls, or cloud fallback.
    with httpx.Client(
        timeout=httpx.Timeout(120, connect=5), trust_env=False, follow_redirects=False
    ) as client:
        require_local_model(client)
        for case in cases:
            try:
                result = evaluate(client, case)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                result = {"id": case["id"], "error": type(exc).__name__, "human_review": "pending"}
            report["cases"].append(result)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{case['id']}: {result.get('checks', result.get('error'))}", flush=True)
    return (
        0
        if all(all(row.get("checks", {"error": False}).values()) for row in report["cases"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
