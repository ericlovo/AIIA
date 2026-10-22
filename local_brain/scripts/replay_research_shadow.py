"""Run one synthetic replay in disposable state; never contacts a service."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from local_brain.command_center.research_shadow import (
    enqueue_replay,
    public_job,
    run_once,
    synthetic_example,
)
from local_brain.command_center.research_shadow_store import ResearchShadowStore


def main():
    config, fixture = synthetic_example()
    with TemporaryDirectory(prefix="aiia-research-shadow-") as directory:
        path = Path(directory) / "shadow.sqlite3"
        store = ResearchShadowStore(path)
        job = enqueue_replay(store, config, fixture)
        result = run_once(store, config.tenant_id, job["id"], "synthetic-worker")
        if result is None or result["state"] != "completed":
            raise RuntimeError("synthetic replay did not complete")
        restarted = ResearchShadowStore(path)
        duplicate = enqueue_replay(restarted, config, fixture)
        if duplicate["id"] != job["id"] or duplicate["receipt"] != result["receipt"]:
            raise RuntimeError("restart changed accepted replay")
        if run_once(restarted, config.tenant_id, job["id"], "second-worker") is not None:
            raise RuntimeError("completed replay executed twice")
        print(json.dumps({"job": public_job(result), "restart_deduplicated": True}, indent=2))


if __name__ == "__main__":
    main()
