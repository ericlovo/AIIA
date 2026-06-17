"""Standalone entry point for the packaged AIIA Brain.

This is what the PyInstaller-frozen `aiia-brain` binary runs when the desktop
app (aiia-console) spawns it as a Tauri sidecar. It differs from
`python -m local_brain.local_api` in two deliberate ways:

1. **Loopback-only by default.** The dev config binds 0.0.0.0 for LAN/tailnet
   use; a shipped single-user product must not expose the Brain to the network
   unless the user opts in. We default LOCAL_BRAIN_HOST to 127.0.0.1 here
   (still overridable via env for users who front it with Tailscale).

2. **Frozen-binary hygiene.** multiprocessing freeze_support (ChromaDB and
   uvicorn both fork workers in places) and an explicit single-process server.

Data stays under ~/.aiia (the config defaults), so memory survives app
updates and reinstalls.
"""

import logging
import multiprocessing
import os


def main() -> None:
    multiprocessing.freeze_support()

    # Loopback-only unless the user explicitly overrides (e.g. Tailscale).
    os.environ.setdefault("LOCAL_BRAIN_HOST", "127.0.0.1")

    import uvicorn

    from local_brain.config import get_config
    from local_brain.local_api import app

    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("aiia.standalone").info(
        "Starting packaged AIIA Brain on %s:%s (data: ~/.aiia)",
        config.api_host,
        config.api_port,
    )
    uvicorn.run(app, host=config.api_host, port=config.api_port, workers=1)


if __name__ == "__main__":
    main()
