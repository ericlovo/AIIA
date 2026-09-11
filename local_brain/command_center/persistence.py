"""Atomic JSON persistence shared by the Agent Studio registries."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class PersistenceError(RuntimeError):
    """Raised when a registry could not store its state."""


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` to ``path`` so readers never observe a partial file.

    The JSON is serialised to a temporary file in the same directory and then
    moved into place with ``os.replace``, which is atomic on POSIX filesystems.
    I/O failures before replacement raise :class:`PersistenceError` and leave
    the previous file untouched. This does not promise power-loss durability
    of the directory entry or coordinate multiple writer processes.
    """
    data = json.dumps(payload, indent=2)
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        raise PersistenceError(f"could not write {path.name}: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
