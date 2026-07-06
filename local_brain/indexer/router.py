"""
File Index API — /v1/files/* endpoints.

POST /v1/files/index   — trigger a re-index (async, returns immediately)
GET  /v1/files/search  — semantic search over indexed local files
GET  /v1/files/status  — indexed file count
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Query

from local_brain.indexer.file_index import run_index, search_files

logger = logging.getLogger("aiia.indexer.router")

router = APIRouter(prefix="/v1/files", tags=["files"])

_index_running = False
_last_result: dict = {}


@router.post("/index")
async def trigger_index(background_tasks: BackgroundTasks):
    global _index_running
    if _index_running:
        return {"status": "already_running"}
    _index_running = True

    async def _run():
        global _index_running, _last_result
        try:
            _last_result = await run_index()
        finally:
            _index_running = False

    background_tasks.add_task(_run)
    return {"status": "started"}


@router.get("/search")
async def file_search(q: str = Query(..., min_length=2), n: int = Query(8, le=20)):
    hits = await search_files(q, n_results=n)
    return {"query": q, "results": hits}


@router.get("/status")
async def index_status():
    return {
        "running": _index_running,
        "last_result": _last_result,
    }
