"""Run the isolated synthetic Desk UI with python -m local_brain.command_center.bd_desk_dev."""

import argparse
import os
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated

import psycopg
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi import Path as APIPath
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .bd_desk_postgres import PostgresDesk
from .bd_routes import DeskDecision, same_origin

DIST = Path(__file__).resolve().parents[2] / "dashboard" / "dist"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def require_local_origin(request: Request):
    if request.headers.get("origin") != f"{request.url.scheme}://{request.headers['host']}":
        raise HTTPException(403, "same_origin_required")
    same_origin(request)


def create_app() -> FastAPI:
    dsn = os.environ.get("STUDIO_DESK_DEV_DSN")
    tenant_id = os.environ.get("STUDIO_DESK_DEV_TENANT_ID")
    if not dsn or not tenant_id:
        raise RuntimeError("Set STUDIO_DESK_DEV_DSN and STUDIO_DESK_DEV_TENANT_ID.")
    try:
        store = PostgresDesk(dsn, tenant_id)
    except (ValueError, PermissionError, psycopg.Error):
        raise RuntimeError("Invalid Desk development database configuration.") from None
    if not (DIST / "index.html").is_file():
        raise RuntimeError("Build dashboard/dist with npm run build before starting the Desk.")
    html = (
        (DIST / "index.html")
        .read_text(encoding="utf-8")
        .replace("</head>", '<meta name="studio-desk-dev" content="local"></head>', 1)
    )
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        try:
            local = request.client is not None and ip_address(request.client.host).is_loopback
        except ValueError:
            local = False
        if not local:
            return JSONResponse({"detail": "loopback_required"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(PermissionError)
    @app.exception_handler(psycopg.Error)
    async def storage_error(request: Request, exc: Exception):
        return JSONResponse({"detail": "desk_storage_unavailable"}, status_code=503)

    @app.get("/api/bd/desk")
    def desk():
        return store.desk()

    @app.get("/api/bd/record")
    def record():
        return store.record()

    @app.get("/api/bd/record/{receipt_id}")
    def record_detail(receipt_id: Annotated[str, APIPath(pattern=UUID_PATTERN)]):
        try:
            return store.detail(receipt_id)
        except KeyError:
            raise HTTPException(404, "receipt_not_found") from None

    @app.post(
        "/api/bd/desk/{bot_id}/{job_id}/{index}/review",
        dependencies=[Depends(require_local_origin)],
    )
    def decide(bot_id: str, job_id: str, index: Annotated[int, APIPath(ge=0)], body: DeskDecision):
        try:
            return store.decide(bot_id, job_id, index, body.expected_tenant_id, body.status)
        except KeyError:
            raise HTTPException(404, "packet_not_found") from None
        except ValueError:
            raise HTTPException(409, "packet_changed") from None

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            },
        )

    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, proxy_headers=False)
