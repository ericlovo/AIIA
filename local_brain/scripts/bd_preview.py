"""Run an isolated conductor and built Studio, without starting legacy services.

python -m local_brain.scripts.bd_preview --database /tmp/bd-preview.sqlite3
"""

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from local_brain.command_center.bd_conductor import BDConductor
from local_brain.command_center.bd_routes import create_router
from local_brain.command_center.bd_service import BDService


def build_app(database, config=None):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="BD conductor isolated preview")
    app.include_router(create_router(lambda: BDService(BDConductor(database, config))))
    dist = Path(__file__).parents[2] / "dashboard" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--config")
    parser.add_argument("--port", type=int, default=8231)
    args = parser.parse_args()
    uvicorn.run(build_app(args.database, args.config), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
