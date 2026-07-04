from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from llm_view.api import create_api_router
from llm_view.core import STATIC_DIR
from llm_view.services import EngineHub


class RevalidatedStaticFiles(StaticFiles):
    """Static files with `no-cache` so browsers revalidate after upgrades.

    Without this, browsers apply heuristic freshness to the JS modules and keep
    serving stale bundles for days after the app changes (304s keep local reloads
    cheap).
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM View",
        description="Interactive visualization of transformer layers, tokens, attention, and logits.",
        version="0.1.0",
    )
    engines = EngineHub()
    app.include_router(create_api_router(engines))

    app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
