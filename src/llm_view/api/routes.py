from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llm_view.core.schemas import ArchitectureResponse, RunRequest, RunResponse
from llm_view.services.engine import EngineHub
from llm_view.services.hardware import detect_hardware
from llm_view.services.model_registry import list_models


def create_api_router(engines: EngineHub) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/hardware")
    def hardware() -> dict:
        return detect_hardware().model_dump()

    @router.get("/models")
    def models() -> dict:
        return {"models": [model.model_dump() for model in list_models()]}

    @router.get("/architecture", response_model=ArchitectureResponse)
    def architecture(model_id: str = "demo-transformer", mode: str = "demo") -> ArchitectureResponse:
        try:
            return engines.architecture(model_id=model_id, mode=mode)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/run", response_model=RunResponse)
    def run(request: RunRequest) -> RunResponse:
        try:
            return engines.run(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
