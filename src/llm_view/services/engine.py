from __future__ import annotations

from llm_view.core.schemas import ArchitectureResponse, RunRequest, RunResponse
from llm_view.services.demo_engine import DemoEngine
from llm_view.services.hf_engine import HuggingFaceEngine, HuggingFaceUnavailable


class EngineHub:
    def __init__(self) -> None:
        self._demo = DemoEngine()
        self._hf: HuggingFaceEngine | None = None

    def architecture(self, model_id: str, mode: str) -> ArchitectureResponse:
        if mode == "real":
            try:
                return self._real().architecture(model_id)
            except HuggingFaceUnavailable:
                return self._demo.architecture("demo-transformer")
        return self._demo.architecture(model_id)

    def run(self, request: RunRequest) -> RunResponse:
        if request.mode == "real":
            try:
                return self._real().run(request)
            except HuggingFaceUnavailable as exc:
                response = self._demo.run(request.model_copy(update={"model_id": "demo-transformer"}))
                response.note = f"{exc}. Showing demo data instead."
                return response
        return self._demo.run(request)

    def _real(self) -> HuggingFaceEngine:
        if self._hf is None:
            self._hf = HuggingFaceEngine()
        return self._hf
