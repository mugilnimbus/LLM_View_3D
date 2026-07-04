from __future__ import annotations

import uvicorn

from llm_view.core import APP_HOST, APP_PORT


def main() -> None:
    uvicorn.run("llm_view.app:create_app", factory=True, host=APP_HOST, port=APP_PORT, reload=False)
