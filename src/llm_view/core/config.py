from __future__ import annotations

import os
from pathlib import Path


APP_HOST = os.getenv("LLM_VIEW_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("LLM_VIEW_PORT", "8999"))
PACKAGE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = PACKAGE_DIR / "static"
