from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from llm_view.core.schemas import ModelInfo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"


_MODELS = {
    "demo-transformer": ModelInfo(
        id="demo-transformer",
        name="Demo Transformer",
        hf_id=None,
        description="Instant synthetic transformer data for exploring the interface.",
        layer_count=12,
        head_count=12,
        hidden_size=768,
        mlp_size=3072,
        parameter_count="124M-shaped",
        supports_real_run=False,
        vocab_size=50257,
    ),
    "distilgpt2": ModelInfo(
        id="distilgpt2",
        name="DistilGPT-2",
        hf_id="distilgpt2",
        description="Small real decoder-only model. Good first target for your RTX 3080 Ti.",
        layer_count=6,
        head_count=12,
        hidden_size=768,
        mlp_size=3072,
        parameter_count="82M",
        supports_real_run=True,
        vocab_size=50257,
    ),
    "gpt2": ModelInfo(
        id="gpt2",
        name="GPT-2 Small",
        hf_id="gpt2",
        description="Classic 12-layer transformer for real attention/logit exploration.",
        layer_count=12,
        head_count=12,
        hidden_size=768,
        mlp_size=3072,
        parameter_count="124M",
        supports_real_run=True,
        vocab_size=50257,
    ),
}


def list_models() -> list[ModelInfo]:
    return [deepcopy(model) for model in _available_models().values()]


def get_model(model_id: str) -> ModelInfo:
    if model_id.startswith("hf:"):
        return _hub_model(model_id)
    models = _available_models()
    if model_id not in models:
        raise ValueError(f"Unknown model id: {model_id}")
    return deepcopy(models[model_id])


def _hub_model(model_id: str) -> ModelInfo:
    """Placeholder entry for an arbitrary `hf:<repo id>` — the Hugging Face engine
    fills in the real layer/head/hidden sizes from the hub config."""
    repo_id = model_id.removeprefix("hf:").strip().strip("/")
    if not repo_id:
        raise ValueError("Hugging Face model id is empty. Use e.g. hf:Qwen/Qwen3-0.6B")
    return ModelInfo(
        id=model_id,
        name=repo_id,
        hf_id=repo_id,
        description=f"Architecture loaded live from huggingface.co/{repo_id} (config only).",
        layer_count=0,
        head_count=0,
        hidden_size=0,
        mlp_size=0,
        parameter_count=_parameter_count_label(repo_id),
        supports_real_run=True,
        vocab_size=0,
    )


def _available_models() -> dict[str, ModelInfo]:
    models = deepcopy(_MODELS)
    models.update(_discover_local_models())
    return models


def _discover_local_models() -> dict[str, ModelInfo]:
    if not MODELS_DIR.exists():
        return {}

    discovered: dict[str, ModelInfo] = {}
    for model_dir in sorted(path for path in MODELS_DIR.iterdir() if path.is_dir()):
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        model_id = _local_model_id(model_dir.name)
        discovered[model_id] = ModelInfo(
            id=model_id,
            name=_display_name(model_dir.name),
            hf_id=str(model_dir),
            description=f"Local model discovered from models/{model_dir.name}.",
            layer_count=_int_from_config(config, "num_hidden_layers", "n_layer", default=0),
            head_count=_int_from_config(config, "num_attention_heads", "n_head", default=0),
            hidden_size=_int_from_config(config, "hidden_size", "n_embd", default=0),
            mlp_size=_int_from_config(config, "intermediate_size", "n_inner", default=0),
            parameter_count=_parameter_count_label(model_dir.name),
            supports_real_run=True,
            vocab_size=_int_from_config(config, "vocab_size", default=0),
        )
    return discovered


def _local_model_id(folder_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", folder_name.lower()).strip("-")
    return f"local-{slug or 'model'}"


def _display_name(folder_name: str) -> str:
    return f"{folder_name.replace('-', ' ')} Local"


def _int_from_config(config: dict[str, Any], *keys: str, default: int) -> int:
    text_config = config.get("text_config")
    sources = [text_config, config] if isinstance(text_config, dict) else [config]
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int):
                return value
    return default


def _parameter_count_label(folder_name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", folder_name)
    if not match:
        return "local"
    number, suffix = match.groups()
    return f"{number}{suffix.upper()}"
