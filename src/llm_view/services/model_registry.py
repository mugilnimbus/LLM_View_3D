from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from llm_view.core.schemas import ModelInfo
from llm_view.services.gguf_metadata import GgufMetadataError, read_gguf_metadata


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
        if config_path.exists():
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
            continue

        gguf_file = _primary_gguf_file(model_dir)
        if gguf_file is None:
            continue

        try:
            metadata = read_gguf_metadata(gguf_file)
        except GgufMetadataError:
            continue

        model_id = _local_model_id(model_dir.name)
        discovered[model_id] = _gguf_model_info(model_id, model_dir, gguf_file, metadata)
    return discovered


def _primary_gguf_file(model_dir: Path) -> Path | None:
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        return None
    main_files = [path for path in gguf_files if not path.name.lower().startswith("mmproj")]
    candidates = main_files or gguf_files
    return max(candidates, key=lambda path: path.stat().st_size)


def _gguf_model_info(
    model_id: str, model_dir: Path, gguf_file: Path, metadata: dict[str, Any]
) -> ModelInfo:
    architecture = str(metadata.get("general.architecture") or "").strip()
    name = str(metadata.get("general.name") or "").strip() or _display_name(model_dir.name)
    size_label = str(metadata.get("general.size_label") or "").strip()
    quant = _gguf_quant_label(gguf_file.name)
    parameter_count = size_label if re.search(r"\d", size_label) else _parameter_count_label(model_dir.name)
    if quant and parameter_count != "local":
        parameter_count = f"{parameter_count} {quant}"
    elif quant:
        parameter_count = quant

    return ModelInfo(
        id=model_id,
        name=f"{name} GGUF" if "gguf" not in name.lower() else name,
        hf_id=str(gguf_file),
        description=(
            f"Architecture metadata read from models/{model_dir.name}/{gguf_file.name}. "
            "GGUF execution is not wired up, so runs use demo tensors."
        ),
        layer_count=_metadata_int(metadata, f"{architecture}.block_count", default=0),
        head_count=_metadata_int(metadata, f"{architecture}.attention.head_count", default=0),
        hidden_size=_metadata_int(metadata, f"{architecture}.embedding_length", default=0),
        mlp_size=_metadata_int(metadata, f"{architecture}.feed_forward_length", default=0),
        parameter_count=parameter_count,
        supports_real_run=False,
        vocab_size=_metadata_int(metadata, "tokenizer.ggml.tokens", default=0),
    )


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


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    value = metadata.get(key)
    if isinstance(value, list):
        numbers = [item for item in value if isinstance(item, int)]
        return max(numbers) if numbers else default
    if isinstance(value, int):
        return value
    return default


def _gguf_quant_label(filename: str) -> str:
    match = re.search(r"(Q\d(?:_[A-Z0-9]+)+)", filename, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _parameter_count_label(folder_name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", folder_name)
    if not match:
        return "local"
    number, suffix = match.groups()
    return f"{number}{suffix.upper()}"
