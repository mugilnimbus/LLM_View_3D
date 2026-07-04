from __future__ import annotations

import math
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from llm_view.core.schemas import (
    ArchitectureResponse,
    AttentionHead,
    ComponentMetric,
    LayerArchitecture,
    LayerRun,
    Prediction,
    RunRequest,
    RunResponse,
    TokenTrace,
)
from llm_view.services.flow import (
    FlowSpec,
    build_layer_flow,
    build_post_flow,
    build_pre_flow,
)
from llm_view.services.model_registry import get_model


class HuggingFaceUnavailable(RuntimeError):
    pass


def _config_int(config: Any, *names: str, default: int = 0) -> int:
    """Return the first present, non-None config attribute as an int.

    ``getattr(config, name, default)`` is unsafe for Hugging Face configs: many
    attributes exist but are set to ``None`` (e.g. GPT-2's ``n_inner`` or
    ``head_dim`` on some architectures), so the supplied default never kicks in
    and ``int(None)`` raises ``TypeError``. This walks the candidate names and
    only falls back to ``default`` when every value is missing or ``None``.
    """
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return default


class HuggingFaceEngine:
    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise HuggingFaceUnavailable(
                "Hugging Face mode needs `uv sync --extra ml` first"
            ) from exc

        self._torch = torch
        self._auto_config = AutoConfig
        self._auto_model = AutoModelForCausalLM
        self._auto_tokenizer = AutoTokenizer
        self._cache: dict[str, tuple[Any, Any]] = {}

    def architecture(self, model_id: str) -> ArchitectureResponse:
        model_meta = get_model(model_id)
        config = self._load_config(model_meta.hf_id)
        text_config = _text_config(config)
        layer_count = _config_int(text_config, "n_layer", "num_hidden_layers")
        head_count = _config_int(text_config, "n_head", "num_attention_heads")
        hidden_size = _config_int(text_config, "n_embd", "hidden_size")
        mlp_size = _config_int(text_config, "n_inner", "intermediate_size")
        vocab_size = _config_int(text_config, "vocab_size") or _config_int(config, "vocab_size")
        layer_types = list(getattr(text_config, "layer_types", []) or [])

        model_meta.layer_count = layer_count or model_meta.layer_count
        model_meta.head_count = head_count or model_meta.head_count
        model_meta.hidden_size = hidden_size or model_meta.hidden_size
        model_meta.mlp_size = mlp_size or model_meta.mlp_size
        model_meta.vocab_size = vocab_size or model_meta.vocab_size

        base_spec = _flow_spec(config, text_config, model_meta)
        layers = [
            LayerArchitecture(
                index=index,
                name=f"Transformer Block {index + 1}",
                components=_components_for_layer(layer_types, index),
                head_count=model_meta.head_count,
                hidden_size=model_meta.hidden_size,
                mlp_size=model_meta.mlp_size,
                attention_type=_attention_type(layer_types, index),
                matrices=_matrix_specs(text_config, model_meta, _attention_type(layer_types, index)),
                flow=build_layer_flow(
                    _layer_spec(base_spec, text_config, _attention_type(layer_types, index))
                ),
            )
            for index in range(model_meta.layer_count)
        ]
        return ArchitectureResponse(
            model=model_meta,
            layers=layers,
            pre_flow=build_pre_flow(base_spec),
            post_flow=build_post_flow(base_spec),
        )

    def run(self, request: RunRequest) -> RunResponse:
        model, tokenizer = self._load(request.model_id)
        config = self._load_config(get_model(request.model_id).hf_id)
        layer_types = list(getattr(_text_config(config), "layer_types", []) or [])
        device = next(model.parameters()).device

        encoded = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=32)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with self._torch.no_grad():
            outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        token_ids = encoded["input_ids"][0].detach().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        attentions = outputs.attentions or []
        hidden_states = outputs.hidden_states or []
        logits = outputs.logits[0, -1].detach().float().cpu().numpy()

        layers = _layers_from_outputs(attentions, hidden_states, tokens, layer_types)
        predictions = _top_predictions(logits, tokenizer)
        traces = _token_traces(tokens, hidden_states)

        architecture = self.architecture(request.model_id)
        return RunResponse(
            model=architecture.model,
            prompt=request.prompt,
            tokens=tokens,
            layers=layers,
            token_traces=traces,
            predictions=predictions,
            source="huggingface",
            note="Real attention, hidden-state norms, and logits from a local Hugging Face forward pass.",
        )

    def _load(self, model_id: str) -> tuple[Any, Any]:
        if model_id in self._cache:
            return self._cache[model_id]
        model_meta = get_model(model_id)
        source = _model_source(model_meta.hf_id)
        local_files_only = _is_local_source(source)
        tokenizer = self._auto_tokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        dtype = self._torch.float16 if self._cuda_has_room() else self._torch.float32
        device_map = "auto" if self._cuda_has_room() else None
        model = self._auto_model.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if device_map is None:
            model = model.to("cpu")
        model.eval()
        self._cache[model_id] = (model, tokenizer)
        return model, tokenizer

    def _load_config(self, model_source: str | None) -> Any:
        source = _model_source(model_source)
        try:
            return self._auto_config.from_pretrained(
                source,
                trust_remote_code=True,
                local_files_only=_is_local_source(source),
            )
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Could not load config for '{source}': {exc}") from exc

    @cached_property
    def _has_cuda(self) -> bool:
        return bool(self._torch.cuda.is_available())

    def _cuda_has_room(self) -> bool:
        if not self._has_cuda:
            return False
        free_bytes, _total_bytes = self._torch.cuda.mem_get_info()
        return free_bytes > 2 * 1024**3


def _model_source(model_source: str | None) -> str:
    if not model_source:
        raise ValueError("Real model is missing a Hugging Face id or local path.")
    return model_source


def _is_local_source(model_source: str) -> bool:
    return Path(model_source).exists()


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


_ACTIVATION_LABELS = {
    "silu": "SiLU",
    "swish": "SiLU",
    "relu": "ReLU",
    "relu2": "ReLU²",
    "gelu_pytorch_tanh": "GELU",
}


def _activation_name(text_config: Any) -> str:
    return str(
        getattr(text_config, "hidden_act", None)
        or getattr(text_config, "activation_function", None)
        or "gelu"
    ).lower()


def _mlp_gated(text_config: Any) -> bool:
    act = _activation_name(text_config)
    return act in {"silu", "swish"} or "glu" in act


def _flow_spec(config: Any, text_config: Any, model_meta: Any) -> FlowSpec:
    """Derive the forward-pass shape spec from a real Hugging Face config."""
    act_raw = _activation_name(text_config)
    gated = _mlp_gated(text_config)
    activation = _ACTIVATION_LABELS.get(
        act_raw, "GELU" if act_raw.startswith("gelu") else act_raw.upper()
    )
    has_rope = any(
        getattr(text_config, key, None) is not None
        for key in ("rope_theta", "rope_parameters", "rope_scaling")
    )
    tied = bool(
        getattr(config, "tie_word_embeddings", None)
        or getattr(text_config, "tie_word_embeddings", None)
    )
    return FlowSpec(
        hidden_size=model_meta.hidden_size,
        mlp_size=model_meta.mlp_size,
        head_count=model_meta.head_count,
        kv_heads=_config_int(
            text_config, "num_key_value_heads", default=model_meta.head_count
        ),
        head_dim=_config_int(
            text_config,
            "head_dim",
            default=model_meta.hidden_size // max(1, model_meta.head_count),
        ),
        vocab_size=model_meta.vocab_size,
        layer_count=model_meta.layer_count,
        norm_name="RMSNorm"
        if getattr(text_config, "rms_norm_eps", None) is not None
        else "LayerNorm",
        gated_mlp=gated,
        activation=activation,
        positional="rope" if has_rope else "learned",
        context_length=_config_int(
            text_config, "max_position_embeddings", "n_positions", default=0
        ),
        tied_embeddings=tied,
    )


def _layer_spec(base: FlowSpec, text_config: Any, attention_type: str) -> FlowSpec:
    if attention_type != "linear attention":
        return base
    key_heads = _config_int(text_config, "linear_num_key_heads", default=base.kv_heads)
    key_dim = _config_int(text_config, "linear_key_head_dim", default=base.head_dim)
    return replace(
        base,
        linear_dims={
            "key_heads": key_heads,
            "key_dim": key_dim,
            "value_heads": _config_int(
                text_config, "linear_num_value_heads", default=key_heads
            ),
            "value_dim": _config_int(
                text_config, "linear_value_head_dim", default=key_dim
            ),
        },
    )


def _components_for_layer(layer_types: list[str], index: int) -> list[str]:
    attention_name = _attention_type(layer_types, index)
    return [
        "RMS/layer norm",
        attention_name,
        "attention projection",
        "residual add",
        "gated MLP",
        "residual add",
    ]


def _attention_type(layer_types: list[str], index: int) -> str:
    layer_type = layer_types[index] if index < len(layer_types) else "masked self-attention"
    return layer_type.replace("_", " ")


def _matrix_specs(text_config: Any, model_meta: Any, attention_type: str) -> list[dict]:
    hidden_size = model_meta.hidden_size
    mlp_size = model_meta.mlp_size
    head_count = model_meta.head_count
    vocab_size = model_meta.vocab_size
    head_dim = _config_int(text_config, "head_dim", default=hidden_size // max(1, head_count))
    key_value_heads = _config_int(text_config, "num_key_value_heads", default=head_count)

    if attention_type == "linear attention":
        key_value_heads = _config_int(text_config, "linear_num_key_heads", default=key_value_heads)
        key_dim = _config_int(text_config, "linear_key_head_dim", default=head_dim)
        value_heads = _config_int(text_config, "linear_num_value_heads", default=key_value_heads)
        value_dim = _config_int(text_config, "linear_value_head_dim", default=head_dim)
        kv_dim = key_value_heads * key_dim
        v_dim = value_heads * value_dim
    else:
        kv_dim = key_value_heads * head_dim
        v_dim = kv_dim

    q_dim = head_count * head_dim
    if _mlp_gated(text_config):
        mlp_matrices = [
            {"name": "Wgate", "shape": [hidden_size, mlp_size], "role": "MLP gate projection"},
            {"name": "Wup", "shape": [hidden_size, mlp_size], "role": "MLP up projection"},
            {"name": "Wdown", "shape": [mlp_size, hidden_size], "role": "MLP down projection"},
        ]
    else:
        mlp_matrices = [
            {"name": "Wup", "shape": [hidden_size, mlp_size], "role": "MLP up projection (fc)"},
            {"name": "Wdown", "shape": [mlp_size, hidden_size], "role": "MLP down projection (proj)"},
        ]
    return [
        {"name": "token_embedding", "shape": [vocab_size, hidden_size], "role": "token id -> residual vector"},
        {"name": "RMSNorm", "shape": [hidden_size], "role": "normalize residual stream"},
        {"name": "Wq", "shape": [hidden_size, q_dim], "role": "residual -> queries"},
        {"name": "Wk", "shape": [hidden_size, kv_dim], "role": "residual -> keys"},
        {"name": "Wv", "shape": [hidden_size, v_dim], "role": "residual -> values"},
        {"name": "attention_scores", "shape": ["tokens", "tokens"], "role": "QK^T causal attention"},
        {"name": "Wo", "shape": [q_dim, hidden_size], "role": "heads -> residual"},
        *mlp_matrices,
        {"name": "lm_head", "shape": [hidden_size, vocab_size], "role": "residual -> token logits"},
    ]


def _layers_from_outputs(
    attentions: tuple[Any, ...] | list[Any],
    hidden_states: tuple[Any, ...],
    tokens: list[str],
    layer_types: list[str],
) -> list[LayerRun]:
    layer_count = max(0, len(hidden_states) - 1)
    if not attentions:
        return [_layer_without_attention(index, hidden_states, tokens) for index in range(layer_count)]

    attention_by_layer = _attention_by_layer(attentions, layer_count, layer_types)
    layers = []
    for index in range(layer_count):
        layer_attention = attention_by_layer.get(index)
        if layer_attention is None:
            layers.append(_layer_without_attention(index, hidden_states, tokens))
        else:
            layers.append(_layer_from_outputs(index, layer_attention, hidden_states, tokens))
    return layers


def _attention_by_layer(
    attentions: tuple[Any, ...] | list[Any], layer_count: int, layer_types: list[str]
) -> dict[int, Any]:
    real_attentions = [attention for attention in attentions if attention is not None]
    if len(real_attentions) == layer_count:
        return dict(enumerate(real_attentions))

    full_attention_layers = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    ]
    if len(full_attention_layers) == len(real_attentions):
        return dict(zip(full_attention_layers, real_attentions, strict=True))

    return dict(enumerate(real_attentions[:layer_count]))


def _layer_from_outputs(
    layer_index: int, layer_attention: Any, hidden_states: tuple[Any, ...], tokens: list[str]
) -> LayerRun:
    attention = layer_attention[0].detach().float().cpu().numpy()
    heads = [
        AttentionHead(
            index=head_index,
            label=f"Head {head_index + 1}",
            attention=np.round(attention[head_index], 4).tolist(),
            focus_score=round(float(np.max(attention[head_index])), 4),
            role="observed attention",
        )
        for head_index in range(attention.shape[0])
    ]

    metrics = []
    if layer_index + 1 < len(hidden_states):
        hidden = hidden_states[layer_index + 1][0].detach().float().cpu().numpy()
        norms = np.linalg.norm(hidden, axis=1)
        metrics.append(ComponentMetric(name="Residual norm", value=round(float(norms.mean()), 4), unit="L2"))
        metrics.append(
            ComponentMetric(
                name="Token norm spread",
                value=round(float(norms.max() - norms.min()), 4),
                unit="L2",
            )
        )
    entropy = _attention_entropy(attention)
    metrics.append(ComponentMetric(name="Attention entropy", value=round(entropy, 4), unit="bits"))

    top_activations = [
        ComponentMetric(name=f"token {token}", value=round(float(score), 4), unit="norm")
        for token, score in zip(tokens, _token_scores(hidden_states, layer_index), strict=False)
    ][:8]
    return LayerRun(index=layer_index, heads=heads, metrics=metrics, top_activations=top_activations)


def _layer_without_attention(
    layer_index: int, hidden_states: tuple[Any, ...], tokens: list[str]
) -> LayerRun:
    metrics = []
    if layer_index + 1 < len(hidden_states):
        hidden = hidden_states[layer_index + 1][0].detach().float().cpu().numpy()
        norms = np.linalg.norm(hidden, axis=1)
        metrics.append(ComponentMetric(name="Residual norm", value=round(float(norms.mean()), 4), unit="L2"))
        metrics.append(
            ComponentMetric(
                name="Token norm spread",
                value=round(float(norms.max() - norms.min()), 4),
                unit="L2",
            )
        )

    top_activations = [
        ComponentMetric(name=f"token {token}", value=round(float(score), 4), unit="norm")
        for token, score in zip(tokens, _token_scores(hidden_states, layer_index), strict=False)
    ][:8]
    return LayerRun(index=layer_index, heads=[], metrics=metrics, top_activations=top_activations)


def _attention_entropy(attention: np.ndarray) -> float:
    safe = np.clip(attention, 1e-8, 1)
    return float(-(safe * np.log2(safe)).sum(axis=-1).mean())


def _token_scores(hidden_states: tuple[Any, ...], layer_index: int) -> list[float]:
    if layer_index + 1 >= len(hidden_states):
        return []
    hidden = hidden_states[layer_index + 1][0].detach().float().cpu().numpy()
    return np.linalg.norm(hidden, axis=1).tolist()


def _top_predictions(logits: np.ndarray, tokenizer: Any) -> list[Prediction]:
    top_indices = np.argsort(logits)[-8:][::-1]
    top_logits = logits[top_indices]
    probabilities = np.exp(top_logits - np.max(top_logits))
    probabilities = probabilities / probabilities.sum()
    return [
        Prediction(token=tokenizer.decode([int(index)]), probability=round(float(probability), 4))
        for index, probability in zip(top_indices, probabilities, strict=True)
    ]


def _token_traces(tokens: list[str], hidden_states: tuple[Any, ...]) -> list[TokenTrace]:
    if not hidden_states:
        return []
    norms_by_layer = []
    for hidden_state in hidden_states[1:]:
        hidden = hidden_state[0].detach().float().cpu().numpy()
        norms = np.linalg.norm(hidden, axis=1)
        max_norm = max(float(norms.max()), 1e-6)
        norms_by_layer.append((norms / max_norm).tolist())

    traces = []
    for token_index, token in enumerate(tokens):
        traces.append(
            TokenTrace(
                index=token_index,
                token=token,
                salience_by_layer=[
                    round(float(layer[token_index]), 4)
                    for layer in norms_by_layer
                    if token_index < len(layer) and math.isfinite(float(layer[token_index]))
                ],
            )
        )
    return traces
