from __future__ import annotations

import hashlib
import math
import re
from random import Random

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


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class DemoEngine:
    """Deterministic transformer-like data for instant exploration without model downloads."""

    def architecture(self, model_id: str = "demo-transformer") -> ArchitectureResponse:
        model = get_model(model_id)
        spec = FlowSpec(
            hidden_size=model.hidden_size,
            mlp_size=model.mlp_size,
            head_count=model.head_count,
            kv_heads=model.head_count,
            head_dim=model.hidden_size // max(1, model.head_count),
            vocab_size=model.vocab_size,
            layer_count=model.layer_count,
            norm_name="RMSNorm",
            gated_mlp=True,
            activation="SiLU",
            positional="rope",
        )
        layer_flow = build_layer_flow(spec)
        layers = [
            LayerArchitecture(
                index=index,
                name=f"Transformer Block {index + 1}",
                components=[
                    "input layer norm",
                    "multi-head self-attention",
                    "attention output projection",
                    "residual stream",
                    "post-attention layer norm",
                    "MLP up projection",
                    "activation",
                    "MLP down projection",
                ],
                head_count=model.head_count,
                hidden_size=model.hidden_size,
                mlp_size=model.mlp_size,
                attention_type="multi-head self-attention",
                matrices=_matrix_specs(
                    hidden_size=model.hidden_size,
                    mlp_size=model.mlp_size,
                    head_count=model.head_count,
                    head_dim=model.hidden_size // max(1, model.head_count),
                    key_value_heads=model.head_count,
                    vocab_size=model.vocab_size,
                ),
                flow=layer_flow,
            )
            for index in range(model.layer_count)
        ]
        return ArchitectureResponse(
            model=model,
            layers=layers,
            pre_flow=build_pre_flow(spec),
            post_flow=build_post_flow(spec),
        )

    def run(self, request: RunRequest) -> RunResponse:
        architecture = self.architecture(request.model_id)
        tokens = _tokenize(request.prompt)
        prompt_seed = _seed_from_prompt(request.prompt)
        layer_runs = [
            _layer_run(layer.index, architecture.model.head_count, tokens, prompt_seed)
            for layer in architecture.layers
        ]
        traces = _token_traces(tokens, len(architecture.layers), prompt_seed)
        predictions = _predictions(tokens, prompt_seed)

        return RunResponse(
            model=architecture.model,
            prompt=request.prompt,
            tokens=tokens,
            layers=layer_runs,
            token_traces=traces,
            predictions=predictions,
            source="demo",
            note=(
                "Synthetic but transformer-shaped data. Install the ml extras and switch to Real Model "
                "for Hugging Face activations."
            ),
        )


def _tokenize(prompt: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(prompt.strip())
    return tokens[:32] or ["The", "model", "visualizes", "attention", "."]


def _seed_from_prompt(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _layer_run(layer_index: int, head_count: int, tokens: list[str], prompt_seed: int) -> LayerRun:
    sequence_length = len(tokens)
    heads = [
        AttentionHead(
            index=head,
            label=f"Head {head + 1}",
            attention=_attention_matrix(sequence_length, layer_index, head, prompt_seed),
            focus_score=round(0.35 + 0.55 * _wave(layer_index, head, prompt_seed), 3),
            role=_head_role(layer_index, head),
        )
        for head in range(head_count)
    ]
    metrics = [
        ComponentMetric(name="Residual norm", value=round(8.5 + layer_index * 0.42, 3), unit="L2"),
        ComponentMetric(
            name="Attention entropy",
            value=round(2.1 + 0.5 * _wave(layer_index, 3, prompt_seed), 3),
            unit="bits",
        ),
        ComponentMetric(
            name="MLP activation density",
            value=round(24 + 42 * _wave(layer_index, 7, prompt_seed), 2),
            unit="%",
        ),
    ]
    activations = _top_activations(layer_index, prompt_seed)
    return LayerRun(index=layer_index, heads=heads, metrics=metrics, top_activations=activations)


def _attention_matrix(size: int, layer_index: int, head_index: int, prompt_seed: int) -> list[list[float]]:
    rng = Random(prompt_seed + layer_index * 101 + head_index * 17)
    matrix: list[list[float]] = []
    for row in range(size):
        weights = []
        for col in range(size):
            if col > row:
                weights.append(0.0)
                continue
            distance = row - col
            local = math.exp(-distance / (1.4 + (head_index % 4)))
            anchor = 1.2 if col == 0 and head_index % 5 == 0 else 0.0
            induction = 1.1 if distance == 1 and head_index % 3 == 1 else 0.0
            layer_wave = 0.25 * math.sin((layer_index + 1) * (col + 1) * 0.31)
            weights.append(max(0.001, local + anchor + induction + layer_wave + rng.random() * 0.08))
        total = sum(weights) or 1.0
        matrix.append([round(value / total, 4) for value in weights])
    return matrix


def _head_role(layer_index: int, head_index: int) -> str:
    roles = [
        "local syntax",
        "previous-token copy",
        "prompt anchor",
        "long-range reference",
        "delimiter tracking",
        "semantic grouping",
    ]
    return roles[(layer_index + head_index) % len(roles)]


def _top_activations(layer_index: int, prompt_seed: int) -> list[ComponentMetric]:
    labels = ["entity feature", "number feature", "syntax feature", "topic feature", "quote feature"]
    return [
        ComponentMetric(
            name=label,
            value=round(0.35 + 0.62 * _wave(layer_index, offset + 11, prompt_seed), 3),
            unit="activation",
        )
        for offset, label in enumerate(labels)
    ]


def _token_traces(tokens: list[str], layer_count: int, prompt_seed: int) -> list[TokenTrace]:
    traces = []
    for index, token in enumerate(tokens):
        salience = [
            round(0.2 + 0.8 * _wave(layer, index + prompt_seed % 13, prompt_seed), 3)
            for layer in range(layer_count)
        ]
        traces.append(TokenTrace(index=index, token=token, salience_by_layer=salience))
    return traces


def _predictions(tokens: list[str], prompt_seed: int) -> list[Prediction]:
    vocabulary = [
        "is",
        "shows",
        "because",
        "the",
        "model",
        "attention",
        "layer",
        "next",
        "visual",
        ".",
    ]
    last = tokens[-1].lower() if tokens else ""
    if last in {"?", "why", "how"}:
        vocabulary = ["because", "the", "model", "uses", "attention", "to", "track", "context", "."]
    rng = Random(prompt_seed)
    raw = np.array([rng.random() + 0.15 * index for index, _ in enumerate(vocabulary)], dtype=float)
    probs = np.exp(raw) / np.exp(raw).sum()
    ranked = sorted(zip(vocabulary, probs, strict=True), key=lambda item: item[1], reverse=True)[:8]
    return [Prediction(token=token, probability=round(float(prob), 4)) for token, prob in ranked]


def _wave(layer_index: int, offset: int, prompt_seed: int) -> float:
    phase = (prompt_seed % 97) / 97
    return (math.sin(layer_index * 0.73 + offset * 1.31 + phase) + 1) / 2


def _matrix_specs(
    hidden_size: int,
    mlp_size: int,
    head_count: int,
    head_dim: int,
    key_value_heads: int,
    vocab_size: int,
) -> list[dict]:
    q_dim = head_count * head_dim
    kv_dim = key_value_heads * head_dim
    return [
        {"name": "token_embedding", "shape": [vocab_size, hidden_size], "role": "token id -> residual vector"},
        {"name": "Wq", "shape": [hidden_size, q_dim], "role": "residual -> queries"},
        {"name": "Wk", "shape": [hidden_size, kv_dim], "role": "residual -> keys"},
        {"name": "Wv", "shape": [hidden_size, kv_dim], "role": "residual -> values"},
        {"name": "attention_scores", "shape": ["tokens", "tokens"], "role": "QK^T causal attention"},
        {"name": "Wo", "shape": [q_dim, hidden_size], "role": "heads -> residual"},
        {"name": "Wgate", "shape": [hidden_size, mlp_size], "role": "MLP gate projection"},
        {"name": "Wup", "shape": [hidden_size, mlp_size], "role": "MLP up projection"},
        {"name": "Wdown", "shape": [mlp_size, hidden_size], "role": "MLP down projection"},
        {"name": "lm_head", "shape": [hidden_size, vocab_size], "role": "residual -> token logits"},
    ]
