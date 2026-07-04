"""Builds the exact top-to-bottom tensor transformation trace of a forward pass.

Every step records which matrix is multiplied by which weights and the
resulting shape, so the UI can show `[T x d] @ Wq [d x h*dh] -> [T x h*dh]`
for the actual dimensions of the loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_view.core.schemas import FlowStep


@dataclass
class FlowSpec:
    hidden_size: int
    mlp_size: int
    head_count: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    layer_count: int
    norm_name: str = "RMSNorm"
    gated_mlp: bool = True
    activation: str = "SiLU"
    positional: str = "rope"  # "rope" | "learned"
    context_length: int = 0
    tied_embeddings: bool = False
    # linear-attention dims (Qwen3.5-style hybrid layers), None for standard layers
    linear_dims: dict[str, int] | None = field(default=None)


def build_pre_flow(spec: FlowSpec) -> list[FlowStep]:
    d = str(spec.hidden_size)
    vocab = str(spec.vocab_size or "V")
    steps = [
        FlowStep(
            stage="input",
            name="Tokenize",
            expr="prompt -> token ids",
            input_shape=["prompt"],
            output_shape=["T"],
            note="the tokenizer splits text into T integer ids from a vocabulary of "
            f"{vocab} tokens",
        ),
        FlowStep(
            stage="input",
            name="Embedding lookup",
            expr="X = E[ids]",
            input_shape=["T"],
            weight_name="E",
            weight_shape=[vocab, d],
            output_shape=["T", d],
            note="each token id selects one row of E - no multiplication, just a lookup",
        ),
    ]
    if spec.positional == "learned":
        ctx = str(spec.context_length or "ctx")
        steps.append(
            FlowStep(
                stage="input",
                name="Position embedding",
                expr="X = X + P[0..T-1]",
                input_shape=["T", d],
                weight_name="P",
                weight_shape=[ctx, d],
                output_shape=["T", d],
                note="adds a learned vector per position so the model knows token order",
            )
        )
    else:
        steps.append(
            FlowStep(
                stage="input",
                name="Position encoding (RoPE)",
                expr="applied inside each block",
                input_shape=["T", d],
                output_shape=["T", d],
                note="no weights added here - rotary embeddings rotate Q/K inside every "
                "attention block instead",
            )
        )
    return steps


def build_post_flow(spec: FlowSpec) -> list[FlowStep]:
    d = str(spec.hidden_size)
    vocab = str(spec.vocab_size or "V")
    tied = " (weights tied to the embedding matrix E)" if spec.tied_embeddings else ""
    return [
        FlowStep(
            stage="output",
            name=f"Final {spec.norm_name}",
            expr=f"X̂ = {spec.norm_name}(X)",
            input_shape=["T", d],
            weight_name="γ",
            weight_shape=[d],
            output_shape=["T", d],
            note="one last normalization of the residual stream",
        ),
        FlowStep(
            stage="output",
            name="LM head",
            expr="logits = X̂ · W_lm",
            input_shape=["T", d],
            weight_name="W_lm",
            weight_shape=[d, vocab],
            output_shape=["T", vocab],
            note=f"projects every token back onto the whole vocabulary{tied}",
        ),
        FlowStep(
            stage="output",
            name="Next-token distribution",
            expr="p = softmax(logits[T])",
            input_shape=[vocab],
            output_shape=[vocab],
            note="only the last row matters for generation - the highest-probability "
            "tokens appear in the Next Token panel",
        ),
    ]


def build_layer_flow(spec: FlowSpec) -> list[FlowStep]:
    steps = [_pre_attention_norm(spec)]
    if spec.linear_dims:
        steps.extend(_linear_attention_steps(spec))
    else:
        steps.extend(_attention_steps(spec))
    steps.append(_residual("X = X + attn_out", spec, "attention output flows back into"))
    steps.append(
        FlowStep(
            stage="norm",
            name=f"Pre-MLP {spec.norm_name}",
            expr=f"X̂ = {spec.norm_name}(X)",
            input_shape=["T", str(spec.hidden_size)],
            weight_name="γ",
            weight_shape=[str(spec.hidden_size)],
            output_shape=["T", str(spec.hidden_size)],
            note="normalize again before the MLP",
        )
    )
    steps.extend(_mlp_steps(spec))
    steps.append(_residual("X = X + mlp_out", spec, "MLP output flows back into"))
    return steps


def _pre_attention_norm(spec: FlowSpec) -> FlowStep:
    d = str(spec.hidden_size)
    return FlowStep(
        stage="norm",
        name=f"Pre-attention {spec.norm_name}",
        expr=f"X̂ = {spec.norm_name}(X)",
        input_shape=["T", d],
        weight_name="γ",
        weight_shape=[d],
        output_shape=["T", d],
        note="rescales each token vector so activations stay stable",
    )


def _attention_steps(spec: FlowSpec) -> list[FlowStep]:
    d = str(spec.hidden_size)
    h, kv, dh = spec.head_count, spec.kv_heads or spec.head_count, spec.head_dim
    q_dim, kv_dim = str(h * dh), str(kv * dh)
    gqa = (
        f" · {kv} KV heads shared by {h} query heads (GQA {h // max(1, kv)}:1)"
        if kv < h
        else ""
    )

    steps = [
        FlowStep(
            stage="attention",
            name="Query projection",
            expr="Q = X̂ · Wq",
            input_shape=["T", d],
            weight_name="Wq",
            weight_shape=[d, q_dim],
            output_shape=["T", q_dim],
            note=f"reshape -> {h} heads × [T × {dh}]",
        ),
        FlowStep(
            stage="attention",
            name="Key projection",
            expr="K = X̂ · Wk",
            input_shape=["T", d],
            weight_name="Wk",
            weight_shape=[d, kv_dim],
            output_shape=["T", kv_dim],
            note=f"reshape -> {kv} heads × [T × {dh}]{gqa}",
        ),
        FlowStep(
            stage="attention",
            name="Value projection",
            expr="V = X̂ · Wv",
            input_shape=["T", d],
            weight_name="Wv",
            weight_shape=[d, kv_dim],
            output_shape=["T", kv_dim],
            note=f"reshape -> {kv} heads × [T × {dh}]",
        ),
    ]
    if spec.positional == "rope":
        steps.append(
            FlowStep(
                stage="attention",
                name="Rotary positions (RoPE)",
                expr="Q, K <- rotate(Q, K, position)",
                input_shape=[str(h), "T", str(dh)],
                output_shape=[str(h), "T", str(dh)],
                note="rotates each query/key by its token position - encodes order "
                "with zero extra weights",
            )
        )
    steps.extend(
        [
            FlowStep(
                stage="attention",
                name="Attention scores",
                expr=f"S = Q · Kᵀ / √{dh}",
                input_shape=[str(h), "T", str(dh)],
                output_shape=[str(h), "T", "T"],
                note="every token scores every earlier token; causal mask hides the future",
            ),
            FlowStep(
                stage="attention",
                name="Softmax",
                expr="A = softmax(S)",
                input_shape=[str(h), "T", "T"],
                output_shape=[str(h), "T", "T"],
                note="each row becomes a probability distribution - this is the heatmap "
                "in the Layer Inspector",
            ),
            FlowStep(
                stage="attention",
                name="Weighted values",
                expr="ctx = A · V",
                input_shape=[str(h), "T", "T"],
                output_shape=[str(h), "T", str(dh)],
                note=f"mixes value vectors by attention weight; concat heads -> "
                f"[T × {q_dim}]",
            ),
            FlowStep(
                stage="attention",
                name="Output projection",
                expr="attn_out = ctx · Wo",
                input_shape=["T", q_dim],
                weight_name="Wo",
                weight_shape=[q_dim, d],
                output_shape=["T", d],
                note="projects the mixed heads back into residual-stream space",
            ),
        ]
    )
    return steps


def _linear_attention_steps(spec: FlowSpec) -> list[FlowStep]:
    d = str(spec.hidden_size)
    dims = spec.linear_dims or {}
    k_heads = dims.get("key_heads", spec.kv_heads)
    k_dim = dims.get("key_dim", spec.head_dim)
    v_heads = dims.get("value_heads", k_heads)
    v_dim = dims.get("value_dim", k_dim)
    qk_dim, v_total = str(k_heads * k_dim), str(v_heads * v_dim)

    return [
        FlowStep(
            stage="attention",
            name="Query projection",
            expr="Q = X̂ · Wq",
            input_shape=["T", d],
            weight_name="Wq",
            weight_shape=[d, qk_dim],
            output_shape=["T", qk_dim],
            note=f"reshape -> {k_heads} heads × [T × {k_dim}]",
        ),
        FlowStep(
            stage="attention",
            name="Key projection",
            expr="K = X̂ · Wk",
            input_shape=["T", d],
            weight_name="Wk",
            weight_shape=[d, qk_dim],
            output_shape=["T", qk_dim],
            note=f"reshape -> {k_heads} heads × [T × {k_dim}]",
        ),
        FlowStep(
            stage="attention",
            name="Value projection",
            expr="V = X̂ · Wv",
            input_shape=["T", d],
            weight_name="Wv",
            weight_shape=[d, v_total],
            output_shape=["T", v_total],
            note=f"reshape -> {v_heads} heads × [T × {v_dim}]",
        ),
        FlowStep(
            stage="attention",
            name="Recurrent state update",
            expr="S_t = decay · S_(t-1) + Kᵀ · V",
            input_shape=[str(k_heads), str(k_dim), str(v_dim)],
            output_shape=[str(k_heads), str(k_dim), str(v_dim)],
            note="linear attention: no T×T score matrix - context is compressed "
            "into a fixed-size state, so cost stays linear in T",
        ),
        FlowStep(
            stage="attention",
            name="State readout",
            expr="ctx = Q · S",
            input_shape=[str(k_heads), "T", str(k_dim)],
            output_shape=["T", v_total],
            note="each query reads from the compressed state instead of attending to "
            "every past token",
        ),
        FlowStep(
            stage="attention",
            name="Output projection",
            expr="attn_out = ctx · Wo",
            input_shape=["T", v_total],
            weight_name="Wo",
            weight_shape=[v_total, d],
            output_shape=["T", d],
            note="projects back into residual-stream space",
        ),
    ]


def _mlp_steps(spec: FlowSpec) -> list[FlowStep]:
    d, m = str(spec.hidden_size), str(spec.mlp_size)
    act = spec.activation
    if spec.gated_mlp:
        return [
            FlowStep(
                stage="mlp",
                name="Gate projection",
                expr="g = X̂ · W_gate",
                input_shape=["T", d],
                weight_name="W_gate",
                weight_shape=[d, m],
                output_shape=["T", m],
                note="decides how much of each expanded feature passes through",
            ),
            FlowStep(
                stage="mlp",
                name="Up projection",
                expr="u = X̂ · W_up",
                input_shape=["T", d],
                weight_name="W_up",
                weight_shape=[d, m],
                output_shape=["T", m],
                note=f"expands each token vector from {d} to {m} features",
            ),
            FlowStep(
                stage="mlp",
                name=f"Gated activation ({act})",
                expr=f"a = {act}(g) ⊙ u",
                input_shape=["T", m],
                output_shape=["T", m],
                note="element-wise: the activated gate switches expanded features on/off",
            ),
            FlowStep(
                stage="mlp",
                name="Down projection",
                expr="mlp_out = a · W_down",
                input_shape=["T", m],
                weight_name="W_down",
                weight_shape=[m, d],
                output_shape=["T", d],
                note=f"compresses back from {m} to {d} residual dimensions",
            ),
        ]
    return [
        FlowStep(
            stage="mlp",
            name="Up projection",
            expr="u = X̂ · W_fc",
            input_shape=["T", d],
            weight_name="W_fc",
            weight_shape=[d, m],
            output_shape=["T", m],
            note=f"expands each token vector from {d} to {m} features",
        ),
        FlowStep(
            stage="mlp",
            name=f"Activation ({act})",
            expr=f"a = {act}(u)",
            input_shape=["T", m],
            output_shape=["T", m],
            note="non-linearity applied element-wise",
        ),
        FlowStep(
            stage="mlp",
            name="Down projection",
            expr="mlp_out = a · W_proj",
            input_shape=["T", m],
            weight_name="W_proj",
            weight_shape=[m, d],
            output_shape=["T", d],
            note=f"compresses back from {m} to {d} residual dimensions",
        ),
    ]


def _residual(expr: str, spec: FlowSpec, what: str) -> FlowStep:
    d = str(spec.hidden_size)
    return FlowStep(
        stage="residual",
        name="Residual add",
        expr=expr,
        input_shape=["T", d],
        output_shape=["T", d],
        note=f"the {what} the residual stream - the block only *edits* the stream, "
        "it never replaces it",
    )
