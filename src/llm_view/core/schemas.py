from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GpuInfo(BaseModel):
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    temperature_c: int


class HardwareInfo(BaseModel):
    os: str
    python: str
    cpu: str
    cpu_cores: int
    logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[GpuInfo]


class ModelInfo(BaseModel):
    id: str
    name: str
    hf_id: str | None
    description: str
    layer_count: int
    head_count: int
    hidden_size: int
    mlp_size: int
    parameter_count: str
    supports_real_run: bool
    vocab_size: int = 0


class MatrixSpec(BaseModel):
    name: str
    shape: list[int | str]
    role: str


class FlowStep(BaseModel):
    """One computation in the forward pass, with exact tensor shapes.

    Shapes are lists of dimension labels ("T" is the symbolic sequence length),
    rendered by the UI as e.g. [T x 768] @ Wq [768 x 2304] -> [T x 2304].
    """

    stage: Literal["input", "norm", "attention", "residual", "mlp", "output"]
    name: str
    expr: str
    input_shape: list[str] = Field(default_factory=list)
    weight_name: str | None = None
    weight_shape: list[str] | None = None
    output_shape: list[str] = Field(default_factory=list)
    note: str = ""


class LayerArchitecture(BaseModel):
    index: int
    name: str
    components: list[str]
    head_count: int
    hidden_size: int
    mlp_size: int
    attention_type: str = "attention"
    matrices: list[MatrixSpec] = Field(default_factory=list)
    flow: list[FlowStep] = Field(default_factory=list)


class ArchitectureResponse(BaseModel):
    model: ModelInfo
    layers: list[LayerArchitecture]
    pre_flow: list[FlowStep] = Field(default_factory=list)
    post_flow: list[FlowStep] = Field(default_factory=list)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    model_id: str = "demo-transformer"
    mode: Literal["demo", "real"] = "demo"


class ComponentMetric(BaseModel):
    name: str
    value: float
    unit: str


class AttentionHead(BaseModel):
    index: int
    label: str
    attention: list[list[float]]
    focus_score: float
    role: str


class LayerRun(BaseModel):
    index: int
    heads: list[AttentionHead]
    metrics: list[ComponentMetric]
    top_activations: list[ComponentMetric]


class TokenTrace(BaseModel):
    index: int
    token: str
    salience_by_layer: list[float]


class Prediction(BaseModel):
    token: str
    probability: float


class RunResponse(BaseModel):
    model: ModelInfo
    prompt: str
    tokens: list[str]
    layers: list[LayerRun]
    token_traces: list[TokenTrace]
    predictions: list[Prediction]
    source: str
    note: str
