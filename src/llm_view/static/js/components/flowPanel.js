const STAGE_LABELS = {
  input: "input",
  norm: "norm",
  attention: "attention",
  residual: "residual",
  mlp: "mlp",
  output: "output",
};

export function renderFlowPanel(element, architecture, run, selectedLayer, onSelectLayer) {
  if (!architecture) {
    element.innerHTML = `<div class="component-pill">Load a model to trace its forward pass.</div>`;
    return;
  }

  const layers = architecture.layers;
  const layer = layers[selectedLayer] ?? layers[0];
  const total = layers.length;
  const index = layer?.index ?? 0;
  const tokenCount = run?.tokens?.length ?? null;

  const before = index > 0 ? collapsedBlocks(1, index, "already transformed the stream") : "";
  const after =
    index < total - 1
      ? collapsedBlocks(index + 2, total, "repeat the same shapes on the updated stream")
      : "";

  element.innerHTML = `
    ${tokenCount ? `<div class="flow-context">T = ${tokenCount} tokens from your prompt</div>` : `<div class="flow-context">T = sequence length (run a prompt to substitute real token counts)</div>`}
    ${section("Input", "tokens enter the residual stream", architecture.pre_flow, tokenCount)}
    <div class="flow-arrow">↓</div>
    ${before}
    <section class="flow-section flow-block">
      <header class="flow-stage-header">
        <button class="flow-nav" data-flow-nav="-1" type="button" ${index === 0 ? "disabled" : ""}>◀</button>
        <span class="flow-stage-title">Transformer Block ${index + 1} <em>of ${total} · ${escapeHtml(layer.attention_type)}</em></span>
        <button class="flow-nav" data-flow-nav="1" type="button" ${index === total - 1 ? "disabled" : ""}>▶</button>
      </header>
      ${steps(layer.flow, tokenCount)}
    </section>
    ${after}
    <div class="flow-arrow">↓</div>
    ${section("Output", "residual stream becomes next-token probabilities", architecture.post_flow, tokenCount)}
  `;

  element.querySelectorAll("[data-flow-nav]").forEach((button) => {
    button.addEventListener("click", () => {
      const next = Math.min(total - 1, Math.max(0, index + Number(button.dataset.flowNav)));
      onSelectLayer(next);
    });
  });
}

function section(title, subtitle, flowSteps, tokenCount) {
  if (!flowSteps?.length) {
    return "";
  }
  return `
    <section class="flow-section">
      <header class="flow-stage-header">
        <span class="flow-stage-title">${escapeHtml(title)} <em>${escapeHtml(subtitle)}</em></span>
      </header>
      ${steps(flowSteps, tokenCount)}
    </section>
  `;
}

function steps(flowSteps, tokenCount) {
  return flowSteps.map((step) => stepTemplate(step, tokenCount)).join(`<div class="flow-arrow">↓</div>`);
}

function stepTemplate(step, tokenCount) {
  // Not everything with weights is a matmul: γ is an element-wise scale,
  // embeddings are row lookups, learned positions are added.
  let op = "@";
  if (step.stage === "norm" || step.name.startsWith("Final")) {
    op = "⊙";
  } else if (step.name.includes("lookup")) {
    op = "rows of";
  } else if (step.name.includes("Position embedding")) {
    op = "+";
  }
  const weight = step.weight_shape
    ? `<span class="flow-op">${op}</span> ${shapeChip(step.weight_shape, "w", tokenCount, step.weight_name)}`
    : "";
  return `
    <div class="flow-step" data-stage="${STAGE_LABELS[step.stage] ?? "input"}">
      <div class="flow-head">
        <strong>${escapeHtml(step.name)}</strong>
        <code class="flow-expr">${escapeHtml(step.expr)}</code>
      </div>
      <div class="flow-shapes">
        ${shapeChip(step.input_shape, "in", tokenCount)}
        ${weight}
        <span class="flow-op">→</span>
        ${shapeChip(step.output_shape, "out", tokenCount)}
      </div>
      ${step.note ? `<div class="flow-note">${escapeHtml(step.note)}</div>` : ""}
    </div>
  `;
}

function shapeChip(dims, kind, tokenCount, label = "") {
  const rendered = (dims ?? [])
    .map((dim) => (tokenCount && dim === "T" ? String(tokenCount) : dim))
    .map(escapeHtml)
    .join(" × ");
  const prefix = label ? `${escapeHtml(label)} ` : "";
  return `<code class="shape ${kind}">${prefix}[${rendered}]</code>`;
}

function collapsedBlocks(from, to, note) {
  const range = from === to ? `Block ${from}` : `Blocks ${from}–${to}`;
  return `
    <div class="flow-collapsed">
      <span class="flow-collapsed-dots">⋮</span>
      ${escapeHtml(range)} — identical shapes, ${escapeHtml(note)}
      <span class="flow-collapsed-dots">⋮</span>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
