import { drawHeatmap } from "./attentionHeatmap.js";

export function renderLayerInspector(element, architecture, run, selectedLayer, selectedHead, onHead) {
  if (!architecture) {
    element.innerHTML = "";
    return;
  }

  const layer = architecture.layers[selectedLayer] ?? architecture.layers[0];
  const layerRun = run?.layers?.find((item) => item.index === layer.index);
  const heads = layerRun?.heads ?? [];
  const head = heads[selectedHead] ?? heads[0];

  element.innerHTML = `
    <div class="inspector-grid">
      <section class="card">
        <h3>Attention heads ${heads.length ? `<span class="hint">${heads.length} live</span>` : ""}</h3>
        <div class="head-grid">
          ${heads.map((item) => headButton(item, selectedHead)).join("") || empty(emptyHeadsText(layer, run))}
        </div>
        <div class="heatmap-wrap">
          <canvas class="heatmap" id="attention-canvas"></canvas>
          <div class="token-axis">
            ${(run?.tokens ?? []).map((token) => `<span class="token-chip">${escapeHtml(token)}</span>`).join("")}
          </div>
        </div>
      </section>
      <section class="card">
        <h3>Block components</h3>
        <div class="component-list">
          ${layer.components.map((component) => `<div class="component-pill">${escapeHtml(component)}</div>`).join("")}
        </div>
        <h3>Metrics</h3>
        <div class="metric-list">
          ${(layerRun?.metrics ?? []).map(metricTemplate).join("") || empty("Run a prompt for layer metrics.")}
        </div>
        <h3>Top activations</h3>
        <div class="activation-list">
          ${(layerRun?.top_activations ?? []).map(metricTemplate).join("") || empty("Activations appear after a run.")}
        </div>
      </section>
    </div>
  `;

  element.querySelectorAll("[data-head]").forEach((button) => {
    button.addEventListener("click", () => onHead(Number(button.dataset.head)));
  });

  drawHeatmap(element.querySelector("#attention-canvas"), head?.attention);
}

function emptyHeadsText(layer, run) {
  if (!run) {
    return "Run a prompt to see live attention heads.";
  }
  if (layer.attention_type === "linear attention") {
    return "Linear attention block: context lives in a recurrent state, so there is no T×T attention matrix to show.";
  }
  return "No attention tensors for this block.";
}

function metricTemplate(metric) {
  return `
    <div class="metric">
      <strong>${escapeHtml(metric.value)} ${escapeHtml(metric.unit)}</strong>
      <span>${escapeHtml(metric.name)}</span>
    </div>
  `;
}

function headButton(head, selectedHead) {
  const active = head.index === selectedHead ? " active" : "";
  return `
    <button class="head-button${active}" data-head="${head.index}" type="button" title="${escapeHtml(head.role)}">
      H${head.index + 1}<br>${Math.round(head.focus_score * 100)}%
    </button>
  `;
}

function empty(text) {
  return `<div class="component-pill">${escapeHtml(text)}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
