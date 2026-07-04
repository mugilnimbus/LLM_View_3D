export function renderPredictions(element, run) {
  if (!run?.predictions?.length) {
    element.innerHTML = `<div class="component-pill">Next-token probabilities appear after a run.</div>`;
    return;
  }

  element.innerHTML = `
    <div class="prediction-list">
      ${run.predictions.map(rowTemplate).join("")}
    </div>
  `;
}

function rowTemplate(prediction) {
  const width = Math.max(2, prediction.probability * 100);
  return `
    <div class="prediction-row">
      <strong>${escapeHtml(prediction.token || " ")}</strong>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
      <span>${Math.round(prediction.probability * 100)}%</span>
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
