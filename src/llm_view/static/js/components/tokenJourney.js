export function renderTokenJourney(element, run) {
  if (!run?.token_traces?.length) {
    element.innerHTML = `<div class="component-pill">Run a prompt to inspect token salience across layers.</div>`;
    return;
  }

  element.innerHTML = `
    <div class="journey-grid">
      ${run.token_traces.map(rowTemplate).join("")}
    </div>
  `;
}

function rowTemplate(trace) {
  return `
    <div class="journey-row">
      <div class="journey-token">${escapeHtml(trace.index)} · ${escapeHtml(trace.token)}</div>
      <div class="sparkline">
        ${trace.salience_by_layer.map((value) => `<span class="spark" style="height:${Math.max(3, value * 26)}px"></span>`).join("")}
      </div>
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
