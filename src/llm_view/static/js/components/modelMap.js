export function renderModelMap(element, architecture, selectedLayer, onSelect, tokenCount) {
  if (!architecture) {
    element.innerHTML = "";
    return;
  }

  const t = tokenCount ?? "T";
  element.innerHTML = architecture.layers
    .map((layer) => {
      const active = layer.index === selectedLayer ? " active" : "";
      return `
        <button class="layer-node${active}" data-layer="${layer.index}" type="button">
          <span class="layer-index">${layer.index + 1}</span>
          <span class="layer-title">
            <strong>${escapeHtml(layer.name)}</strong>
            <span>${escapeHtml(layer.components.slice(0, 3).join(" · "))}</span>
          </span>
          <span class="layer-meta">${layer.head_count} heads<br><span class="layer-io">in [${escapeHtml(t)} × ${layer.hidden_size}]</span></span>
        </button>
      `;
    })
    .join("");

  element.querySelectorAll("[data-layer]").forEach((button) => {
    button.addEventListener("click", () => onSelect(Number(button.dataset.layer)));
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
