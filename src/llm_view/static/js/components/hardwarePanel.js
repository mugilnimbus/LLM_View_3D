export function renderHardwarePanel(element, hardware) {
  const gpu = hardware.gpus?.[0];
  const gpuText = gpu
    ? `${shortGpuName(gpu.name)} · ${Math.round(gpu.memory_free_mb / 1024)} GB free`
    : "no NVIDIA GPU";

  element.innerHTML = [
    chip("CPU", `${hardware.cpu_cores}c / ${hardware.logical_cores}t`),
    chip("RAM", `${Math.round(hardware.ram_available_gb)} / ${Math.round(hardware.ram_total_gb)} GB`),
    chip("GPU", gpuText),
    chip("Py", hardware.python),
  ].join("");
}

function chip(label, value) {
  return `<span class="chip">${escapeHtml(label)} <strong>${escapeHtml(value)}</strong></span>`;
}

function shortGpuName(name) {
  return String(name).replace(/^NVIDIA\s+(GeForce\s+)?/i, "");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
