import { getArchitecture, getHardware, getModels, runPrompt } from "./api.js";
import { Architecture3D } from "./components/architecture3d.js";
import { renderFlowPanel } from "./components/flowPanel.js";
import { renderHardwarePanel } from "./components/hardwarePanel.js";
import { renderLayerInspector } from "./components/layerInspector.js";
import { renderModelMap } from "./components/modelMap.js";
import { renderPredictions } from "./components/predictions.js";
import { renderTokenJourney } from "./components/tokenJourney.js";
import { setState, state, subscribe } from "./state.js";

const elements = {
  status: document.querySelector("#status-pill"),
  hardware: document.querySelector("#hardware-chips"),
  modelSelect: document.querySelector("#model-select"),
  refreshModelsButton: document.querySelector("#refresh-models-button"),
  demoMode: document.querySelector("#demo-mode"),
  realMode: document.querySelector("#real-mode"),
  hfInput: document.querySelector("#hf-input"),
  hfLoadButton: document.querySelector("#hf-load-button"),
  prompt: document.querySelector("#prompt-input"),
  runButton: document.querySelector("#run-button"),
  modelSummary: document.querySelector("#model-summary"),
  modelMap: document.querySelector("#model-map"),
  tabs: [...document.querySelectorAll(".tabs [data-tab]")],
  panels: {
    architecture: document.querySelector("#tab-architecture"),
    flow: document.querySelector("#tab-flow"),
    attention: document.querySelector("#tab-attention"),
    output: document.querySelector("#tab-output"),
  },
  spatialViewport: document.querySelector("#spatial-viewport"),
  spatialDetails: document.querySelector("#spatial-details"),
  selectedLayerLabel: document.querySelector("#selected-layer-label"),
  flowPanel: document.querySelector("#flow-panel"),
  layerInspector: document.querySelector("#layer-inspector"),
  tokenJourney: document.querySelector("#token-journey"),
  predictions: document.querySelector("#predictions"),
};

const architecture3d = new Architecture3D(
  elements.spatialViewport,
  elements.spatialDetails,
  (selectedLayer) => setState({ selectedLayer, selectedHead: 0 }),
  (selectedHead) => setState({ selectedHead }),
);

subscribe(render);
boot();

async function boot() {
  try {
    setStatus("Detecting hardware and loading models...", "busy");
    renderHardwarePanel(elements.hardware, await getHardware());
    await refreshModels();
    await loadArchitecture();
    await runCurrentPrompt();
  } catch (error) {
    setStatus(`Startup error: ${error.message}`, "err");
  }
}

/* ---- data loading ---- */

async function refreshModels() {
  const previousModelId = state.modelId;
  const modelPayload = await getModels();
  const keepPrevious =
    previousModelId.startsWith("hf:") ||
    modelPayload.models.some((model) => model.id === previousModelId);
  const nextModelId = keepPrevious ? previousModelId : (modelPayload.models[0]?.id ?? "demo-transformer");
  setState({ models: modelPayload.models, modelId: nextModelId });
  elements.modelSelect.innerHTML = modelPayload.models
    .map((model) => `<option value="${model.id}">${escapeHtml(model.name)}</option>`)
    .join("");
  if (nextModelId.startsWith("hf:")) {
    ensureModelOption(nextModelId, `${nextModelId.slice(3)} (Hub)`);
  }
  elements.modelSelect.value = nextModelId;
}

async function loadArchitecture() {
  setStatus("Loading architecture...", "busy");
  setState({ architecture: null, run: null, selectedLayer: 0, selectedHead: 0 });
  const architecture = await getArchitecture(state.modelId, state.mode);
  setState({ architecture, selectedLayer: 0, selectedHead: 0 });
  if (architecture.model.id !== state.modelId) {
    // the backend silently falls back to demo data when the ml extras are missing
    setStatus(
      `Showing ${architecture.model.name} instead of ${state.modelId} — real mode needs ` +
        "`uv sync --extra ml` and a server restart.",
      "err",
    );
    return;
  }
  setStatus(`Loaded ${architecture.model.name}.`, "ok");
}

async function runCurrentPrompt() {
  const prompt = elements.prompt.value.trim();
  if (!prompt) {
    setStatus("Enter a prompt first.", "err");
    return;
  }

  elements.runButton.disabled = true;
  elements.runButton.textContent = "Running…";
  setStatus(
    state.mode === "real" ? "Running a real forward pass (first run loads weights)..." : "Generating demo internals...",
    "busy",
  );
  try {
    const run = await runPrompt({ prompt, modelId: state.modelId, mode: state.mode });
    setState({ run, selectedHead: 0 });
    setStatus(`${run.note} Source: ${run.source}.`, "ok");
  } catch (error) {
    setStatus(`Run failed: ${error.message}`, "err");
  } finally {
    elements.runButton.disabled = false;
    elements.runButton.textContent = "Run";
  }
}

async function loadHfModel() {
  const repo = elements.hfInput.value
    .trim()
    .replace(/^https?:\/\/huggingface\.co\//, "")
    .replace(/\/+$/, "");
  if (!repo) {
    setStatus("Enter a Hugging Face id like Qwen/Qwen3-0.6B.", "err");
    return;
  }

  const modelId = `hf:${repo}`;
  setMode("real");
  setState({ mode: "real", modelId });
  setStatus(`Fetching config for ${repo} from the Hugging Face Hub...`, "busy");
  try {
    await loadArchitecture();
    ensureModelOption(modelId, `${repo} (Hub)`);
    elements.modelSelect.value = modelId;
  } catch (error) {
    setStatus(`Could not load ${repo}: ${error.message}`, "err");
  }
}

/* ---- rendering ---- */

function render(current) {
  const architecture = current.architecture;
  elements.modelSummary.textContent = architecture
    ? `${architecture.model.layer_count} layers · ${architecture.model.head_count} heads`
    : "";
  const layer = architecture?.layers?.[current.selectedLayer];
  elements.selectedLayerLabel.textContent = layer ? layer.name : "";

  renderModelMap(
    elements.modelMap,
    architecture,
    current.selectedLayer,
    (selectedLayer) => setState({ selectedLayer, selectedHead: 0 }),
    current.run?.tokens?.length,
  );
  architecture3d.update(current);
  renderFlowPanel(elements.flowPanel, architecture, current.run, current.selectedLayer, (selectedLayer) => {
    setState({ selectedLayer, selectedHead: 0 });
  });
  renderLayerInspector(
    elements.layerInspector,
    architecture,
    current.run,
    current.selectedLayer,
    current.selectedHead,
    (selectedHead) => setState({ selectedHead }),
  );
  renderTokenJourney(elements.tokenJourney, current.run);
  renderPredictions(elements.predictions, current.run);
}

function switchTab(name) {
  setState({ activeTab: name });
  for (const button of elements.tabs) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  for (const [key, panel] of Object.entries(elements.panels)) {
    panel.hidden = key !== name;
  }
  if (name === "architecture") {
    requestAnimationFrame(() => architecture3d.resize());
  }
  if (name === "attention") {
    // the heatmap canvas sizes itself from its container, which was hidden
    render(state);
  }
}

function setMode(mode) {
  elements.realMode.classList.toggle("active", mode === "real");
  elements.demoMode.classList.toggle("active", mode === "demo");
}

function ensureModelOption(id, name) {
  if (![...elements.modelSelect.options].some((option) => option.value === id)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = name;
    elements.modelSelect.appendChild(option);
  }
}

function setStatus(message, kind = "info") {
  elements.status.textContent = message;
  elements.status.dataset.kind = kind;
  elements.status.title = message;
}

function withStatusErrors(handler) {
  return async (...args) => {
    try {
      await handler(...args);
    } catch (error) {
      setStatus(`Error: ${error.message}`, "err");
    }
  };
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

/* ---- events ---- */

elements.tabs.forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

elements.runButton.addEventListener("click", runCurrentPrompt);
elements.hfLoadButton.addEventListener("click", loadHfModel);
elements.hfInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadHfModel();
  }
});
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    runCurrentPrompt();
  }
});

elements.refreshModelsButton.addEventListener(
  "click",
  withStatusErrors(async () => {
    setStatus("Rescanning the models/ folder...", "busy");
    await refreshModels();
    await loadArchitecture();
  }),
);

elements.modelSelect.addEventListener(
  "change",
  withStatusErrors(async () => {
    const selectedModel = state.models.find((model) => model.id === elements.modelSelect.value);
    const isHub = elements.modelSelect.value.startsWith("hf:");
    const nextMode = isHub || selectedModel?.supports_real_run ? "real" : "demo";
    setMode(nextMode);
    setState({ modelId: elements.modelSelect.value, mode: nextMode });
    await loadArchitecture();
  }),
);

elements.demoMode.addEventListener(
  "click",
  withStatusErrors(async () => {
    setMode("demo");
    setState({ mode: "demo", modelId: "demo-transformer" });
    elements.modelSelect.value = "demo-transformer";
    await loadArchitecture();
  }),
);

elements.realMode.addEventListener(
  "click",
  withStatusErrors(async () => {
    setMode("real");
    await refreshModels();
    const localModel = state.models.find((model) => model.id.includes("qwen") && model.supports_real_run);
    const preferred = localModel?.id ?? state.models.find((model) => model.supports_real_run)?.id ?? "distilgpt2";
    setState({ mode: "real", modelId: preferred });
    elements.modelSelect.value = preferred;
    await loadArchitecture();
  }),
);

window.addEventListener("resize", () => {
  architecture3d.resize();
});
