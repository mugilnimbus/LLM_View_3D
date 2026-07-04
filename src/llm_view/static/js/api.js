export async function getHardware() {
  return getJson("/api/hardware");
}

export async function getModels() {
  return getJson("/api/models");
}

export async function getArchitecture(modelId, mode) {
  return getJson(`/api/architecture?model_id=${encodeURIComponent(modelId)}&mode=${mode}`);
}

export async function runPrompt({ prompt, modelId, mode }) {
  return postJson("/api/run", { prompt, model_id: modelId, mode });
}

async function getJson(path) {
  const response = await fetch(path);
  await ensureOk(response);
  return response.json();
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await ensureOk(response);
  return response.json();
}

async function ensureOk(response) {
  if (response.ok) {
    return;
  }
  let message = await response.text();
  try {
    const detail = JSON.parse(message).detail;
    if (typeof detail === "string") {
      message = detail;
    } else if (detail) {
      message = JSON.stringify(detail);
    }
  } catch {
    // keep raw text
  }
  throw new Error(message || `HTTP ${response.status}`);
}
