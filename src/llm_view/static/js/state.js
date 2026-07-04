export const state = {
  models: [],
  architecture: null,
  run: null,
  selectedLayer: 0,
  selectedHead: 0,
  mode: "demo",
  modelId: "demo-transformer",
  activeTab: "architecture",
};

const listeners = new Set();

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setState(patch) {
  Object.assign(state, patch);
  for (const listener of listeners) {
    listener(state);
  }
}
