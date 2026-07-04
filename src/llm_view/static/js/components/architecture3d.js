import * as THREE from "../../vendor/three.module.js";

// floor keeps the camera outside the layer geometry (plates reach ~3.5 from the axis)
const MIN_RADIUS = 4.4;
const MAX_RADIUS = 44;
// fixed vertical distance between blocks: enough room that boxes, labels and
// arrows never collide, however many layers the model has (camera rides the tower)
const LAYER_STEP = 0.85;
// cylindrical camera: (radius, theta, height) around the model's vertical axis,
// with a small fixed downward pitch so layer plates stay visible
const CAMERA_PITCH = 0.22;
const DEG = Math.PI / 180;

const COLORS = {
  background: 0x000000,
  grid: 0x1a2126,
  layer: 0x24313a,
  activeLayer: 0x42d9c8,
  attention: 0xf0b94e,
  mlp: 0xf27360,
  residual: 0x7bd88f,
  weight: 0x9d8cff,
  norm: 0x82c7ff,
  arrow: 0x4fd6c8,
  text: "#f4f7f8",
  muted: "#9cabaf",
  flowText: "#7bd88f",
};

export class Architecture3D {
  constructor(container, detailsElement, onSelectLayer, onSelectHead) {
    this.container = container;
    this.detailsElement = detailsElement;
    this.onSelectLayer = onSelectLayer;
    this.onSelectHead = onSelectHead;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.background);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.pickables = [];
    this.selectedObject = null;
    this.architectureKey = "";
    this.selectedLayer = 0;
    this.selectedHead = 0;
    this.theta = -0.68;
    this.thetaGoal = -0.68;
    this.height = 0;
    this.heightGoal = 0;
    this.heightMin = -8;
    this.heightMax = 8;
    this.radius = 18;
    this.radiusGoal = 18;
    this.layerYs = [];
    this.lastFocusedLayer = null;
    this.drag = { active: false, x: 0, y: 0, moved: 0 };
    this.sliderActive = false;

    this._setupLights();
    this._bindControls();
    this._buildControlBar();
    this.resize();
    this.animate();
  }

  update({ architecture, selectedLayer, selectedHead, run }) {
    this.selectedLayer = selectedLayer ?? 0;
    this.selectedHead = selectedHead ?? 0;
    const key = architecture
      ? `${architecture.model.id}:${architecture.model.layer_count}:${architecture.model.head_count}`
      : "";

    if (key !== this.architectureKey) {
      this.architectureKey = key;
      this._build(architecture);
      this._resetView();
      // show the whole stack on a fresh model; only zoom on explicit selection changes
      this.lastFocusedLayer = this.selectedLayer;
    }

    if (this.selectedLayer !== this.lastFocusedLayer) {
      this.lastFocusedLayer = this.selectedLayer;
      this._focusLayer(this.selectedLayer);
    }

    this._applySelection();
    this._renderDetails(architecture, run);
  }

  _focusLayer(layerIndex) {
    const y = this.layerYs[layerIndex];
    if (y === undefined) {
      return;
    }
    this.heightGoal = y;
    this.radiusGoal = Math.min(this.radiusGoal, 8.5);
  }

  _resetView() {
    this.heightGoal = 0;
    this.radiusGoal = this.defaultRadius ?? 18;
    this.thetaGoal = -0.68;
    this.theta = (((this.theta / DEG + 180) % 360 + 360) % 360 - 180) * DEG;
  }

  resize() {
    const width = Math.max(320, this.container.clientWidth || 320);
    const height = Math.max(360, this.container.clientHeight || 360);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  animate() {
    this.theta += (this.thetaGoal - this.theta) * 0.25;
    this.height += (this.heightGoal - this.height) * 0.15;
    this.radius += (this.radiusGoal - this.radius) * 0.12;
    this._positionCamera();
    this._syncControlBar();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.animate());
  }

  _setupLights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(6, 8, 8);
    const rim = new THREE.DirectionalLight(0x42d9c8, 0.8);
    rim.position.set(-8, 4, -6);
    this.scene.add(ambient, key, rim);

    const grid = new THREE.GridHelper(16, 16, COLORS.grid, COLORS.grid);
    grid.position.y = -6.3;
    this.scene.add(grid);
  }

  _bindControls() {
    this.container.addEventListener("contextmenu", (event) => event.preventDefault());

    this.container.addEventListener("pointerdown", (event) => {
      this.drag = { active: true, x: event.clientX, y: event.clientY, moved: 0 };
      this.container.setPointerCapture(event.pointerId);
    });

    this.container.addEventListener("pointermove", (event) => {
      if (!this.drag.active) {
        this._setHoverCursor(event);
        return;
      }

      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      this.drag.moved += Math.abs(dx) + Math.abs(dy);
      const fine = event.altKey ? 0.2 : 1;
      // cylindrical: horizontal drag rotates around the axis, vertical drag rides up/down
      this.thetaGoal -= dx * 0.005 * fine;
      this.heightGoal = this._clampHeight(this.heightGoal + dy * 0.025 * fine);
    });

    this.container.addEventListener("pointerup", (event) => {
      const wasClick = this.drag.moved < 5;
      this.drag.active = false;
      if (wasClick) {
        this._pick(event);
      }
    });

    this.container.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const fine = event.altKey ? 0.2 : 1;
        // proportional zoom: small steps when close, larger when far out
        const next = this.radiusGoal * (1 + Math.sign(event.deltaY) * 0.09 * fine);
        this.radiusGoal = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, next));
      },
      { passive: false },
    );

    this.container.addEventListener("dblclick", () => this._resetView());

    this.container.addEventListener("keydown", (event) => {
      const fine = event.altKey ? 0.2 : 1;
      const handled = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "_"];
      if (!handled.includes(event.key)) {
        return;
      }
      event.preventDefault();
      if (event.key === "ArrowLeft") this.thetaGoal += 6 * DEG * fine;
      if (event.key === "ArrowRight") this.thetaGoal -= 6 * DEG * fine;
      if (event.key === "ArrowUp") this.heightGoal = this._clampHeight(this.heightGoal + 0.6 * fine);
      if (event.key === "ArrowDown") this.heightGoal = this._clampHeight(this.heightGoal - 0.6 * fine);
      if (event.key === "+" || event.key === "=") {
        this.radiusGoal = Math.max(MIN_RADIUS, this.radiusGoal * (1 - 0.09 * fine));
      }
      if (event.key === "-" || event.key === "_") {
        this.radiusGoal = Math.min(MAX_RADIUS, this.radiusGoal * (1 + 0.09 * fine));
      }
    });
  }

  _clampHeight(value) {
    return Math.max(this.heightMin, Math.min(this.heightMax, value));
  }

  _buildControlBar() {
    const bar = document.createElement("div");
    bar.className = "spatial-controls";
    bar.innerHTML = `
      <label>Rotate <output data-value="theta"></output>
        <input type="range" data-control="theta" min="-180" max="180" step="0.5" />
      </label>
      <label>Height <output data-value="height"></output>
        <input type="range" data-control="height" min="0" max="100" step="0.5" />
      </label>
      <label>Zoom <output data-value="zoom"></output>
        <input type="range" data-control="zoom" min="0" max="100" step="0.25" />
      </label>
      <button type="button" data-control="reset" title="Reset view">⟲</button>
    `;
    // keep slider interaction from starting an orbit drag underneath
    for (const type of ["pointerdown", "pointermove", "pointerup", "wheel", "dblclick", "keydown"]) {
      bar.addEventListener(type, (event) => event.stopPropagation());
    }
    bar.addEventListener("pointerdown", () => {
      this.sliderActive = true;
    });
    window.addEventListener("pointerup", () => {
      this.sliderActive = false;
    });

    this.controls = {
      theta: bar.querySelector('[data-control="theta"]'),
      height: bar.querySelector('[data-control="height"]'),
      zoom: bar.querySelector('[data-control="zoom"]'),
      thetaValue: bar.querySelector('[data-value="theta"]'),
      heightValue: bar.querySelector('[data-value="height"]'),
      zoomValue: bar.querySelector('[data-value="zoom"]'),
    };
    this.controls.theta.addEventListener("input", () => {
      const deg = Number(this.controls.theta.value);
      // wrap current angle into the slider's window so the camera takes the short way
      this.theta = (((this.theta / DEG + 180) % 360 + 360) % 360 - 180) * DEG;
      this.thetaGoal = deg * DEG;
      this.controls.thetaValue.textContent = `${Math.round(deg)}°`;
    });
    this.controls.height.addEventListener("input", () => {
      const percent = Number(this.controls.height.value);
      this.heightGoal = this.heightMin + (percent / 100) * (this.heightMax - this.heightMin);
      this.controls.heightValue.textContent = this._heightLabel(percent);
    });
    this.controls.zoom.addEventListener("input", () => {
      const percent = Number(this.controls.zoom.value);
      this.radiusGoal = MAX_RADIUS - (percent / 100) * (MAX_RADIUS - MIN_RADIUS);
      this.controls.zoomValue.textContent = `${Math.round(percent)}%`;
    });
    bar.querySelector('[data-control="reset"]').addEventListener("click", () => this._resetView());

    this.container.appendChild(bar);
    this._syncControlBar(true);
  }

  _syncControlBar(force = false) {
    if (!this.controls || (this.sliderActive && !force)) {
      return;
    }
    const thetaDeg = ((this.thetaGoal / DEG + 180) % 360 + 360) % 360 - 180;
    const heightSpan = this.heightMax - this.heightMin || 1;
    const heightPercent = Math.max(
      0,
      Math.min(100, ((this.heightGoal - this.heightMin) / heightSpan) * 100),
    );
    const zoomPercent = ((MAX_RADIUS - this.radiusGoal) / (MAX_RADIUS - MIN_RADIUS)) * 100;
    this.controls.theta.value = thetaDeg;
    this.controls.height.value = heightPercent;
    this.controls.zoom.value = zoomPercent;
    this.controls.thetaValue.textContent = `${Math.round(thetaDeg)}°`;
    this.controls.heightValue.textContent = this._heightLabel(heightPercent);
    this.controls.zoomValue.textContent = `${Math.round(zoomPercent)}%`;
  }

  _heightLabel(percent) {
    // show the nearest block when the camera sits at one, otherwise a percentage
    if (this.layerYs.length) {
      const height = this.heightMin + (percent / 100) * (this.heightMax - this.heightMin);
      let nearest = -1;
      let bestDistance = Infinity;
      this.layerYs.forEach((y, index) => {
        const distance = Math.abs(y - height);
        if (distance < bestDistance) {
          bestDistance = distance;
          nearest = index;
        }
      });
      if (bestDistance < 0.35) {
        return `B${nearest + 1}`;
      }
    }
    return `${Math.round(percent)}%`;
  }

  _build(architecture) {
    this._clearRoot();
    this.pickables = [];
    if (!architecture) {
      this.detailsElement.innerHTML = `<div class="component-pill">Load a model to see its 3D architecture.</div>`;
      return;
    }

    const layers = architecture.layers;
    const count = Math.max(1, layers.length);
    const step = LAYER_STEP;
    const startY = -((count - 1) * step) / 2;
    // overview distance grows with the tower so any model fits on reset
    this.defaultRadius = Math.min(MAX_RADIUS - 2, Math.max(16, count * step * 0.72 + 6));
    const width = Math.min(6.2, 3.8 + Math.log2(Math.max(architecture.model.hidden_size, 64)) * 0.22);
    const depth = 2.35;

    const hidden = architecture.model.hidden_size;
    const vocab = architecture.model.vocab_size || "V";
    this.root.add(this._label(`${architecture.model.name}`, 0, startY - 0.92, -2.9, 0.42));
    this.root.add(
      this._label(
        `INPUT FLOW: prompt -> [T] ids -> [T × ${hidden}]\n× ${count} blocks -> [T × ${vocab}] logits`,
        0,
        startY - 1.34,
        2.25,
        0.28,
        COLORS.flowText,
      ),
    );
    // embedding/logits plates grow with vocab size (baseline GPT-2's 50257)
    const vocabScale = _clamp(
      Math.sqrt((architecture.model.vocab_size || 50257) / 50257),
      0.7,
      2.0,
    );
    this.root.add(
      this._stagePlate(
        `Token embeddings\n${architecture.model.vocab_size || "vocab"} x ${architecture.model.hidden_size}`,
        startY - step * 1.4,
        width,
        0x7bd88f,
        vocabScale,
      ),
    );
    this.root.add(
      this._stagePlate(
        `Output logits / lm_head\n${architecture.model.hidden_size} x ${architecture.model.vocab_size || "vocab"}`,
        startY + step * (count + 0.4),
        width,
        0xf27360,
        vocabScale,
      ),
    );
    this.root.add(this._residualSpine(startY - step, startY + step * count));
    this.root.add(
      this._flowArrow(
        new THREE.Vector3(width / 2 + 1.22, startY - step * 1.1, 1.72),
        new THREE.Vector3(width / 2 + 1.22, startY + step * (count + 0.25), 1.72),
        COLORS.residual,
        `INPUT FLOW\n[T × ${hidden}]`,
      ),
    );

    this.layerYs = layers.map((_, index) => startY + index * step);
    // height range spans embeddings plate to logits plate, with a little headroom
    this.heightMin = startY - step * 2 - 0.8;
    this.heightMax = startY + step * (count + 1) + 0.8;
    layers.forEach((layer, index) => {
      const y = startY + index * step;
      this._addLayer(layer, y, width, depth, step);
    });
  }

  _addLayer(layer, y, width, depth, step) {
    const layerGroup = new THREE.Group();
    layerGroup.position.y = y;
    layerGroup.userData = { layerIndex: layer.index };

    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.09, depth),
      this._material(COLORS.layer, 0.88),
    );
    plate.userData = {
      kind: "layer",
      layerIndex: layer.index,
      title: layer.name,
      detail: `${layer.hidden_size} hidden · ${layer.head_count} heads · MLP ${layer.mlp_size}`,
    };
    layerGroup.add(plate);
    this.pickables.push(plate);

    const attentionType = layer.attention_type ?? layer.components.find((component) => component.includes("attention")) ?? "attention";
    const layerLabel = `L${String(layer.index + 1).padStart(2, "0")}\n${attentionType}\nH ${layer.hidden_size}`;
    layerGroup.add(this._label(layerLabel, -width / 2 - 0.95, 0.08, 0.0, 0.15));
    layerGroup.add(this._label("LN", -width * 0.32, 0.26, 0.95, 0.12));
    layerGroup.add(this._label("Q/K/V heads", 0, 0.35, -depth * 0.82, 0.13));
    layerGroup.add(this._label("MLP", width / 2 + 0.58, 0.4, 0, 0.13));
    // block input size on the residual spine: same [T × d] enters and leaves every block
    layerGroup.add(
      this._label(`in/out [T × ${layer.hidden_size}]`, 1.3, 0.18, 1.95, 0.17, COLORS.flowText),
    );

    // norm scale vector is [hidden]: length tracks the residual width
    const norm = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.22, _clamp(0.58 * Math.sqrt(layer.hidden_size / 768), 0.3, 1.0)),
      this._material(COLORS.norm, 0.96),
    );
    norm.position.set(-width * 0.32, 0.08, 0.84);
    norm.userData = {
      kind: "norm",
      layerIndex: layer.index,
      title: `Layer ${layer.index + 1} RMSNorm`,
      detail: `scale vector ${layer.hidden_size}`,
    };
    layerGroup.add(norm);
    this.pickables.push(norm);

    // head cube volume tracks head_dim (baseline 64), derived from the real Wq shape
    const wq = _matrixByName(layer, "Wq");
    const headDim = wq
      ? Number(wq.shape[1]) / Math.max(1, layer.head_count)
      : layer.hidden_size / Math.max(1, layer.head_count);
    const headSize = _clamp(0.18 * Math.sqrt((headDim || 64) / 64), 0.12, 0.3);
    const headCount = Math.min(layer.head_count, 16);
    for (let head = 0; head < headCount; head += 1) {
      const x = ((head / Math.max(1, headCount - 1)) - 0.5) * (width - 0.45);
      const z = -depth * 0.65;
      const headMesh = new THREE.Mesh(
        new THREE.BoxGeometry(headSize, headSize, headSize),
        this._material(COLORS.attention, 1),
      );
      headMesh.position.set(x, 0.21, z);
      headMesh.userData = {
        kind: "head",
        layerIndex: layer.index,
        headIndex: head,
        title: `Layer ${layer.index + 1} head ${head + 1}`,
        detail: attentionType,
      };
      layerGroup.add(headMesh);
      this.pickables.push(headMesh);
      if (head < 8) {
        layerGroup.add(this._label(`H${head + 1}`, x, 0.42, z, 0.085));
      }
    }

    // MLP footprint tracks the expansion ratio (baseline 4x hidden)
    const mlpScale = _clamp(
      Math.sqrt(layer.mlp_size / Math.max(1, 4 * layer.hidden_size)) || 1,
      0.6,
      1.6,
    );
    const mlp = new THREE.Mesh(
      new THREE.BoxGeometry(0.55 * mlpScale, 0.3, 0.72 * mlpScale),
      this._material(COLORS.mlp, 0.94),
    );
    mlp.position.set(width / 2 + 0.55, 0.04, 0.0);
    mlp.userData = {
      kind: "mlp",
      layerIndex: layer.index,
      title: `Layer ${layer.index + 1} MLP`,
      detail: `${layer.hidden_size} -> ${layer.mlp_size} -> ${layer.hidden_size}`,
    };
    layerGroup.add(mlp);
    this.pickables.push(mlp);

    this._addWeightBanks(layerGroup, layer, width, depth);
    this._addLayerFlow(layerGroup, width, depth);

    this.root.add(layerGroup);
  }

  _addWeightBanks(group, layer, width, depth) {
    // box faces scale with sqrt(actual matrix dims), normalized to hidden_size:
    // columns -> length (z), rows -> thickness (x). GQA models show slim Wk/Wv,
    // MLP matrices show their 4x expansion, hybrid layers differ per block.
    const weightLabels = ["Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown"];
    const hidden = Math.max(1, layer.hidden_size);
    const scaled = (value) => Math.sqrt(Math.max(1, Number(value) || hidden) / hidden);
    let z = -1.3;
    for (const label of weightLabels) {
      const matrix = _matrixByName(layer, label);
      if (!matrix) {
        continue;
      }
      const isAttention = ["Wq", "Wk", "Wv", "Wo"].includes(label);
      const thickness = _clamp(0.12 * scaled(matrix.shape[0]), 0.08, 0.3);
      const length = _clamp(0.26 * scaled(matrix.shape[1]), 0.14, 0.8);
      const weight = new THREE.Mesh(
        new THREE.BoxGeometry(thickness, 0.42, length),
        this._material(isAttention ? COLORS.weight : COLORS.mlp, 0.92),
      );
      const zCenter = z + length / 2;
      weight.position.set(-width / 2 - 0.36, 0.05, zCenter);
      weight.userData = {
        kind: "weight",
        layerIndex: layer.index,
        title: `Layer ${layer.index + 1} ${label}`,
        detail: `${matrix.role} · ${_shapeText(matrix.shape)}`,
      };
      group.add(weight);
      this.pickables.push(weight);
      group.add(this._label(`${label}\n${_shapeText(matrix.shape)}`, -width / 2 - 0.82, 0.04, zCenter, 0.1));
      z += length + 0.12;
    }
  }

  _addLayerFlow(group, width, depth) {
    // one numbered path through the block, matching the Forward Flow guide:
    // residual spine -> 1 norm -> 2 attention -> 3 back to residual -> 4 MLP -> 5 back
    const y = 0.3;
    const spine = 1.72;
    const norm = new THREE.Vector3(-width * 0.32, y, 0.95);
    const heads = new THREE.Vector3(0, y, -depth * 0.62);
    const mlp = new THREE.Vector3(width / 2 + 0.5, y, 0.3);
    const mlpOut = new THREE.Vector3(width / 2 + 0.5, y, -0.3);
    const steps = [
      { from: new THREE.Vector3(0, y, spine - 0.1), to: norm, color: COLORS.norm, label: "1 norm" },
      { from: norm.clone().setZ(0.78), to: heads, color: COLORS.attention, label: "2 Q·K·V" },
      { from: heads.clone().setX(0.18), to: new THREE.Vector3(0.18, y, spine - 0.2), color: COLORS.arrow, label: "3 Wo +res" },
      { from: new THREE.Vector3(0.36, y, spine - 0.12), to: mlp, color: COLORS.mlp, label: "4 MLP" },
      { from: mlpOut, to: new THREE.Vector3(0.5, y, spine - 0.22), color: COLORS.mlp, label: "5 +res" },
    ];
    for (const step of steps) {
      group.add(this._flowArrow(step.from, step.to, step.color, step.label));
    }
  }

  _stagePlate(label, y, width, color, vocabScale = 1) {
    const group = new THREE.Group();
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(width * 0.82, 0.08, 1.35 * vocabScale),
      this._material(color, 0.8),
    );
    group.position.y = y;
    group.add(mesh);
    group.add(this._label(label, 0, 0.22, 0, 0.24));
    return group;
  }

  _residualSpine(minY, maxY) {
    const height = maxY - minY;
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.035, height, 12),
      this._material(COLORS.residual, 0.95),
    );
    mesh.position.set(0, minY + height / 2, 1.72);
    return mesh;
  }

  _flowArrow(start, end, color, label = "") {
    const group = new THREE.Group();
    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    if (length <= 0.001) {
      return group;
    }

    const shaftLength = Math.max(0.01, length - 0.2);
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.03, shaftLength, 10),
      this._material(color, 0.9),
    );
    shaft.position.copy(start).addScaledVector(direction, 0.5);
    shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());

    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.1, 0.24, 14),
      this._material(color, 0.95),
    );
    cone.position.copy(end).addScaledVector(direction.clone().normalize(), -0.1);
    cone.quaternion.copy(shaft.quaternion);

    // tagged so selection can dim every other block's arrows
    shaft.userData.isFlow = true;
    cone.userData.isFlow = true;

    group.add(shaft, cone);
    if (label) {
      const midpoint = start.clone().lerp(end, 0.55);
      group.add(this._label(label, midpoint.x + 0.2, midpoint.y + 0.16, midpoint.z, 0.13));
    }
    return group;
  }

  _applySelection() {
    for (const object of this.pickables) {
      const data = object.userData;
      const isSelectedLayer = data.layerIndex === this.selectedLayer;
      const isSelectedHead = data.kind === "head" && data.headIndex === this.selectedHead;
      // subtle bump only: big scale-ups swallow the labels around the layer
      const bump = data.kind === "layer" ? 1.05 : 1.15;
      object.scale.setScalar(isSelectedLayer && (data.kind !== "head" || isSelectedHead) ? bump : 1);
      if (object.material?.emissive) {
        object.material.emissive.setHex(isSelectedLayer ? 0x1d5c56 : 0x000000);
      }
    }

    // labels and flow arrows render through geometry now, so dim the ones on
    // other layers to keep the selected block readable
    for (const group of this.root.children) {
      const layerIndex = group.userData?.layerIndex;
      if (layerIndex === undefined) {
        continue;
      }
      const active = layerIndex === this.selectedLayer;
      group.traverse((object) => {
        if (object.isSprite) {
          object.material.opacity = active ? 1 : 0.22;
        } else if (object.userData?.isFlow && object.material) {
          object.material.opacity = active ? 0.95 : 0.12;
        }
      });
    }
  }

  _renderDetails(architecture, run) {
    if (!architecture) {
      this.detailsElement.innerHTML = "";
      return;
    }

    const layer = architecture.layers[this.selectedLayer] ?? architecture.layers[0];
    const layerRun = run?.layers?.find((item) => item.index === layer.index);
    const headCount = layerRun?.heads?.length ?? 0;
    const picked =
      this.selectedObject && this.selectedObject.layerIndex === layer.index
        ? this.selectedObject
        : {
            title: layer.name,
            detail: layer.components.join(" · "),
          };
    this.detailsElement.innerHTML = `
      <div class="spatial-stat">
        <span>3D Selection</span>
        <strong>${escapeHtml(picked.title)}</strong>
        <em>${escapeHtml(picked.detail)}</em>
      </div>
      <div class="spatial-stat">
        <span>Selected Layer</span>
        <strong>${escapeHtml(layer.name)}</strong>
      </div>
      <div class="spatial-stat">
        <span>Shape</span>
        <strong>${layer.hidden_size}d · ${layer.attention_type} · MLP ${layer.mlp_size}</strong>
      </div>
      <div class="spatial-stat">
        <span>Controls</span>
        <strong>drag ⇄ rotate · drag ⇅ height · wheel zoom · hold Alt for fine steps</strong>
        <em>Sliders below the view give precise rotate/height/zoom. ←/→ rotate, ↑/↓ height, +/− zoom, double-click resets. Selecting a layer rides the camera to it.</em>
      </div>
      <div class="spatial-stat">
        <span>Weights Shown</span>
        <strong>Q/K/V/O + gate/up/down + embeddings + lm_head</strong>
      </div>
      <div class="spatial-stat">
        <span>Visual Scale</span>
        <strong>box size ∝ √(matrix dimensions)</strong>
        <em>length = columns, thickness = rows, normalized to this model's hidden size — e.g. GQA makes Wk/Wv visibly slimmer than Wq, and W_up runs ~2× longer.</em>
      </div>
      <div class="matrix-table">
        <span>Matrix Sizes In This Layer</span>
        ${_matrixRows(layer)}
      </div>
      <div class="flow-guide">
        <span>Forward Flow — numbers match the arrows in the 3D block</span>
        <ol>
          <li><strong>1 norm</strong> — RMSNorm rescales the block input</li>
          <li><strong>2 Q·K·V</strong> — Wq/Wk/Wv project, attention mixes tokens</li>
          <li><strong>3 Wo +res</strong> — output projection adds back into the residual</li>
          <li><strong>4 MLP</strong> — norm, then W_gate/W_up expand the features</li>
          <li><strong>5 +res</strong> — W_down compresses, adds back; the stream rises to the next block, finally lm_head → logits</li>
        </ol>
      </div>
      <div class="spatial-stat">
        <span>Attention Heads</span>
        <strong>${headCount || layer.head_count} ${headCount ? "with live tensors" : "architecture heads"}</strong>
      </div>
      <div class="spatial-legend">
        <span><i style="background:#7bd88f"></i>input flow · residual stream [T × ${architecture.model.hidden_size}]</span>
        <span><i style="background:#f0b94e"></i>attention heads</span>
        <span><i style="background:#9d8cff"></i>attention weights</span>
        <span><i style="background:#f27360"></i>MLP weights</span>
        <span><i style="background:#4fd6c8"></i>flow arrows inside a block</span>
      </div>
    `;
  }

  _pick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.pickables, false)[0];
    if (!hit) {
      return;
    }

    const data = hit.object.userData;
    this.selectedObject = data;
    if (typeof data.layerIndex === "number") {
      this.onSelectLayer(data.layerIndex);
    }
    if (data.kind === "head" && typeof data.headIndex === "number") {
      this.onSelectHead(data.headIndex);
    }
  }

  _setHoverCursor(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.pickables, false)[0];
    this.container.style.cursor = hit ? "pointer" : "grab";
  }

  _positionCamera() {
    // cylindrical polar: camera orbits the vertical axis at the current height,
    // slightly elevated so it looks gently down onto the layer plates
    this.camera.position.set(
      this.radius * Math.sin(this.theta),
      this.height + this.radius * CAMERA_PITCH,
      this.radius * Math.cos(this.theta),
    );
    this.camera.lookAt(0, this.height, 0);
  }

  _clearRoot() {
    while (this.root.children.length) {
      const child = this.root.children.pop();
      child.traverse((object) => {
        object.geometry?.dispose?.();
        if (object.material?.map) {
          object.material.map.dispose();
        }
        object.material?.dispose?.();
      });
    }
  }

  _material(color, opacity = 1) {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: 0.46,
      metalness: 0.12,
      transparent: opacity < 1,
      opacity,
      // translucent parts must not write depth, or they blank out labels and
      // geometry behind them when the camera gets close
      depthWrite: opacity >= 1,
    });
  }

  _label(text, x, y, z, scale, color = COLORS.text) {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 192;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    const lines = String(text).split("\n");
    // shrink the font until the longest line fits, so long labels never clip
    let fontSize = 34;
    context.font = `700 ${fontSize}px Inter, Segoe UI, sans-serif`;
    const widest = Math.max(...lines.map((line) => context.measureText(line).width));
    if (widest > canvas.width - 16) {
      fontSize = Math.max(15, Math.floor((fontSize * (canvas.width - 16)) / widest));
      context.font = `700 ${fontSize}px Inter, Segoe UI, sans-serif`;
    }
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = color;
    const lineHeight = Math.round(fontSize * 1.2);
    lines.forEach((line, index) => {
      context.fillText(
        line,
        canvas.width / 2,
        canvas.height / 2 + (index - (lines.length - 1) / 2) * lineHeight,
      );
    });
    const texture = new THREE.CanvasTexture(canvas);
    // depthTest off + high renderOrder: labels stay readable even when the
    // camera is inside or behind translucent geometry
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(material);
    sprite.renderOrder = 999;
    sprite.position.set(x, y, z);
    sprite.scale.set(scale * 5.2, scale * 1.6, 1);
    return sprite;
  }
}

function _clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function _matrixByName(layer, name) {
  return layer.matrices?.find((matrix) => matrix.name.toLowerCase() === name.toLowerCase());
}

function _shapeText(shape) {
  return shape?.join(" x ") ?? "";
}

function _matrixRows(layer) {
  const matrices = layer.matrices ?? [];
  if (!matrices.length) {
    return `<div class="matrix-row"><strong>No matrix metadata</strong><em>Run with a config-backed model.</em></div>`;
  }

  return matrices
    .map(
      (matrix) => `
        <div class="matrix-row">
          <strong>${escapeHtml(matrix.name)}</strong>
          <code>${escapeHtml(_shapeText(matrix.shape))}</code>
          <em>${escapeHtml(matrix.role)}</em>
        </div>
      `,
    )
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
