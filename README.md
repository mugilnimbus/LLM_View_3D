# LLM View 3D

**Explore Large Language Models in an interactive 3D environment.**

LLM View 3D is a web application that transforms Hugging Face language models into interactive 3D visualizations. Enter or select a model from Hugging Face and explore its architecture, layers, components, and structure through an intuitive visual interface.

The project is designed to make complex LLM architectures easier to understand, explore, and compare for developers, researchers, students, and AI enthusiasts.

## Key Features

* Load and visualize LLMs from Hugging Face
* Interactive 3D model exploration
* Inspect model architecture and individual layers
* View matrix sizes, tensor flow, attention blocks, MLP blocks, and residual paths
* Load local Hugging Face model clones from the `models/` folder
* Read architecture metadata from local GGUF models without loading weights
* Explore models directly in the browser
* Built for AI education, research, and model analysis

**Making large language models easier to see, explore, and understand.**

## Requirements

* Windows PowerShell
* Python 3.12 or newer
* [`uv`](https://docs.astral.sh/uv/)
* Optional: PyTorch and Transformers for real Hugging Face model runs

## Installation and Setup

Clone the repository:

```powershell
git clone <your-repo-url>
cd LLM_View_3D
```

Install the base dependencies:

```powershell
uv sync
```

Start the app:

```powershell
.\scripts\start.ps1
```

Open the app:

```text
http://127.0.0.1:8999
```

Stop the app:

```powershell
.\scripts\stop.ps1
```

Check server status:

```powershell
.\scripts\status.ps1
```

Restart the app:

```powershell
.\scripts\restart.ps1
```

## Optional Real Model Support

The base app works with demo visualization data. To load and run real Hugging Face models, install the optional ML dependencies:

```powershell
uv sync --extra ml
.\scripts\restart.ps1
```

## How to Use

### Use the demo model

1. Start the app.
2. Keep the mode set to **Demo**.
3. Explore the 3D architecture, input flow, attention view, and output view.

### Load a model from Hugging Face

1. Switch to **Real** mode.
2. Enter a Hugging Face model id, such as:

```text
Qwen/Qwen3-0.6B
```

3. Click **Load**.
4. Explore the architecture in 3D.

### Use a local model

Clone Hugging Face models into the `models/` folder:

```powershell
cd models
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B
cd ..
```

Then:

1. Click **Refresh** in the app.
2. Select the local model.
3. Switch to **Real** mode.
4. Press **Run** to execute a prompt if model weights and ML dependencies are available.

The `models/` folder is ignored by Git so local model files and weights are not pushed to GitHub.

### Use a local GGUF model

Put a `.gguf` file inside its own folder under `models/`, then click **Refresh** in the app.
LLM View reads the GGUF header metadata to visualize architecture, layer counts, head counts, hidden
size, MLP size, and vocabulary size. GGUF execution is not wired up yet, so prompt runs for those
entries use demo tensors.

## 3D Controls

* Drag left or right to rotate
* Drag up or down to move through layers
* Use the mouse wheel to zoom
* Use the sliders below the 3D view for precise control
* Click layers, heads, and matrix blocks to inspect them
* Double-click the 3D view to reset the camera

## Development

Run lint checks:

```powershell
uv run ruff check .
```

Build package artifacts:

```powershell
uv build
```
