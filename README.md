# VAMOS 

This is the inference and deployment repo for [VAMOS](https://vamos-vla.github.io/).

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vamos-vla/vamos/blob/main/vamos_demo.ipynb)

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:vamos-vla/vamos.git
   cd vamos
   ```

2. **Set up the environment with uv**:

This repository is configured as a uv project. Use `uv sync` instead of
`uv pip install`.

For a minimal Python/ML installation:

   ```bash
   uv sync
   ```

If you also want the notebook tools:

   ```bash
   uv sync --group dev
   ```

Run commands through the uv environment:

   ```bash
   uv run python vamos_demo.py
   uv run jupyter notebook vamos_demo.ipynb
   ```

The base install contains only what VAMOS inference and the Zenoh nodes need.
Optional groups add the rest:

| Group | Adds | Used by |
| --- | --- | --- |
| `zenoh` | `eclipse-zenoh`, `json5` | everything under `zenoh/` |
| `dev` | Jupyter | `vamos_demo.ipynb` |
| `server` | FastAPI, Uvicorn | `server/vlm_server.py` |
| `ros` | catkin tools, ONNX Runtime, optimum, local `trajectory_*` packages | `vamos_ws` ROS nodes |

The local `trajectory_prediction` and `trajectory_projection` packages are
installed as editable path dependencies by `uv sync --group ros`.

### Jetson AGX Orin

The lock file selects platform-specific Torch builds: Windows AMD64 uses the
CUDA 12.8 index, Linux aarch64 (Jetson AGX Orin, JetPack 7.2 / L4T r39.2) uses
the CUDA 13.2 index, and other Linux hosts use PyPI. On the Jetson:

   ```bash
   uv sync --group zenoh
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
   ```

The second command should print `True (8, 7)`. VAMOS loads in bfloat16, which
Orin (Ampere) supports. Model loading and inference latency on the Jetson have
not been measured yet; see [zenoh/README.md](zenoh/README.md) for the
deployment checklist.

### Older NVIDIA GPUs (Pascal, e.g. P100)

The default Linux x86_64 Torch wheels from PyPI are built for CUDA 13, which no
longer supports Pascal (`sm_60`/`sm_61`). On such hosts opt in to the CUDA 12.6
build with the `cu126` extra, and pass it to every `uv run` as well (otherwise
uv re-syncs to the default build):

   ```bash
   uv sync --group zenoh --extra cu126
   uv run --extra cu126 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
   ```

The arch list must include `sm_60`. Pascal GPUs have no bfloat16, so keep
`node.inference.dtype: "auto"` (it selects float32 there; about 12 GB of
weights, which fits a 16 GB P100). The `pypi-torch` extra only exists to name
the default build in uv's conflict declaration and never needs to be passed.

### Conda Alternative

If you prefer the original Conda flow:

For a minimal installation, either run:
   ```bash
   conda env create -f environment.yml
   conda activate vamos
   ```

Or directly install the necessary packages with pip or your favorite package manager:
```bash
pip3 install torch torchvision transformers peft matplotlib Pillow
```

If you want to test with ROS (Noetic), run the following instead:
   ```bash
   conda env create -f environment_ros.yml
   conda activate vamos
   ```

We use ROS noetic from RoboStack for better portability.

### Quick Test

Test the VLM model with sample images:

```bash
uv run python vamos_demo.py
```

Or use the Jupyter notebook for interactive testing:

```bash
uv run jupyter notebook vamos_demo.ipynb
```

You can access the same notebook on Colab directly from the url at the top of this README.

## ROS Package

VAMOS includes a ROS package. For detailed usage and setup instructions, see [ROS.md](ROS.md).
