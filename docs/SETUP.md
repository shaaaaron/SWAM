# SWAM Installation and Resources

This repository integrates RLinf for training, DreamZero for model implementation and inference, DMuon for optimization, and RoboTwin for simulation. Run all commands below from the repository root.

## 1. Training environment

Create the `main` environment at `.venv/main` for RLinf, DreamZero, and DMuon.

Use Linux x86_64 with uv, an NVIDIA driver compatible with CUDA 12.4, and system FFmpeg shared libraries with AV1 decoding support.

```bash
uv venv --python 3.11.14 --seed .venv/main
source .venv/main/bin/activate
pip install -r requirements/main.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`requirements/main.txt` installs the local RLinf and DMuon packages and the dependencies required by DreamZero.

## 2. RoboTwin environment and assets

Use a separate environment named `robotwin` for task execution, rendering, and robot control.

Follow the official RoboTwin 2.0 installation and asset download instructions for the bundled code. Place the assets under `RoboTwin/assets/`. An existing compatible environment can be reused. [Official installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

For inference and evaluation, set `ROBOTWIN_PYTHON` to the Python executable in this environment.

## 3. Data and weights

### Datasets

Download the pretraining and post-training data from Hugging Face and place them in a `data/` directory alongside the training repository. [Hugging Face dataset](https://huggingface.co/datasets/FreDERicia/SWAM-Pretrain)

```text
workspace/
├── SWAM/                 # Training repository
└── data/
    ├── <pretraining-data>/
    └── <post-training-data>/
```

### Base model weights

| Resource | Source | Suggested location |
|---|---|---|
| UMT5 tokenizer | [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) | `models/umt5-xxl/` |
| Wan2.2-TI2V-5B | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `models/Wan2.2-TI2V-5B/` |

The base weights initialize training. After downloading them, set their paths in the terminal used to launch training.

```bash
export DREAMZERO_UMT5_PATH="$PWD/models/umt5-xxl"
export DREAMZERO_WAN22_PATH="$PWD/models/Wan2.2-TI2V-5B"
```

### Metadata

The repository provides dataset statistics in `.assets/`. Refer to the selected training config for the metadata it uses.

Continue with the training or inference and evaluation guide. [Training](TRAINING.md) · [Inference and evaluation](EVALUATION.md)
