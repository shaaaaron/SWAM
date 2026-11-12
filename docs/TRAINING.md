# SWAM Training

Prepare the environments, data, and base weights first. Run all commands below from the repository root. [Installation and resources](SETUP.md)

## 1. Preparation

The examples use four nodes with eight GPUs each. The training platform runs the same commands on every node and supplies `PET_NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. For manual launches, set these three variables for each node before running the commands.

Activate the environment and set the model paths in the current terminal.

```bash
source .venv/main/bin/activate
export DREAMZERO_UMT5_PATH="$PWD/models/umt5-xxl"
export DREAMZERO_WAN22_PATH="$PWD/models/Wan2.2-TI2V-5B"
```

The training launchers determine the repository root automatically and default to `dreamzero/`, `.venv/main`, and `logs/`. Override these paths with `RLINF_ROOT`, `DREAMZERO_PATH`, `VENV_PATH`, and `LOG_ROOT`. The examples use the default GPU list and sharding strategy. Configure machine-specific settings, such as the network interface, for your environment.

## 2. Post-training

Initialize RoboTwin Clean post-training from converted pretraining weights. Replace the weights path with your actual directory.

```bash
unset WARM_START_MODEL_PATH RESUME_DIR

export PET_NNODES=4
export CONFIG_NAME="posttrain"
export EXPERIMENT_NAME=posttrain_example
export MAX_STEPS=20000
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128
export WARM_START_MODEL_PATH=/path/to/converted_pretrain_checkpoint
export AUTO_CONVERT_FINAL_CKPT=true

bash examples/sft/train_dreamzero.sh
```

Training logs and checkpoints are written to `logs/posttrain_example/` by default. After training, the script automatically produces evaluation weights under `logs-eval/posttrain_example/global_step_20000/`.

## 3. Pretraining → Post-training

Run Robotics pretraining, weight conversion, and RoboTwin post-training in one pipeline. Extract the selected SWAM-Pretrain subsets into `../data/robotics/`, following the directory layout on the dataset page. Pretraining starts from the base weights prepared above. Set the experiment name, step count, and batch sizes for each stage.

```bash
unset WARM_START_MODEL_PATH RESUME_DIR

export PET_NNODES=4
export PIPELINE_RUN_ID=pretrain_posttrain_r01

export PRETRAIN_CONFIG_NAME="pretrain"
export PRETRAIN_EXPERIMENT_NAME=pretrain_example
export PRETRAIN_STEPS=20000
export PRETRAIN_MICRO_BATCH_SIZE=8
export PRETRAIN_GLOBAL_BATCH_SIZE=256

export POSTTRAIN_CONFIG_NAME="posttrain"
export POSTTRAIN_EXPERIMENT_NAME=posttrain_from_pretrain
export POSTTRAIN_STEPS=20000
export POSTTRAIN_MICRO_BATCH_SIZE=4
export POSTTRAIN_GLOBAL_BATCH_SIZE=128

bash examples/sft/train_dreamzero_pretrain_posttrain.sh
```

Pretraining and post-training outputs are written to `logs/pretrain_example/` and `logs/posttrain_from_pretrain/`, respectively. The script automatically converts the pretraining weights for post-training, then produces evaluation weights under `logs-eval/posttrain_from_pretrain/global_step_20000/` after post-training.

See the guide for inference and evaluation. [Inference and evaluation](EVALUATION.md)
