# SWAM Inference and Evaluation

The bundled simulator uses RoboTwin 2.0, based on upstream commit `c3ddfa8`. [Upstream version](https://github.com/RoboTwin-Platform/RoboTwin/commit/c3ddfa8b97d5519efa828b075999bd0006778e5e)

It adds DreamZero evaluation support, fixed evaluation cases and instruction checks, and result and multi-view video output. Upstream fixes keep object selection consistent for the same random seed, initialize task state correctly, and prevent clutter placement from drifting across repeated scene setups.

The model server uses the main training environment, and the simulator client uses the RoboTwin environment. Prepare the model weights and simulator assets first. Run all commands below from the repository root. [Installation and resources](SETUP.md)

## 1. Convert a checkpoint

After training, activate the main training environment and convert a complete RLinf checkpoint to the DreamZero evaluation format.

```bash
source .venv/main/bin/activate
python dreamzero/tools/convert_rlinf_dcp_to_dreamzero_eval.py \
  --output-mode eval \
  --rlinf-checkpoint /path/to/global_step_1000 \
  --output-dir /path/to/eval_checkpoint
```

Specify your checkpoint path and a new output directory. Conversion uses the model configuration and metadata exported during training to generate `config.json`, model weights, and `experiment_cfg/`.

If you already have a complete evaluation checkpoint, proceed to the next step.

## 2. Run a RoboTwin task

This example runs `stack_blocks_two` in Clean mode. The model server and simulator client share GPU 0.

```bash
export DREAMZERO_PYTHON="$PWD/.venv/main/bin/python"
export ROBOTWIN_PYTHON=/path/to/robotwin/bin/python
export ROBOTWIN_RESULT_ROOT="$PWD/results/my_eval"
export DEPLOY_POLICY_TEMPLATE_PATH="$PWD/dreamzero/robotwin_eval/deploy_policy_robotwin_320x256.yml"

bash dreamzero/robotwin_eval/start_eval.sh \
  -m demo_clean -n my_eval -c /path/to/eval_checkpoint \
  -g 0 --ngpu 1 --eval-gpu 0 -p 8000 \
  stack_blocks_two
```

Set `DREAMZERO_PYTHON` to the Python executable in `main` for the model server, and `ROBOTWIN_PYTHON` to the executable in `robotwin` for the simulator client. Pass the converted evaluation checkpoint with `-c`.

`start_eval.sh` starts the DreamZero server, then runs the task through the RoboTwin client. Explicit Python paths take precedence over Conda environment activation.

The example uses `deploy_policy_robotwin_320x256.yml`, which requires a checkpoint with 320×256 image inputs and 16-dimensional end-effector pose actions. Use a matching deployment template for other models.

## 3. Modes and results

Use `demo_clean` for Clean evaluation or `demo_randomized` for randomized evaluation. The script selects the corresponding language instruction type.

Evaluation results are saved under `ROBOTWIN_RESULT_ROOT`. The launcher prints the log location.

## 4. Evaluate all 50 tasks

| Setup | Script |
|---|---|
| One node, eight GPUs | `dreamzero/robotwin_eval/run_block_full50.sh` |
| Four nodes | `dreamzero/robotwin_eval/run_block_full50_4n.sh` |

For one node with eight GPUs, fill in the RoboTwin Python and checkpoint paths, then run the following commands.

```bash
export DREAMZERO_PYTHON="$PWD/.venv/main/bin/python"
export ROBOTWIN_PYTHON=/path/to/robotwin/bin/python
export CKPT=/path/to/eval_checkpoint
export EXP_ID=swam_full50
export RUN_TAG=20261111_r01
export EXPECTED_HOST="$(hostname)"
export RESULT_ROOT="$PWD/results/full50"
export LOG_BASE="$PWD/logs/full50/$RUN_TAG"
export ROBOTWIN_TEST_NUM=50

for gpu in {0..7}; do
  bash dreamzero/robotwin_eval/run_block_full50.sh "$gpu" run &
done
wait
```

GPUs 0–2 run Clean evaluation, and GPUs 3–7 run randomized evaluation. Each mode covers all 50 tasks. This example runs 50 episodes per task per mode, for 5,000 episodes in total. The script saves logs and results under the directories above. Use a new `RUN_TAG` when rerunning evaluation.

Validation so far covers RoboTwin scene rendering and robot control. The commands have been checked against the launch scripts. Full model policy evaluation remains to be validated.
