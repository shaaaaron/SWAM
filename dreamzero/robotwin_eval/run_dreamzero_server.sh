#!/usr/bin/env bash
set -euo pipefail

# Launch DreamZero policy server for RoboTwin evaluation.
#
# Usage:
#   bash run_dreamzero_server.sh <ckpt_path> [gpu_ids] [port] [ngpu]
#
# Examples:
#   bash run_dreamzero_server.sh /path/to/checkpoint 0,1 8000 2
#   bash run_dreamzero_server.sh /path/to/checkpoint 0 8000 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_dreamzero_server.sh <ckpt_path> [gpu_ids] [port] [ngpu]" >&2
    exit 1
fi

ckpt_path="$1"
gpu_ids="${2:-${DREAMZERO_SERVER_GPUS:-0,1}}"
port="${3:-${DREAMZERO_SERVER_PORT:-8000}}"
ngpu="${4:-${DREAMZERO_NGPU:-2}}"
dreamzero_python="${DREAMZERO_PYTHON:-python}"
# Deploy policy YAML: default v2 (deploy_policy.yml); override to v1 via env var
# DEPLOY_POLICY_TEMPLATE_PATH (e.g. deploy_policy_robotwin_320x256.yml) to switch embodiment.
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

echo "[INFO] Starting DreamZero RoboTwin policy server"
echo "[INFO] checkpoint: ${ckpt_path}"
echo "[INFO] GPUs: ${gpu_ids} (nproc=${ngpu})"
echo "[INFO] port: ${port}"
echo "[INFO] deploy_policy_template: ${deploy_policy_template}"

cd "${DREAMZERO_ROOT}"

export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"

torchrun_args=(--nproc_per_node="${ngpu}")
torchrun_args+=(--standalone)
echo "[INFO] torchrun rendezvous: standalone auto free port"

server_args=(
    --port "${port}"
    --model-path "${ckpt_path}"
    --config "${deploy_policy_template}"
)
if [[ -n "${ACTION_TRAIN_SIGMA_SHIFT:-}" ]]; then
    server_args+=(--action-train-sigma-shift "${ACTION_TRAIN_SIGMA_SHIFT}")
fi
if [[ -n "${ACTION_INFERENCE_SIGMA_SHIFT:-}" ]]; then
    server_args+=(--action-inference-sigma-shift "${ACTION_INFERENCE_SIGMA_SHIFT}")
fi
if [[ -n "${DREAMZERO_EVAL_INDEX:-}" ]]; then
    server_args+=(--index "${DREAMZERO_EVAL_INDEX}")
fi
if [[ -n "${DREAMZERO_VIDEO_OUTPUT_ROOT:-}" ]]; then
    server_args+=(--video-output-root "${DREAMZERO_VIDEO_OUTPUT_ROOT}")
    echo "[INFO] video output root: ${DREAMZERO_VIDEO_OUTPUT_ROOT}"
fi

CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    "$dreamzero_python" -m torch.distributed.run "${torchrun_args[@]}" \
    "${SCRIPT_DIR}/dreamzero_robotwin_server.py" \
    "${server_args[@]}"
