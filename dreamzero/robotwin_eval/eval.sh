#!/usr/bin/env bash
set -euo pipefail

# Evaluate a single RoboTwin task using the DreamZero policy server.
#
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [port] [host]
#
# This script:
#   1. Sets PYTHONPATH to include dreamzero, RoboTwin, and robotwin_eval
#   2. Patches deploy_policy.yml with the correct host/port
#   3. Calls RoboTwin's eval_policy.py

if [[ $# -lt 6 ]]; then
    echo "Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [port] [host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RELEASE_ROOT="${RLINF_RELEASE_ROOT:-$(cd "${DREAMZERO_ROOT}/.." && pwd)}"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-${RELEASE_ROOT}/RoboTwin}"
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
    echo "ROBOTWIN_PATH does not exist: ${ROBOTWIN_PATH}" >&2
    exit 1
fi

robotwin_eval_script="${ROBOTWIN_PATH}/script/eval_policy.py"
if [[ ! -f "${robotwin_eval_script}" ]]; then
    echo "RoboTwin eval entry does not exist: ${robotwin_eval_script}" >&2
    exit 1
fi

policy_name="${ROBOTWIN_POLICY_NAME:-dreamzero2robotwin_interface}"
task_name="$1"
task_config="$2"
case "${task_config}" in
    demo_clean)      instruction_type=seen ;;
    demo_randomized) instruction_type=unseen ;;
    *)
        echo "Unsupported task_config: ${task_config}" >&2
        exit 1
        ;;
esac

ckpt_setting="${3:-dreamzero}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${DREAMZERO_SERVER_PORT:-8000}}"
policy_host="${8:-${DREAMZERO_SERVER_HOST:-127.0.0.1}}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

# Create runtime config with substituted host/port
runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/dreamzero_deploy_policy.XXXXXX.yml")"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

sed \
    -e "s/^host:.*/host: \"${policy_host}\"/" \
    -e "s/^port:.*/port: ${policy_port}/" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Set PYTHONPATH: RoboTwin + dreamzero + robotwin_eval (for the interface module)
export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

cd "${ROBOTWIN_PATH}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"

    # --policy_ckpt_path "${policy_ckpt_path}" \
PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" script/eval_policy.py --config "${runtime_deploy_policy}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --instruction_type "${instruction_type}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
