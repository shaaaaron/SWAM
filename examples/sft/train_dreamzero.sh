#!/bin/bash
set -euo pipefail

export RLINF_ROOT="${RLINF_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DREAMZERO_PATH="${DREAMZERO_PATH:-${RLINF_ROOT}/dreamzero}"
export VENV_PATH="${VENV_PATH:-${RLINF_ROOT}/.venv/main}"
export LOG_ROOT="${LOG_ROOT:-${RLINF_ROOT}/logs}"

# 每次启动使用独立缓存，避免删除同机其他任务的缓存。
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/rlinf-inductor-$$}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/rlinf-triton-$$}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Qizhi runs this launcher once on every node. Ray owns the per-GPU workers, so
# only node rank 0 starts the RLinf training driver.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-net0}"

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "[ERROR] Required environment variable ${name} is not set."
        exit 1
    fi
}

require_dir() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        echo "[ERROR] Required directory is missing: ${path}"
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [ ! -s "${path}" ]; then
        echo "[ERROR] Required file is missing or empty: ${path}"
        exit 1
    fi
}

if [ -n "${WARM_START_MODEL_PATH:-}" ] && [ -n "${RESUME_DIR:-}" ]; then
    echo "[ERROR] WARM_START_MODEL_PATH and RESUME_DIR cannot be used together."
    exit 1
fi

require_env PET_NNODES
require_env PET_NODE_RANK
require_env MASTER_ADDR
require_env MASTER_PORT

case "${PET_NNODES}" in
    ''|*[!0-9]*) echo "[ERROR] PET_NNODES must be a positive integer: ${PET_NNODES}"; exit 1 ;;
esac
case "${PET_NODE_RANK}" in
    ''|*[!0-9]*) echo "[ERROR] PET_NODE_RANK must be a non-negative integer: ${PET_NODE_RANK}"; exit 1 ;;
esac
case "${MASTER_PORT}" in
    ''|*[!0-9]*) echo "[ERROR] MASTER_PORT must be an integer: ${MASTER_PORT}"; exit 1 ;;
esac
if [ "${PET_NNODES}" -lt 1 ] || [ "${PET_NODE_RANK}" -ge "${PET_NNODES}" ]; then
    echo "[ERROR] Invalid node topology: PET_NNODES=${PET_NNODES}, PET_NODE_RANK=${PET_NODE_RANK}"
    exit 1
fi

if [ ! -d "${RLINF_ROOT}" ]; then
    echo "[ERROR] RLINF_ROOT not found: ${RLINF_ROOT}"
    exit 1
fi
if [ ! -d "${DREAMZERO_PATH}" ]; then
    echo "[ERROR] DREAMZERO_PATH not found: ${DREAMZERO_PATH}"
    exit 1
fi
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found: ${VENV_PATH}"
    exit 1
fi

# Relative paths in the training configs are resolved from the RLinf root.
cd "${RLINF_ROOT}"

# Ray workers inherit the environment present when ray start is executed.
# shellcheck disable=SC1091
. "${VENV_PATH}/bin/activate"
if [ -d "${VENV_PATH}/lib/ffmpeg" ]; then
    export LD_LIBRARY_PATH="${VENV_PATH}/lib/ffmpeg:${LD_LIBRARY_PATH:-}"
fi

export REPO_PATH="${REPO_PATH:-${RLINF_ROOT}}"
export DREAMZERO_PATH
export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-${VENV_PATH}/bin/python}"
export PYTHONPATH="${DREAMZERO_PATH}:${REPO_PATH}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export AV_LOG_FORCE_NOCOLOR="${AV_LOG_FORCE_NOCOLOR:-1}"
export LIBAV_LOG_LEVEL="${LIBAV_LOG_LEVEL:-quiet}"
export AV_LOG_LEVEL="${AV_LOG_LEVEL:-quiet}"

if [ ! -x "${PYTHON_EXECUTABLE}" ]; then
    echo "[ERROR] PYTHON_EXECUTABLE is not executable: ${PYTHON_EXECUTABLE}"
    exit 1
fi
if ! command -v ray >/dev/null 2>&1; then
    echo "[ERROR] ray is not available after activating ${VENV_PATH}"
    exit 1
fi

NGPU_PER_NODE="${NGPU_PER_NODE:-${PET_NPROC_PER_NODE:-8}}"
case "${NGPU_PER_NODE}" in
    ''|*[!0-9]*) echo "[ERROR] NGPU_PER_NODE must be a positive integer: ${NGPU_PER_NODE}"; exit 1 ;;
esac
if [ "${NGPU_PER_NODE}" -lt 1 ]; then
    echo "[ERROR] NGPU_PER_NODE must be positive: ${NGPU_PER_NODE}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [ "${#visible_gpus[@]}" -ne "${NGPU_PER_NODE}" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES exposes ${#visible_gpus[@]} GPUs, expected NGPU_PER_NODE=${NGPU_PER_NODE}."
    exit 1
fi

export RLINF_NODE_RANK="${PET_NODE_RANK}"
export PET_NPROC_PER_NODE="${NGPU_PER_NODE}"
export LOCAL_WORLD_SIZE="${NGPU_PER_NODE}"
export NGPU="${NGPU_PER_NODE}"

CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME must be set}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set and identical on every node}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
DATALOADER_PROFILE="${DATALOADER_PROFILE:-false}"
WRITE_RUN_LOG="${WRITE_RUN_LOG:-true}"
AUTO_CONVERT_FINAL_CKPT="${AUTO_CONVERT_FINAL_CKPT:-true}"
WARM_START_MODEL_PATH="${WARM_START_MODEL_PATH:-}"
DREAMZERO_UMT5_PATH="${DREAMZERO_UMT5_PATH:-}"
DREAMZERO_WAN22_PATH="${DREAMZERO_WAN22_PATH:-}"
DREAMZERO_WAN21_IMAGE_ENCODER_PATH="${DREAMZERO_WAN21_IMAGE_ENCODER_PATH:-}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-hybrid_shard}"
MAX_WAIT="${MAX_WAIT:-600}"
HEAD_FAILURE_LIMIT="${HEAD_FAILURE_LIMIT:-12}"

case "${AUTO_CONVERT_FINAL_CKPT}" in
    true|false) ;;
    *) echo "[ERROR] AUTO_CONVERT_FINAL_CKPT must be true or false."; exit 1 ;;
esac

case "${HEAD_FAILURE_LIMIT}" in
    ''|*[!0-9]*) echo "[ERROR] HEAD_FAILURE_LIMIT must be a positive integer: ${HEAD_FAILURE_LIMIT}"; exit 1 ;;
esac
if [ "${HEAD_FAILURE_LIMIT}" -lt 1 ]; then
    echo "[ERROR] HEAD_FAILURE_LIMIT must be positive: ${HEAD_FAILURE_LIMIT}"
    exit 1
fi

WORLD_SIZE_TOTAL=$((PET_NNODES * NGPU_PER_NODE))
if [ "$((GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * WORLD_SIZE_TOTAL)))" -ne 0 ]; then
    echo "[ERROR] GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * WORLD_SIZE_TOTAL."
    echo "[ERROR] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}, WORLD_SIZE_TOTAL=${WORLD_SIZE_TOTAL}"
    echo "[ERROR] For 2 nodes x 8 GPUs, use micro=8/global=128 to preserve the old global batch,"
    echo "[ERROR] or micro=16/global=256 to preserve the old per-GPU micro batch."
    exit 1
fi

if [ -n "${WARM_START_MODEL_PATH}" ]; then
    if [ ! -s "${WARM_START_MODEL_PATH}/config.json" ] || [ ! -s "${WARM_START_MODEL_PATH}/model.safetensors.index.json" ]; then
        echo "[ERROR] WARM_START_MODEL_PATH is not a converted HF weights directory: ${WARM_START_MODEL_PATH}"
        exit 1
    fi
    shopt -s nullglob
    warm_start_shards=("${WARM_START_MODEL_PATH}"/model-*.safetensors)
    shopt -u nullglob
    if [ "${#warm_start_shards[@]}" -eq 0 ]; then
        echo "[ERROR] WARM_START_MODEL_PATH has no model-*.safetensors shards: ${WARM_START_MODEL_PATH}"
        exit 1
    fi
fi

if [ -n "${DREAMZERO_UMT5_PATH}" ]; then
    require_dir "${DREAMZERO_UMT5_PATH}"
fi
if [ -n "${DREAMZERO_WAN22_PATH}" ]; then
    require_dir "${DREAMZERO_WAN22_PATH}"
    require_file "${DREAMZERO_WAN22_PATH}/models_t5_umt5-xxl-enc-bf16.pth"
    require_file "${DREAMZERO_WAN22_PATH}/Wan2.2_VAE.pth"
fi
if [ -n "${DREAMZERO_WAN21_IMAGE_ENCODER_PATH}" ]; then
    if [ ! -d "${DREAMZERO_WAN21_IMAGE_ENCODER_PATH}" ]; then
        require_file "${DREAMZERO_WAN21_IMAGE_ENCODER_PATH}"
    fi
fi

RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/${EXPERIMENT_NAME}/logs}"
RUN_LOG_FILE="${RUN_LOG_FILE:-${RUN_LOG_DIR}/train_multi_node_$(date +%Y%m%d-%H%M%S).log}"
MULTI_NODE_RUN_ID="${MULTI_NODE_RUN_ID:-${EXPERIMENT_NAME}}"
case "${MULTI_NODE_RUN_ID}" in
    ''|*[!A-Za-z0-9._-]*)
        echo "[ERROR] MULTI_NODE_RUN_ID may contain only letters, digits, dot, underscore, and dash."
        exit 1
        ;;
esac
RUN_READY_FILE="${RUN_READY_FILE:-${RUN_LOG_DIR}/multi_node_${MULTI_NODE_RUN_ID}.ready}"
RUN_STATUS_FILE="${RUN_STATUS_FILE:-${RUN_LOG_DIR}/multi_node_${MULTI_NODE_RUN_ID}.status}"
mkdir -p "${RUN_LOG_DIR}"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    ray stop --force >/dev/null 2>&1 || true
    if [ "${PET_NODE_RANK}" -eq 0 ]; then
        local status_tmp="${RUN_STATUS_FILE}.tmp.$$"
        printf '%s %s\n' "${MULTI_NODE_RUN_ID}" "${status}" > "${status_tmp}" || true
        mv -f "${status_tmp}" "${RUN_STATUS_FILE}" || true
    fi
    exit "${status}"
}
trap 'exit 130' INT TERM
trap cleanup EXIT

if [ "${PET_NODE_RANK}" -eq 0 ]; then
    rm -f "${RUN_READY_FILE}" "${RUN_STATUS_FILE}"
fi
ray stop --force >/dev/null 2>&1 || true

echo "[INFO] host=$(hostname) node_rank=${PET_NODE_RANK}/${PET_NNODES}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[INFO] NGPU_PER_NODE=${NGPU_PER_NODE} WORLD_SIZE_TOTAL=${WORLD_SIZE_TOTAL}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] CONFIG_NAME=${CONFIG_NAME} EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[INFO] FSDP_SHARDING_STRATEGY=${FSDP_SHARDING_STRATEGY}"

if [ "${PET_NODE_RANK}" -eq 0 ]; then
    echo "[INFO] Starting Ray head."
    ray start --head --node-ip-address="${MASTER_ADDR}" --port="${MASTER_PORT}" --num-gpus="${NGPU_PER_NODE}" --dashboard-host=0.0.0.0
    ready_tmp="${RUN_READY_FILE}.tmp.$$"
    printf '%s %s %s\n' "${MULTI_NODE_RUN_ID}" "${MASTER_ADDR}" "${MASTER_PORT}" > "${ready_tmp}"
    mv -f "${ready_tmp}" "${RUN_READY_FILE}"
else
    echo "[INFO] Waiting for rank 0 readiness marker: ${RUN_READY_FILE}"
    ready_waited=0
    while true; do
        if [ -s "${RUN_READY_FILE}" ]; then
            read -r ready_run_id ready_addr ready_port < "${RUN_READY_FILE}" || true
            if [ "${ready_run_id:-}" = "${MULTI_NODE_RUN_ID}" ] && \
               [ "${ready_addr:-}" = "${MASTER_ADDR}" ] && \
               [ "${ready_port:-}" = "${MASTER_PORT}" ]; then
                break
            fi
        fi
        if [ -s "${RUN_STATUS_FILE}" ]; then
            read -r status_run_id early_status < "${RUN_STATUS_FILE}" || true
            if [ "${status_run_id:-}" = "${MULTI_NODE_RUN_ID}" ]; then
                echo "[ERROR] Rank 0 exited before publishing readiness; status=${early_status:-1}."
                exit "${early_status:-1}"
            fi
        fi
        if [ "${ready_waited}" -ge "${MAX_WAIT}" ]; then
            echo "[ERROR] Timed out waiting for rank 0 readiness marker."
            exit 1
        fi
        sleep 1
        ready_waited=$((ready_waited + 1))
    done

    echo "[INFO] Waiting for Ray head at ${MASTER_ADDR}:${MASTER_PORT}."
    "${PYTHON_EXECUTABLE}" - "${MASTER_ADDR}" "${MASTER_PORT}" "${MAX_WAIT}" <<'PY'
import socket
import sys
import time

host, port, max_wait = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for _ in range(max_wait):
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"Ray head connection timed out: {host}:{port}")
PY
    echo "[INFO] Joining Ray cluster."
    ray start --address="${MASTER_ADDR}:${MASTER_PORT}" --num-gpus="${NGPU_PER_NODE}"
    echo "[INFO] Ray worker joined; waiting for ${RUN_STATUS_FILE}"
    head_failures=0
    while true; do
        if [ -s "${RUN_STATUS_FILE}" ]; then
            read -r status_run_id train_status < "${RUN_STATUS_FILE}" || true
            if [ "${status_run_id:-}" = "${MULTI_NODE_RUN_ID}" ]; then
                break
            fi
        fi
        if "${PYTHON_EXECUTABLE}" - "${MASTER_ADDR}" "${MASTER_PORT}" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2):
        pass
except OSError:
    raise SystemExit(1)
PY
        then
            head_failures=0
        else
            head_failures=$((head_failures + 1))
            if [ "${head_failures}" -ge "${HEAD_FAILURE_LIMIT}" ]; then
                echo "[ERROR] Ray head is unreachable and no completion status was published."
                exit 1
            fi
        fi
        sleep 10
    done
    case "${train_status:-}" in
        ''|*[!0-9]*)
            echo "[ERROR] Invalid rank 0 status marker: ${RUN_STATUS_FILE}"
            exit 1
            ;;
    esac
    echo "[INFO] Rank 0 completion status detected: ${train_status}."
    exit "${train_status}"
fi

echo "[INFO] Waiting for all ${PET_NNODES} Ray nodes."
"${PYTHON_EXECUTABLE}" - "${PET_NNODES}" "${MAX_WAIT}" <<'PY'
import sys
import time

import ray

expected, max_wait = int(sys.argv[1]), int(sys.argv[2])
ray.init(address="auto", namespace="RLinf-launcher", ignore_reinit_error=True)
for _ in range(max_wait):
    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    if alive >= expected:
        print(f"[INFO] Ray cluster ready: {alive}/{expected} nodes alive.")
        break
    time.sleep(1)
else:
    raise SystemExit(f"Ray cluster timed out: {alive}/{expected} nodes alive")
ray.shutdown()
PY

echo "[INFO] MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "[INFO] GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * WORLD_SIZE_TOTAL)))"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"

# Only checkpoints absent before this training invocation are eligible for conversion.
declare -A checkpoints_before=()
checkpoint_root="${LOG_ROOT}/${EXPERIMENT_NAME}/checkpoints"
if [ "${AUTO_CONVERT_FINAL_CKPT}" = "true" ]; then
    for checkpoint in "${checkpoint_root}"/global_step_*; do
        if [ -d "${checkpoint}" ]; then
            checkpoints_before["${checkpoint##*/}"]=1
        fi
    done
fi

TRAIN_CMD=(
    "${PYTHON_EXECUTABLE}" examples/sft/train_vla_sft.py
    --config-name "${CONFIG_NAME}"
    "cluster.num_nodes=${PET_NNODES}"
    "cluster.component_placement.actor=all"
    "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
    "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "actor.fsdp_config.gradient_checkpointing=true"
    "data.dataloader_profile=${DATALOADER_PROFILE}"
    "runner.logger.log_path=${LOG_ROOT}"
    "runner.logger.experiment_name=${EXPERIMENT_NAME}"
)
[ -n "${MAX_STEPS:-}" ] && TRAIN_CMD+=("runner.max_steps=${MAX_STEPS}")
[ -n "${SAVE_INTERVAL:-}" ] && TRAIN_CMD+=("runner.save_interval=${SAVE_INTERVAL}")
[ -n "${SAVE_START_STEP:-}" ] && TRAIN_CMD+=("++runner.save_start_step=${SAVE_START_STEP}")
[ -n "${TRAIN_DATA_PATHS:-}" ] && TRAIN_CMD+=("data.train_data_paths=${TRAIN_DATA_PATHS}")
[ -n "${DREAMZERO_METADATA_PATH:-}" ] && TRAIN_CMD+=("actor.model.metadata_json_path=${DREAMZERO_METADATA_PATH}")
[ -n "${DREAMZERO_UMT5_PATH}" ] && TRAIN_CMD+=("actor.model.tokenizer_path=${DREAMZERO_UMT5_PATH}")
if [ -n "${DREAMZERO_WAN22_PATH}" ]; then
    TRAIN_CMD+=("actor.model.diffusion_model_pretrained_path=${DREAMZERO_WAN22_PATH}")
    TRAIN_CMD+=("actor.model.text_encoder_pretrained_path=${DREAMZERO_WAN22_PATH}/models_t5_umt5-xxl-enc-bf16.pth")
    TRAIN_CMD+=("actor.model.vae_pretrained_path=${DREAMZERO_WAN22_PATH}/Wan2.2_VAE.pth")
fi
[ -n "${DREAMZERO_WAN21_IMAGE_ENCODER_PATH}" ] && TRAIN_CMD+=("actor.model.image_encoder_pretrained_path=${DREAMZERO_WAN21_IMAGE_ENCODER_PATH}")
[ -n "${RESUME_DIR:-}" ] && TRAIN_CMD+=("runner.resume_dir=${RESUME_DIR}")
[ -n "${ENABLE_OFFLOAD:-}" ] && TRAIN_CMD+=("actor.enable_offload=${ENABLE_OFFLOAD}")
[ -n "${WARM_START_MODEL_PATH}" ] && TRAIN_CMD+=("actor.model.model_path=${WARM_START_MODEL_PATH}")
TRAIN_CMD+=("actor.fsdp_config.sharding_strategy=${FSDP_SHARDING_STRATEGY}")

printf '[INFO] Training command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

set +e
if [ "${WRITE_RUN_LOG}" = "true" ]; then
    : > "${RUN_LOG_FILE}"
    "${TRAIN_CMD[@]}" 2>&1 | tee -a "${RUN_LOG_FILE}"
    status=${PIPESTATUS[0]}
elif [ "${WRITE_RUN_LOG}" = "false" ]; then
    "${TRAIN_CMD[@]}"
    status=$?
else
    echo "[ERROR] WRITE_RUN_LOG must be true or false."
    status=1
fi
set -e

if [ "${status}" -ne 0 ] || [ "${AUTO_CONVERT_FINAL_CKPT}" = "false" ]; then
    exit "${status}"
fi

if [ -n "${MAX_STEPS:-}" ]; then
    checkpoint_name="global_step_${MAX_STEPS}"
    if [ -n "${checkpoints_before[${checkpoint_name}]:-}" ]; then
        echo "[ERROR] Training completed, but ${checkpoint_name} existed before this run; refusing to convert an old checkpoint."
        exit 1
    fi
else
    latest_step=-1
    checkpoint_name=""
    for checkpoint in "${checkpoint_root}"/global_step_*; do
        [ -d "${checkpoint}" ] || continue
        name="${checkpoint##*/}"
        [[ "${name}" =~ ^global_step_([0-9]+)$ ]] || continue
        [ -z "${checkpoints_before[${name}]:-}" ] || continue
        step="${BASH_REMATCH[1]}"
        if (( 10#${step} > latest_step )); then
            latest_step=$((10#${step}))
            checkpoint_name="${name}"
        fi
    done
    if [ -z "${checkpoint_name}" ]; then
        echo "[ERROR] Training completed, but no new global_step_* checkpoint was produced."
        exit 1
    fi
fi

checkpoint="${checkpoint_root}/${checkpoint_name}"
eval_root="${EVAL_CKPT_ROOT:-${LOG_ROOT%/}-eval}"
eval_dir="${eval_root}/${EXPERIMENT_NAME}/${checkpoint_name}"
eval_tmp="${eval_dir}.tmp.${MULTI_NODE_RUN_ID}"
echo "[INFO] Training completed. Converting ${checkpoint} -> ${eval_dir}"

set +e
(
    set -euo pipefail
    require_dir "${checkpoint}"
    if [ -d "${checkpoint}/actor/dcp_checkpoint" ]; then
        require_file "${checkpoint}/actor/dcp_checkpoint/.metadata"
    elif [ -d "${checkpoint}/actor/dmuon_checkpoint" ]; then
        require_file "${checkpoint}/actor/dmuon_checkpoint/manifest.pt"
        compgen -G "${checkpoint}/actor/dmuon_checkpoint/rank_*.pt" > /dev/null
    else
        echo "[ERROR] Checkpoint has no DCP or DMuON actor: ${checkpoint}"
        exit 1
    fi
    require_file "${checkpoint}/actor/data.pt"
    require_file "${LOG_ROOT}/${EXPERIMENT_NAME}/config.yaml"
    require_file "${LOG_ROOT}/${EXPERIMENT_NAME}/experiment_cfg/conf.yaml"
    require_file "${LOG_ROOT}/${EXPERIMENT_NAME}/experiment_cfg/metadata.json"
    require_file "${DREAMZERO_PATH}/tools/convert_rlinf_dcp_to_dreamzero_eval.py"
    if [ -e "${eval_dir}" ] || [ -e "${eval_tmp}" ]; then
        echo "[ERROR] Eval output or temporary directory already exists: ${eval_dir} / ${eval_tmp}"
        exit 1
    fi
    mkdir -p "${eval_root}/${EXPERIMENT_NAME}"
    env -u LD_LIBRARY_PATH "${PYTHON_EXECUTABLE}" \
        "${DREAMZERO_PATH}/tools/convert_rlinf_dcp_to_dreamzero_eval.py" \
        --output-mode eval \
        --rlinf-checkpoint "${checkpoint}" \
        --output-dir "${eval_tmp}" \
        --rlinf-root "${RLINF_ROOT}" \
        --dreamzero-root "${DREAMZERO_PATH}" \
        --torch-dtype bf16 \
        --strict-load true
    require_file "${eval_tmp}/config.json"
    require_file "${eval_tmp}/model.safetensors.index.json"
    require_file "${eval_tmp}/experiment_cfg/conf.yaml"
    require_file "${eval_tmp}/experiment_cfg/metadata.json"
    compgen -G "${eval_tmp}/model-*.safetensors" > /dev/null
    for shard in "${eval_tmp}"/model-*.safetensors; do
        require_file "${shard}"
    done
    mv -nT -- "${eval_tmp}" "${eval_dir}"
    [ ! -e "${eval_tmp}" ]
)
convert_status=$?
set -e
if [ "${convert_status}" -ne 0 ]; then
    echo "[ERROR] Training completed, but eval conversion failed (status=${convert_status}); original checkpoint is intact."
    exit "${convert_status}"
fi
echo "[OK] Eval checkpoint: ${eval_dir}"
