#!/bin/bash
set -euo pipefail

export RLINF_ROOT="${RLINF_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DREAMZERO_PATH="${DREAMZERO_PATH:-${RLINF_ROOT}/dreamzero}"
export VENV_PATH="${VENV_PATH:-${RLINF_ROOT}/.venv/main}"
export LOG_ROOT="${LOG_ROOT:-${RLINF_ROOT}/logs}"

# Run this entrypoint once on every node of the same distributed job. It runs
# 2-, 4-, or 8-node pretrain, converts the final checkpoint on rank 0, then
# warm-starts RoboTwin posttrain with the same topology.

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "[ERROR] Required environment variable ${name} is not set."
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

require_dir() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        echo "[ERROR] Required directory is missing: ${path}"
        exit 1
    fi
}

require_positive_int() {
    local name="$1"
    local value="${!name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}"
        exit 1
    fi
}

reject_env() {
    local name="$1"
    if [[ -v "${name}" ]]; then
        echo "[ERROR] ${name} is not accepted by this pipeline. Run: unset ${name}"
        exit 1
    fi
}

reject_env WARM_START_MODEL_PATH
reject_env RESUME_DIR
require_env PET_NNODES
require_env PET_NODE_RANK
require_env MASTER_ADDR
require_env MASTER_PORT
require_env PIPELINE_RUN_ID
require_env PRETRAIN_CONFIG_NAME
require_env PRETRAIN_EXPERIMENT_NAME
require_env POSTTRAIN_CONFIG_NAME
require_env POSTTRAIN_EXPERIMENT_NAME

case "${PIPELINE_RUN_ID}" in
    ''|*[!A-Za-z0-9._-]*)
        echo "[ERROR] PIPELINE_RUN_ID may contain only letters, digits, dot, underscore, and dash."
        exit 1
        ;;
esac

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-${VENV_PATH}/bin/python}"
CONVERTED_ROOT="${CONVERTED_ROOT:-${LOG_ROOT}-warm}"
EVAL_CKPT_ROOT="${EVAL_CKPT_ROOT:-${LOG_ROOT%/}-eval}"

PRETRAIN_STEPS="${PRETRAIN_STEPS:-20000}"
POSTTRAIN_STEPS="${POSTTRAIN_STEPS:-20000}"
PRETRAIN_GLOBAL_BATCH_SIZE="${PRETRAIN_GLOBAL_BATCH_SIZE:-256}"
POSTTRAIN_GLOBAL_BATCH_SIZE="${POSTTRAIN_GLOBAL_BATCH_SIZE:-128}"
NGPU_PER_NODE="${NGPU_PER_NODE:-${PET_NPROC_PER_NODE:-8}}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-hybrid_shard}"
CONVERSION_MAX_WAIT="${CONVERSION_MAX_WAIT:-14400}"

case "${PET_NNODES}" in
    4)
        DEFAULT_PRETRAIN_MICRO_BATCH_SIZE=8
        DEFAULT_POSTTRAIN_MICRO_BATCH_SIZE=4
        ;;
    8)
        DEFAULT_PRETRAIN_MICRO_BATCH_SIZE=4
        DEFAULT_POSTTRAIN_MICRO_BATCH_SIZE=2
        ;;
    *)
        DEFAULT_PRETRAIN_MICRO_BATCH_SIZE=16
        DEFAULT_POSTTRAIN_MICRO_BATCH_SIZE=8
        ;;
esac
PRETRAIN_MICRO_BATCH_SIZE="${PRETRAIN_MICRO_BATCH_SIZE:-${DEFAULT_PRETRAIN_MICRO_BATCH_SIZE}}"
POSTTRAIN_MICRO_BATCH_SIZE="${POSTTRAIN_MICRO_BATCH_SIZE:-${DEFAULT_POSTTRAIN_MICRO_BATCH_SIZE}}"

for integer_name in \
    PET_NNODES MASTER_PORT NGPU_PER_NODE \
    PRETRAIN_STEPS POSTTRAIN_STEPS \
    PRETRAIN_MICRO_BATCH_SIZE PRETRAIN_GLOBAL_BATCH_SIZE \
    POSTTRAIN_MICRO_BATCH_SIZE POSTTRAIN_GLOBAL_BATCH_SIZE \
    CONVERSION_MAX_WAIT; do
    require_positive_int "${integer_name}"
done
if ! [[ "${PET_NODE_RANK}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] PET_NODE_RANK must be a non-negative integer, got: ${PET_NODE_RANK}"
    exit 1
fi
if [ "${PET_NODE_RANK}" -ge "${PET_NNODES}" ]; then
    echo "[ERROR] PET_NODE_RANK=${PET_NODE_RANK} is invalid for PET_NNODES=${PET_NNODES}."
    exit 1
fi
if { [ "${PET_NNODES}" -ne 2 ] && [ "${PET_NNODES}" -ne 4 ] && [ "${PET_NNODES}" -ne 8 ]; } || [ "${NGPU_PER_NODE}" -ne 8 ]; then
    echo "[ERROR] This pipeline requires 2, 4, or 8 nodes x 8 GPUs."
    echo "[ERROR] PET_NNODES=${PET_NNODES}, NGPU_PER_NODE=${NGPU_PER_NODE}"
    exit 1
fi

WORLD_SIZE_TOTAL=$((PET_NNODES * NGPU_PER_NODE))
if [ "$((PRETRAIN_GLOBAL_BATCH_SIZE % (PRETRAIN_MICRO_BATCH_SIZE * WORLD_SIZE_TOTAL)))" -ne 0 ]; then
    echo "[ERROR] Invalid pretrain batch sizes for world size ${WORLD_SIZE_TOTAL}."
    exit 1
fi
if [ "$((POSTTRAIN_GLOBAL_BATCH_SIZE % (POSTTRAIN_MICRO_BATCH_SIZE * WORLD_SIZE_TOTAL)))" -ne 0 ]; then
    echo "[ERROR] Invalid posttrain batch sizes for world size ${WORLD_SIZE_TOTAL}."
    exit 1
fi

TRAIN_LAUNCHER="${RLINF_ROOT}/examples/sft/train_dreamzero.sh"
CONVERTER="${DREAMZERO_PATH}/tools/convert_rlinf_dcp_to_dreamzero_eval.py"
PRETRAIN_RUN_DIR="${LOG_ROOT}/${PRETRAIN_EXPERIMENT_NAME}"
PRETRAIN_CHECKPOINT="${PRETRAIN_RUN_DIR}/checkpoints/global_step_${PRETRAIN_STEPS}"
CONVERTED_DIR="${CONVERTED_ROOT}/${PRETRAIN_EXPERIMENT_NAME}_global_step_${PRETRAIN_STEPS}"
CONVERTED_TMP_DIR="${CONVERTED_DIR}.tmp.${PIPELINE_RUN_ID}"
POSTTRAIN_RUN_DIR="${LOG_ROOT}/${POSTTRAIN_EXPERIMENT_NAME}"
POSTTRAIN_CHECKPOINT="${POSTTRAIN_RUN_DIR}/checkpoints/global_step_${POSTTRAIN_STEPS}"
POSTTRAIN_EVAL_CHECKPOINT="${EVAL_CKPT_ROOT}/${POSTTRAIN_EXPERIMENT_NAME}/global_step_${POSTTRAIN_STEPS}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-${PRETRAIN_RUN_DIR}/logs}"
PIPELINE_LOG_FILE="${PIPELINE_LOG_FILE:-${PIPELINE_LOG_DIR}/pipeline_${PIPELINE_RUN_ID}_rank${PET_NODE_RANK}.log}"
CONVERT_LOG_FILE="${CONVERT_LOG_FILE:-${PIPELINE_LOG_DIR}/convert_global_step_${PRETRAIN_STEPS}_${PIPELINE_RUN_ID}.log}"
CONVERT_STATUS_FILE="${PIPELINE_LOG_DIR}/pipeline_${PIPELINE_RUN_ID}_convert.status"
FINAL_AUDIT_STATUS_FILE="${PIPELINE_LOG_DIR}/pipeline_${PIPELINE_RUN_ID}_final_audit.status"

mkdir -p "${PIPELINE_LOG_DIR}"

pipeline_log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*" | tee -a "${PIPELINE_LOG_FILE}"
}

write_status() {
    local path="$1"
    local status="$2"
    local tmp="${path}.tmp.$$"
    printf '%s %s\n' "${PIPELINE_RUN_ID}" "${status}" > "${tmp}"
    mv -f "${tmp}" "${path}"
}

wait_for_status() {
    local path="$1"
    local timeout="$2"
    local waited=0
    local marker_run_id marker_status
    while true; do
        if [ -s "${path}" ]; then
            read -r marker_run_id marker_status < "${path}" || true
            if [ "${marker_run_id:-}" = "${PIPELINE_RUN_ID}" ]; then
                case "${marker_status:-}" in
                    ''|*[!0-9]*)
                        pipeline_log "[ERROR] Invalid status marker: ${path}"
                        return 1
                        ;;
                esac
                if [ "${marker_status}" -gt 255 ]; then
                    pipeline_log "[ERROR] Out-of-range status marker: ${path}"
                    return 1
                fi
                return "${marker_status}"
            fi
        fi
        if [ "${waited}" -ge "${timeout}" ]; then
            pipeline_log "[ERROR] Timed out waiting for status marker: ${path}"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}

require_file "${TRAIN_LAUNCHER}"
require_file "${CONVERTER}"
if [ ! -x "${PYTHON_EXECUTABLE}" ]; then
    echo "[ERROR] PYTHON_EXECUTABLE is not executable: ${PYTHON_EXECUTABLE}"
    exit 1
fi
require_file "${RLINF_ROOT}/examples/sft/config/${PRETRAIN_CONFIG_NAME}.yaml"
require_file "${RLINF_ROOT}/examples/sft/config/${POSTTRAIN_CONFIG_NAME}.yaml"

if [ "${PET_NODE_RANK}" -eq 0 ]; then
    rm -f "${CONVERT_STATUS_FILE}" "${FINAL_AUDIT_STATUS_FILE}"
fi

pipeline_log "[PIPELINE START] id=${PIPELINE_RUN_ID} rank=${PET_NODE_RANK}/${PET_NNODES}"
pipeline_log "[PIPELINE 1/3] Pretrain ${PRETRAIN_EXPERIMENT_NAME}: steps=${PRETRAIN_STEPS}, micro/global=${PRETRAIN_MICRO_BATCH_SIZE}/${PRETRAIN_GLOBAL_BATCH_SIZE}"

set +e
RLINF_ROOT="${RLINF_ROOT}" \
DREAMZERO_PATH="${DREAMZERO_PATH}" \
VENV_PATH="${VENV_PATH}" \
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
LOG_ROOT="${LOG_ROOT}" \
NGPU_PER_NODE="${NGPU_PER_NODE}" \
CONFIG_NAME="${PRETRAIN_CONFIG_NAME}" \
EXPERIMENT_NAME="${PRETRAIN_EXPERIMENT_NAME}" \
MULTI_NODE_RUN_ID="${PIPELINE_RUN_ID}_pretrain" \
MICRO_BATCH_SIZE="${PRETRAIN_MICRO_BATCH_SIZE}" \
GLOBAL_BATCH_SIZE="${PRETRAIN_GLOBAL_BATCH_SIZE}" \
MAX_STEPS="${PRETRAIN_STEPS}" \
AUTO_CONVERT_FINAL_CKPT=false \
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY}" \
bash "${TRAIN_LAUNCHER}"
pretrain_status=$?
set -e
if [ "${pretrain_status}" -ne 0 ]; then
    pipeline_log "[PIPELINE 1/3] Pretrain failed with status=${pretrain_status}."
    exit "${pretrain_status}"
fi
pipeline_log "[PIPELINE 1/3] Pretrain completed."

if [ "${PET_NODE_RANK}" -eq 0 ]; then
    pipeline_log "[PIPELINE 2/3] Converting ${PRETRAIN_CHECKPOINT} -> ${CONVERTED_DIR}"
    set +e
    (
        set -euo pipefail
        if [ -d "${PRETRAIN_CHECKPOINT}/actor/dcp_checkpoint" ]; then
            require_file "${PRETRAIN_CHECKPOINT}/actor/dcp_checkpoint/.metadata"
        elif [ -d "${PRETRAIN_CHECKPOINT}/actor/dmuon_checkpoint" ]; then
            require_file "${PRETRAIN_CHECKPOINT}/actor/dmuon_checkpoint/manifest.pt"
            if ! compgen -G "${PRETRAIN_CHECKPOINT}/actor/dmuon_checkpoint/rank_*.pt" > /dev/null; then
                echo "[ERROR] DMuON checkpoint contains no rank shards: ${PRETRAIN_CHECKPOINT}/actor/dmuon_checkpoint"
                exit 1
            fi
        else
            echo "[ERROR] Required checkpoint directory missing: expected actor/dcp_checkpoint or actor/dmuon_checkpoint under ${PRETRAIN_CHECKPOINT}/actor"
            exit 1
        fi
        require_file "${PRETRAIN_CHECKPOINT}/actor/data.pt"
        require_file "${PRETRAIN_RUN_DIR}/config.yaml"
        if [ -e "${CONVERTED_DIR}" ]; then
            echo "[ERROR] Converted weights directory already exists: ${CONVERTED_DIR}"
            exit 1
        fi
        if [ -e "${CONVERTED_TMP_DIR}" ]; then
            echo "[ERROR] Temporary conversion directory already exists: ${CONVERTED_TMP_DIR}"
            exit 1
        fi
        mkdir -p "${CONVERTED_ROOT}"
        env -u LD_LIBRARY_PATH "${PYTHON_EXECUTABLE}" "${CONVERTER}" \
            --output-mode weights \
            --rlinf-checkpoint "${PRETRAIN_CHECKPOINT}" \
            --output-dir "${CONVERTED_TMP_DIR}" \
            --rlinf-root "${RLINF_ROOT}" \
            --dreamzero-root "${DREAMZERO_PATH}" \
            --torch-dtype bf16 \
            --strict-load true \
            2>&1 | tee -a "${CONVERT_LOG_FILE}"
        require_file "${CONVERTED_TMP_DIR}/config.json"
        require_file "${CONVERTED_TMP_DIR}/model.safetensors.index.json"
        if [ -e "${CONVERTED_TMP_DIR}/experiment_cfg" ]; then
            echo "[ERROR] Converted warm-start weights unexpectedly contain experiment_cfg."
            exit 1
        fi
        if ! compgen -G "${CONVERTED_TMP_DIR}/model-*.safetensors" >/dev/null; then
            echo "[ERROR] Converted weights contain no model shards: ${CONVERTED_TMP_DIR}"
            exit 1
        fi
        mv "${CONVERTED_TMP_DIR}" "${CONVERTED_DIR}"
    )
    convert_status=$?
    set -e
    write_status "${CONVERT_STATUS_FILE}" "${convert_status}"
else
    pipeline_log "[PIPELINE 2/3] Waiting for rank 0 conversion."
    set +e
    wait_for_status "${CONVERT_STATUS_FILE}" "${CONVERSION_MAX_WAIT}"
    convert_status=$?
    set -e
fi

if [ "${convert_status}" -ne 0 ]; then
    pipeline_log "[PIPELINE 2/3] Conversion failed with status=${convert_status}."
    exit "${convert_status}"
fi
pipeline_log "[PIPELINE 2/3] Conversion completed and validated."

pipeline_log "[PIPELINE 3/3] Posttrain ${POSTTRAIN_EXPERIMENT_NAME}: steps=${POSTTRAIN_STEPS}, micro/global=${POSTTRAIN_MICRO_BATCH_SIZE}/${POSTTRAIN_GLOBAL_BATCH_SIZE}"
set +e
RLINF_ROOT="${RLINF_ROOT}" \
DREAMZERO_PATH="${DREAMZERO_PATH}" \
VENV_PATH="${VENV_PATH}" \
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
LOG_ROOT="${LOG_ROOT}" \
NGPU_PER_NODE="${NGPU_PER_NODE}" \
CONFIG_NAME="${POSTTRAIN_CONFIG_NAME}" \
EXPERIMENT_NAME="${POSTTRAIN_EXPERIMENT_NAME}" \
MULTI_NODE_RUN_ID="${PIPELINE_RUN_ID}_posttrain" \
MICRO_BATCH_SIZE="${POSTTRAIN_MICRO_BATCH_SIZE}" \
GLOBAL_BATCH_SIZE="${POSTTRAIN_GLOBAL_BATCH_SIZE}" \
MAX_STEPS="${POSTTRAIN_STEPS}" \
AUTO_CONVERT_FINAL_CKPT=true \
EVAL_CKPT_ROOT="${EVAL_CKPT_ROOT}" \
WARM_START_MODEL_PATH="${CONVERTED_DIR}" \
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY}" \
bash "${TRAIN_LAUNCHER}"
posttrain_status=$?
set -e
if [ "${posttrain_status}" -ne 0 ]; then
    pipeline_log "[PIPELINE 3/3] Posttrain launcher failed (training or eval conversion), status=${posttrain_status}."
    exit "${posttrain_status}"
fi

if [ "${PET_NODE_RANK}" -eq 0 ]; then
    set +e
    (
        set -euo pipefail
        if [ -d "${POSTTRAIN_CHECKPOINT}/actor/dcp_checkpoint" ]; then
            require_file "${POSTTRAIN_CHECKPOINT}/actor/dcp_checkpoint/.metadata"
        elif [ -d "${POSTTRAIN_CHECKPOINT}/actor/dmuon_checkpoint" ]; then
            require_file "${POSTTRAIN_CHECKPOINT}/actor/dmuon_checkpoint/manifest.pt"
            if ! compgen -G "${POSTTRAIN_CHECKPOINT}/actor/dmuon_checkpoint/rank_*.pt" > /dev/null; then
                echo "[ERROR] DMuON checkpoint contains no rank shards: ${POSTTRAIN_CHECKPOINT}/actor/dmuon_checkpoint"
                exit 1
            fi
        else
            echo "[ERROR] Required checkpoint directory missing: expected actor/dcp_checkpoint or actor/dmuon_checkpoint under ${POSTTRAIN_CHECKPOINT}/actor"
            exit 1
        fi
        require_file "${POSTTRAIN_CHECKPOINT}/actor/data.pt"
        require_file "${POSTTRAIN_EVAL_CHECKPOINT}/config.json"
        require_file "${POSTTRAIN_EVAL_CHECKPOINT}/model.safetensors.index.json"
        require_file "${POSTTRAIN_EVAL_CHECKPOINT}/experiment_cfg/conf.yaml"
        require_file "${POSTTRAIN_EVAL_CHECKPOINT}/experiment_cfg/metadata.json"
        if ! compgen -G "${POSTTRAIN_EVAL_CHECKPOINT}/model-*.safetensors" > /dev/null; then
            echo "[ERROR] Final eval checkpoint contains no model shards: ${POSTTRAIN_EVAL_CHECKPOINT}"
            exit 1
        fi
        for shard in "${POSTTRAIN_EVAL_CHECKPOINT}"/model-*.safetensors; do
            require_file "${shard}"
        done
    )
    audit_status=$?
    set -e
    write_status "${FINAL_AUDIT_STATUS_FILE}" "${audit_status}"
else
    set +e
    wait_for_status "${FINAL_AUDIT_STATUS_FILE}" 600
    audit_status=$?
    set -e
fi

if [ "${audit_status}" -ne 0 ]; then
    pipeline_log "[ERROR] Final posttrain checkpoint audit failed."
    exit "${audit_status}"
fi

pipeline_log "[OK] Pretrain, conversion, posttrain, and eval packaging completed."
pipeline_log "[OK] Pretrain checkpoint: ${PRETRAIN_CHECKPOINT}"
pipeline_log "[OK] Converted warm-start weights: ${CONVERTED_DIR}"
pipeline_log "[OK] Posttrain checkpoint: ${POSTTRAIN_CHECKPOINT}"
pipeline_log "[OK] Posttrain eval checkpoint: ${POSTTRAIN_EVAL_CHECKPOINT}"
