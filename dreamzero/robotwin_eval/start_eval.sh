#!/usr/bin/env bash
set -euo pipefail

# Multi-task DreamZero RoboTwin evaluation orchestrator.
#
# Launches a DreamZero policy server, then runs multiple RoboTwin tasks
# sequentially (one server shared across tasks since DreamZero requires
# multiple GPUs for distributed inference).
#
# Usage:
#   bash start_eval.sh -m <mode> -n <policy_name> -c <ckpt_path> [options] <tasks...>
#
# Examples:
#   bash start_eval.sh -m demo_clean -n dreamzero -c /path/to/checkpoint all
#   bash start_eval.sh -m demo_clean -n dreamzero -c /path/to/checkpoint adjust_bottle open_laptop
#   bash start_eval.sh -m demo_randomized -n dreamzero -c /path/to/checkpoint task_list.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Explicit Python paths are authoritative. The Conda names remain as a
# compatibility fallback for older launch commands that do not set them.
if [[ -n "${DREAMZERO_PYTHON:-}" ]]; then
    SERVER_CONDA_ENV=""
else
    SERVER_CONDA_ENV="${SERVER_CONDA_ENV-dreamzero}"
fi
if [[ -n "${ROBOTWIN_PYTHON:-}" ]]; then
    EVAL_CONDA_ENV=""
else
    EVAL_CONDA_ENV="${EVAL_CONDA_ENV-RoboTwin}"
fi

# Activate a conda env in the CURRENT shell. MUST be called inside a subshell
# so it does not pollute the outer start_eval.sh shell. No-op if env is empty.
_activate_conda() {
    local env_name="$1"
    if [[ -z "${env_name}" ]]; then
        return 0
    fi
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        echo "[WARN] conda not found; cannot activate '${env_name}'. Running in current shell env." >&2
        return 0
    fi
    # conda activate touches unset vars; temporarily relax nounset.
    set +u
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    set -u
}

# adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic
# hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
# place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
# place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet
# rotate_qrcode scan_object shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch


ROBOTWIN_ALL_TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle_horizontally
    shake_bottle
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

SERVER_PID=""
FAILED_TASKS=()

join_arr() {
    local sep="$1"; shift
    local out=""
    for item in "$@"; do
        out="${out:+${out}${sep}}${item}"
    done
    printf '%s' "${out}"
}

kill_descendants() {
    local target_pid="$1"
    local sig="${2:-TERM}"
    local child_pids
    child_pids="$(ps -o pid= --ppid "${target_pid}" 2>/dev/null)" || true
    local cpid
    for cpid in ${child_pids}; do
        kill_descendants "${cpid}" "${sig}"
    done
    kill -"${sig}" "${target_pid}" 2>/dev/null || true
}

cleanup() {
    trap '' INT TERM EXIT
    echo "[INFO] Cleaning up..."
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill_descendants "${SERVER_PID}" TERM
        sleep 2
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill_descendants "${SERVER_PID}" KILL
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

usage() {
    cat >&2 <<'EOF'
Usage:
  bash start_eval.sh -m <mode> -n <policy_name> -c <ckpt_path> [options] <tasks...>

Required flags:
  -m, --mode              Eval mode: demo_clean or demo_randomized
  -n, --name              Policy name (used for log directory naming)
  -c, --ckpt              Path to the checkpoint directory

Tasks (positional):
  Remaining arguments after flags are treated as task names, the keyword
  `all` (all RoboTwin tasks), or a single task-list file (one per line).

Optional flags:
  -s, --seed              Eval seed (default: 0)
  -g, --gpus              GPU IDs for the server (default: 0,1)
      --ngpu              Number of GPUs for torchrun (default: 2)
  -p, --port              Server port (default: 8000)
      --server-timeout    Seconds to wait for server (default: 600)
      --eval-gpu          GPU for RoboTwin eval client (default: 0)
  -h, --help              Show this help message
EOF
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "${value}"
}

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
    else
        python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',${port})); s.close()" 2>/dev/null
    fi
}

process_running() {
    local pid="$1"
    local stat
    stat="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    [[ -n "${stat}" && "${stat}" != *Z* ]]
}

wait_for_server() {
    local port="$1"
    local timeout_s="${2:-600}"
    local server_pid="${3:-}"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if port_in_use "${port}"; then
            return 0
        fi
        if [[ -n "${server_pid}" ]] && ! process_running "${server_pid}"; then
            return 2
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

resolve_tasks() {
    local -a raw_inputs=("$@")
    local -a resolved=()
    local input="" task="" line=""

    if (( ${#raw_inputs[@]} == 1 )) && [[ -f "${raw_inputs[0]}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"
            if [[ -n "${line}" ]]; then
                resolved+=("${line}")
            fi
        done < "${raw_inputs[0]}"
    else
        for input in "${raw_inputs[@]}"; do
            if [[ "${input}" == "all" ]]; then
                resolved+=("${ROBOTWIN_ALL_TASKS[@]}")
                continue
            fi
            IFS=',' read -ra split_inputs <<< "${input}"
            for task in "${split_inputs[@]}"; do
                task="$(trim "${task}")"
                if [[ -n "${task}" ]]; then
                    resolved+=("${task}")
                fi
            done
        done
    fi

    if (( ${#resolved[@]} == 0 )); then
        echo "No RoboTwin tasks were resolved from input." >&2
        return 1
    fi

    printf '%s\n' "${resolved[@]}"
}

# --- Argument parsing ---

TASK_CONFIG=""
POLICY_NAME=""
CKPT_PATH=""
opt_seed="0"
opt_gpus="0,1"
opt_ngpu="2"
opt_port="8000"
opt_timeout="600"
opt_eval_gpu="0"

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)          TASK_CONFIG="$2"; shift 2 ;;
        -n|--name)          POLICY_NAME="$2"; shift 2 ;;
        -c|--ckpt)          CKPT_PATH="$2"; shift 2 ;;
        -s|--seed)          opt_seed="$2"; shift 2 ;;
        -g|--gpus)          opt_gpus="$2"; shift 2 ;;
        --ngpu)             opt_ngpu="$2"; shift 2 ;;
        -p|--port)          opt_port="$2"; shift 2 ;;
        --server-timeout)   opt_timeout="$2"; shift 2 ;;
        --eval-gpu)         opt_eval_gpu="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        -*)                 echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)                  break ;;
    esac
done

if [[ -z "${TASK_CONFIG}" || -z "${POLICY_NAME}" || -z "${CKPT_PATH}" ]]; then
    echo "Missing required flags: -m/--mode, -n/--name, -c/--ckpt" >&2
    usage
    exit 1
fi

if [[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" ]]; then
    echo "Unsupported mode: ${TASK_CONFIG} (expected demo_clean or demo_randomized)" >&2
    exit 1
fi

if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "Checkpoint directory does not exist: ${CKPT_PATH}" >&2
    exit 1
fi

if (( $# == 0 )); then
    echo "No tasks specified." >&2
    usage
    exit 1
fi

ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"

mapfile -t TASKS < <(resolve_tasks "$@")
TOTAL_TASKS=${#TASKS[@]}

# Log directory
ckpt_name="$(basename "${CKPT_PATH}")"
timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROBOTWIN_LOG_ROOT:-$(dirname "${CKPT_PATH}")/robotwin_eval_logs/${POLICY_NAME}_${TASK_CONFIG}_${ckpt_name}_${timestamp}}"
mkdir -p "${LOG_DIR}"

echo "[INFO] mode=${TASK_CONFIG}  name=${POLICY_NAME}  seed=${opt_seed}"
echo "[INFO] ckpt=${CKPT_PATH}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] server GPUs=${opt_gpus}  ngpu=${opt_ngpu}  port=${opt_port}"
echo "[INFO] eval GPU=${opt_eval_gpu}"
echo "[INFO] conda envs: server='${SERVER_CONDA_ENV:-<current>}'  eval='${EVAL_CONDA_ENV:-<current>}'"
echo "[INFO] tasks (${TOTAL_TASKS}): $(join_arr ', ' "${TASKS[@]}")"

# --- Launch DreamZero server ---

server_log="${LOG_DIR}/server.log"
echo "[INFO] Starting DreamZero server (log: ${server_log})"

(
    _activate_conda "${SERVER_CONDA_ENV}"
    exec bash "${SCRIPT_DIR}/run_dreamzero_server.sh" \
        "${CKPT_PATH}" "${opt_gpus}" "${opt_port}" "${opt_ngpu}"
) > "${server_log}" 2>&1 &
SERVER_PID=$!

echo "[INFO] Waiting for server on port ${opt_port} (timeout: ${opt_timeout}s)..."
if wait_for_server "${opt_port}" "${opt_timeout}" "${SERVER_PID}"; then
    echo "[INFO] Server is ready on port ${opt_port}"
else
    wait_status=$?
    if (( wait_status == 2 )); then
        echo "[ERROR] DreamZero server exited before listening on port ${opt_port}. See ${server_log}" >&2
    else
        echo "[ERROR] DreamZero server failed to start within ${opt_timeout}s. See ${server_log}" >&2
    fi
    exit 1
fi

# --- Run tasks sequentially ---

completed=0
for task_name in "${TASKS[@]}"; do
    eval_log="${LOG_DIR}/${task_name}_${TASK_CONFIG}_eval.log"
    echo "[INFO] Running task=${task_name} (${completed}/${TOTAL_TASKS} done)"

    if (
        _activate_conda "${EVAL_CONDA_ENV}"
        bash "${SCRIPT_DIR}/eval.sh" \
            "${task_name}" \
            "${TASK_CONFIG}" \
            "${POLICY_NAME}" \
            "${opt_seed}" \
            "${opt_eval_gpu}" \
            "${CKPT_PATH}" \
            "${opt_port}"
    ) > >(tee "${eval_log}" | grep --line-buffered "Success rate" | sed -u "s/^/[RESULT] ${task_name}: /") 2>&1; then
        echo "[INFO] Finished task=${task_name}"
    else
        FAILED_TASKS+=("${task_name}")
        echo "[ERROR] Task ${task_name} failed. See ${eval_log}" >&2
    fi
    completed=$((completed + 1))
done

# --- Summary ---

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[ERROR] Evaluation finished with failures: $(join_arr ', ' "${FAILED_TASKS[@]}")" >&2
    echo "[ERROR] Logs: ${LOG_DIR}" >&2
    exit 1
fi

echo "[INFO] All ${TOTAL_TASKS} tasks completed successfully"
echo "[INFO] Logs: ${LOG_DIR}"
