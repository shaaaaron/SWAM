#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "usage: $0 EXP_ID MODE NAME PORT GPU CKPT RESULT_ROOT LOG_ROOT TASK..." >&2
  exit 2
fi

EXP_ID="$1"
MODE="$2"
NAME="$3"
PORT="$4"
GPU="$5"
CKPT="$6"
RESULT_ROOT="$7"
LOG_ROOT_ARG="$8"
shift 8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RELEASE_ROOT="${RLINF_RELEASE_ROOT:-$(cd "${DREAMZERO_ROOT}/.." && pwd)}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${RELEASE_ROOT}/RoboTwin}"
EVAL_PY="${EVAL_PY:-${ROBOTWIN_PYTHON:-}}"
TMP_DIR="${TMP_DIR:-/dev/shm/dreamzero_eval_${EXP_ID//-/_}_g${GPU}}"

if [[ -z "${EVAL_SHARD:-}" ]]; then
  EVAL_SHARD="${NAME##*_}"
fi

LOG_ROOT="${LOG_ROOT_ARG%/}"
if [[ "${LOG_ROOT##*/}" != "$EVAL_SHARD" ]]; then
  LOG_ROOT="$LOG_ROOT/$EVAL_SHARD"
fi

if [[ "${CKPT}" == */checkpoint ]]; then
  EXP_ROOT="${CKPT%/checkpoint}"
else
  EXP_ROOT="$(dirname "${CKPT}")"
fi

VIDEO_PRED_ROOT="${VIDEO_PRED_ROOT:-${EXP_ROOT}/video_pred/${EVAL_SHARD}}"

cd "$DREAMZERO_ROOT"
unset LD_LIBRARY_PATH

mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$TMP_DIR" "$VIDEO_PRED_ROOT"
export ROBOTWIN_PATH="$ROBOTWIN_ROOT"
export ROBOTWIN_RESULT_ROOT="$RESULT_ROOT"
export ROBOTWIN_LOG_ROOT="$LOG_ROOT"
export DREAMZERO_VIDEO_OUTPUT_ROOT="$VIDEO_PRED_ROOT"
export DEPLOY_POLICY_TEMPLATE_PATH="$DREAMZERO_ROOT/robotwin_eval/deploy_policy_robotwin_320x256.yml"
if [[ -n "${DREAMZERO_PYTHON:-}" ]]; then
  export SERVER_CONDA_ENV=""
else
  export SERVER_CONDA_ENV="${SERVER_CONDA_ENV:-dreamzero}"
fi
if [[ -n "${EVAL_PY}" ]]; then
  export ROBOTWIN_PYTHON="$EVAL_PY"
fi
if [[ -n "${ROBOTWIN_PYTHON:-}" ]]; then
  export EVAL_CONDA_ENV=""
else
  export EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-RoboTwin}"
fi
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-50}"
export TMPDIR="$TMP_DIR"

echo "[LAUNCH] experiment=$EXP_ID name=$NAME mode=$MODE gpu=$GPU port=$PORT"
echo "[LAUNCH] checkpoint=$CKPT"
echo "[LAUNCH] result_root=$RESULT_ROOT"
echo "[LAUNCH] log_root=$LOG_ROOT"
echo "[LAUNCH] video_pred_root=$VIDEO_PRED_ROOT"
echo "[LAUNCH] tasks=$*"

bash robotwin_eval/start_eval.sh \
  -m "$MODE" \
  -n "$NAME" \
  -p "$PORT" \
  -g "$GPU" \
  --ngpu 1 \
  --eval-gpu "$GPU" \
  --server-timeout 60000 \
  -s 0 \
  -c "$CKPT" \
  "$@"
