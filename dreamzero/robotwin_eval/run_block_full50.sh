#!/usr/bin/env bash
set -euo pipefail

# 全50任务单机八卡模板，每卡运行一组固定任务。
# GPU 0-2 运行 Clean，GPU 3-7 运行 Aug，每卡调用一次 launcher。
# 通过下方环境变量指定 checkpoint、运行标识和输出目录。
# 用法：bash robotwin_eval/run_block_full50.sh <GPU编号0-7> [describe|run]。
# 在各 GPU 上分别调用本脚本，默认执行 run。
# describe 预览分桶与启动命令，run 执行评测。
# ROBOTWIN_TEST_NUM 指定每任务、每模式的 episode 数，默认50。
# 每个 RUN_TAG 对应一次完整评测，重试时使用新标识。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EXP_ID="${EXP_ID:-}"                 # 实验编号
RUN_TAG="${RUN_TAG:-}"               # 本次运行标识，例如日期；重试时使用新标识
EXPECTED_HOST="${EXPECTED_HOST:-}"   # 本次分配机器的 hostname
CKPT="${CKPT:-}"                     # 完整 eval checkpoint 目录
RESULT_ROOT="${RESULT_ROOT:-}"       # 本实验的正式结果根目录
LOG_BASE="${LOG_BASE:-}"             # 本次评测日志根目录

usage() {
  printf '用法：bash %s <GPU编号0-7> [describe|run]\n' "$0"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  exit 0
fi
if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[0-7]$ ]]; then
  usage >&2
  exit 2
fi
GPU="$1"
ACTION="${2:-run}"
if [[ "$ACTION" != describe && "$ACTION" != run ]]; then
  usage >&2
  exit 2
fi

export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM-50}"
if [[ ! "$ROBOTWIN_TEST_NUM" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROBOTWIN_TEST_NUM 必须为正整数（例如25或50）。\n' >&2
  exit 2
fi

SCOPE=full50
SCOPE_TASK_COUNT=50
case "$GPU" in
  0) MODE=demo_clean; SHARD=full50_c01; PORT=5000; TASKS=(hanging_mug handover_mic place_bread_basket place_a2b_right place_bread_skillet beat_block_hammer lift_pot move_playingcard_away put_bottles_dustbin place_dual_shoes stack_bowls_two place_mouse_pad turn_switch rotate_qrcode place_object_stand shake_bottle_horizontally) ;;
  1) MODE=demo_clean; SHARD=full50_c02; PORT=5001; TASKS=(open_microwave blocks_ranking_rgb move_stapler_pad place_burger_fries pick_diverse_bottles place_a2b_left adjust_bottle click_alarmclock click_bell stack_bowls_three place_can_basket scan_object place_cans_plasticbox place_object_scale place_fan place_shoe place_container_plate shake_bottle) ;;
  2) MODE=demo_clean; SHARD=full50_c03; PORT=5002; TASKS=(blocks_ranking_size handover_block open_laptop dump_bin_bigbin move_pillbottle_pad move_can_pot pick_dual_bottles grab_roller put_object_cabinet stack_blocks_three place_object_basket stack_blocks_two stamp_seal place_phone_stand place_empty_cup press_stapler) ;;
  3) MODE=demo_randomized; SHARD=full50_a01; PORT=5003; TASKS=(open_microwave dump_bin_bigbin beat_block_hammer click_alarmclock put_bottles_dustbin stack_bowls_two place_object_scale turn_switch shake_bottle_horizontally) ;;
  4) MODE=demo_randomized; SHARD=full50_a02; PORT=5004; TASKS=(blocks_ranking_size place_bread_skillet open_laptop move_playingcard_away grab_roller stack_blocks_three scan_object stack_blocks_two place_empty_cup press_stapler) ;;
  5) MODE=demo_randomized; SHARD=full50_a03; PORT=5005; TASKS=(hanging_mug place_burger_fries move_can_pot place_a2b_right adjust_bottle put_object_cabinet place_dual_shoes rotate_qrcode place_shoe shake_bottle) ;;
  6) MODE=demo_randomized; SHARD=full50_a04; PORT=5006; TASKS=(handover_block place_bread_basket move_stapler_pad lift_pot pick_diverse_bottles stack_bowls_three place_cans_plasticbox place_mouse_pad place_phone_stand place_container_plate) ;;
  7) MODE=demo_randomized; SHARD=full50_a05; PORT=5007; TASKS=(blocks_ranking_rgb handover_mic move_pillbottle_pad place_a2b_left pick_dual_bottles click_bell place_can_basket place_object_basket stamp_seal place_fan place_object_stand) ;;
esac

# 显式保留范围，避免 launcher 从名称末尾只提取 c01/a01 后混用目录。
export EVAL_SHARD="${SCOPE}_${SHARD##*_}"
NAME="${RUN_TAG}_${EXP_ID}_ep${ROBOTWIN_TEST_NUM}_${EVAL_SHARD}"
LOG_DIR="${LOG_BASE%/}/${NAME}/${EVAL_SHARD}"
export VIDEO_PRED_ROOT="$(dirname "${CKPT%/}")/video_pred/${NAME}"
TASK_COUNT=${#TASKS[@]}
SHARD_EPISODES=$((TASK_COUNT * ROBOTWIN_TEST_NUM))
SCOPE_EPISODES=$((SCOPE_TASK_COUNT * 2 * ROBOTWIN_TEST_NUM))
LAUNCHER="$DREAMZERO_ROOT/robotwin_eval/launch_eval_shard.sh"
COMMAND=(bash "$LAUNCHER" "$EXP_ID" "$MODE" "$NAME" "$PORT" "$GPU"
         "$CKPT" "$RESULT_ROOT" "$LOG_DIR" "${TASKS[@]}")

if [[ "$ACTION" == describe ]]; then
  printf 'SCOPE=%s\nGPU=%s\nMODE=%s\nSHARD=%s\nPORT=%s\nTASKS=%s\n' \
    "$SCOPE" "$GPU" "$MODE" "$SHARD" "$PORT" "${TASKS[*]}"
  printf 'EVAL_SHARD=%s\nVIDEO_PRED_ROOT=%s\nROBOTWIN_TEST_NUM=%s\n' \
    "$EVAL_SHARD" "$VIDEO_PRED_ROOT" "$ROBOTWIN_TEST_NUM"
  printf 'NAME=%s\nTASK_COUNT=%s\nSHARD_EPISODES=%s\nSCOPE_TASK_COUNT=%s\nSCOPE_EPISODES=%s\n' \
    "$NAME" "$TASK_COUNT" "$SHARD_EPISODES" "$SCOPE_TASK_COUNT" "$SCOPE_EPISODES"
  printf '启动命令：ROBOTWIN_TEST_NUM=%q ' "$ROBOTWIN_TEST_NUM"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

for key in EXP_ID RUN_TAG EXPECTED_HOST CKPT RESULT_ROOT LOG_BASE; do
  if [[ -z "${!key}" ]]; then
    printf '请先填写实验参数：%s\n' "$key" >&2
    exit 2
  fi
done
if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  printf '当前机器与 EXPECTED_HOST 不一致。\n' >&2
  exit 2
fi
if [[ ! -d "$CKPT" || ! -f "$LAUNCHER" ]]; then
  printf 'checkpoint 目录或统一 launcher 不存在。\n' >&2
  exit 2
fi
if [[ -e "$LOG_DIR" ]]; then
  printf '日志目录已存在，请检查历史运行并使用新的 RUN_TAG：%s\n' "$LOG_DIR" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/launcher.log" 2>&1
trap 'rc=$?; printf "[BLOCK_EXIT] rc=%s time=%s\n" "$rc" "$(date -Is)"; exit "$rc"' EXIT
printf '[BLOCK_PLAN] scope=%s mode=%s tasks=%s episodes_per_task=%s shard_episodes=%s scope_episodes=%s\n' \
  "$SCOPE" "$MODE" "$TASK_COUNT" "$ROBOTWIN_TEST_NUM" "$SHARD_EPISODES" "$SCOPE_EPISODES"
cd "$DREAMZERO_ROOT"
"${COMMAND[@]}"
