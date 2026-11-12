#!/usr/bin/env bash
set -euo pipefail

# 全50任务四机分片模板，四台机器使用同一 checkpoint。
# 各机 GPU 0-2 运行 Clean，GPU 3-7 运行 Aug，每卡调用一次 launcher。
# 用法：bash robotwin_eval/run_block_full50_4n.sh <机器编号1-4> <GPU编号0-7> [describe|run]。
# 在各机器、各 GPU 上分别调用本脚本，默认执行 run。
# describe 预览分桶与启动命令，run 执行评测。
# ROBOTWIN_TEST_NUM 指定每任务、每模式的 episode 数，默认100。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EXP_ID="${EXP_ID:-}"                 # 实验编号
RUN_TAG="${RUN_TAG:-}"               # 本次运行标识，例如日期；重试时使用新标识
EXPECTED_HOST="${EXPECTED_HOST:-}"   # 本次分配机器的 hostname
CKPT="${CKPT:-}"                     # 完整 eval checkpoint 目录
RESULT_ROOT="${RESULT_ROOT:-}"       # 本实验的正式结果根目录
LOG_BASE="${LOG_BASE:-}"             # 本次评测日志根目录

usage() {
  printf '用法：bash %s <机器编号1-4> <GPU编号0-7> [describe|run]\n' "$0"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  usage
  exit 0
fi
if (( $# < 2 || $# > 3 )) || [[ ! "$1" =~ ^[1-4]$ ]]; then
  usage >&2
  exit 2
fi
MACHINE="$1"
shift
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

export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM-100}"
if [[ ! "$ROBOTWIN_TEST_NUM" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROBOTWIN_TEST_NUM 必须为正整数（例如25或50）。\n' >&2
  exit 2
fi

SCOPE=full50_4n
SCOPE_TASK_COUNT=50
case "$MACHINE:$GPU" in
  1:0) MODE=demo_clean; SHARD=full50_4n_m1_c01; PORT=5000; TASKS=(blocks_ranking_rgb place_object_basket place_bread_skillet stamp_seal shake_bottle) ;;
  1:1) MODE=demo_clean; SHARD=full50_4n_m1_c02; PORT=5001; TASKS=(stack_blocks_three stack_blocks_two move_can_pot place_empty_cup click_alarmclock) ;;
  1:2) MODE=demo_clean; SHARD=full50_4n_m1_c03; PORT=5002; TASKS=(stack_bowls_three dump_bin_bigbin place_phone_stand place_container_plate) ;;
  1:3) MODE=demo_randomized; SHARD=full50_4n_m1_a01; PORT=5003; TASKS=(open_microwave) ;;
  1:4) MODE=demo_randomized; SHARD=full50_4n_m1_a02; PORT=5004; TASKS=(put_object_cabinet place_object_stand) ;;
  1:5) MODE=demo_randomized; SHARD=full50_4n_m1_a03; PORT=5005; TASKS=(handover_mic move_pillbottle_pad grab_roller) ;;
  1:6) MODE=demo_randomized; SHARD=full50_4n_m1_a04; PORT=5006; TASKS=(scan_object stamp_seal place_fan) ;;
  1:7) MODE=demo_randomized; SHARD=full50_4n_m1_a05; PORT=5007; TASKS=(stack_bowls_two place_phone_stand place_object_scale) ;;
  2:0) MODE=demo_clean; SHARD=full50_4n_m2_c01; PORT=5000; TASKS=(place_can_basket stack_bowls_two open_laptop place_fan move_playingcard_away) ;;
  2:1) MODE=demo_clean; SHARD=full50_4n_m2_c02; PORT=5001; TASKS=(hanging_mug place_object_scale shake_bottle_horizontally) ;;
  2:2) MODE=demo_clean; SHARD=full50_4n_m2_c03; PORT=5002; TASKS=(put_object_cabinet place_bread_basket rotate_qrcode place_object_stand) ;;
  2:3) MODE=demo_randomized; SHARD=full50_4n_m2_a01; PORT=5003; TASKS=(place_object_basket place_a2b_left place_container_plate) ;;
  2:4) MODE=demo_randomized; SHARD=full50_4n_m2_a02; PORT=5004; TASKS=(place_dual_shoes turn_switch shake_bottle_horizontally) ;;
  2:5) MODE=demo_randomized; SHARD=full50_4n_m2_a03; PORT=5005; TASKS=(place_cans_plasticbox open_laptop pick_dual_bottles) ;;
  2:6) MODE=demo_randomized; SHARD=full50_4n_m2_a04; PORT=5006; TASKS=(place_bread_basket rotate_qrcode pick_diverse_bottles) ;;
  2:7) MODE=demo_randomized; SHARD=full50_4n_m2_a05; PORT=5007; TASKS=(place_burger_fries stack_blocks_two move_can_pot) ;;
  3:0) MODE=demo_clean; SHARD=full50_4n_m3_c01; PORT=5000; TASKS=(place_dual_shoes move_stapler_pad move_pillbottle_pad place_shoe grab_roller) ;;
  3:1) MODE=demo_clean; SHARD=full50_4n_m3_c02; PORT=5001; TASKS=(open_microwave place_a2b_left beat_block_hammer) ;;
  3:2) MODE=demo_clean; SHARD=full50_4n_m3_c03; PORT=5002; TASKS=(put_bottles_dustbin pick_diverse_bottles pick_dual_bottles) ;;
  3:3) MODE=demo_randomized; SHARD=full50_4n_m3_a01; PORT=5003; TASKS=(handover_block place_empty_cup click_bell) ;;
  3:4) MODE=demo_randomized; SHARD=full50_4n_m3_a02; PORT=5004; TASKS=(put_bottles_dustbin adjust_bottle) ;;
  3:5) MODE=demo_randomized; SHARD=full50_4n_m3_a03; PORT=5005; TASKS=(place_can_basket press_stapler) ;;
  3:6) MODE=demo_randomized; SHARD=full50_4n_m3_a04; PORT=5006; TASKS=(dump_bin_bigbin place_bread_skillet lift_pot) ;;
  3:7) MODE=demo_randomized; SHARD=full50_4n_m3_a05; PORT=5007; TASKS=(stack_blocks_three beat_block_hammer) ;;
  4:0) MODE=demo_clean; SHARD=full50_4n_m4_c01; PORT=5000; TASKS=(handover_block scan_object place_mouse_pad place_a2b_right click_bell) ;;
  4:1) MODE=demo_clean; SHARD=full50_4n_m4_c02; PORT=5001; TASKS=(blocks_ranking_size place_burger_fries press_stapler lift_pot) ;;
  4:2) MODE=demo_clean; SHARD=full50_4n_m4_c03; PORT=5002; TASKS=(handover_mic place_cans_plasticbox turn_switch adjust_bottle) ;;
  4:3) MODE=demo_randomized; SHARD=full50_4n_m4_a01; PORT=5003; TASKS=(blocks_ranking_rgb place_a2b_right shake_bottle) ;;
  4:4) MODE=demo_randomized; SHARD=full50_4n_m4_a02; PORT=5004; TASKS=(blocks_ranking_size click_alarmclock) ;;
  4:5) MODE=demo_randomized; SHARD=full50_4n_m4_a03; PORT=5005; TASKS=(stack_bowls_three place_shoe) ;;
  4:6) MODE=demo_randomized; SHARD=full50_4n_m4_a04; PORT=5006; TASKS=(move_stapler_pad place_mouse_pad move_playingcard_away) ;;
  4:7) MODE=demo_randomized; SHARD=full50_4n_m4_a05; PORT=5007; TASKS=(hanging_mug) ;;
esac

# 显式保留范围，避免 launcher 从名称末尾只提取 c01/a01 后混用目录。
export EVAL_SHARD="$SHARD"
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
  printf 'MACHINE=M%s\n' "$MACHINE"
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
