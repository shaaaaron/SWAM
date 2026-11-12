#!/usr/bin/env bash
set -euo pipefail

# Run video-only inference over multiple LeRobot episodes.
#
# Examples:
#   MODEL_PATH=../ckpts/0520_video_wo_clip_lr_1e-5/checkpoint-50000 \
#   OUTPUT_DIR=.visual/0520_video_wo_clip_lr_1e-5/checkpoint-50000 \
#   EPISODE_START=0 EPISODE_END=50 \
#   bash scripts/eval/infer_video_only_loop.sh

#   MODEL_PATH=../ckpts/0520_video_wo_clip_lr_2e-5/checkpoint-50000 \
#   OUTPUT_DIR=.visual/0520_video_wo_clip_lr_2e-5/checkpoint-50000 \
#   EPISODE_START=0 EPISODE_END=50 \
#   bash scripts/eval/infer_video_only_loop.sh

#   MODEL_PATH=../ckpts/0520_video_lr_1e-5/checkpoint-50000 \
#   OUTPUT_DIR=.visual/0520_video_lr_1e-5/checkpoint-50000 \
#   EPISODE_START=0 EPISODE_END=50 \
#   bash scripts/eval/infer_video_only_loop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT=${DREAMZERO_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
INFER_SCRIPT="$DREAMZERO_ROOT/scripts/eval/infer_video_only_simple.py"

MODEL_PATH=${MODEL_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT}
TOKENIZER=${TOKENIZER:?Set TOKENIZER}

EMBODIMENT_TAG=${EMBODIMENT_TAG:-oxe_droid}
MODALITY_MAP=${MODALITY_MAP:-oxe_droid}
EVAL_VIEWS=${EVAL_VIEWS:-three}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_CMD=${PYTHON_CMD:-python}

IMAGE_HEIGHT=${IMAGE_HEIGHT:-160}
IMAGE_WIDTH=${IMAGE_WIDTH:-320}
FPS=${FPS:-5}

if [ -z "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT is required."
    echo "Example: DATA_ROOT=/path/to/lerobot EPISODES=\"0 1 2\" bash $0"
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is required."
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR is required."
    exit 1
fi

if [ ! -f "$INFER_SCRIPT" ]; then
    echo "ERROR: inference script not found: $INFER_SCRIPT"
    exit 1
fi

if [ -z "${EPISODES:-}" ]; then
    EPISODE_START=${EPISODE_START:-0}
    EPISODE_END=${EPISODE_END:-0}
    if [ "$EPISODE_END" -lt "$EPISODE_START" ]; then
        echo "ERROR: EPISODE_END ($EPISODE_END) must be >= EPISODE_START ($EPISODE_START)."
        exit 1
    fi
    mapfile -t EPISODE_LIST < <(seq "$EPISODE_START" "$EPISODE_END")
else
    read -r -a EPISODE_LIST <<< "$EPISODES"
fi

echo "Running video-only inference loop"
echo "  model_path=$MODEL_PATH"
echo "  output_dir=$OUTPUT_DIR"
echo "  data_root=$DATA_ROOT"
echo "  episodes=${EPISODE_LIST[*]}"
echo "  cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

for EPISODE_IDX in "${EPISODE_LIST[@]}"; do
    echo
    echo "===== Episode $EPISODE_IDX ====="

    CMD=(
        "$PYTHON_CMD" "$INFER_SCRIPT"
        --model_path "$MODEL_PATH"
        --output_dir "$OUTPUT_DIR"
        --embodiment_tag "$EMBODIMENT_TAG"
        --modality_map "$MODALITY_MAP"
        --eval_views "$EVAL_VIEWS"
        --data_root "$DATA_ROOT"
        --episode_idx "$EPISODE_IDX"
        --image_height "$IMAGE_HEIGHT"
        --image_width "$IMAGE_WIDTH"
        --fps "$FPS"
    )

    if [ -n "$TOKENIZER" ]; then
        CMD+=(--tokenizer_path "$TOKENIZER")
    fi
    if [ -n "${FRAMES_PER_CHUNK:-}" ]; then
        CMD+=(--frames_per_chunk "$FRAMES_PER_CHUNK")
    fi
    if [ -n "${MAX_CHUNKS:-}" ]; then
        CMD+=(--max_chunks "$MAX_CHUNKS")
    fi
    if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then
        CMD+=(--num_inference_steps "$NUM_INFERENCE_STEPS")
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${CMD[@]}"
done

echo
echo "Done. Videos are saved under: $OUTPUT_DIR"
