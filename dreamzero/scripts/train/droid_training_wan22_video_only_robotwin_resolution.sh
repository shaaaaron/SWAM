#!/bin/bash
# DreamZero DROID video-only full fine-tuning with RoboTwin-aligned video resolution.

export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# ============ USER CONFIGURATION ============
DROID_DATA_ROOT=${DROID_DATA_ROOT:?Set DROID_DATA_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-"../checkpoints/droid_video_only_robotwin_resolution"}

PER_DEVICE_BS=${PER_DEVICE_BS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
TOTAL_GPUS=$((NNODES * NUM_GPUS))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((TOTAL_GPUS * PER_DEVICE_BS))}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"./checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"groot/vla/configs/deepspeed/zero2.json"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (for CLIP only)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    exit 1
fi

# Resolution contract for the 3-view layout:
# - image_resolution_*: per-view resize before composition (W x H = 160 x 128)
# - target_video_*: composed frame passed to VAE/DiT (W x H = 320 x 256)
# - frame_seqlen: Wan2.2 latent patch tokens per frame at 320 x 256 (= 80)
# ============ LAUNCH TRAINING ============
EXPERIMENT_PY="./groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi

TORCHRUN_ARGS=(
    --nproc_per_node "$NUM_GPUS"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)
RUN_CMD=( python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" "$EXPERIMENT_PY" )

"${RUN_CMD[@]}" \
    report_to=tensorboard \
    data=dreamzero/droid_relative_wan22_video_only \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=0 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LEARNING_RATE \
    +training_args.max_grad_norm=1.0 \
    training_args.deepspeed=$DEEPSPEED_CONFIG \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    save_steps=2000 \
    max_steps=20000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=2 \
    image_resolution_width=160 \
    image_resolution_height=128 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=80 \
    action_head_cfg.config.target_video_width=320 \
    action_head_cfg.config.target_video_height=256 \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
