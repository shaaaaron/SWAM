#!/bin/bash
# DreamZero RoboTwin Training Script with Wan2.2-TI2V-5B backbone, without CLIP image context.

export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export NCCL_TIMEOUT=3600000 # robotwin可能存在超时问题

# ============ USER CONFIGURATION ============
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-"./data/robotwin-clean-and-aug-lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_robotwin_wan22_wo_clip_lr1e_4"}

PER_DEVICE_BS=${PER_DEVICE_BS:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_GPUS=${NUM_GPUS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"./checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"groot/vla/configs/deepspeed/zero2.json"}

# Which subsets to include (space-separated)
SUBSETS=${SUBSETS:-"lerobot_robotwin_eef_aug_500 lerobot_robotwin_eef_clean_50"}
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
    echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (kept for checkpoint/config compatibility; CLIP forward is disabled)."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$ROBOTWIN_DATA_ROOT" ]; then
    echo "ERROR: RoboTwin dataset not found at $ROBOTWIN_DATA_ROOT"
    echo "Please set ROBOTWIN_DATA_ROOT to the directory containing lerobot_robotwin_eef_aug_500/ etc."
    exit 1
fi

# ============ AUTO-DISCOVER TASK PATHS ============
echo "Discovering RoboTwin task folders..."
ROBOTWIN_TASK_PATHS=$(python3 -c "
import os, sys
root = '$ROBOTWIN_DATA_ROOT'
subsets = '$SUBSETS'.split()
paths = []
for subset in subsets:
    subset_dir = os.path.join(root, subset)
    if not os.path.isdir(subset_dir):
        print(f'[WARN] Subset not found: {subset_dir}', file=sys.stderr)
        continue
    for task in sorted(os.listdir(subset_dir)):
        task_dir = os.path.join(subset_dir, task)
        if os.path.isfile(os.path.join(task_dir, 'meta', 'info.json')):
            paths.append(task_dir)
if not paths:
    print('ERROR: No valid task folders found!', file=sys.stderr)
    sys.exit(1)
print(f'Found {len(paths)} task folders', file=sys.stderr)
print('[' + ','.join(paths) + ']')
")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to discover task paths"
    exit 1
fi

echo "Task paths: ${ROBOTWIN_TASK_PATHS:0:200}..."

# Resolution contract for the 3-view layout:
# - image_resolution_*: per-view resize before composition (W x H = 160 x 128);
#   this matches resize_cfg_robotwin in base_48_wan_fine_aug_relative.yaml.
# - target_video_*: composed frame passed to VAE/DiT (W x H = 320 x 256)
# - frame_seqlen: Wan2.2 latent patch tokens per frame at 320 x 256 (= 80)
# ============ LAUNCH TRAINING ============
torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=tensorboard \
    data=dreamzero/robotwin \
    wandb_project=dreamzero_robotwin_wan2_2 \
    train_architecture=full \
    num_frames=33 \
    action_horizon=48 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_wo_clip \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=48 \
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
    max_action_dim=16 \
    max_state_dim=16 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=80 \
    action_head_cfg.config.target_video_width=320 \
    action_head_cfg.config.target_video_height=256 \
    robotwin_task_paths="$ROBOTWIN_TASK_PATHS" \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
